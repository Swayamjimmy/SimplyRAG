import os
from typing import TypedDict, Literal, Annotated
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

from src.llm import llm
from langchain_core.messages import HumanMessage, AIMessage

from src.embeddings import get_collection
from src.pipeline import RerankedRAGPipeline, CitedRAGPipeline

load_dotenv()

# ============================================================
# INITIALIZE RAG PIPELINE
# ============================================================
BASE_PIPELINE = RerankedRAGPipeline()
PIPELINE = CitedRAGPipeline(
    reranker=BASE_PIPELINE.reranker,
    hybrid_retriever=BASE_PIPELINE.retriever
)
collection = get_collection()

# ============================================================
# AGENT STATE
# ============================================================
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    intent: str
    retrieved_documents: list
    citations: list
    current_document_context: str

# ============================================================
# ROUTER OUTPUT SCHEMA
# ============================================================
class RouteIntent(BaseModel):
    intent: Literal[
        "retrieval",
        "summarization",
        "comparison",
        "table_analysis",
    ] = Field(description="Intent classification")

def build_conversational_query(messages):
    history = []
    # Optimized: Limit context to the last 4 messages (2 conversational turns)
    for msg in messages[-4:]:
        if isinstance(msg, HumanMessage):
            history.append(f"User: {msg.content}")
        elif isinstance(msg, AIMessage):
            history.append(f"Assistant: {msg.content}")
    return "\n".join(history)

# ============================================================
# ROUTER
# ============================================================
def router(state: AgentState):
    question = build_conversational_query(state["messages"])
    structured_llm = llm.with_structured_output(RouteIntent, method="json_mode")
    prompt = f"""
You are an intent classifier.
Classify the following conversation into exactly ONE of these intents:
- retrieval
- summarization
- comparison
- table_analysis

Return ONLY valid JSON.
Required format:
{{
    "intent": "retrieval"
}}

Conversation:
{question}
"""
    result = structured_llm.invoke(prompt)
    return {"intent": result.intent}

# ============================================================
# RAG PIPELINE EXECUTOR
# ============================================================
def handle_query(state: AgentState, instruction: str = ""):
    latest_question = state["messages"][-1].content
    search_query = f"{latest_question} {instruction}" if instruction else latest_question

    # Fast Mode: verify=False disabled for high-speed responsiveness
    result = PIPELINE.query(search_query, verify=False)

    citations_dicts = [
        {
            "source": c.source_doc,
            "page_number": c.page_number,
            "passage": c.passage,
            "verified": c.verified
        }
        for c in result.citations
    ]

    return {
        "retrieved_documents": citations_dicts,
        "current_document_context": result.answer,
        "citations": citations_dicts,
        "messages": [AIMessage(content=result.answer)],
    }

def retrieval_node(state: AgentState):
    return handle_query(state)

def summarization_node(state: AgentState):
    return handle_query(state, instruction="Provide a detailed summary.")

def comparison_node(state: AgentState):
    return handle_query(state, instruction="Compare the relevant concepts, methods, papers, or approaches.")

def table_analysis_node(state: AgentState):
    query = build_conversational_query(state["messages"])
    multimodal_results = collection.query(
        query_texts=[query],
        n_results=5,
        where={"content_type": {"$in": ["image", "table"]}}
    )

    docs = multimodal_results["documents"][0] if multimodal_results["documents"] else []
    context = "\n\n".join(docs)

    prompt = f"""
Analyze the following tables, figures and numerical data.
Question:
{query}

Context:
{context}
"""
    response = llm.invoke(prompt)
    return {
        "retrieved_documents": docs,
        "current_document_context": response.content,
        "citations": [],
        "messages": [AIMessage(content=response.content)]
    }

# ============================================================
# QUERY GRADING / REWRITER
# ============================================================
def grade_documents(state: AgentState) -> Literal["generate", "rewrite"]:
    docs = state.get("retrieved_documents", [])
    if not docs:
        return "rewrite"
    return "generate"

def rewrite_question(state: AgentState):
    query = build_conversational_query(state["messages"])
    prompt = f"Rewrite the following user query to improve document retrieval quality. Return only the rewritten query.\n\nConversation:\n{query}"
    response = llm.invoke(prompt)
    return {"messages": [HumanMessage(content=response.content)]}

def route_to_handler(state: AgentState):
    return state["intent"]

# ============================================================
# BUILD LANGGRAPH
# ============================================================
workflow = StateGraph(AgentState)

workflow.add_node("router", router)
workflow.add_node("retrieval", retrieval_node)
workflow.add_node("summarization", summarization_node)
workflow.add_node("comparison", comparison_node)
workflow.add_node("table_analysis", table_analysis_node)
workflow.add_node("rewrite_question", rewrite_question)

workflow.add_edge(START, "router")
workflow.add_conditional_edges(
    "router",
    route_to_handler,
    {"retrieval": "retrieval", "summarization": "summarization", "comparison": "comparison", "table_analysis": "table_analysis"}
)

for node in ["retrieval", "summarization", "comparison", "table_analysis"]:
    workflow.add_conditional_edges(
        node,
        grade_documents,
        {"generate": END, "rewrite": "rewrite_question"}
    )

workflow.add_edge("rewrite_question", "router")

memory = MemorySaver()
graph = workflow.compile(checkpointer=memory)

def build_graph():
    return graph