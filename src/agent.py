import os
from typing import TypedDict, Literal, Annotated

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage

from src.embeddings import get_collection
from src.pipeline import RerankedRAGPipeline, CitedRAGPipeline


# Load environment variables
load_dotenv()


# ============================================================
# INITIALIZE RAG PIPELINE
# ============================================================

# Bootstrap the base pipeline
BASE_PIPELINE = RerankedRAGPipeline()

# Wrap the base components inside the cited pipeline
PIPELINE = CitedRAGPipeline(
    reranker=BASE_PIPELINE.reranker,
    hybrid_retriever=BASE_PIPELINE.retriever
)

# Load Chroma collection
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
    ] = Field(
        description="Intent classification"
    )


# ============================================================
# LLM
# ============================================================

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0,
)


# ============================================================
# CONVERSATION HELPER
# ============================================================

def build_conversational_query(messages):
    history = []

    for msg in messages[-6:]:

        if isinstance(msg, HumanMessage):
            history.append(
                f"User: {msg.content}"
            )

        elif isinstance(msg, AIMessage):
            history.append(
                f"Assistant: {msg.content}"
            )

    return "\n".join(history)


# ============================================================
# ROUTER
# ============================================================

def router(state: AgentState):

    question = build_conversational_query(
        state["messages"]
    )

    # IMPORTANT:
    # Use JSON mode instead of tool/function calling.
    structured_llm = llm.with_structured_output(
        RouteIntent,
        method="json_mode"
    )

    prompt = f"""
You are an intent classifier.

Classify the following conversation into exactly ONE of these intents:

- retrieval
- summarization
- comparison
- table_analysis

Definitions:

retrieval:
Use when the user is asking for factual information,
explanations, concepts, definitions, or information that
should be retrieved from the document knowledge base.

summarization:
Use when the user explicitly asks to summarize content,
a document, paper, section, or information.

comparison:
Use when the user asks to compare concepts, methods,
models, approaches, techniques, or papers.

table_analysis:
Use when the user asks about metrics, statistics,
tables, figures, charts, or numerical comparisons.

Return ONLY valid JSON.

Required format:

{{
    "intent": "retrieval"
}}

Conversation:

{question}
"""

    result = structured_llm.invoke(prompt)

    return {
        "intent": result.intent
    }


# ============================================================
# RAG PIPELINE
# ============================================================

def run_rag(query: str):
    return PIPELINE.query(query)


def handle_query(
    state: AgentState,
    instruction: str = ""
):

    latest_question = state["messages"][-1].content

    # Only pass the raw/latest question to the retriever
    search_query = latest_question

    if instruction:
        search_query = (
            f"{latest_question} {instruction}"
        )

    # Run the cited RAG pipeline
    result = run_rag(search_query)

    # Convert Pydantic Citation objects to dictionaries
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
        "messages": [
            AIMessage(
                content=result.answer
            )
        ],
    }


# ============================================================
# RETRIEVAL NODE
# ============================================================

def retrieval_node(state: AgentState):
    return handle_query(state)


# ============================================================
# SUMMARIZATION NODE
# ============================================================

def summarization_node(state: AgentState):

    return handle_query(
        state,
        instruction="Provide a detailed summary."
    )


# ============================================================
# COMPARISON NODE
# ============================================================

def comparison_node(state: AgentState):

    return handle_query(
        state,
        instruction=(
            "Compare the relevant concepts, "
            "methods, papers, or approaches."
        ),
    )


# ============================================================
# TABLE ANALYSIS NODE
# ============================================================

def table_analysis_node(state: AgentState):

    query = build_conversational_query(
        state["messages"]
    )

    multimodal_results = collection.query(
        query_texts=[query],
        n_results=5,
        where={
            "content_type": {
                "$in": ["image", "table"]
            }
        }
    )

    docs = multimodal_results["documents"][0]

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
        "messages": [
            AIMessage(
                content=response.content
            )
        ]
    }


# ============================================================
# DOCUMENT GRADER
# ============================================================

def grade_documents(
    state: AgentState,
) -> Literal["generate", "rewrite"]:

    docs = state.get(
        "retrieved_documents",
        []
    )

    if not docs:
        return "rewrite"

    return "generate"


# ============================================================
# QUERY REWRITER
# ============================================================

def rewrite_question(state: AgentState):

    query = build_conversational_query(
        state["messages"]
    )

    prompt = f"""
Rewrite the following user query to improve
document retrieval quality.

Return only the rewritten query.

Conversation:

{query}
"""

    response = llm.invoke(prompt)

    return {
        "messages": [
            HumanMessage(
                content=response.content
            )
        ]
    }


# ============================================================
# ROUTING FUNCTION
# ============================================================

def route_to_handler(
    state: AgentState,
):
    return state["intent"]


# ============================================================
# BUILD LANGGRAPH
# ============================================================

workflow = StateGraph(
    AgentState
)


workflow.add_node(
    "router",
    router,
)

workflow.add_node(
    "retrieval",
    retrieval_node,
)

workflow.add_node(
    "summarization",
    summarization_node,
)

workflow.add_node(
    "comparison",
    comparison_node,
)

workflow.add_node(
    "table_analysis",
    table_analysis_node,
)

workflow.add_node(
    "rewrite_question",
    rewrite_question,
)


# Start with router
workflow.add_edge(
    START,
    "router",
)


# Route based on detected intent
workflow.add_conditional_edges(
    "router",
    route_to_handler,
    {
        "retrieval": "retrieval",
        "summarization": "summarization",
        "comparison": "comparison",
        "table_analysis": "table_analysis",
    },
)


# Retrieval flow
workflow.add_conditional_edges(
    "retrieval",
    grade_documents,
    {
        "generate": END,
        "rewrite": "rewrite_question",
    },
)


# Summarization flow
workflow.add_conditional_edges(
    "summarization",
    grade_documents,
    {
        "generate": END,
        "rewrite": "rewrite_question",
    },
)


# Comparison flow
workflow.add_conditional_edges(
    "comparison",
    grade_documents,
    {
        "generate": END,
        "rewrite": "rewrite_question",
    },
)


# Table analysis flow
workflow.add_conditional_edges(
    "table_analysis",
    grade_documents,
    {
        "generate": END,
        "rewrite": "rewrite_question",
    },
)


# Retry retrieval with rewritten query
workflow.add_edge(
    "rewrite_question",
    "router",
)


# ============================================================
# MEMORY
# ============================================================

memory = MemorySaver()


# Compile graph
graph = workflow.compile(
    checkpointer=memory
)


# ============================================================
# STREAM QUERY
# ============================================================

def stream_query(
    question: str,
    thread_id: str = "default",
):

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    input_state = {
        "messages": [
            HumanMessage(
                content=question
            )
        ]
    }

    for chunk in graph.stream(
        input_state,
        config,
        stream_mode="updates",
    ):

        for value in chunk.values():

            if (
                isinstance(value, dict)
                and "messages" in value
            ):

                latest = value["messages"][-1]

                if (
                    hasattr(latest, "content")
                    and latest.content
                ):

                    print("\nAssistant:")

                    print(
                        latest.content
                    )


# ============================================================
# EXPORT GRAPH
# ============================================================

def build_graph():
    return graph