import os
from typing import TypedDict, Literal
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

from src.pipeline import RerankedRAGPipeline
from src.pipeline import CitedRAGPipeline
from src.hybrid_retriever import HybridRetriever
from src.reranker import CrossEncoderReranker

load_dotenv()

reranker = CrossEncoderReranker()

# Define the state that flows between all nodes in the graph
class AgentState(TypedDict):
    messages: list
    intent: str
    retrieved_documents: list
    citations: list
    current_document_context: str

# Structured output schema for intent classification
class RouteIntent(BaseModel):
    """The classified intent of the user's question."""
    intent: str = Field(
        description="One of: retrieval, summarization, comparison, table_analysis"
    )

# Initialize the Groq LLM for all agent nodes
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY")
)

# Router node: classifies user intent into one of four categories
def router(state: AgentState) -> dict:
    """Classify the user's question into a specific intent."""
    messages = state["messages"]
    last_message = messages[-1].content if hasattr(messages[-1], 'content') else str(messages[-1])

    # Use structured output to guarantee a valid intent classification
    structured_llm = llm.with_structured_output(RouteIntent)

    route_prompt = f"""Classify the following question into exactly one intent:
    - retrieval: Factual Q&A requiring specific passages from documents
    - summarization: Requests to summarize sections or entire documents
    - comparison: Questions comparing information across multiple documents
    - table_analysis: Questions about tabular data, numbers, or statistics

    Question: {last_message}"""

    result = structured_llm.invoke(route_prompt)
    return {"intent": result.intent}

# Retrieval node: handles factual Q&A with hybrid search + reranking + citations
def retrieval_node(state: AgentState) -> dict:
    """Handle factual Q&A using the full hybrid + reranking + citation pipeline."""
    messages = state["messages"]
    query = messages[-1].content if hasattr(messages[-1], 'content') else str(messages[-1])

    # Use the reranked pipeline for precise document retrieval
    pipeline = RerankedRAGPipeline()
    reranked_docs = pipeline.retrieve(query)

    # Generate a cited response from the reranked documents
    cited_pipeline = CitedRAGPipeline()
    response = cited_pipeline.generate(query, reranked_docs)

    return {
        "retrieved_documents": reranked_docs,
        "citations": response.citations,
        "current_document_context": response.answer
    }

# Summarization node: retrieves broader context and produces summaries
def summarization_node(state: AgentState) -> dict:
    """Retrieve broader context and produce summaries with section references."""
    messages = state["messages"]
    query = messages[-1].content if hasattr(messages[-1], 'content') else str(messages[-1])

    # Retrieve more chunks for comprehensive summarization
    hybrid = HybridRetriever()
    docs = hybrid.retrieve(query, k=10)

    context = "\n\n".join([doc["content"] for doc in docs])

    summary_prompt = f"""Summarize the following content in response to the user's request.
    Include section references (e.g., [Page X, Section Y]) for each major point.

    User request: {query}

    Content:
    {context}"""

    response = llm.invoke(summary_prompt)

    return {
        "retrieved_documents": docs,
        "current_document_context": response.content
    }

# Comparison node: retrieves from multiple documents and produces structured comparison
def comparison_node(state: AgentState) -> dict:
    """Retrieve from multiple documents and produce a structured comparison."""
    messages = state["messages"]
    query = messages[-1].content if hasattr(messages[-1], 'content') else str(messages[-1])

    # Retrieve a broad set of chunks from multiple sources
    hybrid = HybridRetriever()
    docs = hybrid.retrieve(query, k=15)

    # Group documents by source file
    sources = {}
    for doc in docs:
        source = doc.get("metadata", {}).get("source", "unknown")
        if source not in sources:
            sources[source] = []
        sources[source].append(doc["content"])

    comparison_prompt = f"""Compare the information from these different sources.
    Structure your response with:
    - Similarities
    - Differences
    - Include citations [Source: filename, Page X] for each point.

    Question: {query}

    Sources:
    {chr(10).join([f"Source: {s}" + chr(10) + chr(10).join(chunks) for s, chunks in sources.items()])}"""

    response = llm.invoke(comparison_prompt)

    return {
        "retrieved_documents": docs,
        "current_document_context": response.content
    }

# Table analysis node: focuses on extracting and reasoning over numerical data
def table_analysis_node(state: AgentState) -> dict:
    """Extract and reason over tabular and numerical data from documents."""
    messages = state["messages"]
    query = messages[-1].content if hasattr(messages[-1], 'content') else str(messages[-1])

    # Retrieve documents likely containing tabular/numerical content
    hybrid = HybridRetriever()
    docs = hybrid.retrieve(query, k=10)

    table_prompt = f"""Analyze the following content to answer the user's question about data, statistics, or tables.
    Extract relevant numbers, present them clearly, and explain any trends or patterns.
    Cite sources for each data point.

    Question: {query}

    Content:
    {chr(10).join([doc["content"] for doc in docs])}"""

    response = llm.invoke(table_prompt)

    return {
        "retrieved_documents": docs,
        "current_document_context": response.content
    }

# Structured output schema for grading document relevance
class GradeDocuments(BaseModel):
    """Grade whether retrieved documents are relevant to the question."""
    binary_score: str = Field(
        description="Relevance score: 'yes' if relevant, or 'no' if not relevant"
    )

# Conditional edge: assess if retrieved documents are relevant
def grade_documents(state: AgentState) -> Literal["generate", "rewrite"]:
    """Determine whether retrieved documents are relevant to the query."""
    docs = state["retrieved_documents"]
    messages = state["messages"]
    query = messages[-1].content if hasattr(messages[-1], 'content') else str(messages[-1])

    if not docs:
        return "rewrite"

    # Check relevance of the top retrieved document
    doc_content = docs[0]["content"] if isinstance(docs[0], dict) else str(docs[0])

    structured_llm = llm.with_structured_output(GradeDocuments)

    grade_prompt = f"""You are a grader assessing relevance of a retrieved document to a user question.

    Retrieved document: {doc_content}

    User question: {query}

    Give a binary score 'yes' or 'no' to indicate whether the document is relevant to the question."""

    result = structured_llm.invoke(grade_prompt)

    if result.binary_score == "yes":
        return "generate"
    return "rewrite"

# Rewrite node: reformulates the query when documents are irrelevant
def rewrite_question(state: AgentState) -> dict:
    """Reformulate the query to improve retrieval results."""
    messages = state["messages"]
    query = messages[-1].content if hasattr(messages[-1], 'content') else str(messages[-1])

    rewrite_prompt = f"""The original question did not retrieve relevant documents.
    Reformulate it to be more specific and searchable.

    Original question: {query}

    Rewritten question:"""

    response = llm.invoke(rewrite_prompt)

    # Replace the last message with the rewritten query
    return {"messages": [HumanMessage(content=response.content)]}

# Route to the correct handler based on classified intent
def route_to_handler(state: AgentState) -> str:
    """Direct to the appropriate handler based on classified intent."""
    return state["intent"]

# Assemble the full LangGraph state graph
workflow = StateGraph(AgentState)

# Add all nodes to the graph
workflow.add_node("router", router)
workflow.add_node("retrieval", retrieval_node)
workflow.add_node("summarization", summarization_node)
workflow.add_node("comparison", comparison_node)
workflow.add_node("table_analysis", table_analysis_node)
workflow.add_node("rewrite_question", rewrite_question)

# Entry point: every query starts at the router
workflow.add_edge(START, "router")

# Router dispatches to the correct handler based on intent
workflow.add_conditional_edges(
    "router",
    route_to_handler,
    {
        "retrieval": "retrieval",
        "summarization": "summarization",
        "comparison": "comparison",
        "table_analysis": "table_analysis",
    }
)

# After each handler, grade document relevance
workflow.add_conditional_edges("retrieval", grade_documents, {"generate": END, "rewrite": "rewrite_question"})
workflow.add_conditional_edges("summarization", grade_documents, {"generate": END, "rewrite": "rewrite_question"})
workflow.add_conditional_edges("comparison", grade_documents, {"generate": END, "rewrite": "rewrite_question"})
workflow.add_conditional_edges("table_analysis", grade_documents, {"generate": END, "rewrite": "rewrite_question"})

# After rewriting, loop back to the router for re-classification
workflow.add_edge("rewrite_question", "router")

# Compile the graph into a runnable agent
graph = workflow.compile()