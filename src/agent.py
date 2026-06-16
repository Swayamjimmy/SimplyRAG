import os
from typing import TypedDict, Literal, Annotated

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

from langchain_groq import ChatGroq
from src.embeddings import get_collection
from langchain_core.messages import HumanMessage, AIMessage

from src.pipeline import RerankedRAGPipeline

load_dotenv()

# Load once
PIPELINE = RerankedRAGPipeline()
collection = get_collection()


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    intent: str
    retrieved_documents: list
    citations: list
    current_document_context: str


class RouteIntent(BaseModel):
    intent: Literal[
        "retrieval",
        "summarization",
        "comparison",
        "table_analysis",
    ] = Field(
        description="Intent classification"
    )


llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0,
)


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


def router(state: AgentState):

    question = build_conversational_query(
        state["messages"]
    )

    structured_llm = llm.with_structured_output(
        RouteIntent
    )

    prompt = f"""
Classify the following conversation into exactly one intent.

retrieval:
    factual lookup

summarization:
    summarize content

comparison:
    compare concepts, methods, models or papers

table_analysis:
    analyze metrics, statistics or tables

Conversation:

{question}
"""

    result = structured_llm.invoke(prompt)

    return {
        "intent": result.intent
    }


def run_rag(query: str):
    return PIPELINE.query(query)


def handle_query(
    state: AgentState,
    instruction: str = ""
):

    latest_question = (
        state["messages"][-1].content
    )

    history = build_conversational_query(
        state["messages"][:-1]
    )

    query = f"""
Conversation History:

{history}

Current User Question:

{latest_question}

{instruction}
"""

    result = run_rag(query)

    return {
        "retrieved_documents": result["sources"],
        "current_document_context": result["answer"],
        "messages": [
            AIMessage(
                content=result["answer"]
            )
        ],
    }


def retrieval_node(state: AgentState):
    return handle_query(state)


def summarization_node(state: AgentState):
    return handle_query(
        state,
        instruction="Provide a detailed summary."
    )


def comparison_node(state: AgentState):
    return handle_query(
        state,
        instruction=(
            "Compare the relevant concepts, "
            "methods, papers, or approaches."
        ),
    )


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
        "messages": [
            AIMessage(content=response.content)
        ]
    }


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


def rewrite_question(state: AgentState):

    query = build_conversational_query(
        state["messages"]
    )

    prompt = f"""
Rewrite this query to improve retrieval quality.

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


def route_to_handler(
    state: AgentState,
):
    return state["intent"]


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

workflow.add_edge(
    START,
    "router",
)

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

workflow.add_conditional_edges(
    "retrieval",
    grade_documents,
    {
        "generate": END,
        "rewrite": "rewrite_question",
    },
)

workflow.add_conditional_edges(
    "summarization",
    grade_documents,
    {
        "generate": END,
        "rewrite": "rewrite_question",
    },
)

workflow.add_conditional_edges(
    "comparison",
    grade_documents,
    {
        "generate": END,
        "rewrite": "rewrite_question",
    },
)

workflow.add_conditional_edges(
    "table_analysis",
    grade_documents,
    {
        "generate": END,
        "rewrite": "rewrite_question",
    },
)

workflow.add_edge(
    "rewrite_question",
    "router",
)

memory = MemorySaver()

graph = workflow.compile(
    checkpointer=memory
)


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
                    print(
                        "\nAssistant:"
                    )
                    print(
                        latest.content
                    )

def build_graph():
    return graph