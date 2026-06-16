import gradio as gr
import pandas as pd
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from src.agent import build_graph
from src.embeddings import store_chunks
from src.ingest import ingest_pdf

# Load environment variables
load_dotenv()

# Build the agent graph
graph = build_graph()

# Intent emoji mapping for router decisions
INTENT_EMOJIS = {
    "retrieval": "\U0001F50D",      # magnifying glass
    "summarization": "\U0001F4DD",  # memo
    "comparison": "\u2696\uFE0F",    # balance scale
    "table_analysis": "\U0001F4CA"  # bar chart
}


def process_upload(file):

    if file is None:
        return "No file uploaded."

    chunks = ingest_pdf(file)

    store_chunks(chunks)

    return (
        f"Indexed {len(chunks)} chunks "
        f"from {file.split('/')[-1]}"
    )

def chat(message, history):
    """Chat function for Gradio ChatInterface with streaming."""
    config = {"configurable": {"thread_id": "gradio-session"}}
    full_response = ""
    intent = ""
    citations = []

    # Stream the agent response
    for event in graph.stream(
            {
        "messages": [
            HumanMessage(content=message)
        ]
    },
        config=config,
        stream_mode="updates"
    ):
        for node_name, node_output in event.items():
            # Capture intent from router
            if "intent" in node_output:
                intent = node_output["intent"]
            # Stream message content
            if "messages" in node_output:
                for msg in node_output["messages"]:
                    if hasattr(msg, "content") and msg.content:
                        full_response += msg.content
                        # Prefix with intent emoji on first token
                        emoji = INTENT_EMOJIS.get(intent, "")
                        prefix = f"{emoji} Mode: {intent.title()}\n\n" if intent else ""
                        yield prefix + full_response
            # Capture citations
            if "citations" in node_output:
                citations = node_output["citations"]

    # Append citation cards at the end
    if citations:
        citation_text = "\n\n---\n**Citations:**\n"
        for i, cite in enumerate(citations, 1):
            source = cite.get("source", "Unknown")
            page = cite.get("page_number", "?")
            passage = cite.get("passage", "")[:150]
            verified = "Verified" if cite.get("verified", False) else "Unverified"
            citation_text += f"\n[{i}] **{source}** (p.{page}) - {verified}\n> {passage}...\n"
        emoji = INTENT_EMOJIS.get(intent, "")
        prefix = f"{emoji} Mode: {intent.title()}\n\n" if intent else ""
        yield prefix + full_response + citation_text

def load_benchmark_data():
    """Load benchmark report as a DataFrame for the metrics tab."""
    try:
        data = {
            "Pipeline": ["Basic RAG", "Hybrid Search", "Hybrid + Reranking", "Agentic (Full)"],
            "Faithfulness": ["-", "-", "-", "-"],
            "Context Precision": ["-", "-", "-", "-"],
            "Answer Relevancy": ["-", "-", "-", "-"],
            "Citation Accuracy": ["-", "-", "-", "-"]
        }
        # Try to load actual results from benchmark report
        if pd.io.common.file_exists("evals/benchmark_report.md"):
            # Parse markdown table if it exists
            pass
        return pd.DataFrame(data)
    except Exception:
        return pd.DataFrame()


# --- Build the Gradio interface ---
with gr.Blocks(title="Document Intelligence Agent") as demo:
    gr.Markdown("# Document Intelligence Agent")
    gr.Markdown("Upload PDFs and ask questions with intelligent routing, citations, and streaming.")

    with gr.Tab("Chat"):
        with gr.Row():
            # Sidebar for file upload
            with gr.Column(scale=1):
                gr.Markdown("### Upload Documents")
                file_upload = gr.File(
                    label="Upload PDF",
                    file_types=[".pdf"],
                    type="filepath"
                )
                upload_status = gr.Textbox(
                    label="Status",
                    interactive=False
                )
                file_upload.upload(
                    fn=process_upload,
                    inputs=file_upload,
                    outputs=upload_status
                )

            # Main chat area
            with gr.Column(scale=3):
                gr.ChatInterface(
                    fn=chat,
                    title="Ask Your Documents",
                    examples=[
                        "What methods were used in this paper?",
                        "Summarize the key findings.",
                        "Compare the approaches in the uploaded documents.",
                        "What were the accuracy scores?"
                    ]
                )

    with gr.Tab("Metrics"):
        gr.Markdown("### RAG Pipeline Benchmark Results")
        benchmark_table = gr.DataFrame(
            value=load_benchmark_data(),
            label="Pipeline Comparison"
        )
        gr.Markdown("*Run the /evaluate endpoint or re-run benchmarks to populate with real scores.*")


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)