import os
import json
import logging
import tempfile
from typing import AsyncGenerator

from fastapi import FastAPI, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage

from src.ingest import ingest_pdf
from src.agent import build_graph

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="RAG System API")

graph = build_graph()


class QueryRequest(BaseModel):
    question: str
    conversation_id: str = "default"


class QueryResponse(BaseModel):
    answer: str
    citations: list[dict]
    intent: str


@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.post("/ingest")
async def ingest_document(file: UploadFile):

    logger.info(f"Ingesting file: {file.filename}")

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ) as tmp:

        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:

        chunks = ingest_pdf(
            os.path.dirname(tmp_path)
        )

        return {
            "filename": file.filename,
            "chunks_indexed": len(chunks)
        }

    finally:

        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.post("/query")
async def query_documents(
    request: QueryRequest
):

    logger.info(
        f"Query: {request.question}"
    )

    async def generate() -> AsyncGenerator[str, None]:

        config = {
            "configurable": {
                "thread_id":
                request.conversation_id
            }
        }

        intent = ""
        citations = []

        for event in graph.stream(
            {
                "messages": [
                    HumanMessage(
                        content=request.question
                    )
                ]
            },
            config=config,
            stream_mode="updates",
        ):

            for node_name, node_output in event.items():

                if (
                    isinstance(node_output, dict)
                    and "intent" in node_output
                ):
                    intent = node_output["intent"]

                if (
                    isinstance(node_output, dict)
                    and "citations" in node_output
                ):
                    citations = (
                        node_output["citations"]
                    )

                if (
                    isinstance(node_output, dict)
                    and "messages" in node_output
                ):

                    messages = (
                        node_output["messages"]
                    )

                    if not messages:
                        continue

                    latest = messages[-1]

                    if (
                        hasattr(
                            latest,
                            "content"
                        )
                        and latest.content
                    ):
                        yield latest.content

        yield (
            "\n\n"
            + json.dumps(
                {
                    "_meta": {
                        "intent": intent,
                        "citations": citations,
                    }
                }
            )
        )

    return StreamingResponse(
        generate(),
        media_type="text/plain",
    )


@app.websocket("/ws/query")
async def websocket_query(
    websocket: WebSocket
):

    await websocket.accept()

    logger.info(
        "WebSocket connected"
    )

    try:

        while True:

            data = await websocket.receive_text()

            request_data = json.loads(
                data
            )

            question = (
                request_data.get(
                    "question",
                    ""
                )
            )

            conversation_id = (
                request_data.get(
                    "conversation_id",
                    "default"
                )
            )

            config = {
                "configurable": {
                    "thread_id":
                    conversation_id
                }
            }

            for event in graph.stream(
                {
                    "messages": [
                        HumanMessage(
                            content=question
                        )
                    ]
                },
                config=config,
                stream_mode="updates",
            ):

                for (
                    node_name,
                    node_output,
                ) in event.items():

                    if (
                        isinstance(
                            node_output,
                            dict
                        )
                        and "messages"
                        in node_output
                    ):

                        messages = (
                            node_output[
                                "messages"
                            ]
                        )

                        if not messages:
                            continue

                        latest = messages[-1]

                        if (
                            hasattr(
                                latest,
                                "content"
                            )
                            and latest.content
                        ):
                            await websocket.send_text(
                                latest.content
                            )

            await websocket.send_text(
                "[END]"
            )

    except WebSocketDisconnect:

        logger.info(
            "WebSocket disconnected"
        )