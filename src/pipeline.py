# src/pipeline.py

import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

from src.retriever import BasicRetriever
from src.hybrid_retriever import HybridRetriever
from src.reranker import CrossEncoderReranker
from src.ingest import ingest_pdf
from src.embeddings import get_collection, get_embedding_function

from src.citations import (
    Citation,
    CitationResponse,
    extract_citations,
    verify_citation,
    compute_citation_accuracy,
)


load_dotenv()


# ============================================================
# Gemini Configuration
# ============================================================

GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.8-flash"
)


def get_gemini_llm():
    """
    Create a Gemini chat model for text generation.

    GEMINI_API_KEY must be present in the environment.
    """

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY is not set. "
            "Add it to your .env file or deployment secrets."
        )

    return ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=api_key,
        temperature=0,
        max_retries=2,
    )


# Shared Gemini instance for this module.
gemini_llm = get_gemini_llm()


# ============================================================
# Basic RAG
# ============================================================

class BasicRAGPipeline:
    """
    Basic RAG pipeline:

        Query
          ↓
        Vector Retrieval
          ↓
        Context Formatting
          ↓
        Gemini
          ↓
        Answer
    """

    def __init__(self):
        self.retriever = BasicRetriever(top_k=5)
        self.llm = gemini_llm

    def format_prompt(self, question, context_chunks):
        """
        Build a grounded prompt using retrieved context.
        """

        context = "\n\n".join(
            [
                (
                    f"Source: {chunk['metadata'].get('source', 'Unknown')}, "
                    f"Page: {chunk['metadata'].get('page', chunk['metadata'].get('page_num', 0))}\n"
                    f"{chunk['text']}"
                )
                for chunk in context_chunks
            ]
        )

        prompt = f"""
Answer the following question based ONLY on the provided context.

If the context does not contain enough information to answer the
question, clearly say that the information is not available in
the provided context.

Context:
{context}

Question:
{question}

Answer:
"""

        return prompt.strip()

    def query(self, question):
        """
        Run the full basic RAG pipeline.
        """

        chunks = self.retriever.retrieve(question)

        prompt = self.format_prompt(
            question,
            chunks
        )

        response = self.llm.invoke(prompt)

        return {
            "answer": response.content,
            "retrieved_chunks": chunks,
        }


# ============================================================
# Hybrid RAG
# ============================================================

class HybridRAGPipeline:
    """
    RAG pipeline using:

        BM25
          +
        Vector Search
          ↓
        Hybrid Retrieval
          ↓
        Gemini
          ↓
        Answer
    """

    def __init__(
        self,
        chunks,
        chroma_collection,
        embedding_function,
        llm_client=None,
    ):
        self.retriever = HybridRetriever(
            chunks,
            chroma_collection,
            embedding_function
        )

        # Keep the optional llm_client argument for backward
        # compatibility with existing benchmark/test code.
        self.llm = llm_client or gemini_llm

    def query(self, question):
        """
        Retrieve relevant chunks with hybrid search,
        then generate an answer with Gemini.
        """

        retrieved = self.retriever.retrieve(
            question,
            k=5
        )

        context = "\n\n".join(
            [
                chunk["text"]
                for chunk in retrieved
            ]
        )

        prompt = f"""
Answer the following question based ONLY on the provided context.

If the context does not contain enough information to answer the
question, clearly say that the information is not available in
the provided context.

Context:
{context}

Question:
{question}

Answer:
"""

        response = self.llm.invoke(
            prompt.strip()
        )

        return {
            "answer": response.content,
            "retrieved_chunks": retrieved,
        }


# ============================================================
# Hybrid + Reranking RAG
# ============================================================

class RerankedRAGPipeline:
    """
    RAG pipeline using:

        BM25 + Vector Retrieval
                ↓
           Top 20 candidates
                ↓
        Cross-Encoder Reranking
                ↓
            Top 5 chunks
                ↓
             Gemini
                ↓
             Answer
    """

    def __init__(self):

        # Ingest/load existing documents.
        chunks = ingest_pdf("data/")

        chroma_collection = get_collection()

        embedding_function = get_embedding_function()

        self.retriever = HybridRetriever(
            chunks,
            chroma_collection,
            embedding_function
        )

        self.reranker = CrossEncoderReranker()

        self.llm = gemini_llm

        # Active document filter.
        self.current_source_filter = None

    def query(self, question):
        """
        Retrieve top-20 candidates,
        rerank to top-5,
        generate grounded answer.
        """

        candidates = self.retriever.retrieve(
            question,
            k=20,
            source_filter=self.current_source_filter
        )

        top_docs = self.reranker.rerank(
            question,
            candidates,
            top_n=5
        )

        context = "\n\n".join(
            [
                doc["text"]
                for doc in top_docs
            ]
        )

        prompt = f"""
Answer the following question based ONLY on the provided context.

If the context does not contain enough information to answer the
question, clearly say that the information is not available in
the provided context.

Context:
{context}

Question:
{question}

Answer:
"""

        response = self.llm.invoke(
            prompt.strip()
        )

        return {
            "answer": response.content,
            "sources": top_docs,
        }


# ============================================================
# Cited RAG
# ============================================================

class CitedRAGPipeline:
    """
    RAG pipeline with:

        Hybrid Retrieval
              ↓
        Cross-Encoder Reranking
              ↓
          Gemini Generation
              ↓
        Inline [N] citations
              ↓
        Citation Verification
    """

    def __init__(
        self,
        reranker,
        hybrid_retriever
    ):

        self.reranker = reranker

        self.hybrid_retriever = hybrid_retriever

        # Keep this alias because app.py may access
        # PIPELINE.retriever.refresh_bm25().
        self.retriever = hybrid_retriever

        self.llm = gemini_llm

        self.current_source_filter = None

    def format_sources(
        self,
        chunks: list[dict]
    ) -> str:
        """
        Convert retrieved chunks into numbered sources
        that Gemini can reference with [N].
        """

        sources = []

        for i, chunk in enumerate(
            chunks,
            start=1
        ):
            sources.append(
                f'Source [{i}]: "{chunk["text"]}"'
            )

        return "\n\n".join(
            sources
        )

    def query(
        self,
        question: str
    ) -> CitationResponse:
        """
        Run cited RAG.

        Every factual claim should contain [N],
        where N corresponds to the retrieved source.
        """

        # ----------------------------------------------------
        # Hybrid retrieval
        # ----------------------------------------------------

        raw_results = self.hybrid_retriever.retrieve(
            question,
            k=20,
            source_filter=self.current_source_filter
        )

        # ----------------------------------------------------
        # Cross-encoder reranking
        # ----------------------------------------------------

        reranked = self.reranker.rerank(
            question,
            raw_results,
            top_n=5
        )

        # ----------------------------------------------------
        # Format sources
        # ----------------------------------------------------

        sources_text = self.format_sources(
            reranked
        )

        # ----------------------------------------------------
        # Gemini prompt
        # ----------------------------------------------------

        prompt = (
            "Answer the question using ONLY the provided sources.\n\n"

            "Citation rules:\n"
            "1. Every factual claim must have an inline citation "
            "marker such as [1] or [2].\n"
            "2. The citation number must correspond to the source "
            "number provided below.\n"
            "3. Do not invent source numbers.\n"
            "4. If the sources do not contain enough information, "
            "say so instead of guessing.\n\n"

            f"Sources:\n{sources_text}\n\n"

            f"Question:\n{question}\n\n"

            "Answer with inline citations:"
        )

        response = self.llm.invoke(
            prompt
        )

        answer_text = response.content

        # ----------------------------------------------------
        # Extract citations
        # ----------------------------------------------------

        extracted = extract_citations(
            answer_text
        )

        citations = []

        # ----------------------------------------------------
        # Verify citations
        # ----------------------------------------------------

        for claim, idx in extracted:

            if 1 <= idx <= len(reranked):

                chunk = reranked[
                    idx - 1
                ]

                is_verified = verify_citation(
                    claim,
                    chunk["text"],
                    self.llm
                )

                # Safely extract metadata regardless
                # of whether the chunk came from normal
                # text, image or table processing.

                meta = chunk.get(
                    "metadata",
                    {}
                )

                page_val = meta.get(
                    "page",
                    meta.get(
                        "page_num",
                        0
                    )
                )

                source_val = meta.get(
                    "source",
                    "Unknown"
                )

                citations.append(
                    Citation(
                        source_doc=str(
                            source_val
                        ),
                        page_number=int(
                            page_val
                        ),
                        passage=chunk["text"],
                        verified=is_verified,
                    )
                )

        return CitationResponse(
            answer=answer_text,
            citations=citations,
        )