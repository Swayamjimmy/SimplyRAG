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
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")

def get_gemini_llm():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set. Add it to your .env file.")
    return ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=api_key,
        temperature=0,
        max_retries=2,
    )

gemini_llm = get_gemini_llm()

# ============================================================
# Basic RAG
# ============================================================
class BasicRAGPipeline:
    def __init__(self):
        self.retriever = BasicRetriever(top_k=5)
        self.llm = gemini_llm

    def format_prompt(self, question, context_chunks):
        context = "\n\n".join([
            (
                f"Source: {chunk['metadata'].get('source', 'Unknown')}, "
                f"Page: {chunk['metadata'].get('page', chunk['metadata'].get('page_num', 0))}\n"
                f"{chunk['text']}"
            )
            for chunk in context_chunks
        ])
        prompt = f"""
Answer the following question based ONLY on the provided context.
If the context does not contain enough information to answer the question, clearly say that the information is not available.

Context:
{context}

Question:
{question}

Answer:
"""
        return prompt.strip()

    def query(self, question):
        chunks = self.retriever.retrieve(question)
        prompt = self.format_prompt(question, chunks)
        response = self.llm.invoke(prompt)
        return {
            "answer": response.content,
            "retrieved_chunks": chunks,
        }

# ============================================================
# Hybrid RAG
# ============================================================
class HybridRAGPipeline:
    def __init__(self, chunks, chroma_collection, embedding_function, llm_client=None):
        self.retriever = HybridRetriever(chunks, chroma_collection, embedding_function)
        self.llm = llm_client or gemini_llm

    def query(self, question):
        retrieved = self.retriever.retrieve(question, k=5)
        context = "\n\n".join([chunk["text"] for chunk in retrieved])
        prompt = f"""
Answer the following question based ONLY on the provided context.
If the context does not contain enough information to answer the question, clearly say so.

Context:
{context}

Question:
{question}

Answer:
"""
        response = self.llm.invoke(prompt.strip())
        return {
            "answer": response.content,
            "retrieved_chunks": retrieved,
        }

# ============================================================
# Hybrid + Reranking RAG
# ============================================================
class RerankedRAGPipeline:
    def __init__(self):
        chunks = ingest_pdf("data/")
        chroma_collection = get_collection()
        embedding_function = get_embedding_function()

        self.retriever = HybridRetriever(chunks, chroma_collection, embedding_function)
        self.reranker = CrossEncoderReranker()
        self.llm = gemini_llm
        self.current_source_filter = None

    def query(self, question):
        # Optimized: Only pull top 10 candidates instead of 20
        candidates = self.retriever.retrieve(
            question,
            k=10,
            source_filter=self.current_source_filter
        )
        
        # Optimized: Rerank down to top 3 instead of 5 for token savings
        top_docs = self.reranker.rerank(question, candidates, top_n=3)
        context = "\n\n".join([doc["text"] for doc in top_docs])

        prompt = f"""
Answer the following question based ONLY on the provided context.
If the context does not contain enough information to answer the question, clearly say so.

Context:
{context}

Question:
{question}

Answer:
"""
        response = self.llm.invoke(prompt.strip())
        return {
            "answer": response.content,
            "sources": top_docs,
        }

# ============================================================
# Cited RAG
# ============================================================
class CitedRAGPipeline:
    def __init__(self, reranker, hybrid_retriever):
        self.reranker = reranker
        self.hybrid_retriever = hybrid_retriever
        self.retriever = hybrid_retriever
        self.llm = gemini_llm
        self.current_source_filter = None

    def format_sources(self, chunks: list[dict]) -> str:
        sources = []
        for i, chunk in enumerate(chunks, start=1):
            sources.append(f'Source [{i}]: "{chunk["text"]}"')
        return "\n\n".join(sources)

    def query(self, question: str, verify: bool = False) -> CitationResponse:
        """
        Run cited RAG. Optional `verify` flag disabled by default to save API latency/tokens.
        """
        
        # 1. Optimized Retrieval (10 candidates)
        raw_results = self.hybrid_retriever.retrieve(
            question,
            k=10, 
            source_filter=self.current_source_filter
        )

        # 2. Optimized Reranking (top 3 chunks)
        reranked = self.reranker.rerank(
            question,
            raw_results,
            top_n=3
        )

        sources_text = self.format_sources(reranked)

        prompt = (
            "Answer the question using ONLY the provided sources.\n\n"
            "Citation rules:\n"
            "1. Every factual claim must have an inline citation marker such as [1] or [2].\n"
            "2. The citation number must correspond to the source number provided below.\n"
            "3. Do not invent source numbers.\n"
            "4. If the sources do not contain enough information, say so instead of guessing.\n\n"
            f"Sources:\n{sources_text}\n\n"
            f"Question:\n{question}\n\n"
            "Answer with inline citations:"
        )

        response = self.llm.invoke(prompt)
        content = response.content

        # Safely extract response text
        if isinstance(content, str):
            answer_text = content
        elif isinstance(content, list):
            answer_text = "\n".join(
                item.get("text", "") if isinstance(item, dict) else str(item)
                for item in content
            )
        else:
            answer_text = str(content)

        extracted = extract_citations(answer_text)
        citations = []

        # 3. Skip sequential LLM verification for speed unless strictly requested
        for claim, idx in extracted:
            if 1 <= idx <= len(reranked):
                chunk = reranked[idx - 1]
                
                # Turn off sequential LLM calls by default
                is_verified = verify_citation(claim, chunk["text"], self.llm) if verify else False

                meta = chunk.get("metadata", {})
                page_val = meta.get("page", meta.get("page_num", 0))
                source_val = meta.get("source", "Unknown")

                citations.append(
                    Citation(
                        source_doc=str(source_val),
                        page_number=int(page_val),
                        passage=chunk["text"],
                        verified=is_verified,
                    )
                )

        return CitationResponse(
            answer=answer_text,
            citations=citations,
        )