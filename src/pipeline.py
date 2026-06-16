import os
from dotenv import load_dotenv
from groq import Groq
from src.retriever import BasicRetriever
from src.hybrid_retriever import HybridRetriever
from src.reranker import CrossEncoderReranker
from src.ingest import ingest_pdf
from src.embeddings import get_collection, get_embedding_function
from src.citations import (
    Citation, CitationResponse, extract_citations,
    verify_citation, compute_citation_accuracy
)

load_dotenv()

class BasicRAGPipeline:
    """Full retrieval chain: query -> retrieve -> format prompt -> call Groq -> answer."""

    def __init__(self):
        self.retriever = BasicRetriever(top_k=5)
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    def format_prompt(self, question, context_chunks):
        """Build a prompt that grounds the LLM in retrieved context."""
        context = "\n\n".join(
            [f"Source: {chunk['metadata']['source']}, Page {chunk['metadata']['page']}\n{chunk['text']}"
             for chunk in context_chunks]
        )
        prompt = f"""Answer the following question based ONLY on the provided context.
If the context doesn't contain enough information, say so.

Context:
{context}

Question: {question}

Answer:"""
        return prompt

    def query(self, question):
        """Run the full RAG pipeline: retrieve, format, generate."""
        # Retrieve relevant chunks from ChromaDB
        chunks = self.retriever.retrieve(question)

        # Format the prompt with retrieved context
        prompt = self.format_prompt(question, chunks)

        # Call Groq LLM for generation
        response = self.client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        return response.choices[0].message.content

class HybridRAGPipeline:
    """RAG pipeline using hybrid BM25 + vector retrieval."""

    def __init__(self, chunks, chroma_collection, embedding_function, llm_client):
        # Initialize hybrid retriever with both BM25 and vector search
        self.retriever = HybridRetriever(chunks, chroma_collection, embedding_function)
        self.llm_client = llm_client

    def query(self, question):
        """Retrieve relevant chunks with hybrid search, then generate answer."""
        # Get top-k chunks using hybrid retrieval
        retrieved = self.retriever.retrieve(question, k=5)

        # Format context from retrieved chunks
        context = "\n\n".join([chunk["text"] for chunk in retrieved])

        # Build grounded prompt
        prompt = f"""Answer the following question based ONLY on the provided context.
If the context does not contain enough information, say so.

Context:
{context}

Question: {question}

Answer:"""

        # Call Groq LLM
        response = self.llm_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}]
        )
        return {
            "answer": response.choices[0].message.content,
            "retrieved_chunks": retrieved
        }


# ... existing classes (BasicRAGPipeline, HybridRAGPipeline) ...

class RerankedRAGPipeline:
    """RAG pipeline with hybrid search and cross-encoder reranking."""

    def __init__(self):
        chunks = ingest_pdf("data/")
        chroma_collection = get_collection()
        embedding_function = get_embedding_function()

        self.retriever = HybridRetriever(chunks, chroma_collection, embedding_function)
        self.reranker = CrossEncoderReranker()
        self.llm_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        
        # State variable to hold the active document filter
        self.current_source_filter = None

    def query(self, question):
        """Retrieve top-20, rerank to top-5, then generate answer."""
        
        # Pass the active filter down to the retriever
        candidates = self.retriever.retrieve(
            question, 
            k=20, 
            source_filter=self.current_source_filter
        )

        # Rerank: score each candidate against the query, keep top-5
        top_docs = self.reranker.rerank(question, candidates, top_n=5)

        # Format context from reranked documents
        context = "\n\n".join([doc["text"] for doc in top_docs])

        # Generate answer using Groq LLM
        prompt = f"""Answer the following question based ONLY on the provided context.
If the context does not contain enough information, say so.

Context:
{context}

Question: {question}

Answer:"""

        response = self.llm_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}]
        )

        # The crucial return statement that ensures the Agent gets its data back!
        return {
            "answer": response.choices[0].message.content,
            "sources": top_docs
        }
    
class CitedRAGPipeline:
    """RAG pipeline with inline citation grounding and verification."""

    def __init__(self, reranker, hybrid_retriever):
        self.reranker = reranker
        self.hybrid_retriever = hybrid_retriever
        self.llm = Groq(api_key=os.getenv("GROQ_API_KEY"))

    def format_sources(self, chunks: list[dict]) -> str:
        """Format retrieved chunks as numbered sources for the prompt."""
        sources = []
        for i, chunk in enumerate(chunks, 1):
            sources.append(f'Source [{i}]: "{chunk["text"]}"')
        return "\n\n".join(sources)

    def query(self, question: str) -> CitationResponse:
        """Run the full cited RAG pipeline with verification."""
        # Retrieve and rerank documents
        raw_results = self.hybrid_retriever.retrieve(question, k=20)
        reranked = self.reranker.rerank(question, raw_results, top_n=5)

        # Format sources with numbered markers
        sources_text = self.format_sources(reranked)

        # Instruct the LLM to cite sources inline
        prompt = (
            "Answer the question using ONLY the provided sources.\n"
            "For every factual claim, add an inline citation marker [N] "
            "referencing the source number.\n\n"
            f"Sources:\n{sources_text}\n\n"
            f"Question: {question}\n\n"
            "Answer with inline citations:"
        )

        response = self.llm.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        answer_text = response.choices[0].message.content

        # Extract citations and verify each one
        extracted = extract_citations(answer_text)
        citations = []

        for claim, idx in extracted:
            if 1 <= idx <= len(reranked):
                chunk = reranked[idx - 1]
                # Verify the source actually supports the claim
                is_verified = verify_citation(
                    claim, chunk["text"], self.llm
                )
                citations.append(Citation(
                    source_doc=chunk["metadata"]["source"],
                    page_number=chunk["metadata"]["page"],
                    passage=chunk["text"],
                    verified=is_verified
                ))

        return CitationResponse(answer=answer_text, citations=citations)