import os
from dotenv import load_dotenv
from groq import Groq
from src.retriever import BasicRetriever
from src.hybrid_retriever import HybridRetriever
from src.reranker import CrossEncoderReranker
from src.ingest import ingest_pdf
from src.embeddings import get_collection, get_embedding_function

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
        # 1. Load required shared resources internally
        chunks = ingest_pdf("data/")
        chroma_collection = get_collection()
        embedding_function = get_embedding_function()

        # 2. Initialize retriever with its required dependencies
        self.retriever = HybridRetriever(chunks, chroma_collection, embedding_function)
        self.reranker = CrossEncoderReranker()
        self.llm_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    def query(self, question):
        """Retrieve top-20, rerank to top-5, then generate answer."""
        # Over-retrieve: get 20 candidates from hybrid search
        candidates = self.retriever.retrieve(question, k=20)

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

        return {
            "answer": response.choices[0].message.content,
            "sources": top_docs
        }

    
