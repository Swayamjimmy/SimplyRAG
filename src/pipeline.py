import os
from dotenv import load_dotenv
from groq import Groq
from src.retriever import BasicRetriever
from src.hybrid_retriever import HybridRetriever

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