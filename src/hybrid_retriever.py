from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever as LangChainBM25
from langchain_core.documents import Document


class HybridRetriever:
    """Combines BM25 keyword search and ChromaDB vector search using Reciprocal Rank Fusion."""

    def __init__(self, chunks, chroma_collection, embedding_function):
        # Convert chunks to LangChain Document format for BM25
        documents = [
            Document(page_content=c["text"], metadata=c["metadata"])
            for c in chunks
        ]

        # Initialize LangChain BM25 retriever from documents
        self.bm25_retriever = LangChainBM25.from_documents(documents)
        self.bm25_retriever.k = 5

        # Initialize ChromaDB vector retriever via LangChain interface
        from langchain_community.vectorstores import Chroma
        vectorstore = Chroma(
            collection_name=chroma_collection.name,
            persist_directory="chroma_db",
            embedding_function=embedding_function
        )
        self.vector_retriever = vectorstore.as_retriever(
            search_kwargs={"k": 5}
        )

        # Combine with EnsembleRetriever using equal weights and RRF
        self.ensemble = EnsembleRetriever(
            retrievers=[self.bm25_retriever, self.vector_retriever],
            weights=[0.5, 0.5]
        )

    def retrieve(self, query, k=5):
        """Return top-k results merged with Reciprocal Rank Fusion."""
        # EnsembleRetriever handles RRF fusion internally
        results = self.ensemble.invoke(query)
        # Convert back to project format
        return [
            {"text": doc.page_content, "metadata": doc.metadata}
            for doc in results[:k]
        ]