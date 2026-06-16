from collections import defaultdict

from langchain_community.retrievers import BM25Retriever as LangChainBM25
from langchain_core.documents import Document


class HybridRetriever:
    """
    Hybrid retriever combining:
    1. BM25 keyword retrieval
    2. ChromaDB vector retrieval
    using Reciprocal Rank Fusion (RRF).
    """

    def __init__(self, chunks, chroma_collection, embedding_function):
        # Convert project chunks into LangChain Documents
        documents = self.load_documents_from_chroma()

        # BM25 Retriever
        self.bm25_retriever = LangChainBM25.from_documents(documents)
        self.bm25_retriever.k = 5

        # Direct ChromaDB access
        self.chroma_collection = chroma_collection

        # SentenceTransformer model
        self.embedding_model = embedding_function

    def _vector_search(self, query, k=5):
        """
        Query ChromaDB directly using SentenceTransformer embeddings.
        """

        query_embedding = self.embedding_model.encode(query).tolist()

        results = self.chroma_collection.query(
            query_embeddings=[query_embedding],
            n_results=k
        )

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]

        vector_results = []

        for doc, metadata in zip(documents, metadatas):
            vector_results.append(
                Document(
                    page_content=doc,
                    metadata=metadata
                )
            )

        return vector_results

    def _rrf_fusion(self, bm25_results, vector_results, rrf_k=60):
        """
        Reciprocal Rank Fusion.

        Score(doc) = Σ 1 / (rrf_k + rank)
        """

        scores = defaultdict(float)
        doc_lookup = {}

        # BM25 rankings
        for rank, doc in enumerate(bm25_results, start=1):
            doc_id = doc.page_content

            scores[doc_id] += 1.0 / (rrf_k + rank)
            doc_lookup[doc_id] = doc

        # Vector rankings
        for rank, doc in enumerate(vector_results, start=1):
            doc_id = doc.page_content

            scores[doc_id] += 1.0 / (rrf_k + rank)
            doc_lookup[doc_id] = doc

        ranked_docs = sorted(
            scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return [
            doc_lookup[doc_id]
            for doc_id, _ in ranked_docs
        ]

    def retrieve(self, query, k=5):
        """
        Hybrid retrieval:
        BM25 + Vector Search + RRF
        """

        bm25_results = self.bm25_retriever.invoke(query)

        vector_results = self._vector_search(
            query=query,
            k=k
        )

       

        return [
            {
                "text": doc.page_content,
                "metadata": doc.metadata
            }
            for doc in vector_results[:k]
        ]
    def load_documents_from_chroma(self):

        data = self.chroma_collection.get(
            include=["documents", "metadatas"]
        )

        return [
            Document(
                page_content=doc,
                metadata=meta
            )
            for doc, meta in zip(
                data["documents"],
                data["metadatas"]
            )
        ]