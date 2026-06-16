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
            # 1. Assign attributes FIRST so load_documents_from_chroma can use them
            self.chroma_collection = chroma_collection
            self.embedding_model = embedding_function
            
            # 2. Build the initial BM25 index
            self.refresh_bm25()

    def refresh_bm25(self):
        """Rebuilds the BM25 index from the latest state of ChromaDB."""
        documents = self.load_documents_from_chroma()
        from langchain_community.retrievers import BM25Retriever as LangChainBM25
        self.bm25_retriever = LangChainBM25.from_documents(documents)
        self.bm25_retriever.k = 5

    def _vector_search(self, query, k=5, source_filter=None):
        query_embedding = self.embedding_model.encode(query).tolist()
        
        # Build query arguments dynamically to include the filter if it exists
        query_kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": k
        }
        
        # Tell ChromaDB to ONLY look at chunks from a specific file
        if source_filter:
            query_kwargs["where"] = {"source": source_filter}

        results = self.chroma_collection.query(**query_kwargs)

        documents = results["documents"][0] if results["documents"] else []
        metadatas = results["metadatas"][0] if results["metadatas"] else []

        vector_results = []
        for doc, metadata in zip(documents, metadatas):
            vector_results.append(Document(page_content=doc, metadata=metadata))

        return vector_results

    def retrieve(self, query, k=5, source_filter=None):
        # Fetch extra BM25 results so we have enough left over after filtering
        self.bm25_retriever.k = k * 10
        raw_bm25 = self.bm25_retriever.invoke(query)
        
        # Filter BM25 results manually based on the filename
        if source_filter:
            bm25_results = [doc for doc in raw_bm25 if doc.metadata.get("source") == source_filter][:k]
        else:
            bm25_results = raw_bm25[:k]

        # Filter Vector results directly via ChromaDB
        vector_results = self._vector_search(query=query, k=k, source_filter=source_filter)

        # Execute Reciprocal Rank Fusion
        fused_docs = self._rrf_fusion(bm25_results, vector_results)

        return [
            {
                "text": doc.page_content,
                "metadata": doc.metadata
            }
            for doc in fused_docs[:k]
        ]

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

    def load_documents_from_chroma(self):
        # FIX 1: Add a massive limit to guarantee ChromaDB returns ALL chunks, 
        # ensuring the BM25 index sees your newly uploaded files.
        data = self.chroma_collection.get(
            include=["documents", "metadatas"],
            limit=100000 
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

    