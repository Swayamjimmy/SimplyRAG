from src.embeddings import embedding_model, collection

class BasicRetriever:
    """Retrieves top-k relevant chunks from ChromaDB using similarity search."""

    def __init__(self, top_k=5):
        self.top_k = top_k

    def retrieve(self, query):
        """Embed the query and search ChromaDB for similar chunks."""
        # Generate embedding for the query
        query_embedding = embedding_model.encode([query]).tolist()

        # Search ChromaDB for the most similar chunks
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=self.top_k
        )

        # Format results with text and metadata
        documents = []
        for i in range(len(results["documents"][0])):
            documents.append({
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i]
            })
        return documents