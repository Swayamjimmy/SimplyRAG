from sentence_transformers import CrossEncoder


class CrossEncoderReranker:
    """Reranks retrieved documents using a cross-encoder model."""

    def __init__(
        self,
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
        max_length=512
    ):
        self.model = CrossEncoder(
            model_name,
            max_length=max_length
        )

    def rerank(self, query, docs, top_n=5):
        """Score each query-document pair and return the top_n most relevant."""

        pairs = [
            (query, doc["text"])
            for doc in docs
        ]

        scores = self.model.predict(pairs)

        for i, doc in enumerate(docs):
            doc["rerank_score"] = float(scores[i])

        ranked_docs = sorted(
            docs,
            key=lambda x: x["rerank_score"],
            reverse=True
        )

        return ranked_docs[:top_n]