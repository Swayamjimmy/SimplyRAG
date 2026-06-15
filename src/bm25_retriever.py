from rank_bm25 import BM25Okapi


class BM25Retriever:
    """Keyword-based retriever using BM25 scoring."""

    def __init__(self, chunks):
        # Store original chunks for retrieval
        self.chunks = chunks
        # Tokenize all chunk texts for BM25 indexing
        self.tokenized_corpus = self.tokenize_corpus(chunks)
        # Build the BM25 index over tokenized documents
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def tokenize_corpus(self, chunks):
        """Split each chunk's text into lowercase tokens."""
        return [chunk["text"].lower().split() for chunk in chunks]

    def retrieve(self, query, k=5):
        """Return top-k chunks ranked by BM25 keyword score."""
        # Tokenize the query the same way as the corpus
        tokenized_query = query.lower().split()
        # Get BM25 scores for all documents
        scores = self.bm25.get_scores(tokenized_query)
        # Get indices of top-k scoring documents
        top_indices = scores.argsort()[-k:][::-1]
        # Return chunks with their BM25 scores
        results = []
        for idx in top_indices:
            result = self.chunks[idx].copy()
            result["score"] = float(scores[idx])
            results.append(result)
        return results