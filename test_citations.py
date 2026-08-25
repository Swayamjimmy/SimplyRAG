# test_citations.py

import json
from dotenv import load_dotenv

load_dotenv()

from src.hybrid_retriever import HybridRetriever
from src.reranker import CrossEncoderReranker
from src.pipeline import CitedRAGPipeline
from src.ingest import ingest_pdf
from src.embeddings import get_collection, get_embedding_function
from src.citations import compute_citation_accuracy


# Initialize components
chunks = ingest_pdf("data/")
chroma_collection = get_collection()
embedding_function = get_embedding_function()

hybrid_retriever = HybridRetriever(
    chunks,
    chroma_collection,
    embedding_function
)

reranker = CrossEncoderReranker()

pipeline = CitedRAGPipeline(
    reranker=reranker,
    hybrid_retriever=hybrid_retriever
)

# Load evaluation questions
with open("evals/test_set.json", "r") as f:
    test_set = json.load(f)

results = []

print(f"Running cited benchmark on {len(test_set)} questions...\n")

for i, item in enumerate(test_set, start=1):
    question = item["question"]

    print(f"[{i}/{len(test_set)}] {question}")

    result = pipeline.query(question)

    citation_accuracy = compute_citation_accuracy(
        result.citations
    )

    results.append({
        "question": question,
        "answer": result.answer,

        # RAGAS expects retrieved_contexts
        "retrieved_contexts": [
            citation.passage
            for citation in result.citations
        ],

        "citation_accuracy": citation_accuracy
    })

# Save results
with open("evals/results_cited.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nSaved: evals/results_cited.json")

avg_accuracy = sum(
    r["citation_accuracy"]
    for r in results
) / len(results)

print(
    f"Average Citation Accuracy: "
    f"{avg_accuracy:.2%}"
)