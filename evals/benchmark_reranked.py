import json
from src.pipeline import RerankedRAGPipeline

# Load the 20-question test set
with open("evals/test_set.json", "r") as f:
    test_set = json.load(f)

# Initialize the reranked pipeline
pipeline = RerankedRAGPipeline()

# Run all 20 questions through the reranked pipeline
results = []
for item in test_set:
    question = item["question"]
    result = pipeline.query(question)
    results.append({
        "question": question,
        "answer": result["answer"],
        "sources": result["sources"],
        "reference": item["reference_answer"]
    })
    print(f"Processed: {question[:50]}...")

# Save results to file
with open("evals/results_reranked.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\nCompleted {len(results)} questions. Results saved to evals/results_reranked.json")