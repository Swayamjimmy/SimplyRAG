import json
import os
from dotenv import load_dotenv
from groq import Groq
from src.ingest import ingest_pdf
from src.embeddings import get_collection, get_embedding_function, store_chunks
from src.retriever import BasicRetriever
from src.pipeline import BasicRAGPipeline, HybridRAGPipeline

load_dotenv()

# Initialize shared resources
chunks = ingest_pdf("data/")
store_chunks(chunks)
collection = get_collection()
embedding_function = get_embedding_function()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Load test set
with open("evals/test_set.json", "r") as f:
    test_set = json.load(f)

# Run basic RAG pipeline
print("Running BasicRAGPipeline...")
basic_pipeline = BasicRAGPipeline()
basic_results = []
for item in test_set:
    result = basic_pipeline.query(item["question"])
    if isinstance(result, dict):
        answer = result.get("answer")
        retrieved_chunks = result.get("retrieved_chunks", [])
    else:
        answer = result
        retrieved_chunks = basic_pipeline.retriever.retrieve(item["question"])

    basic_results.append({
        "question": item["question"],
        "answer": answer,
        "retrieved_chunks": [c["text"][:200] for c in retrieved_chunks]
    })
    print(f"  Done: {item['question'][:50]}...")

# Save basic results
with open("evals/results_basic.json", "w") as f:
    json.dump(basic_results, f, indent=2)
print("Saved evals/results_basic.json")

# Run hybrid RAG pipeline
print("\nRunning HybridRAGPipeline...")
hybrid_pipeline = HybridRAGPipeline(chunks, collection, embedding_function, client)
hybrid_results = []
for item in test_set:
    result = hybrid_pipeline.query(item["question"])
    hybrid_results.append({
        "question": item["question"],
        "answer": result["answer"],
        "retrieved_chunks": [c["text"][:200] for c in result["retrieved_chunks"]]
    })
    print(f"  Done: {item['question'][:50]}...")

# Save hybrid results
with open("evals/results_hybrid.json", "w") as f:
    json.dump(hybrid_results, f, indent=2)
print("Saved evals/results_hybrid.json")

print("\nBenchmark complete! Results saved to evals/ directory.")