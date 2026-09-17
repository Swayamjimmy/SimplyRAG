# evals/run_benchmark.py

import json

from dotenv import load_dotenv

from src.ingest import ingest_pdf
from src.embeddings import (
    get_collection,
    get_embedding_function,
    store_chunks,
)

from src.pipeline import (
    BasicRAGPipeline,
    HybridRAGPipeline,
)


# ============================================================
# Environment
# ============================================================

load_dotenv()


# ============================================================
# Initialize Shared Resources
# ============================================================

print(
    "Loading documents..."
)

chunks = ingest_pdf(
    "data/"
)


print(
    f"Loaded {len(chunks)} chunks."
)


store_chunks(
    chunks
)


collection = get_collection()


embedding_function = (
    get_embedding_function()
)


# ============================================================
# Load Test Set
# ============================================================

with open(
    "evals/test_set.json",
    "r"
) as f:

    test_set = json.load(
        f
    )


print(
    f"Loaded {len(test_set)} benchmark questions."
)


# ============================================================
# Basic RAG Benchmark
# ============================================================

print(
    "\n" + "=" * 60
)

print(
    "Running BasicRAGPipeline..."
)

print(
    "=" * 60
)


basic_pipeline = (
    BasicRAGPipeline()
)


basic_results = []


for item in test_set:

    question = item[
        "question"
    ]

    result = (
        basic_pipeline.query(
            question
        )
    )

    # Current BasicRAGPipeline returns
    # a dictionary.
    answer = result.get(
        "answer",
        ""
    )

    retrieved_chunks = result.get(
        "retrieved_chunks",
        []
    )

    basic_results.append(
        {
            "question": question,

            "answer": answer,

            "retrieved_chunks": [
                {
                    "text": chunk.get(
                        "text",
                        ""
                    ),
                    "metadata": chunk.get(
                        "metadata",
                        {}
                    ),
                }
                for chunk in retrieved_chunks
            ],
        }
    )

    print(
        f"  Done: {question[:70]}..."
    )


# ============================================================
# Save Basic Results
# ============================================================

with open(
    "evals/results_basic.json",
    "w"
) as f:

    json.dump(
        basic_results,
        f,
        indent=2
    )


print(
    "Saved evals/results_basic.json"
)


# ============================================================
# Hybrid RAG Benchmark
# ============================================================

print(
    "\n" + "=" * 60
)

print(
    "Running HybridRAGPipeline..."
)

print(
    "=" * 60
)


hybrid_pipeline = (
    HybridRAGPipeline(
        chunks,
        collection,
        embedding_function,
    )
)


hybrid_results = []


for item in test_set:

    question = item[
        "question"
    ]

    result = (
        hybrid_pipeline.query(
            question
        )
    )

    hybrid_results.append(
        {
            "question": question,

            "answer": result.get(
                "answer",
                ""
            ),

            "retrieved_chunks": [
                {
                    "text": chunk.get(
                        "text",
                        ""
                    ),
                    "metadata": chunk.get(
                        "metadata",
                        {}
                    ),
                }
                for chunk in result.get(
                    "retrieved_chunks",
                    []
                )
            ],
        }
    )

    print(
        f"  Done: {question[:70]}..."
    )


# ============================================================
# Save Hybrid Results
# ============================================================

with open(
    "evals/results_hybrid.json",
    "w"
) as f:

    json.dump(
        hybrid_results,
        f,
        indent=2
    )


print(
    "Saved evals/results_hybrid.json"
)


# ============================================================
# Benchmark Complete
# ============================================================

print(
    "\n" + "=" * 60
)

print(
    "Benchmark complete!"
)

print(
    "=" * 60
)

print(
    "Results saved to evals/"
)