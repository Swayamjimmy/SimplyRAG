import json
import time
import os
import pandas as pd
from dotenv import load_dotenv

# RAGAS imports for evaluation
from ragas import SingleTurnSample, EvaluationDataset, evaluate
from ragas.metrics import Faithfulness, ResponseRelevancy, LLMContextPrecisionWithoutReference
from ragas.llms import LangchainLLMWrapper

# LangChain Groq wrapper for the evaluator LLM
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()

# Configure the evaluator LLM using Groq via LangChain
groq_llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0
)

# Wrap for RAGAS compatibility
evaluator_llm = LangchainLLMWrapper(groq_llm)

# Load the test set and pipeline results
with open("evals/test_set.json", "r") as f:
    test_set = json.load(f)

with open("evals/results_basic.json", "r") as f:
    results_basic = json.load(f)

with open("evals/results_hybrid.json", "r") as f:
    results_hybrid = json.load(f)

with open("evals/results_reranked.json", "r") as f:
    results_reranked = json.load(f)

with open("evals/results_cited.json", "r") as f:
    results_cited = json.load(f)


def build_eval_dataset(test_set, results):
    """Convert test set + results into a RAGAS EvaluationDataset."""
    samples = []
    for item, result in zip(test_set, results):
        sample = SingleTurnSample(
            user_input=item["question"],
            response=result["answer"],
            retrieved_contexts=result["retrieved_contexts"],
            reference=item["reference_answer"]
        )
        samples.append(sample)
    return EvaluationDataset(samples=samples)


def evaluate_pipeline(name, test_set, results):
    """Run RAGAS metrics on a single pipeline's results."""
    print(f"\nEvaluating: {name}...")
    dataset = build_eval_dataset(test_set, results)

    # Define the three core RAG metrics
    metrics = [
        Faithfulness(),
        LLMContextPrecisionWithoutReference(),
        ResponseRelevancy()
    ]

    # Time the evaluation
    start_time = time.time()
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=evaluator_llm
    )
    eval_time = time.time() - start_time

    scores = {
        "pipeline": name,
        "faithfulness": result["faithfulness"],
        "context_precision": result["llm_context_precision_without_reference"],
        "answer_relevancy": result["answer_relevancy"],
        "eval_time_seconds": round(eval_time, 2)
    }
    return scores


# Run evaluation on all four pipelines
all_scores = []
all_scores.append(evaluate_pipeline("Basic RAG", test_set, results_basic))
all_scores.append(evaluate_pipeline("Hybrid Search", test_set, results_hybrid))
all_scores.append(evaluate_pipeline("Hybrid + Reranking", test_set, results_reranked))
all_scores.append(evaluate_pipeline("Cited RAG", test_set, results_cited))

# Append citation accuracy for the cited pipeline
if "citation_accuracy" in results_cited[0]:
    all_scores[3]["citation_accuracy"] = sum(
        r["citation_accuracy"] for r in results_cited
    ) / len(results_cited)

print("\n" + "=" * 60)
print("EVALUATION COMPLETE")
print("=" * 60)

# Build DataFrame from all scores
df = pd.DataFrame(all_scores)
df = df.set_index("pipeline")

# Calculate improvement percentages over basic RAG
basic_scores = df.loc["Basic RAG"]
improvements = {}
for pipeline in ["Hybrid Search", "Hybrid + Reranking", "Cited RAG"]:
    row = df.loc[pipeline]
    imp = {}
    for metric in ["faithfulness", "context_precision", "answer_relevancy"]:
        if basic_scores[metric] > 0:
            pct = ((row[metric] - basic_scores[metric]) / basic_scores[metric]) * 100
            imp[metric] = f"{pct:+.1f}%"
        else:
            imp[metric] = "N/A"
    improvements[pipeline] = imp

# Print formatted summary to the console
print("\n" + "=" * 60)
print("BENCHMARK RESULTS")
print("=" * 60)
print(df.to_string())

print("\n\nIMPROVEMENT OVER BASIC RAG:")
for pipeline, imp in improvements.items():
    print(f"  {pipeline}:")
    for metric, value in imp.items():
        print(f"    {metric}: {value}")

# Generate markdown report
report_lines = [
    "# RAG Pipeline Benchmark Report\n",
    "## Metric Comparison\n",
    "| Pipeline | Faithfulness | Context Precision | Answer Relevancy | Citation Accuracy |",
    "|----------|-------------|-------------------|------------------|-------------------|",
]

for pipeline in ["Basic RAG", "Hybrid Search", "Hybrid + Reranking", "Cited RAG"]:
    row = df.loc[pipeline]
    faith = f"{row['faithfulness']:.4f}"
    ctx = f"{row['context_precision']:.4f}"
    rel = f"{row['answer_relevancy']:.4f}"
    cit_val = row.get("citation_accuracy")
    cit = f"{cit_val:.4f}" if cit_val is not None and pd.notna(cit_val) else "-"
    report_lines.append(f"| {pipeline} | {faith} | {ctx} | {rel} | {cit} |")

report_lines.append("\n## Improvement Over Basic RAG\n")
for pipeline, imp in improvements.items():
    parts = [f"{metric} {value}" for metric, value in imp.items()]
    report_lines.append(f"- **{pipeline}**: " + ", ".join(parts))

report_lines.append("\n## Latency Comparison\n")
report_lines.append("| Pipeline | Evaluation Time (s) |")
report_lines.append("|----------|--------------------:|")
for pipeline in ["Basic RAG", "Hybrid Search", "Hybrid + Reranking", "Cited RAG"]:
    row = df.loc[pipeline]
    report_lines.append(f"| {pipeline} | {row['eval_time_seconds']:.2f} |")

# Save the report to a markdown file
with open("evals/benchmark_report.md", "w") as f:
    f.write("\n".join(report_lines))

print("\n\nBenchmark report saved to evals/benchmark_report.md")
print("These are your resume metrics!")