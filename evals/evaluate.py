import json
import time
import os
import pandas as pd
from dotenv import load_dotenv

# RAGAS imports
from ragas import SingleTurnSample, EvaluationDataset, evaluate
from ragas.metrics import (
    Faithfulness,
    ResponseRelevancy,
    LLMContextPrecisionWithoutReference,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

# LangChain
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

# Load environment variables
load_dotenv()

# -----------------------------
# Evaluator LLM (Groq)
# -----------------------------
groq_llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0,
)

evaluator_llm = LangchainLLMWrapper(groq_llm)

# -----------------------------
# Evaluator Embeddings (Local)
# -----------------------------
hf_embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

evaluator_embeddings = LangchainEmbeddingsWrapper(
    hf_embeddings
)

# -----------------------------
# Load datasets/results
# -----------------------------
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


# ----------------------------------
# Use only first 4 samples for testing
# ----------------------------------

MAX_SAMPLES = 1

test_set = test_set[:MAX_SAMPLES]
results_basic = results_basic[:MAX_SAMPLES]
results_hybrid = results_hybrid[:MAX_SAMPLES]
results_reranked = results_reranked[:MAX_SAMPLES]
results_cited = results_cited[:MAX_SAMPLES]

print(f"Running evaluation on {MAX_SAMPLES} samples per pipeline")


def get_contexts(result):
    """
    Support both:
    - retrieved_contexts
    - retrieved_chunks
    """
    if "retrieved_contexts" in result:
        return result["retrieved_contexts"]

    if "retrieved_chunks" in result:
        return result["retrieved_chunks"]

    return []


def build_eval_dataset(test_set, results):
    """
    Convert benchmark data into a RAGAS EvaluationDataset.
    """
    samples = []

    for item, result in zip(test_set, results):
        sample = SingleTurnSample(
            user_input=item["question"],
            response=result["answer"],
            retrieved_contexts=get_contexts(result),
            reference=item["reference_answer"],
        )

        samples.append(sample)

    return EvaluationDataset(samples=samples)


def evaluate_pipeline(name, test_set, results):
    """
    Evaluate a single pipeline using RAGAS.
    """
    print(f"\nEvaluating: {name}...")

    dataset = build_eval_dataset(test_set, results)

    metrics = [
        Faithfulness(),
        # LLMContextPrecisionWithoutReference(),
        # ResponseRelevancy(),
    ]

    start_time = time.time()

    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
    )

    eval_time = time.time() - start_time

    def scalar(value):
        if isinstance(value, list):
            if len(value) == 0:
                return None
            return sum(value) / len(value)
        return value

    scores = {
        "pipeline": name,
        "faithfulness": scalar(
            result._scores_dict.get("faithfulness")
        ),
        "context_precision": scalar(
            result._scores_dict.get(
                "llm_context_precision_without_reference",
                result._scores_dict.get("context_precision")
            )
        ),
        "answer_relevancy": scalar(
            result._scores_dict.get(
                "answer_relevancy",
                result._scores_dict.get("response_relevancy")
            )
        ),
        "eval_time_seconds": round(eval_time, 2),
    }

    print(scores)

    return scores


# -----------------------------
# Run Evaluations
# -----------------------------
all_scores = []

all_scores.append(
    evaluate_pipeline(
        "Basic RAG",
        test_set,
        results_basic,
    )
)

all_scores.append(
    evaluate_pipeline(
        "Hybrid Search",
        test_set,
        results_hybrid,
    )
)

all_scores.append(
    evaluate_pipeline(
        "Hybrid + Reranking",
        test_set,
        results_reranked,
    )
)

all_scores.append(
    evaluate_pipeline(
        "Cited RAG",
        test_set,
        results_cited,
    )
)

# Citation Accuracy
if (
    len(results_cited) > 0
    and "citation_accuracy" in results_cited[0]
):
    all_scores[3]["citation_accuracy"] = (
        sum(
            r["citation_accuracy"]
            for r in results_cited
        )
        / len(results_cited)
    )

print("\n" + "=" * 60)
print("EVALUATION COMPLETE")
print("=" * 60)

# -----------------------------
# DataFrame
# -----------------------------
df = pd.DataFrame(all_scores)
df = df.set_index("pipeline")

# -----------------------------
# Improvements
# -----------------------------
basic_scores = df.loc["Basic RAG"]

improvements = {}

metric_columns = [
    col
    for col in df.columns
    if col not in ["eval_time_seconds", "citation_accuracy"]
]

for pipeline in [
    "Hybrid Search",
    "Hybrid + Reranking",
    "Cited RAG",
]:
    row = df.loc[pipeline]
    imp = {}

    for metric in metric_columns:
        base = basic_scores.get(metric)
        current = row.get(metric)

        if (
            base is not None
            and current is not None
            and pd.notna(base)
            and pd.notna(current)
            and base > 0
        ):
            pct = ((current - base) / base) * 100
            imp[metric] = f"{pct:+.1f}%"
        else:
            imp[metric] = "N/A"

    improvements[pipeline] = imp

# -----------------------------
# Console Output
# -----------------------------
print("\n" + "=" * 60)
print("BENCHMARK RESULTS")
print("=" * 60)

print(df.to_string())

print("\n\nIMPROVEMENT OVER BASIC RAG:")

for pipeline, imp in improvements.items():
    print(f"  {pipeline}:")

    for metric, value in imp.items():
        print(f"    {metric}: {value}")

# -----------------------------
# Markdown Report
# -----------------------------
report_lines = [
    "# RAG Pipeline Benchmark Report\n",
    "## Metric Comparison\n",
]

report_metrics = [
    col
    for col in df.columns
]

report_lines.append(
    "| Pipeline | " +
    " | ".join(report_metrics) +
    " |"
)

report_lines.append(
    "|" +
    "|".join(["---"] * (len(report_metrics) + 1))
    + "|"
)

for pipeline in df.index:
    row = df.loc[pipeline]

    values = []

    for metric in report_metrics:
        value = row.get(metric)

        if value is None or pd.isna(value):
            values.append("-")
        elif isinstance(value, float):
            values.append(f"{value:.4f}")
        else:
            values.append(str(value))

    report_lines.append(
        f"| {pipeline} | " +
        " | ".join(values) +
        " |"
    )

report_lines.append("\n## Improvement Over Basic RAG\n")

for pipeline, imp in improvements.items():
    parts = [
        f"{metric} {value}"
        for metric, value in imp.items()
    ]

    report_lines.append(
        f"- **{pipeline}**: " +
        ", ".join(parts)
    )

report_lines.append("\n## Latency Comparison\n")
report_lines.append("| Pipeline | Evaluation Time (s) |")
report_lines.append("|----------|--------------------:|")

for pipeline in df.index:
    row = df.loc[pipeline]

    report_lines.append(
        f"| {pipeline} | {row['eval_time_seconds']:.2f} |"
    )

with open("evals/benchmark_report.md", "w") as f:
    f.write("\n".join(report_lines))