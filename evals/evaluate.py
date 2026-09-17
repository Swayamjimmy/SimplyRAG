# evals/evaluate.py

import json
import os
import time

import pandas as pd

from dotenv import load_dotenv

# ============================================================
# RAGAS
# ============================================================

from ragas import (
    SingleTurnSample,
    EvaluationDataset,
    evaluate,
)

from ragas.metrics import (
    Faithfulness,
    ResponseRelevancy,
    LLMContextPrecisionWithoutReference,
)

from ragas.llms import (
    LangchainLLMWrapper,
)

from ragas.embeddings import (
    LangchainEmbeddingsWrapper,
)


# ============================================================
# LangChain
# ============================================================

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
)

from langchain_huggingface import (
    HuggingFaceEmbeddings,
)


# ============================================================
# Environment
# ============================================================

load_dotenv()


GEMINI_API_KEY = os.getenv(
    "GEMINI_API_KEY"
)

GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.8-flash"
)


if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY is not set. "
        "Add it to your .env file."
    )


# ============================================================
# Gemini Evaluator LLM
# ============================================================

gemini_llm = ChatGoogleGenerativeAI(
    model=GEMINI_MODEL,
    google_api_key=GEMINI_API_KEY,
    temperature=0,
    max_retries=2,
)


# RAGAS expects a LangChain-compatible LLM wrapper.
evaluator_llm = LangchainLLMWrapper(
    gemini_llm
)


# ============================================================
# Evaluator Embeddings
# ============================================================

# Keep the existing local embedding model.
#
# Gemini is being used for generation/evaluation,
# NOT for your existing retrieval embeddings.

hf_embeddings = HuggingFaceEmbeddings(
    model_name=(
        "sentence-transformers/"
        "all-MiniLM-L6-v2"
    )
)

evaluator_embeddings = (
    LangchainEmbeddingsWrapper(
        hf_embeddings
    )
)


# ============================================================
# Load Benchmark Data
# ============================================================

with open(
    "evals/test_set.json",
    "r"
) as f:
    test_set = json.load(f)


# ============================================================
# Load Pipeline Results
# ============================================================

def load_json_file(
    path
):
    """
    Load a JSON result file.

    Returns an empty list when the file
    does not exist.
    """

    if not os.path.exists(path):
        print(
            f"Warning: {path} does not exist."
        )
        return []

    with open(
        path,
        "r"
    ) as f:
        return json.load(f)


results_basic = load_json_file(
    "evals/results_basic.json"
)

results_hybrid = load_json_file(
    "evals/results_hybrid.json"
)

results_reranked = load_json_file(
    "evals/results_reranked.json"
)

results_cited = load_json_file(
    "evals/results_cited.json"
)


# ============================================================
# Context Extraction
# ============================================================

def get_contexts(
    result
):
    """
    Extract retrieved contexts from a benchmark result.

    Supports the result formats used by the existing
    benchmark scripts.
    """

    # Cited RAG
    if "retrieved_contexts" in result:

        return result[
            "retrieved_contexts"
        ]

    # Basic / Hybrid RAG
    if "retrieved_chunks" in result:

        chunks = result[
            "retrieved_chunks"
        ]

        contexts = []

        for chunk in chunks:

            if isinstance(
                chunk,
                dict
            ):
                contexts.append(
                    chunk.get(
                        "text",
                        ""
                    )
                )

            else:
                contexts.append(
                    str(chunk)
                )

        return contexts

    # Reranked RAG
    if "sources" in result:

        sources = result[
            "sources"
        ]

        contexts = []

        for source in sources:

            if isinstance(
                source,
                dict
            ):
                contexts.append(
                    source.get(
                        "text",
                        ""
                    )
                )

            else:
                contexts.append(
                    str(source)
                )

        return contexts

    return []


# ============================================================
# Build RAGAS Dataset
# ============================================================

def build_eval_dataset(
    test_set,
    results
):
    """
    Convert benchmark results into
    a RAGAS EvaluationDataset.
    """

    samples = []

    for item, result in zip(
        test_set,
        results
    ):

        sample = SingleTurnSample(
            user_input=item[
                "question"
            ],

            response=result[
                "answer"
            ],

            retrieved_contexts=get_contexts(
                result
            ),

            reference=item[
                "reference_answer"
            ],
        )

        samples.append(
            sample
        )

    return EvaluationDataset(
        samples=samples
    )


# ============================================================
# Evaluate Pipeline
# ============================================================

def evaluate_pipeline(
    name,
    test_set,
    results
):
    """
    Evaluate a single pipeline using RAGAS.
    """

    print(
        f"\nEvaluating: {name}..."
    )

    if not results:
        print(
            f"No results available for {name}."
        )

        return {
            "pipeline": name,
            "faithfulness": None,
            "context_precision": None,
            "answer_relevancy": None,
            "eval_time_seconds": None,
        }

    dataset = build_eval_dataset(
        test_set,
        results
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    metrics = [
        Faithfulness(),

        # Uncomment these if you want to enable
        # additional RAGAS metrics.
        #
        # LLMContextPrecisionWithoutReference(),
        # ResponseRelevancy(),
    ]

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    start_time = time.time()

    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
    )

    eval_time = (
        time.time()
        - start_time
    )

    # --------------------------------------------------------
    # Scalar helper
    # --------------------------------------------------------

    def scalar(value):

        if isinstance(
            value,
            list
        ):

            if len(value) == 0:
                return None

            return sum(value) / len(value)

        return value

    # --------------------------------------------------------
    # Scores
    # --------------------------------------------------------

    scores = {
        "pipeline": name,

        "faithfulness": scalar(
            result._scores_dict.get(
                "faithfulness"
            )
        ),

        "context_precision": scalar(
            result._scores_dict.get(
                "llm_context_precision_without_reference",
                result._scores_dict.get(
                    "context_precision"
                )
            )
        ),

        "answer_relevancy": scalar(
            result._scores_dict.get(
                "answer_relevancy",
                result._scores_dict.get(
                    "response_relevancy"
                )
            )
        ),

        "eval_time_seconds": round(
            eval_time,
            2
        ),
    }

    print(
        scores
    )

    return scores


# ============================================================
# Run Evaluations
# ============================================================

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


# ============================================================
# Citation Accuracy
# ============================================================

if (
    len(results_cited) > 0
    and "citation_accuracy"
    in results_cited[0]
):

    all_scores[3][
        "citation_accuracy"
    ] = (
        sum(
            result[
                "citation_accuracy"
            ]
            for result in results_cited
        )
        / len(results_cited)
    )


# ============================================================
# Evaluation Complete
# ============================================================

print(
    "\n" + "=" * 60
)

print(
    "EVALUATION COMPLETE"
)

print(
    "=" * 60
)


# ============================================================
# DataFrame
# ============================================================

df = pd.DataFrame(
    all_scores
)

df = df.set_index(
    "pipeline"
)


# ============================================================
# Improvements Over Basic RAG
# ============================================================

if "Basic RAG" in df.index:

    basic_scores = df.loc[
        "Basic RAG"
    ]

else:

    basic_scores = None


improvements = {}


metric_columns = [
    col
    for col in df.columns
    if col not in [
        "eval_time_seconds",
        "citation_accuracy",
    ]
]


if basic_scores is not None:

    for pipeline in [
        "Hybrid Search",
        "Hybrid + Reranking",
        "Cited RAG",
    ]:

        if pipeline not in df.index:
            continue

        row = df.loc[
            pipeline
        ]

        imp = {}

        for metric in metric_columns:

            base = basic_scores.get(
                metric
            )

            current = row.get(
                metric
            )

            if (
                base is not None
                and current is not None
                and pd.notna(base)
                and pd.notna(current)
                and base > 0
            ):

                pct = (
                    (current - base)
                    / base
                ) * 100

                imp[metric] = (
                    f"{pct:+.1f}%"
                )

            else:

                imp[metric] = "N/A"

        improvements[
            pipeline
        ] = imp


# ============================================================
# Console Output
# ============================================================

print(
    "\n" + "=" * 60
)

print(
    "BENCHMARK RESULTS"
)

print(
    "=" * 60
)

print(
    df.to_string()
)


print(
    "\n\nIMPROVEMENT OVER BASIC RAG:"
)


for pipeline, imp in improvements.items():

    print(
        f"  {pipeline}:"
    )

    for metric, value in imp.items():

        print(
            f"    {metric}: {value}"
        )


# ============================================================
# Markdown Report
# ============================================================

report_lines = [
    "# RAG Pipeline Benchmark Report\n",
    "## Metric Comparison\n",
]


report_metrics = [
    col
    for col in df.columns
]


report_lines.append(
    "| Pipeline | "
    + " | ".join(
        report_metrics
    )
    + " |"
)


report_lines.append(
    "|"
    + "|".join(
        ["---"]
        * (
            len(report_metrics)
            + 1
        )
    )
    + "|"
)


for pipeline in df.index:

    row = df.loc[
        pipeline
    ]

    values = []

    for metric in report_metrics:

        value = row.get(
            metric
        )

        if (
            value is None
            or pd.isna(value)
        ):

            values.append("-")

        elif isinstance(
            value,
            float
        ):

            values.append(
                f"{value:.4f}"
            )

        else:

            values.append(
                str(value)
            )

    report_lines.append(
        f"| {pipeline} | "
        + " | ".join(values)
        + " |"
    )


report_lines.append(
    "\n## Improvement Over Basic RAG\n"
)


for pipeline, imp in improvements.items():

    parts = [
        f"{metric} {value}"
        for metric, value in imp.items()
    ]

    report_lines.append(
        f"- **{pipeline}**: "
        + ", ".join(parts)
    )


report_lines.append(
    "\n## Latency Comparison\n"
)

report_lines.append(
    "| Pipeline | Evaluation Time (s) |"
)

report_lines.append(
    "|---|---:|"
)


for pipeline in df.index:

    value = df.loc[
        pipeline,
        "eval_time_seconds"
    ]

    if (
        value is None
        or pd.isna(value)
    ):
        value = "-"

    report_lines.append(
        f"| {pipeline} | {value} |"
    )


# ============================================================
# Save Report
# ============================================================

with open(
    "evals/benchmark_report.md",
    "w"
) as f:

    f.write(
        "\n".join(
            report_lines
        )
    )


print(
    "\nSaved: "
    "evals/benchmark_report.md"
)