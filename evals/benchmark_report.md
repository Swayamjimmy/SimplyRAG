# RAG Pipeline Benchmark Report

## Metric Comparison

| Pipeline | faithfulness | context_precision | answer_relevancy | eval_time_seconds | citation_accuracy |
|---|---|---|---|---|---|
| Basic RAG | 0.8000 | - | - | 130.1000 | - |
| Hybrid Search | 0.7500 | - | - | 129.3100 | - |
| Hybrid + Reranking | 0.0000 | - | - | 125.3500 | - |
| Cited RAG | 1.0000 | - | - | 132.3100 | 0.8333 |

## Improvement Over Basic RAG

- **Hybrid Search**: faithfulness -6.3%, context_precision N/A, answer_relevancy N/A
- **Hybrid + Reranking**: faithfulness -100.0%, context_precision N/A, answer_relevancy N/A
- **Cited RAG**: faithfulness +25.0%, context_precision N/A, answer_relevancy N/A

## Latency Comparison

| Pipeline | Evaluation Time (s) |
|----------|--------------------:|
| Basic RAG | 130.10 |
| Hybrid Search | 129.31 |
| Hybrid + Reranking | 125.35 |
| Cited RAG | 132.31 |