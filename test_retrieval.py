from src.pipeline import RerankedRAGPipeline

pipeline = RerankedRAGPipeline()

docs = pipeline.retriever.retrieve(
    "What is Swayam's name?",
    k=5
)

for d in docs:
    print(d["metadata"])
    print(d["text"][:300])
    print()