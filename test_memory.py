from src.embeddings import get_collection

collection = get_collection()

print("COUNT:", collection.count())

results = collection.query(
    query_texts=["Swayam"],
    n_results=5
)

print(results["documents"])