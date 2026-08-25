import uuid

from sentence_transformers import SentenceTransformer
import chromadb

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

chroma_client = chromadb.PersistentClient(path="chroma_db")
collection = chroma_client.get_or_create_collection(name="documents")


def get_collection():
    return collection


def get_embedding_function():
    return embedding_model


def generate_embeddings(texts):
    embeddings = embedding_model.encode(texts)
    return embeddings.tolist()


def store_chunks(chunks):
    texts = [chunk["text"] for chunk in chunks]
    metadatas = [chunk["metadata"] for chunk in chunks]

    ids = [
        str(uuid.uuid4())
        for _ in chunks
    ]
    embeddings = generate_embeddings(texts)

    collection.upsert(
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )

    print(f"Stored {len(chunks)} chunks in ChromaDB")