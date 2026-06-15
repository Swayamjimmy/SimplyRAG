from sentence_transformers import SentenceTransformer
import chromadb

# Load the embedding model locally (384-dimensional vectors)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize ChromaDB with persistent storage
chroma_client = chromadb.PersistentClient(path="chroma_db")
collection = chroma_client.get_or_create_collection(name="documents")

def generate_embeddings(texts):
    """Generate 384-dim embeddings for a list of texts."""
    embeddings = embedding_model.encode(texts)
    return embeddings.tolist()

def store_chunks(chunks):
    """Store chunks with embeddings and metadata in ChromaDB."""
    texts = [chunk["text"] for chunk in chunks]
    metadatas = [chunk["metadata"] for chunk in chunks]
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    embeddings = generate_embeddings(texts)

    # Upsert allows re-running without duplicate ID errors
    collection.upsert(
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )
    print(f"Stored {len(chunks)} chunks in ChromaDB")