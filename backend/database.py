import os
from datetime import datetime, timezone
from motor.motor_asyncio import AsyncIOMotorClient
import chromadb

# MongoDB Setup
MONGO_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
client = AsyncIOMotorClient(MONGO_URI)
db = client.agentic_rag
feedback_collection = db.feedback
chat_collection = db.chat_history

# ChromaDB Setup
chroma_client = chromadb.PersistentClient(path="./chroma_db")
vector_store = chroma_client.get_or_create_collection(name="documents")


def add_document_to_chroma(doc_id: str, text: str, metadata: dict = None):
    """Upsert a document into the vector store with a freshness timestamp."""
    base_meta = {"freshness": datetime.now(timezone.utc).isoformat()}
    if metadata:
        base_meta.update(metadata)
    vector_store.upsert(
        documents=[text],
        ids=[doc_id],
        metadatas=[base_meta],
    )


def search_chroma(query: str, n_results: int = 3):
    """Query the vector store. Returns documents, distances, and metadata."""
    # Guard: don't query more than what's stored
    count = vector_store.count()
    safe_n = min(n_results, count) if count > 0 else 1
    results = vector_store.query(
        query_texts=[query],
        n_results=max(1, safe_n),
        include=["documents", "distances", "metadatas"],
    )
    return results


def get_collection_stats() -> dict:
    """Return basic stats about the vector store."""
    count = vector_store.count()
    return {"document_count": count, "collection_name": vector_store.name}
