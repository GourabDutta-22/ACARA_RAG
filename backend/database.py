import os
import chromadb
from datetime import datetime, timezone
from motor.motor_asyncio import AsyncIOMotorClient
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings

# MongoDB Setup
MONGO_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
client = AsyncIOMotorClient(MONGO_URI)
db = client.agentic_rag
feedback_collection = db.feedback
chat_collection = db.chat_history

# Vector Store Logic (Hybrid: Pinecone for Cloud / Chroma for Local)
PINECONE_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "acara-index")

USE_PINECONE = PINECONE_KEY is not None and PINECONE_KEY != "your_pinecone_api_key_here"

# Initialize Embeddings globally for reuse
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")

if USE_PINECONE:
    pc = Pinecone(api_key=PINECONE_KEY)
    vector_store = pc.Index(PINECONE_INDEX_NAME)
    print(f"🌲 Using Pinecone Cloud Index: {PINECONE_INDEX_NAME}")
else:
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    vector_store = chroma_client.get_or_create_collection(name="documents")
    print("💾 Using Local ChromaDB (Ephemeral)")


def add_document_to_vector_store(doc_id: str, text: str, metadata: dict = None):
    """Upsert a document into the vector store with a freshness timestamp."""
    base_meta = {
        "freshness": datetime.now(timezone.utc).isoformat(),
        "text": text
    }
    if metadata:
        base_meta.update(metadata)
    
    if USE_PINECONE:
        # Generate embedding on the fly if not provided
        emb = embeddings_model.embed_query(text)
        vector_store.upsert(
            vectors=[{"id": doc_id, "values": emb, "metadata": base_meta}]
        )
    else:
        vector_store.upsert(
            documents=[text],
            ids=[doc_id],
            metadatas=[base_meta],
        )

def search_vector_store(query_text: str, n_results: int = 3, embedding: list = None):
    """Query the selected vector store."""
    if USE_PINECONE:
        # Use provided embedding or generate one
        query_emb = embedding or embeddings_model.embed_query(query_text)
        
        results = vector_store.query(
            vector=query_emb,
            top_k=n_results,
            include_metadata=True
        )
        # Reformat to match ChromaDB's structure for agent compatibility
        docs = [match.metadata.get("text", "") for match in results.matches]
        metas = [match.metadata for match in results.matches]
        dists = [1 - match.score for match in results.matches] # convert to distance
        return {"documents": [docs], "metadatas": [metas], "distances": [dists]}
    else:
        count = vector_store.count()
        safe_n = min(n_results, count) if count > 0 else 1
        return vector_store.query(
            query_texts=[query_text],
            n_results=max(1, safe_n),
            include=["documents", "distances", "metadatas"],
        )

def get_collection_stats() -> dict:
    """Return basic stats about the vector store."""
    if USE_PINECONE:
        stats = vector_store.describe_index_stats()
        return {"document_count": stats.total_vector_count, "collection_name": PINECONE_INDEX_NAME}
    else:
        count = vector_store.count()
        return {"document_count": count, "collection_name": vector_store.name}
