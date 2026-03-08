import os
import time
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, Distance, PointStruct
)

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
VECTOR_DIM = 384
INDEX_DIR = "indexes"
MAX_CHUNKS = 500  # middle ground for large repos


def get_client():
    if not QDRANT_URL or not QDRANT_API_KEY:
        raise RuntimeError("QDRANT_URL and QDRANT_API_KEY must be set in environment variables")
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60)


def ensure_collection(repo_name):
    """Create collection if it doesn't exist."""
    client = get_client()
    collections = [c.name for c in client.get_collections().collections]
    if repo_name not in collections:
        try:
            client.create_collection(
                collection_name=repo_name,
                vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE)
            )
            print(f"Created Qdrant collection: {repo_name}")
        except Exception as e:
            print(f"Warning: Failed to create collection: {e}")
            raise


def delete_collection(repo_name):
    """Delete all vectors for a repo (used on full re-index)."""
    client = get_client()
    collections = [c.name for c in client.get_collections().collections]
    if repo_name in collections:
        try:
            client.delete_collection(collection_name=repo_name)
            print(f"Deleted Qdrant collection: {repo_name}")
        except Exception as e:
            print(f"Warning: Failed to delete collection: {e}")
            raise


def upsert_vectors(repo_name, embeddings, texts, functions, id_offset=0):
    """Upsert vectors to Qdrant with retry logic.
    
    Args:
        repo_name: Collection name
        embeddings: List of embedding vectors (numpy arrays)
        texts: List of text chunks
        functions: List of function metadata dicts
        id_offset: Starting ID for this batch (ensures globally unique IDs)
    """
    client = get_client()

    points = []
    for i, (emb, text, func) in enumerate(zip(embeddings, texts, functions)):
        point_id = id_offset + i
        points.append(PointStruct(
            id=point_id,
            vector=emb.tolist(),
            payload={"text": text, "function": func, "repo": repo_name}
        ))

    # Upload in batches of 32 with retry
    batch_size = 32
    for i in range(0, len(points), batch_size):
        batch = points[i:i + batch_size]
        for attempt in range(3):
            try:
                client.upsert(collection_name=repo_name, points=batch)
                break
            except Exception as e:
                if attempt == 2:
                    print(f"Failed to upsert batch after 3 attempts: {e}")
                    raise
                print(f"Upsert retry {attempt + 1}/2 after error: {e}")
                time.sleep(2)


def search_repo(query_embedding, repo_name, k=3, query=None):
    client = get_client()
    results = client.search(
        collection_name=repo_name,
        query_vector=query_embedding.tolist(),
        limit=k
    )
    return [
        {
            "text": r.payload["text"],
            "function": r.payload["function"],
            "repo": repo_name,
            "score": r.score
        }
        for r in results
    ]


def search_all_repos(query_embedding, k=3, query=None):
    client = get_client()
    collections = [c.name for c in client.get_collections().collections]

    all_results = []
    for repo_name in collections:
        try:
            results = search_repo(query_embedding, repo_name, k=k * 2)
            all_results.extend(results)
        except Exception as e:
            print(f"Error searching '{repo_name}': {e}")

    all_results.sort(key=lambda x: x["score"], reverse=True)
    return all_results[:k]


def list_indexed_repos():
    try:
        client = get_client()
        return [c.name for c in client.get_collections().collections]
    except Exception:
        if not os.path.exists(INDEX_DIR):
            return []
        return [f.replace(".index", "") for f in os.listdir(INDEX_DIR) if f.endswith(".index")]