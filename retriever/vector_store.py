import os
import json
import faiss
import numpy as np
from sentence_transformers import CrossEncoder

INDEX_DIR = "indexes"

# Re-ranker model - loads once at import time
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def save_index(repo_name, index, meta):
    os.makedirs(INDEX_DIR, exist_ok=True)
    faiss.write_index(index, os.path.join(INDEX_DIR, f"{repo_name}.index"))
    with open(os.path.join(INDEX_DIR, f"{repo_name}_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


def load_index(repo_name):
    index_path = os.path.join(INDEX_DIR, f"{repo_name}.index")
    meta_path = os.path.join(INDEX_DIR, f"{repo_name}_meta.json")

    if not os.path.exists(index_path):
        raise FileNotFoundError(f"No index found for repo: '{repo_name}'. Ingest it first.")

    index = faiss.read_index(index_path)
    with open(meta_path, "r") as f:
        meta = json.load(f)

    return index, meta


def build_index(embeddings):
    embeddings_array = np.array(embeddings).astype("float32")
    faiss.normalize_L2(embeddings_array)

    index = faiss.IndexFlatIP(embeddings_array.shape[1])
    index.add(embeddings_array)
    return index


def rerank(query, results, top_k=3):
    """Re-rank results using a cross-encoder for better relevance."""
    pairs = [(query, r["text"]) for r in results]
    scores = reranker.predict(pairs)

    for i, result in enumerate(results):
        result["rerank_score"] = float(scores[i])

    results.sort(key=lambda x: x["rerank_score"], reverse=True)
    return results[:top_k]


def search_repo(query_embedding, repo_name, k=3, query=None):
    index, meta = load_index(repo_name)

    query_arr = np.array([query_embedding]).astype("float32")
    faiss.normalize_L2(query_arr)

    fetch_k = k * 4 if query else k
    distances, indices = index.search(query_arr, fetch_k)

    results = []
    for i, idx in enumerate(indices[0]):
        results.append({
            "text": meta["texts"][idx],
            "function": meta["functions"][idx],
            "repo": repo_name,
            "score": float(distances[0][i])
        })

    if query:
        results = rerank(query, results, top_k=k)

    return results


def search_all_repos(query_embedding, k=3, query=None):
    all_results = []

    for file in os.listdir(INDEX_DIR):
        if not file.endswith(".index"):
            continue
        repo_name = file.replace(".index", "")
        try:
            results = search_repo(query_embedding, repo_name, k=k * 2, query=None)
            all_results.extend(results)
        except Exception as e:
            print(f"Error searching '{repo_name}': {e}")

    if query:
        all_results = rerank(query, all_results, top_k=k)
    else:
        all_results.sort(key=lambda x: x["score"], reverse=True)
        all_results = all_results[:k]

    return all_results


def list_indexed_repos():
    if not os.path.exists(INDEX_DIR):
        return []
    return [f.replace(".index", "") for f in os.listdir(INDEX_DIR) if f.endswith(".index")]