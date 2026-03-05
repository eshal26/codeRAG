import io
import hashlib
import os
import requests

from parser.ast_parser import extract_functions_from_zip
from parser.js_parser import extract_js_from_zip
from embeddings.embedder import embed_functions
from retriever.vector_store import build_index, save_index, list_indexed_repos, INDEX_DIR
from database.db import get_hashes, save_hashes, upsert_repo, repo_exists


def parse_github_url(github_url):
    parts = github_url.rstrip("/").split("/")
    return parts[-2], parts[-1]


def get_default_branch(owner, repo):
    api_url = f"https://api.github.com/repos/{owner}/{repo}"
    response = requests.get(api_url)
    response.raise_for_status()
    return response.json().get("default_branch", "main")


def download_zip(github_url):
    owner, repo = parse_github_url(github_url)
    branch = get_default_branch(owner, repo)
    zip_url = f"https://github.com/{owner}/{repo}/archive/refs/heads/{branch}.zip"
    print(f"Downloading {zip_url}...")
    response = requests.get(zip_url, stream=True)
    response.raise_for_status()
    return io.BytesIO(response.content), repo


def _hash_file(content: bytes) -> str:
    return hashlib.md5(content).hexdigest()


def ingest_repo(github_url, force=False):
    import zipfile
    import numpy as np

    repo_name = github_url.rstrip("/").split("/")[-1]
    is_update = repo_exists(repo_name)

    if is_update and not force:
        print(f"'{repo_name}' already indexed. Checking for changes...")
    elif not is_update:
        print(f"New repo: '{repo_name}'. Full index build.")

    zip_bytes, repo_name = download_zip(github_url)

    old_hashes = get_hashes(repo_name) if is_update else {}
    new_hashes = {}
    changed_files = []
    unchanged_files = []

    with zipfile.ZipFile(zip_bytes) as zf:
        # Identify changed files
        for zip_path in zf.namelist():
            if not (zip_path.endswith(".py") or zip_path.endswith(".js")):
                continue
            content = zf.read(zip_path)
            file_hash = _hash_file(content)
            new_hashes[zip_path] = file_hash

            if old_hashes.get(zip_path) != file_hash:
                changed_files.append(zip_path)
            else:
                unchanged_files.append(zip_path)

        if is_update and not changed_files:
            print(f"No changes detected in '{repo_name}'. Skipping re-index.")
            return repo_name

        if is_update:
            print(f"Changed files: {len(changed_files)} | Unchanged: {len(unchanged_files)}")
        
        # Parse only changed files
        new_chunks = []
        for zip_path in changed_files:
            with zf.open(zip_path) as f:
                try:
                    source_code = f.read().decode("utf-8")
                except UnicodeDecodeError:
                    continue

            if zip_path.endswith(".py"):
                from parser.ast_parser import extract_functions_from_file_content
                new_chunks.extend(extract_functions_from_file_content(source_code, zip_path))
            elif zip_path.endswith(".js"):
                from parser.js_parser import extract_functions_from_js
                new_chunks.extend(extract_functions_from_js(source_code, zip_path))

    if not new_chunks and not is_update:
        print("No functions found.")
        return None

    print(f"Embedding {len(new_chunks)} chunks from changed files...")
    new_embeddings, new_texts = embed_functions(new_chunks)

    # If updating, load old index and merge keeping unchanged file chunks
    if is_update:
        from retriever.vector_store import load_index
        import faiss

        old_index, old_meta = load_index(repo_name)

        # Keep chunks from unchanged files
        kept_chunks = [
            (old_meta["texts"][i], old_meta["functions"][i])
            for i in range(len(old_meta["functions"]))
            if old_meta["functions"][i]["file_path"] not in changed_files
        ]

        kept_texts = [c[0] for c in kept_chunks]
        kept_functions = [c[1] for c in kept_chunks]

        # Reconstruct old embeddings for kept chunks
        old_vectors = faiss.rev_swig_ptr(old_index.get_xb(), old_index.ntotal * old_index.d)
        old_vectors = old_vectors.reshape(old_index.ntotal, old_index.d)
        kept_indices = [
            i for i in range(len(old_meta["functions"]))
            if old_meta["functions"][i]["file_path"] not in changed_files
        ]
        kept_vectors = old_vectors[kept_indices]

        # Embed new chunks and normalize
        new_emb_array = np.array(new_embeddings).astype("float32")
        faiss.normalize_L2(new_emb_array)

        # Merge
        all_vectors = np.vstack([kept_vectors, new_emb_array])
        all_texts = kept_texts + new_texts
        all_functions = kept_functions + [
            {
                "function_name": c["function_name"],
                "class_name": c["class_name"],
                "file_path": c["file_path"],
                "start_line": c["start_line"],
                "end_line": c["end_line"],
                "docstring": c["docstring"],
                "code": c.get("code", ""),
            }
            for c in new_chunks
        ]
    else:
        # Full index build
        import numpy as np
        import faiss as faiss_mod

        all_vectors = np.array(new_embeddings).astype("float32")
        faiss_mod.normalize_L2(all_vectors)
        all_texts = new_texts
        all_functions = [
            {
                "function_name": c["function_name"],
                "class_name": c["class_name"],
                "file_path": c["file_path"],
                "start_line": c["start_line"],
                "end_line": c["end_line"],
                "docstring": c["docstring"],
                "code": c.get("code", ""),
            }
            for c in new_chunks
        ]

    import faiss
    dim = all_vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(all_vectors)

    meta = {
        "repo_name": repo_name,
        "github_url": github_url,
        "total_functions": len(all_functions),
        "texts": all_texts,
        "functions": all_functions
    }

    save_index(repo_name, index, meta)
    save_hashes(repo_name, new_hashes)
    upsert_repo(repo_name, github_url, len(all_functions))

    action = "updated" if is_update else "indexed"
    print(f"Done. '{repo_name}' {action} with {len(all_functions)} total chunks.")
    return repo_name