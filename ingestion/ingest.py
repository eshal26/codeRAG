import io
import hashlib
import os
import re
import requests

from parser.ast_parser import extract_functions_from_zip
from embeddings.embedder import embed_batch, create_embedding_text
from retriever.vector_store import upsert_vectors, list_indexed_repos, ensure_collection
from database.db import get_hashes, save_hashes, upsert_repo, repo_exists


def parse_github_url(github_url):
    """Parse GitHub URL to extract owner and repo name.
    
    Args:
        github_url: URL like https://github.com/owner/repo
        
    Returns:
        Tuple of (owner, repo_name)
        
    Raises:
        ValueError: If URL is not a valid GitHub URL
    """
    # Validate it's a GitHub URL
    if not re.match(r'https?://(www\.)?github\.com/[^/]+/[^/]+/?$', github_url):
        raise ValueError(f"Invalid GitHub URL: {github_url}. Expected format: https://github.com/owner/repo")
    
    parts = github_url.rstrip("/").split("/")
    if len(parts) < 2:
        raise ValueError(f"Invalid GitHub URL format: {github_url}")
    
    return parts[-2], parts[-1]


def get_default_branch(owner, repo):
    api_url = f"https://api.github.com/repos/{owner}/{repo}"
    headers = {}
    token = os.getenv("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"token {token}"
    response = requests.get(api_url, headers=headers)
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

    repo_name = github_url.rstrip("/").split("/")[-1]

    try:
        is_update = repo_exists(repo_name)
    except Exception as e:
        print(f"DB check failed, falling back to file check: {e}")
        is_update = repo_name in list_indexed_repos()

    # Verify Qdrant collection actually exists — if not, force full re-index
    if is_update:
        try:
            from retriever.vector_store import get_client
            client = get_client()
            collections = [c.name for c in client.get_collections().collections]
            if repo_name not in collections:
                print(f"Qdrant collection missing for '{repo_name}', forcing full re-index.")
                is_update = False
        except Exception as e:
            print(f"Qdrant check failed: {e}")

    if is_update and not force:
        print(f"'{repo_name}' already indexed. Checking for changes...")
    elif not is_update:
        print(f"New repo: '{repo_name}'. Full index build.")

    zip_bytes, repo_name = download_zip(github_url)

    try:
        old_hashes = get_hashes(repo_name) if is_update else {}
    except Exception:
        old_hashes = {}

    new_hashes = {}
    changed_files = []
    unchanged_files = []

    with zipfile.ZipFile(zip_bytes) as zf:
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

    MAX_CHUNKS = 800
    if len(new_chunks) > MAX_CHUNKS:
        print(f"Capping chunks from {len(new_chunks)} to {MAX_CHUNKS}")
        new_chunks = new_chunks[:MAX_CHUNKS]
        capped = True
    else:
        capped = False

    print(f"Embedding and uploading {len(new_chunks)} chunks...")
    
    # Delete old collection if full re-index (not an update)
    if not is_update:
        try:
            from retriever.vector_store import delete_collection
            from database.db import delete_repo_metadata
            delete_collection(repo_name)
            delete_repo_metadata(repo_name)  # Clean up PostgreSQL metadata too
        except Exception:
            pass  # Collection may not exist on first index
    
    ensure_collection(repo_name)

    # Embed and upload in batches — keeps RAM low
    BATCH_SIZE = 32
    all_functions = []
    all_texts = []
    id_counter = 0  # Global ID counter to avoid collisions

    for i in range(0, len(new_chunks), BATCH_SIZE):
        batch = new_chunks[i:i + BATCH_SIZE]
        texts = [create_embedding_text(c) for c in batch]
        embeddings = embed_batch(texts)

        functions = [
            {
                "function_name": c["function_name"],
                "class_name": c.get("class_name", ""),
                "file_path": c["file_path"],
                "start_line": c["start_line"],
                "end_line": c["end_line"],
                "docstring": c.get("docstring", ""),
                "code": c.get("code", ""),
            }
            for c in batch
        ]

        upsert_vectors(repo_name, embeddings, texts, functions, id_offset=id_counter)
        id_counter += len(batch)
        all_functions.extend(functions)
        all_texts.extend(texts)
        print(f"Processed {min(i + BATCH_SIZE, len(new_chunks))}/{len(new_chunks)} chunks")

    try:
        save_hashes(repo_name, new_hashes)
        upsert_repo(repo_name, github_url, len(all_functions))
    except Exception as e:
        print(f"DB save failed (non-fatal): {e}")

    action = "updated" if is_update else "indexed"
    cap_msg = " (capped at 800 chunks)" if capped else ""
    print(f"Index complete. '{repo_name}' {action} with {len(all_functions)} total chunks{cap_msg}.")
    return repo_name