from dotenv import load_dotenv
load_dotenv()

import os
import json
import uuid
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import uvicorn

from ingestion.ingest import ingest_repo
from retriever.vector_store import search_repo, search_all_repos, list_indexed_repos
from generator.answer import stream_answer
from embeddings.embedder import model as embedding_model
from database.db import init_db, get_all_repos, save_query

limiter = Limiter(key_func=get_remote_address)

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield

app = FastAPI(lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class IngestRequest(BaseModel):
    github_url: str


class QueryRequest(BaseModel):
    question: str
    repo: str
    history: list = []


@app.get("/")
def root():
    path = os.path.join(BASE_DIR, "static", "index.html")
    with open(path, "r", encoding="utf-8") as f:
        html = f.read()
    return HTMLResponse(content=html)


def run_ingest(job_id: str, github_url: str):
    """Background task — runs ingestion and updates job status."""
    try:
        jobs[job_id] = {"status": "downloading", "message": "Downloading repo..."}
        repo_name = ingest_repo(github_url)
        if not repo_name:
            jobs[job_id] = {"status": "error", "message": "No functions found."}
        else:
            jobs[job_id] = {"status": "done", "message": f"'{repo_name}' indexed successfully.", "repo": repo_name}
    except Exception as e:
        jobs[job_id] = {"status": "error", "message": str(e)}


@app.post("/ingest")
@limiter.limit("5/minute")
def ingest(req: IngestRequest, request: Request, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "queued", "message": "Starting..."}
    background_tasks.add_task(run_ingest, job_id, req.github_url)
    return {"job_id": job_id, "status": "queued"}


@app.get("/status/{job_id}")
def status(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    return job


@app.post("/stream")
@limiter.limit("20/minute")
def stream(req: QueryRequest, request: Request):
    query_embedding = embedding_model.encode(req.question)

    if req.repo == "all":
        results = search_all_repos(query_embedding, k=3, query=req.question)
    else:
        results = search_repo(query_embedding, req.repo, k=3, query=req.question)

    if not results:
        raise HTTPException(status_code=404, detail="No results found.")

    retrieved_texts = [r["text"] for r in results]

    sources = []
    for r in results:
        f = r["function"]
        class_prefix = f"{f['class_name']}." if f.get("class_name") else ""
        sources.append({
            "repo": r["repo"],
            "function": f"{class_prefix}{f['function_name']}",
            "file": f["file_path"].split("/")[-1],
            "line": f["start_line"],
            "score": round(r["score"], 3),
            "code": f.get("code", "")
        })

    full_answer = []

    def event_stream():
        yield f"data: {json.dumps({'sources': sources})}\n\n"
        for chunk in stream_answer(req.question, retrieved_texts, repo_name=req.repo, history=req.history):
            full_answer.append(chunk)
            yield f"data: {json.dumps({'chunk': chunk})}\n\n"
        save_query(req.question, "".join(full_answer), req.repo)
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/reindex")
@limiter.limit("5/minute")
def reindex(req: IngestRequest, request: Request, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "queued", "message": "Starting re-index..."}
    background_tasks.add_task(run_ingest, job_id, req.github_url)
    return {"job_id": job_id, "status": "queued"}


@app.get("/repos")
def repos():
    return {"repos": get_all_repos()}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)