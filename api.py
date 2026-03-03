import os
import json
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
import uvicorn

from ingestion.ingest import ingest_repo
from retriever.vector_store import search_repo, search_all_repos, list_indexed_repos
from generator.answer import generate_answer, stream_answer
from embeddings.embedder import model as embedding_model

app = FastAPI()
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


@app.post("/ingest")
def ingest(req: IngestRequest):
    try:
        repo_name = ingest_repo(req.github_url)
        if not repo_name:
            raise HTTPException(status_code=400, detail="No Python functions found.")
        return {"status": "success", "repo": repo_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/stream")
def stream(req: QueryRequest):
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

    def event_stream():
        yield f"data: {json.dumps({'sources': sources})}\n\n"
        for chunk in stream_answer(req.question, retrieved_texts, repo_name=req.repo, history=req.history):
            yield f"data: {json.dumps({'chunk': chunk})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/reindex")
def reindex(req: IngestRequest):
    try:
        repo_name = ingest_repo(req.github_url, force=False)
        if not repo_name:
            raise HTTPException(status_code=400, detail="No functions found.")
        return {"status": "success", "repo": repo_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/repos")
def repos():
    return {"repos": list_indexed_repos()}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)