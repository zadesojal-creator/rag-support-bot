"""
api.py (Cloud Version)
----------------------
On startup: checks Pinecone index. If empty → builds it.
If populated → just loads it. This prevents re-processing PDFs on every cold start.
"""

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from rag_core import (
    get_or_create_pinecone_index,
    index_is_empty,
    build_vectorstore,
    load_vectorstore,
    answer_question,
)

load_dotenv()

app = FastAPI(title="RAG Support Bot")

# CORS: allow browser access from anywhere
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── STARTUP: build or load vector store ─────────────────
print("Starting up...")
pinecone_index = get_or_create_pinecone_index()

if index_is_empty(pinecone_index):
    print("Index is empty. Building from PDFs...")
    vs = build_vectorstore()
else:
    stats = pinecone_index.describe_index_stats()
    print(f"Found {stats.get('total_vector_count', 0)} existing vectors. Loading...")
    vs = load_vectorstore()


# ── SCHEMAS ─────────────────────────────────────────────
class Query(BaseModel):
    question: str


class Answer(BaseModel):
    answer: str
    sources: list[str]


# ── ROUTES ──────────────────────────────────────────────
@app.post("/query", response_model=Answer)
def query(q: Query):
    result = answer_question(q.question, vs)
    return Answer(answer=result["answer"], sources=result["sources"])


@app.get("/health")
def health():
    """Health check endpoint — keeps the server warm and lets Render know it's alive."""
    return {"status": "ok"}


@app.get("/")
def serve_frontend():
    return FileResponse("index.html")