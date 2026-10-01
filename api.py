"""
api.py
------
FastAPI server that:
  - Builds or loads the Chroma vector store on startup
  - Exposes POST /query  → runs the RAG loop
  - Serves GET  /        → the HTML chat frontend
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from rag_core import (
    build_vectorstore,
    load_vectorstore,
    vectorstore_exists,
    answer_question,
)

app = FastAPI(title="RAG Support Bot")

# CORS: allow the browser to call this API from anywhere.
# For local learning, "*" is fine. Restrict this in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------
# STARTUP: build or load the vector store once
# -----------------------------------------------------------------
print("Starting up...")
if vectorstore_exists():
    print("Found existing Chroma DB. Loading...")
    vs = load_vectorstore()
else:
    print("No Chroma DB found. Building from PDFs...")
    vs = build_vectorstore()


# -----------------------------------------------------------------
# REQUEST / RESPONSE SCHEMAS
# -----------------------------------------------------------------
class Query(BaseModel):
    question: str


class Answer(BaseModel):
    answer: str
    sources: list[str]


# -----------------------------------------------------------------
# ROUTES
# -----------------------------------------------------------------
@app.post("/query", response_model=Answer)
def query(q: Query):
    """Run the full RAG loop for a user question."""
    result = answer_question(q.question, vs)
    return Answer(answer=result["answer"], sources=result["sources"])


@app.get("/")
def serve_frontend():
    """Serve the chat UI."""
    return FileResponse("index.html")