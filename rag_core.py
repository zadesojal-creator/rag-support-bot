"""
rag_core.py
-----------
The RAG engine. Contains:
  - build_vectorstore()  → Steps 1-4 (load, chunk, embed, store)
  - load_vectorstore()   → Load existing Chroma DB
  - answer_question()    → Steps 5-8 (embed query, search, prompt, LLM)
"""

import os
from dotenv import load_dotenv

load_dotenv()

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import TokenTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

# ---------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------
KNOWLEDGE_DIR = "./knowledge"
CHROMA_DIR = "./chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # 384-dim, tiny (~80 MB), CPU-friendly
LLM_MODEL = "openai/gpt-oss-20b"     # Groq's fast Llama 3.1 8B
CHUNK_SIZE = 300                       # tokens per chunk
CHUNK_OVERLAP = 50                     # overlap to avoid cutting sentences
TOP_K = 4                              # how many chunks to retrieve

# Fail fast if the API key is missing
if not os.getenv("GROQ_API_KEY"):
    raise ValueError(
        "GROQ_API_KEY is not set. Get one free at https://console.groq.com"
    )


# ===============================================================
# STEPS 1-4: LOAD → CHUNK → EMBED → STORE
# ===============================================================
def build_vectorstore():
    """
    Reads all PDFs in ./knowledge, chunks them, embeds them,
    and stores them in a local ChromaDB.
    Run this ONCE (or whenever your PDFs change).
    """

    # ---- STEP 1: LOAD -------------------------------------------------
    print("[1/4] Loading PDFs from", KNOWLEDGE_DIR)
    loader = PyPDFDirectoryLoader(KNOWLEDGE_DIR)
    documents = loader.load()

    if not documents:
        raise ValueError("No PDFs found. Add some to ./knowledge")

    print(f"      Loaded {len(documents)} pages.")

    # ---- STEP 2: CHUNK ------------------------------------------------
    # TokenTextSplitter splits by tokens (not characters), which matches
    # how the LLM actually sees text. Overlap prevents a sentence from
    # being cut in half between two chunks.
    print(f"[2/4] Chunking into {CHUNK_SIZE}-token pieces "
          f"(overlap={CHUNK_OVERLAP})")
    splitter = TokenTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(documents)
    print(f"      Created {len(chunks)} chunks.")

    # ---- STEP 3: EMBED ------------------------------------------------
    # The embedding model converts text → 384 numbers (a vector).
    # Similar meanings produce similar vectors.
    print(f"[3/4] Embedding with {EMBEDDING_MODEL} "
          f"(first run downloads ~80 MB)")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    # ---- STEP 4: STORE ------------------------------------------------
    # ChromaDB writes the vectors + original text to ./chroma_db
    print(f"[4/4] Storing in ChromaDB at {CHROMA_DIR}")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
    )
    print("      Done. Vector store built.\n")
    return vectorstore


def load_vectorstore():
    """Load an already-built ChromaDB from disk (fast, no re-embedding)."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
    )


def vectorstore_exists() -> bool:
    """Check if ./chroma_db already exists and has data."""
    return os.path.isdir(CHROMA_DIR) and len(os.listdir(CHROMA_DIR)) > 0


# ===============================================================
# STEPS 5-8: EMBED QUERY → SEARCH → PROMPT → LLM
# ===============================================================

# ---- STEP 7: THE STRICT PROMPT ------------------------------------
# This is the most important part of RAG. It forces the LLM to answer
# ONLY from the retrieved context. Without this, the LLM would make
# things up (hallucinate) using its training data.
PROMPT_TEMPLATE = """You are an AI customer support assistant for our company.
Answer the user's question using the company information provided below.


Context:
{context}

Question: {question}

Answer:"""


def answer_question(question: str, vectorstore) -> dict:
    """
    Runs steps 5-8 for a single question.
    Returns a dict with the answer and the source chunks (for transparency).
    """

    # ---- STEP 5 + 6: EMBED QUERY + SEARCH -----------------------------
    # The question is embedded with the SAME model, then Chroma finds
    # the TOP_K chunks whose vectors are closest (cosine similarity).
    docs = vectorstore.similarity_search(question, k=TOP_K)

    # Join the retrieved chunks into one block of context text.
    # The separator makes it obvious where one chunk ends and another begins.
    context = "\n\n---\n\n".join(doc.page_content for doc in docs)

    # ---- STEP 7: BUILD THE PROMPT -------------------------------------
    prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)
    formatted_prompt = prompt.format(context=context, question=question)

    # ---- STEP 8: SEND TO GROQ -----------------------------------------
    # temperature=0 makes the model deterministic (no creative drift).
    llm = ChatGroq(model_name=LLM_MODEL, temperature=0)
    response = llm.invoke(formatted_prompt)

    return {
        "answer": response.content,
        "sources": [doc.page_content for doc in docs],
    }