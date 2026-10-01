"""
rag_core.py (Cloud Version)
---------------------------
Uses Pinecone instead of ChromaDB for cloud deployment.
"""

import os
from dotenv import load_dotenv

load_dotenv()  # Loads GROQ_API_KEY and PINECONE_API_KEY from .env

from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import TokenTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
import time

# ── CONFIG ──────────────────────────────────────────────
KNOWLEDGE_DIR = "./knowledge"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "openai/gpt-oss-20b"          # Current free-tier default on Groq
PINECONE_INDEX_NAME = "rag-knowledge-base"
EMBEDDING_DIMENSION = 384                  # all-MiniLM-L6-v2 outputs 384-dim vectors

if not os.getenv("GROQ_API_KEY"):
    raise ValueError("GROQ_API_KEY not set. Add it to .env or Render environment variables.")
if not os.getenv("PINECONE_API_KEY"):
    raise ValueError("PINECONE_API_KEY not set. Add it to .env or Render environment variables.")


# ── PINECONE INDEX MANAGEMENT ───────────────────────────
def vectorstore_exists() -> bool:
    """Return True only when the Pinecone index exists and contains vectors."""
    try:
        pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        existing_indexes = [idx.name for idx in pc.list_indexes()]
        if PINECONE_INDEX_NAME not in existing_indexes:
            return False

        index = pc.Index(PINECONE_INDEX_NAME)
        stats = index.describe_index_stats()
        return stats.get("total_vector_count", 0) > 0
    except Exception as exc:
        print(f"Warning: unable to check Pinecone index: {exc}")
        return False


def get_or_create_pinecone_index():
    """Create the Pinecone index if it doesn't exist, then return it."""
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

    existing_indexes = [idx.name for idx in pc.list_indexes()]

    if PINECONE_INDEX_NAME not in existing_indexes:
        print(f"Creating Pinecone index '{PINECONE_INDEX_NAME}' ...")
        try:
            pc.create_index(
                name=PINECONE_INDEX_NAME,
                dimension=EMBEDDING_DIMENSION,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
        except Exception as exc:
            if "ALREADY_EXISTS" not in str(exc):
                raise
            print(f"Index '{PINECONE_INDEX_NAME}' already exists. Continuing.")

        # Wait until the index is ready; fail gracefully if it is already ready.
        deadline = time.time() + 60
        while time.time() < deadline:
            try:
                status = pc.describe_index(PINECONE_INDEX_NAME).status
                if status.get("ready"):
                    print("Index ready.")
                    break
            except Exception:
                pass
            time.sleep(1)
        else:
            print("Index creation timed out; continuing anyway.")
    else:
        print(f"Index '{PINECONE_INDEX_NAME}' already exists.")

    return pc.Index(PINECONE_INDEX_NAME)


def index_is_empty(pinecone_index) -> bool:
    """Check if the Pinecone index has any vectors."""
    stats = pinecone_index.describe_index_stats()
    return stats.get("total_vector_count", 0) == 0


# ── BUILD VECTOR STORE ──────────────────────────────────
def build_vectorstore():
    """Load PDFs → chunk → embed → upload to Pinecone. Run ONCE."""
    if vectorstore_exists():
        print("Vector store already exists; loading it instead of rebuilding.")
        return load_vectorstore()

    print("[1/4] Loading PDFs from", KNOWLEDGE_DIR)
    loader = PyPDFDirectoryLoader(KNOWLEDGE_DIR)
    documents = loader.load()

    if not documents:
        raise ValueError("No PDFs found in ./knowledge")

    print(f"      Loaded {len(documents)} pages.")

    print("[2/4] Chunking into 300-token pieces (overlap=50)")
    splitter = TokenTextSplitter(chunk_size=300, chunk_overlap=50)
    chunks = splitter.split_documents(documents)
    print(f"      Created {len(chunks)} chunks.")

    print(f"[3/4] Embedding with {EMBEDDING_MODEL}")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    print(f"[4/4] Uploading to Pinecone index '{PINECONE_INDEX_NAME}'")
    get_or_create_pinecone_index()  # Ensure the index exists
    vectorstore = PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=PINECONE_INDEX_NAME,
    )
    print("      Done. Vector store built.")
    return vectorstore


def load_vectorstore():
    """Load an existing Pinecone index (fast, no re-embedding)."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return PineconeVectorStore(
        index_name=PINECONE_INDEX_NAME,
        embedding=embeddings,
    )


# ── STRICT PROMPT ───────────────────────────────────────
PROMPT_TEMPLATE = """You are an AI customer support assistant for our company.
Answer the user's question using ONLY the company information provided below.
If the answer is not in the provided context, politely say:
"I do not know based on the provided documents."

Context:
{context}

Question: {question}

Answer:"""


# ── QUERY ───────────────────────────────────────────────
def answer_question(question: str, vectorstore) -> dict:
    """Retrieve top 4 chunks, build prompt, call Groq LLM."""
    docs = vectorstore.similarity_search(question, k=4)
    context = "\n\n---\n\n".join(doc.page_content for doc in docs)

    prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)
    formatted_prompt = prompt.format(context=context, question=question)

    llm = ChatGroq(model_name=LLM_MODEL, temperature=0)
    response = llm.invoke(formatted_prompt)

    return {
        "answer": response.content,
        "sources": [doc.page_content for doc in docs],
    }