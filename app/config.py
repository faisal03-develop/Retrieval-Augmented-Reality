"""Application configuration loaded from environment variables."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")

# Chat LLM: "ollama" (local) or "openai"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower()
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")

# Local embedding model (BAAI/bge-small-en-v1.5 — ~133 MB, strong for retrieval)
MODELS_DIR = BASE_DIR / "models"
LOCAL_EMBEDDING_MODEL = os.getenv("LOCAL_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
LOCAL_EMBEDDING_PATH = BASE_DIR / os.getenv(
    "LOCAL_EMBEDDING_PATH", "models/embeddings/bge-small-en-v1.5"
)
USE_LOCAL_EMBEDDINGS = os.getenv("USE_LOCAL_EMBEDDINGS", "true").lower() in {
    "1",
    "true",
    "yes",
}

MODELS_DIR.mkdir(parents=True, exist_ok=True)

DOCUMENTS_DIR = BASE_DIR / os.getenv("DOCUMENTS_DIR", "data/documents")
VECTOR_STORE_DIR = BASE_DIR / os.getenv("VECTOR_STORE_DIR", "storage/vector_store")

DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
