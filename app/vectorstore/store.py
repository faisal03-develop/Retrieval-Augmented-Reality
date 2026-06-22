"""Vector store creation and persistence."""

from pathlib import Path

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from app.config import EMBEDDING_MODEL, OPENAI_API_KEY


def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=OPENAI_API_KEY)


def create_vector_store(chunks: list, persist_dir: Path) -> Chroma:
    """Create and persist a Chroma vector store from document chunks."""
    return Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        persist_directory=str(persist_dir),
    )


def load_vector_store(persist_dir: Path) -> Chroma:
    """Load an existing Chroma vector store from disk."""
    return Chroma(
        persist_directory=str(persist_dir),
        embedding_function=get_embeddings(),
    )
