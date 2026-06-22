"""Vector store creation and persistence."""

from pathlib import Path

from langchain_chroma import Chroma

from app.models.embeddings import get_embeddings

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


def vector_store_exists(persist_dir: Path) -> bool:
    """Return True if a persisted Chroma database is present."""
    return (persist_dir / "chroma.sqlite3").exists()
