"""Local and remote embedding model factory."""

from pathlib import Path

from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings

from app.config import (
    EMBEDDING_MODEL,
    LOCAL_EMBEDDING_PATH,
    OPENAI_API_KEY,
    USE_LOCAL_EMBEDDINGS,
)

BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


def is_local_model_ready(path: Path | None = None) -> bool:
    """Return True if the local embedding model has been downloaded."""
    model_path = path or LOCAL_EMBEDDING_PATH
    return (model_path / "config.json").exists()


def get_embeddings() -> Embeddings:
    """Return the configured embedding model (local by default)."""
    if USE_LOCAL_EMBEDDINGS:
        if not is_local_model_ready():
            raise FileNotFoundError(
                f"Local embedding model not found at {LOCAL_EMBEDDING_PATH}. "
                "Run: python -m scripts.download_models"
            )
        from langchain_huggingface import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(
            model_name=str(LOCAL_EMBEDDING_PATH),
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
            query_encode_kwargs={
                "prompt": BGE_QUERY_INSTRUCTION,
                "normalize_embeddings": True,
            },
        )

    return OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=OPENAI_API_KEY)
