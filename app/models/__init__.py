"""Local model loading for RAG."""

from app.models.embeddings import get_embeddings, is_local_model_ready

__all__ = ["get_embeddings", "is_local_model_ready"]
