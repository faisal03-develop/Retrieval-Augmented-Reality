"""Download and cache local models for RAG."""

from app.config import LOCAL_EMBEDDING_MODEL, LOCAL_EMBEDDING_PATH
from app.models.embeddings import is_local_model_ready


def download_embedding_model() -> None:
    """Download the embedding model and save it under models/embeddings/."""
    if is_local_model_ready():
        print(f"Embedding model already present at {LOCAL_EMBEDDING_PATH}")
        return

    from sentence_transformers import SentenceTransformer

    print(f"Downloading {LOCAL_EMBEDDING_MODEL} (~133 MB)...")
    LOCAL_EMBEDDING_PATH.parent.mkdir(parents=True, exist_ok=True)

    model = SentenceTransformer(LOCAL_EMBEDDING_MODEL)
    model.save(str(LOCAL_EMBEDDING_PATH))

    vector = model.encode("RAG retrieval test")
    print(f"Saved to {LOCAL_EMBEDDING_PATH}")
    print(f"Embedding dimension: {len(vector)}")


if __name__ == "__main__":
    download_embedding_model()
    print("Local models are ready.")
