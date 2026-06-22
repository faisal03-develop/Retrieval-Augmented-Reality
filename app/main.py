"""CLI entry point for the RAG chatbot."""

from app.config import DOCUMENTS_DIR, VECTOR_STORE_DIR
from app.ingestion.loader import chunk_documents, load_documents
from app.rag.pipeline import ask, build_rag_chain
from app.vectorstore.store import create_vector_store, load_vector_store


def ingest() -> None:
    """Load documents, chunk them, and build the vector store."""
    print(f"Loading documents from {DOCUMENTS_DIR}...")
    documents = load_documents(DOCUMENTS_DIR)

    if not documents:
        print("No documents found. Add PDF or TXT files to data/documents/.")
        return

    print(f"Chunking {len(documents)} document(s)...")
    chunks = chunk_documents(documents)

    print(f"Building vector store at {VECTOR_STORE_DIR}...")
    create_vector_store(chunks, VECTOR_STORE_DIR)
    print("Ingestion complete.")


def chat() -> None:
    """Interactive chat loop using the RAG pipeline."""
    vector_store = load_vector_store(VECTOR_STORE_DIR)
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    chain = build_rag_chain(retriever)

    print("RAG Chatbot ready. Type 'quit' to exit.\n")

    while True:
        question = input("You: ").strip()
        if not question:
            continue
        if question.lower() in {"quit", "exit", "q"}:
            print("Goodbye!")
            break

        answer = ask(chain, question)
        print(f"Bot: {answer}\n")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2 or sys.argv[1] not in {"ingest", "chat"}:
        print("Usage: python -m app.main [ingest|chat]")
        sys.exit(1)

    if sys.argv[1] == "ingest":
        ingest()
    else:
        chat()
