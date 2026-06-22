"""Load and chunk documents for indexing."""

from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_documents(documents_dir: Path) -> list:
    """Load PDF and text files from the documents directory."""
    loaders = [
        DirectoryLoader(str(documents_dir), glob="**/*.pdf", loader_cls=PyPDFLoader),
        DirectoryLoader(str(documents_dir), glob="**/*.txt", loader_cls=TextLoader),
    ]

    documents = []
    for loader in loaders:
        documents.extend(loader.load())

    return documents


def chunk_documents(documents: list, chunk_size: int = 1000, chunk_overlap: int = 200) -> list:
    """Split documents into smaller chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(documents)
