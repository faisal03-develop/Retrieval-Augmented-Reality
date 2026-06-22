"""RAG pipeline: retrieve context and generate answers."""

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from app.config import CHAT_MODEL, OPENAI_API_KEY


SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on the provided context.
Use only the context below to answer. If the answer is not in the context, say you don't know.

Context:
{context}
"""


def build_rag_chain(retriever):
    """Build a retrieval-augmented generation chain."""
    llm = ChatOpenAI(model=CHAT_MODEL, api_key=OPENAI_API_KEY, temperature=0)

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{input}"),
    ])

    document_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, document_chain)


def ask(chain, question: str) -> str:
    """Run a question through the RAG chain and return the answer."""
    response = chain.invoke({"input": question})
    return response["answer"]
