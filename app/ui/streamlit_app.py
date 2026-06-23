"""Streamlit chat UI for the RAG chatbot."""

import streamlit as st

from app.config import (
    DOCUMENTS_DIR,
    LLM_PROVIDER,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    CHAT_MODEL,
    VECTOR_STORE_DIR,
)
from app.models.llm import is_chat_llm_ready
from app.rag.pipeline import ask, build_rag_chain
from app.vectorstore.store import load_vector_store, vector_store_exists


@st.cache_resource(show_spinner="Loading RAG pipeline...")
def get_rag_chain():
    vector_store = load_vector_store(VECTOR_STORE_DIR)
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    return build_rag_chain(retriever)


def chat_model_label() -> str:
    if LLM_PROVIDER == "ollama":
        return f"ollama/{OLLAMA_MODEL}"
    return CHAT_MODEL


st.set_page_config(page_title="RAG Chatbot", page_icon="💬", layout="centered")

st.title("RAG Chatbot")
st.caption("Ask questions about your ingested documents.")

with st.sidebar:
    st.header("Status")
    st.write(f"**Chat provider:** `{LLM_PROVIDER}`")
    st.write(f"**Chat model:** `{chat_model_label()}`")

    if not is_chat_llm_ready():
        if LLM_PROVIDER == "ollama":
            st.error("Set `OLLAMA_MODEL` in your `.env` file.")
        else:
            st.error("Set `OPENAI_API_KEY` in your `.env` file.")
    elif LLM_PROVIDER == "ollama":
        st.caption(f"Ollama server: `{OLLAMA_BASE_URL}`")
        st.success("LLM configured (Ollama)")
    else:
        st.success("LLM configured (OpenAI)")

    if vector_store_exists(VECTOR_STORE_DIR):
        st.success("Vector store ready")
    else:
        st.warning("No vector store found.")

    st.divider()
    st.markdown("**Documents folder**")
    st.code(str(DOCUMENTS_DIR), language=None)

    st.markdown("**Ingest documents**")
    st.code("python -m app.main ingest", language="bash")

    if LLM_PROVIDER == "ollama":
        st.markdown("**Pull Ollama model**")
        st.code(f"ollama pull {OLLAMA_MODEL}", language="bash")

    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

ready = is_chat_llm_ready() and vector_store_exists(VECTOR_STORE_DIR)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question...", disabled=not ready):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                answer = ask(get_rag_chain(), prompt)
            except Exception as exc:
                answer = f"Something went wrong: {exc}"
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})

if not ready and not st.session_state.messages:
    if not is_chat_llm_ready():
        if LLM_PROVIDER == "ollama":
            st.info(
                "Start the Ollama desktop app, pull your model with "
                f"`ollama pull {OLLAMA_MODEL}`, then restart this app."
            )
        else:
            st.info("Add your OpenAI API key to `.env`, then restart the app.")
    elif not vector_store_exists(VECTOR_STORE_DIR):
        st.info(
            "Add PDF or TXT files to the documents folder, then run "
            "`python -m app.main ingest` before chatting."
        )
