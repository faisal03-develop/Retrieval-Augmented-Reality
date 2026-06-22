"""Streamlit chat UI for the RAG chatbot."""

import streamlit as st

from app.config import CHAT_MODEL, DOCUMENTS_DIR, OPENAI_API_KEY, VECTOR_STORE_DIR
from app.rag.pipeline import ask, build_rag_chain
from app.vectorstore.store import load_vector_store, vector_store_exists


@st.cache_resource(show_spinner="Loading RAG pipeline...")
def get_rag_chain():
    vector_store = load_vector_store(VECTOR_STORE_DIR)
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    return build_rag_chain(retriever)


st.set_page_config(page_title="RAG Chatbot", page_icon="💬", layout="centered")

st.title("RAG Chatbot")
st.caption("Ask questions about your ingested documents.")

with st.sidebar:
    st.header("Status")
    st.write(f"**Chat model:** `{CHAT_MODEL}`")

    if not OPENAI_API_KEY:
        st.error("Set `OPENAI_API_KEY` in your `.env` file.")
    elif vector_store_exists(VECTOR_STORE_DIR):
        st.success("Vector store ready")
    else:
        st.warning("No vector store found.")

    st.divider()
    st.markdown("**Documents folder**")
    st.code(str(DOCUMENTS_DIR), language=None)

    st.markdown("**Ingest documents**")
    st.code("python -m app.main ingest", language="bash")

    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

ready = bool(OPENAI_API_KEY) and vector_store_exists(VECTOR_STORE_DIR)

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
    if not OPENAI_API_KEY:
        st.info("Add your OpenAI API key to `.env`, then restart the app.")
    elif not vector_store_exists(VECTOR_STORE_DIR):
        st.info(
            "Add PDF or TXT files to the documents folder, then run "
            "`python -m app.main ingest` before chatting."
        )
