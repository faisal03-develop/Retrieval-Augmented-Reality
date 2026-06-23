"""Chat model factory for OpenAI and local Ollama."""

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

from app.config import (
    CHAT_MODEL,
    LLM_PROVIDER,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    OPENAI_API_KEY,
)


def is_chat_llm_ready() -> bool:
    """Return True when the configured chat provider has required settings."""
    if LLM_PROVIDER == "ollama":
        return bool(OLLAMA_MODEL)
    return bool(OPENAI_API_KEY)


def get_chat_model() -> BaseChatModel:
    """Return the configured chat model (Ollama by default)."""
    if LLM_PROVIDER == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=0,
        )

    if not OPENAI_API_KEY:
        raise ValueError("Set OPENAI_API_KEY in your .env file when LLM_PROVIDER=openai.")

    return ChatOpenAI(model=CHAT_MODEL, api_key=OPENAI_API_KEY, temperature=0)
