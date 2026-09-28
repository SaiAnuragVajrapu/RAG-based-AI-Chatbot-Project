"""Provider factory for embeddings and the chat LLM.

Keeps the assignment's OpenAI path as the default while allowing a free
Gemini provider for local end-to-end testing. The active provider is chosen
by the LLM_PROVIDER environment variable (see src/config.py).
"""

from src.config import (
    LLM_PROVIDER,
    OPENAI_API_KEY,
    GOOGLE_API_KEY,
    EMBEDDING_MODEL,
    LLM_MODEL,
)


def get_embeddings():
    """Return an embeddings client for the active provider."""
    if LLM_PROVIDER == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        return GoogleGenerativeAIEmbeddings(
            model=EMBEDDING_MODEL, google_api_key=GOOGLE_API_KEY
        )

    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=OPENAI_API_KEY)


def get_llm():
    """Return a chat LLM client for the active provider (temperature=0)."""
    if LLM_PROVIDER == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=LLM_MODEL, temperature=0, google_api_key=GOOGLE_API_KEY
        )

    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=LLM_MODEL, temperature=0, api_key=OPENAI_API_KEY)
