"""Environment setup & constants.

Loads API credentials from the .env file and defines the constants used
across the ingestion pipeline and the RAG graph.
"""

import os
from dotenv import load_dotenv

# Load variables from the .env file into the process environment.
load_dotenv()

# --- Provider selection ---
# "openai"  -> the provider specified by the assignment (default).
# "gemini"  -> Google Gemini, a free-tier alternative for local testing.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()

# --- API credentials (read from .env) ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# --- Pinecone index configuration ---
# The default index name follows the active provider so OpenAI (1536-dim) and
# Gemini (3072-dim) vectors never share an index with a mismatched dimension.
_DEFAULT_INDEX = "agentic-ai-index" if LLM_PROVIDER == "openai" else "agentic-ai-index-gemini"
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", _DEFAULT_INDEX)
# Cloud/region for the Pinecone serverless index (free tier defaults).
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")

# --- Model configuration (per provider) ---
if LLM_PROVIDER == "gemini":
    # Google Gemini free-tier models.
    EMBEDDING_MODEL = "models/gemini-embedding-001"
    EMBEDDING_DIMENSION = 3072
    # flash-lite has a more generous free-tier daily quota than 2.5-flash.
    LLM_MODEL = "gemini-2.5-flash-lite"
else:
    # OpenAI (assignment default). text-embedding-3-small -> 1536 dims.
    EMBEDDING_MODEL = "text-embedding-3-small"
    EMBEDDING_DIMENSION = 1536
    LLM_MODEL = "gpt-4o-mini"

# --- Chunking configuration ---
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# --- Retrieval configuration ---
TOP_K = 3

# --- Ingestion batching (to respect provider rate limits) ---
# Gemini's free tier caps embedding requests per minute (~100), so we embed in
# smaller batches with a pause between them. OpenAI has no such tight limit.
if LLM_PROVIDER == "gemini":
    EMBED_BATCH_SIZE = 50
    EMBED_BATCH_DELAY = 40  # seconds between batches
else:
    EMBED_BATCH_SIZE = 100
    EMBED_BATCH_DELAY = 0

# --- Paths ---
PDF_PATH = os.path.join("data", "Ebook-Agentic-AI.pdf")


def validate_env():
    """Raise a clear error if required API keys are missing for the active provider."""
    missing = []
    if LLM_PROVIDER == "gemini":
        if not GOOGLE_API_KEY:
            missing.append("GOOGLE_API_KEY")
    else:
        if not OPENAI_API_KEY:
            missing.append("OPENAI_API_KEY")
    if not PINECONE_API_KEY:
        missing.append("PINECONE_API_KEY")
    if missing:
        raise EnvironmentError(
            "Missing required environment variables: "
            + ", ".join(missing)
            + ". Create a .env file (see .env.example)."
        )
