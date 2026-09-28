"""PDF loading, splitting & Pinecone index setup.

ETL pipeline for the Agentic AI eBook:
  1. Load the PDF document into memory.
  2. Split it into overlapping text chunks.
  3. Create the Pinecone index (dimension 1536, cosine distance) if needed.
  4. Convert chunks to embeddings and upsert them into Pinecone.
"""

import os
import time

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from src.config import (
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    PINECONE_CLOUD,
    PINECONE_REGION,
    EMBEDDING_MODEL,
    EMBEDDING_DIMENSION,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    PDF_PATH,
    EMBED_BATCH_SIZE,
    EMBED_BATCH_DELAY,
    validate_env,
)
from src.providers import get_embeddings


def _ensure_index(index_name: str):
    """Create the Pinecone index with dimension 1536 and cosine metric if it does not exist."""
    pc = Pinecone(api_key=PINECONE_API_KEY)
    existing = [idx["name"] for idx in pc.list_indexes()]
    if index_name not in existing:
        print(f"Creating Pinecone index '{index_name}' (dim={EMBEDDING_DIMENSION}, metric=cosine)...")
        pc.create_index(
            name=index_name,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
        )
    else:
        print(f"Pinecone index '{index_name}' already exists. Reusing it.")


def run_ingestion(pdf_path: str = PDF_PATH, index_name: str = PINECONE_INDEX_NAME):
    """Load the PDF, chunk it, and store embeddings in Pinecone."""
    validate_env()

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found at '{pdf_path}'.")

    # 1. Load document
    print(f"Loading PDF from '{pdf_path}'...")
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    print(f"Loaded {len(docs)} pages.")

    # 2. Chunk document
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = text_splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks.")

    # 3. Ensure the Pinecone index exists
    _ensure_index(index_name)

    # 4. Create embeddings & store in Pinecone.
    # Upsert in batches with a short pause between them. This keeps us within
    # provider rate limits (e.g. the Gemini free tier caps embedding requests
    # per minute) and makes ingestion resilient for larger documents.
    print(f"Embedding chunks with '{EMBEDDING_MODEL}' and upserting to Pinecone...")
    embeddings = get_embeddings()

    vector_store = PineconeVectorStore(index_name=index_name, embedding=embeddings)

    total = len(chunks)
    for start in range(0, total, EMBED_BATCH_SIZE):
        batch = chunks[start:start + EMBED_BATCH_SIZE]
        vector_store.add_documents(batch)
        done = min(start + EMBED_BATCH_SIZE, total)
        print(f"  Upserted {done}/{total} chunks.")
        # Pause between batches to respect per-minute rate limits (skip after the last batch).
        if done < total and EMBED_BATCH_DELAY > 0:
            time.sleep(EMBED_BATCH_DELAY)

    print("Ingestion complete. Vectors stored in Pinecone.")
    return vector_store


if __name__ == "__main__":
    run_ingestion()
