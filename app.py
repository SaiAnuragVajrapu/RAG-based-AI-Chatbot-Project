"""FastAPI application exposing the RAG chatbot.

POST /chat  -> { "query": "..." }
Returns     -> { final_answer, retrieved_context, confidence_score }
"""

import os

from fastapi import FastAPI
from pydantic import BaseModel

from src.graph import build_rag_graph

app = FastAPI(title="Agentic AI RAG API")

# Build (compile) the LangGraph workflow once at startup.
graph = build_rag_graph(index_name=os.getenv("PINECONE_INDEX_NAME", "agentic-ai-index"))


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    final_answer: str
    retrieved_context: list[str]
    confidence_score: float


@app.get("/")
def root():
    return {"message": "Agentic AI RAG API. POST a JSON {'query': '...'} to /chat."}


@app.post("/chat", response_model=QueryResponse)
async def chat_endpoint(request: QueryRequest):
    initial_state = {"question": request.query, "context": [], "scores": [], "answer": "", "score": 0.0}
    result = graph.invoke(initial_state)

    return QueryResponse(
        final_answer=result["answer"],
        retrieved_context=result["context"],
        confidence_score=result["score"],
    )
