"""LangGraph workflow for the Agentic AI RAG chatbot."""

from typing import TypedDict, Any

from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore

from src.config import (
    GOOGLE_API_KEY,
    LLM_MODEL,
    PINECONE_INDEX_NAME,
    TOP_K,
    validate_env,
)
from src.providers import get_embeddings


# --------------------------------------------------
# 1. Define the graph state
# --------------------------------------------------

class AgentState(TypedDict, total=False):
    question: str
    context: str
    scores: list[float]
    answer: str
    score: float


# --------------------------------------------------
# 2. Validate configuration and initialize Gemini
# --------------------------------------------------

validate_env()

llm = ChatGoogleGenerativeAI(
    model=LLM_MODEL,
    google_api_key=GOOGLE_API_KEY,
    temperature=0,
)


# --------------------------------------------------
# 3. Connect to the existing Pinecone index
# --------------------------------------------------

def get_vector_store():
    """Connect to the existing Pinecone index."""
    embeddings = get_embeddings()

    return PineconeVectorStore(
        index_name=PINECONE_INDEX_NAME,
        embedding=embeddings,
    )


# --------------------------------------------------
# 4. Retrieve relevant document chunks
# --------------------------------------------------

def retrieve_node(state: AgentState) -> dict[str, Any]:
    question = state.get("question", "").strip()

    if not question:
        return {
            "context": "",
            "scores": [],
            "score": 0.0,
        }

    vector_store = get_vector_store()

    results = vector_store.similarity_search_with_score(
        query=question,
        k=TOP_K,
    )

    context_parts = []
    scores = []

    for document, similarity_score in results:
        if document.page_content:
            context_parts.append(document.page_content)

        scores.append(float(similarity_score))

    context = "\n\n".join(context_parts)

    average_score = (
        sum(scores) / len(scores)
        if scores
        else 0.0
    )

    return {
        "context": context,
        "scores": scores,
        "score": average_score,
    }


# --------------------------------------------------
# 5. Generate a plain-text answer
# --------------------------------------------------

def generate_node(state: AgentState) -> dict[str, Any]:
    question = state.get("question", "").strip()
    context = state.get("context", "")

    if not question:
        return {
            "answer": "Please enter a question."
        }

    if not context:
        return {
            "answer": (
                "I couldn't find relevant information "
                "in the Agentic AI eBook for this question."
            )
        }

    prompt = f"""
You are an Agentic AI assistant answering questions
about the provided Agentic AI eBook.

Instructions:
- Answer using the retrieved context.
- Explain concepts clearly.
- Do not invent facts.
- If the context does not contain the answer,
  say that the information is not available in the retrieved text.

Retrieved context:
{context}

User question:
{question}

Answer:
"""

    response = llm.invoke(prompt)
    raw_answer = getattr(response, "content", response)

    # Convert Gemini's content blocks into readable text.
    if isinstance(raw_answer, str):
        answer = raw_answer.strip()

    elif isinstance(raw_answer, list):
        text_parts = []

        for item in raw_answer:
            if isinstance(item, str):
                text_parts.append(item)

            elif isinstance(item, dict):
                text = item.get("text")

                if isinstance(text, str):
                    text_parts.append(text)

        answer = "\n\n".join(text_parts).strip()

    elif isinstance(raw_answer, dict):
        answer = str(
            raw_answer.get("text")
            or raw_answer.get("content")
            or ""
        ).strip()

    else:
        answer = str(raw_answer).strip()

    if not answer:
        answer = "The model returned an empty response. Please try again."

    return {
        "answer": answer,
    }


# --------------------------------------------------
# 6. Build the RAG graph
# --------------------------------------------------

def build_rag_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)

    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile()