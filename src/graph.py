"""LangGraph workflow definition & state logic.

Builds a stateful RAG graph:  START -> retrieve -> generate -> END
  - retrieve: queries Pinecone for the top-k relevant chunks.
  - generate: answers strictly from the retrieved context and assigns a
    confidence score.
"""

from typing import List, TypedDict

from langgraph.graph import StateGraph, START, END
from langchain_pinecone import PineconeVectorStore

from src.config import PINECONE_INDEX_NAME, TOP_K
from src.providers import get_embeddings, get_llm


class AgentState(TypedDict):
    question: str
    context: List[str]
    scores: List[float]
    answer: str
    score: float


# If the average retrieval relevance is below this, we treat the question as
# not covered by the document (low confidence).
RELEVANCE_THRESHOLD = 0.5


def build_rag_graph(index_name: str = PINECONE_INDEX_NAME):
    embeddings = get_embeddings()
    vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)
    llm = get_llm()

    # --- Nodes ---
    def retrieve_node(state: AgentState):
        # similarity_search_with_score returns (Document, relevance_score) pairs.
        # For a cosine index, higher score = more relevant.
        results = vectorstore.similarity_search_with_score(state["question"], k=TOP_K)
        context_texts = [doc.page_content for doc, _ in results]
        scores = [float(score) for _, score in results]
        return {"context": context_texts, "scores": scores}

    def generate_node(state: AgentState):
        context_str = "\n\n".join(state["context"])
        prompt = f"""You are a strict assistant. Answer the question relying ONLY on the context below.
If the context does not contain enough info, state 'I cannot answer based on the provided document.'

Context:
{context_str}

Question: {state['question']}"""

        response = llm.invoke(prompt)

        # Confidence = average relevance of the retrieved chunks (clamped to [0, 1]).
        # This makes the score meaningful: in-document questions retrieve highly
        # relevant chunks (high score); off-topic questions retrieve poor matches
        # (low score). Falls back to 0.0 when nothing is retrieved.
        scores = state.get("scores") or []
        if scores:
            avg = sum(scores) / len(scores)
            confidence = max(0.0, min(1.0, round(avg, 4)))
        else:
            confidence = 0.0

        return {"answer": response.content, "score": confidence}

    # --- Build Graph ---
    workflow = StateGraph(AgentState)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)

    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile()
