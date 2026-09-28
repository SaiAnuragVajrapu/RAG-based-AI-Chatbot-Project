"""Streamlit UI for the Agentic AI RAG chatbot.

A lightweight chat interface with a side panel displaying the retrieved
context chunks and the confidence score.

Run with:  streamlit run streamlit_app.py
"""

import streamlit as st

from src.graph import build_rag_graph

st.set_page_config(page_title="Agentic AI RAG Chatbot", layout="wide")
st.title("Agentic AI RAG Chatbot")
st.caption("Answers strictly from the Agentic AI eBook.")


@st.cache_resource
def get_graph():
    return build_rag_graph()


graph = get_graph()

query = st.chat_input("Ask a question about the Agentic AI eBook...")

if query:
    with st.spinner("Retrieving and generating..."):
        initial_state = {"question": query, "context": [], "scores": [], "answer": "", "score": 0.0}
        result = graph.invoke(initial_state)

    st.chat_message("user").write(query)
    st.chat_message("assistant").write(result["answer"])

    with st.sidebar:
        st.subheader("Confidence score")
        st.metric(label="Score", value=f"{result['score']:.2f}")

        st.subheader("Retrieved context chunks")
        for i, chunk in enumerate(result["context"], start=1):
            with st.expander(f"Chunk {i}"):
                st.write(chunk)
