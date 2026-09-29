"""
Streamlit UI for the Agentic AI RAG Chatbot.

Features:
- Chat interface with conversation history
- Gemini response formatting
- Retrieved context chunks
- Retrieval score
- Clear chat button
- Error handling

Run:
    streamlit run streamlit_app.py
"""

import os
import streamlit as st


# ==================================================
# LOAD STREAMLIT CLOUD SECRETS
# ==================================================

try:
    cloud_secrets = st.secrets

    for key in [
        "GOOGLE_API_KEY",
        "PINECONE_API_KEY",
        "PINECONE_INDEX_NAME",
        "LLM_PROVIDER",
    ]:
        if key in cloud_secrets:
            os.environ[key] = str(cloud_secrets[key])

except (FileNotFoundError, RuntimeError):
    # Locally, configuration can be loaded from .env
    pass


# Import AFTER loading Streamlit secrets.
# src.graph calls validate_env() during import.
from src.graph import build_rag_graph


# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Agentic AI RAG Chatbot",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 Agentic AI RAG Chatbot")

st.caption(
    "Ask questions about the Agentic AI eBook. "
    "Answers are generated using retrieved document content."
)


# ==================================================
# INITIALIZE LANGGRAPH
# ==================================================

@st.cache_resource
def get_graph():
    """Build and cache the LangGraph workflow."""
    return build_rag_graph()


try:
    graph = get_graph()

except Exception as e:
    st.error("Unable to initialize the RAG chatbot.")
    st.exception(e)
    st.stop()


# ==================================================
# SESSION STATE
# ==================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_context" not in st.session_state:
    st.session_state.last_context = []

if "last_score" not in st.session_state:
    st.session_state.last_score = 0.0


# ==================================================
# HELPER: EXTRACT PLAIN TEXT
# ==================================================

def extract_answer(raw_answer):
    """
    Convert model output into a readable string.

    Handles:
    - Plain strings
    - LangChain message objects
    - Gemini content-block lists
    - Dictionaries containing text/content
    """

    # Handle LangChain message objects
    if hasattr(raw_answer, "content"):
        raw_answer = raw_answer.content

    # Handle Gemini list-based content
    if isinstance(raw_answer, list):
        text_parts = []

        for item in raw_answer:
            if isinstance(item, str):
                text_parts.append(item)

            elif isinstance(item, dict):
                text = item.get("text")

                if isinstance(text, str) and text.strip():
                    text_parts.append(text)

        return "\n\n".join(text_parts).strip()

    # Handle dictionary responses
    if isinstance(raw_answer, dict):
        text = raw_answer.get(
            "text",
            raw_answer.get("content", ""),
        )

        if isinstance(text, list):
            return extract_answer(text)

        if isinstance(text, str):
            return text.strip()

        return ""

    # Handle plain strings
    if isinstance(raw_answer, str):
        return raw_answer.strip()

    return ""


# ==================================================
# HELPER: CLEAN CONTEXT
# ==================================================

def clean_context(raw_context):
    """
    Normalize retrieved context into a list of
    non-empty strings.
    """

    if raw_context is None:
        return []

    if isinstance(raw_context, str):
        raw_context = [raw_context]

    if not isinstance(raw_context, (list, tuple)):
        return []

    cleaned_chunks = []

    for chunk in raw_context:
        if chunk is None:
            continue

        # Handle Document-like objects
        if hasattr(chunk, "page_content"):
            chunk = chunk.page_content

        # Handle dictionaries containing text
        elif isinstance(chunk, dict):
            chunk = chunk.get(
                "page_content",
                chunk.get("text", ""),
            )

        if not isinstance(chunk, str):
            chunk = str(chunk)

        chunk = chunk.strip()

        if chunk:
            cleaned_chunks.append(chunk)

    return cleaned_chunks


# ==================================================
# SIDEBAR
# ==================================================

with st.sidebar:

    st.header("📚 Retrieved Information")

    st.metric(
        label="Retrieval Score",
        value=f"{st.session_state.last_score:.2f}",
    )

    st.caption(
        "This is the average retrieval score returned by "
        "the vector database. It is not a calibrated "
        "probability that the answer is correct."
    )

    st.divider()

    st.subheader("Retrieved Context Chunks")

    context_chunks = st.session_state.last_context

    if context_chunks:
        st.success(
            f"{len(context_chunks)} chunk(s) retrieved"
        )

        for i, chunk in enumerate(
            context_chunks,
            start=1,
        ):
            with st.expander(
                f"Chunk {i}",
                expanded=False,
            ):
                st.text(chunk)

    else:
        st.info(
            "Ask a question to view retrieved document chunks."
        )

    st.divider()

    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True,
    ):
        st.session_state.messages = []
        st.session_state.last_context = []
        st.session_state.last_score = 0.0

        st.rerun()


# ==================================================
# DISPLAY CHAT HISTORY
# ==================================================

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ==================================================
# CHAT INPUT
# ==================================================

query = st.chat_input(
    "Ask a question about the Agentic AI eBook..."
)


# ==================================================
# PROCESS QUESTION
# ==================================================

if query:

    # ----------------------------------------------
    # DISPLAY USER QUESTION
    # ----------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": query,
        }
    )

    with st.chat_message("user"):
        st.markdown(query)

    try:

        # ------------------------------------------
        # RUN RAG WORKFLOW
        # ------------------------------------------

        with st.spinner(
            "Searching the eBook and generating an answer..."
        ):

            initial_state = {
                "question": query,
                "context": [],
                "scores": [],
                "answer": "",
                "score": 0.0,
            }

            result = graph.invoke(initial_state)

        # ------------------------------------------
        # EXTRACT ANSWER
        # ------------------------------------------

        raw_answer = result.get("answer", "")

        answer = extract_answer(raw_answer)

        if not answer:
            answer = (
                "I could not generate a readable answer. "
                "Please try asking your question again."
            )

        # ------------------------------------------
        # EXTRACT AND CLEAN CONTEXT
        # ------------------------------------------

        raw_context = result.get("context", [])

        context_chunks = clean_context(raw_context)

        # ------------------------------------------
        # EXTRACT RETRIEVAL SCORE
        # ------------------------------------------

        score = result.get("score", 0.0)

        try:
            score = float(score)

        except (TypeError, ValueError):
            score = 0.0

        # ------------------------------------------
        # SAVE RESULTS
        # ------------------------------------------

        st.session_state.last_context = context_chunks
        st.session_state.last_score = score

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
            }
        )

        # ------------------------------------------
        # DISPLAY ANSWER
        # ------------------------------------------

        with st.chat_message("assistant"):
            st.markdown(answer)

        # Refresh sidebar with the latest results
        st.rerun()

    except Exception as e:
        st.error(
            "An error occurred while processing your question."
        )

        st.exception(e)