# RAG-based AI Chatbot — Agentic AI eBook

A Retrieval-Augmented Generation (RAG) chatbot that answers questions **strictly** from the
*Agentic AI* eBook. Built with **Python**, **LangGraph**, **Pinecone**, **OpenAI**, and
**FastAPI / Streamlit**.

If a question is not covered by the document, the chatbot responds:
`I cannot answer based on the provided document.`

## Objective

- **Ingest & store** — parse the PDF, split it into chunks, embed them, and index them in Pinecone.
- **Orchestrate** — use LangGraph to build a stateful `retrieve → generate` workflow.
- **Ground strictly** — answer using only the retrieved document context; refuse otherwise.
- **Expose output** — a FastAPI endpoint and a Streamlit UI that return the answer, the retrieved
  context chunks, and a confidence score.

## Tech stack

| Layer | Choice |
|-------|--------|
| Language | Python 3.10+ |
| Orchestration | LangGraph (stateful graph) |
| Vector store | Pinecone (serverless, cosine) |
| Embeddings + LLM | OpenAI `text-embedding-3-small` + `gpt-4o-mini` (Gemini optional, see §2) |
| API | FastAPI (`POST /chat`) |
| UI | Streamlit |

---

## 1. Architecture

```
                 INGESTION (run once)
  data/Ebook-Agentic-AI.pdf
            │
            ▼
     PyPDFLoader (load pages)
            │
            ▼
  RecursiveCharacterTextSplitter (chunk_size=1000, overlap=200)
            │
            ▼
  OpenAIEmbeddings (text-embedding-3-small, 1536 dims)
            │
            ▼
     Pinecone index (cosine)  ◄──────────────┐
                                             │
                 QUERY TIME                  │
  User question                              │
            │                                │
            ▼                                │
   ┌──────────────── LangGraph ─────────────────┐
   │  START ─► retrieve ─► generate ─► END       │
   │            │            │                   │
   │   top-k similar     LLM (gpt-4o-mini)       │
   │   chunks ───────────┘  strict grounding     │
   └─────────────────────────────────────────────┘
            │
            ▼
  { final_answer, retrieved_context, confidence_score }
            │
   ┌────────┴─────────┐
   ▼                  ▼
FastAPI /chat     Streamlit UI
```

### Components

| File | Responsibility |
|------|----------------|
| `src/config.py` | Loads env variables and defines constants (models, chunk sizes, index name). |
| `src/providers.py` | Factory that returns the embeddings + LLM clients for the active provider. |
| `src/ingestion.py` | Loads the PDF, chunks it, creates the Pinecone index, and upserts embeddings. |
| `src/graph.py` | Defines the LangGraph `AgentState` and the `retrieve → generate` workflow. |
| `app.py` | FastAPI backend exposing `POST /chat`. |
| `streamlit_app.py` | Streamlit chat UI with a side panel for context chunks and score. |
| `tests_sample_queries.py` | Runs the 5–6 benchmark queries against the graph. |

### The LangGraph workflow

`AgentState` carries `question`, `context`, `scores`, `answer`, and `score` between nodes.

- **retrieve** — embeds the question and queries Pinecone for the top-k (default 3) most similar
  chunks, capturing each chunk's cosine relevance score.
- **generate** — builds a strict system prompt that instructs the LLM to answer **only** from the
  retrieved context, then computes a confidence score.

Graph flow: `START → retrieve → generate → END`.

### How the confidence score is computed

The confidence score reflects **how relevant the retrieved chunks are to the question**. It is
computed from Pinecone's cosine similarity — not hardcoded. Step by step:

1. **Embed the question.** The question is converted to a vector using the same embedding model
   used during ingestion, so it lives in the same vector space as the document chunks.
2. **Retrieve with scores.** Pinecone returns the top-k chunks (default 3), each with a cosine
   similarity score (higher = more semantically similar to the question). See
   `similarity_search_with_score` in `src/graph.py`.
3. **Average the top-k scores.** The per-chunk scores are averaged into a single value:
   `confidence = mean(top_k_scores)`.
4. **Clamp and round.** The result is clamped to `[0, 1]` and rounded to 4 decimals. If nothing is
   retrieved, the confidence is `0.0`.

**What the number means in practice** (observed in testing):

| Question type | Example | Confidence |
|---|---|---|
| In the eBook | "What is Agentic AI?" | ≈ 0.79 |
| In the eBook | "Core components of an Agentic Architecture?" | ≈ 0.78 |
| Not in the eBook | "Who won the 2022 FIFA World Cup?" | ≈ 0.53 |
| Not in the eBook | "What is the capital of France?" | ≈ 0.53 |

In-document questions retrieve highly relevant chunks and score high; off-topic questions retrieve
poor matches and score lower. Together with the strict prompt, this is why the bot answers grounded
questions and refuses everything else.

**Notes / honesty about the metric:**
- It measures *retrieval relevance*, not factual correctness of the answer.
- Cosine similarity on embeddings rarely reaches 0, so off-topic scores sit around ≈0.53 rather than
  near zero — the meaningful signal is the *gap* between in-document (~0.78) and off-topic (~0.53).
- This is a deliberate improvement over the reference snippet's placeholder heuristic
  (`0.95 if context else 0.0`), which returned a fixed value for every query.

---

## 2. Setup

### Prerequisites

- Python 3.10 or higher
- An OpenAI API key
- A Pinecone API key (free tier is sufficient)

### Steps

```bash
# 1. Create and activate a virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS / Linux:
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment variables
#    Copy the template and fill in your keys.
copy .env.example .env       # Windows
# cp .env.example .env       # macOS / Linux
```

Edit `.env`:

```env
OPENAI_API_KEY=your_openai_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=agentic-ai-index
```

### Optional: free provider for testing (Google Gemini)

The project uses **OpenAI** by default, exactly as the assignment specifies. For running the
pipeline at no cost, it also supports **Google Gemini** (free tier, no credit card). The provider
is selected with a single environment variable:

```env
# In .env — switch the provider without touching any code
LLM_PROVIDER=gemini
GOOGLE_API_KEY=your_google_api_key
```

Notes when using Gemini:
- Get a free key at https://ai.google.dev (no card required).
- Models used: `models/gemini-embedding-001` (embeddings) and `gemini-2.5-flash-lite` (chat).
- Gemini embeddings are **3072** dimensions (vs OpenAI's 1536); the index dimension adjusts
  automatically, and a separate default index name (`agentic-ai-index-gemini`) is used so the two
  never collide. Re-run ingestion after switching providers.
- The free tier has daily/per-minute request caps, so ingestion embeds in batches with short pauses.

Leave `LLM_PROVIDER=openai` (or unset) to use the assignment's default OpenAI path.

> **Note on this submission:** the code is built OpenAI-first per the assignment. The pipeline was
> verified end-to-end using the free Gemini tier (to avoid paid OpenAI credits), which is why Gemini
> support is included. Nothing about the two approaches differs except the provider — the LangGraph
> workflow, Pinecone storage, API, UI, and grounding are identical.

---

## 3. Ingestion (run once)

This loads the PDF, chunks it, creates the Pinecone index, and stores the embeddings.

```bash
python -m src.ingestion
```

You only need to run this again if the source document changes.

---

## 4. Run

### Option A — FastAPI backend

```bash
uvicorn app:app --reload
```

Then send a query:

```bash
curl -X POST http://127.0.0.1:8000/chat ^
  -H "Content-Type: application/json" ^
  -d "{\"query\": \"What is Agentic AI according to the eBook?\"}"
```

Interactive API docs are available at `http://127.0.0.1:8000/docs`.

Example response:

```json
{
  "final_answer": "Agentic AI refers to ...",
  "retrieved_context": ["chunk 1 text ...", "chunk 2 text ...", "chunk 3 text ..."],
  "confidence_score": 0.787
}
```

### Option B — Streamlit UI

```bash
streamlit run streamlit_app.py
```

A chat box appears with a side panel showing the retrieved context chunks and confidence score.

---

## 5. Testing

Run the 5–6 benchmark queries:

```bash
python tests_sample_queries.py
```

The queries:

1. What is Agentic AI according to the eBook?
2. How do AI agents differ from traditional automation systems?
3. What are the core components of an Agentic Architecture?
4. What role does memory play in Agentic AI workflows?
5. Who won the 2022 FIFA World Cup? — **validation test**: the chatbot should refuse or state
   that the context lacks this information.

---

## 6. Project Structure

```
RAG-based-AI-Chatbot-Project/
├── data/
│   └── Ebook-Agentic-AI.pdf     # Source document
├── src/
│   ├── __init__.py
│   ├── config.py                # Env setup & constants
│   ├── providers.py             # Embeddings + LLM factory (OpenAI / Gemini)
│   ├── ingestion.py             # PDF load, split & Pinecone index setup
│   └── graph.py                 # LangGraph workflow & state logic
├── app.py                       # FastAPI application
├── streamlit_app.py             # Streamlit UI (alternative interface)
├── requirements.txt             # Python dependencies
├── .env.example                 # Environment variable template
├── .gitignore
├── README.md
└── tests_sample_queries.py      # 5–6 sample test queries
```
