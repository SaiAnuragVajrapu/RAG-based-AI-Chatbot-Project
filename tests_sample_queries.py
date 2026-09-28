"""Script containing the 5-6 sample test queries.

Runs the benchmark queries directly against the compiled RAG graph and
prints the answer, retrieved context chunks, and confidence score for each.
The final query (FIFA World Cup) validates that the chatbot refuses to
answer questions the document does not cover.

Run with:  python tests_sample_queries.py
"""

from src.graph import build_rag_graph

SAMPLE_QUERIES = [
    "What is Agentic AI according to the eBook?",
    "How do AI agents differ from traditional automation systems?",
    "What are the core components of an Agentic Architecture?",
    "What role does memory play in Agentic AI workflows?",
    "Who won the 2022 FIFA World Cup?",  # Validation: should refuse / say not in document.
]


def run():
    graph = build_rag_graph()

    for i, query in enumerate(SAMPLE_QUERIES, start=1):
        print("=" * 80)
        print(f"Query {i}: {query}")
        print("-" * 80)

        initial_state = {"question": query, "context": [], "scores": [], "answer": "", "score": 0.0}
        result = graph.invoke(initial_state)

        print(f"Answer:\n{result['answer']}\n")
        print(f"Confidence score: {result['score']}")
        print(f"Retrieved chunks: {len(result['context'])}")
        for j, chunk in enumerate(result["context"], start=1):
            preview = chunk[:200].replace("\n", " ")
            print(f"  [Chunk {j}] {preview}...")
        print()


if __name__ == "__main__":
    run()
