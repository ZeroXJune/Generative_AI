"""
Checkpoint 2 end-to-end demonstration.

Run:  python src/checkpoint2_demo.py

Exercises all four Checkpoint 2 deliverables in one pass:
  1. The system prompt library
  2. Chat Completion API integration
  3. Chroma vector indexing and retrieval
  4. A pointer to the distance metric comparison experiment
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from build_index import build_index, load_embedder
from llm.chat_client import ChatClient
from prompts.system_prompts import PROMPT_LIBRARY, build_rag_messages
from retrieval.retriever import RAGRetriever

DEMO_QUESTIONS = [
    "When is the capstone final defense?",
    "What is retrieval augmented generation and why does it reduce hallucination?",
    "Which distance metric should a vector database use for text embeddings?",
    "Explain what queries keys and values do in self attention.",
    "How much butter does the chocolate chip cookie recipe need?",
    "What is the WiFi password for the campus network?",
]


def section(title: str):
    """Print a section banner."""
    print("\n" + "=" * 68)
    print(title)
    print("=" * 68)


def demo_prompts():
    """Show the registered system prompts and their design rationale."""
    section("1. PROMPT ARCHITECTURE")
    for prompt in PROMPT_LIBRARY.values():
        print(f"\n[{prompt.name}] {prompt.version}  (temperature={prompt.temperature})")
        print(f"  Role:  {prompt.role}")
        print(f"  Fixes: {prompt.failure_mode_addressed}")
        for note in prompt.design_notes:
            print(f"    - {note}")


def demo_chat_api():
    """Show the Chat Completion integration and its configuration."""
    section("2. CHAT COMPLETION API INTEGRATION")
    client = ChatClient()
    print(f"Configuration: {json.dumps(client.get_info(), indent=2)}")

    if client.is_live:
        print("\nLive endpoint configured - responses come from the model.")
    else:
        print(
            "\nNo OPENAI_API_KEY set. Using the offline extractive responder so the\n"
            "chain still runs end to end. Set OPENAI_API_KEY to call a real model."
        )

    messages = build_rag_messages(
        "When is Checkpoint 2 due?",
        [{"doc_id": "schedule_semester_2026", "text": "Checkpoint 2 is due September 19, 2026."}],
    )
    response = client.complete(messages)
    print(f"\nSample call -> {response.content}")
    print(f"Call metadata: {response.summary()}")
    return client


def demo_retrieval(client: ChatClient):
    """Run the full RAG loop over the indexed corpus."""
    section("3. VECTOR INDEXING AND RETRIEVAL")
    store = build_index()
    embedder, _ = load_embedder()

    # The embedder must be fitted on the same corpus the index was built from,
    # otherwise IDF weights - and therefore the query vectors - would not match.
    from build_index import build_chunks

    if hasattr(embedder, "fit"):
        embedder.fit([chunk["text"] for chunk in build_chunks()])

    retriever = RAGRetriever(vector_store=store, embedder=embedder, chat_client=client, top_k=4)

    section("4. END-TO-END RAG QUERIES")
    for question in DEMO_QUESTIONS:
        result = retriever.answer(question)
        print(f"\nQ: {question}")
        print(f"A: {result['answer']}")
        sources = ", ".join(
            f"{source['doc_id']}(score={source['score']:.3f})" for source in result["sources"]
        )
        print(f"   Sources: {sources or 'none above threshold'}")
        print(f"   Prompt: {result['prompt']['name']} {result['prompt']['version']}")


def main():
    """Run the full Checkpoint 2 demonstration."""
    demo_prompts()
    client = demo_chat_api()
    demo_retrieval(client)

    section("5. DISTANCE METRIC COMPARISON")
    print("Run the full experiment with:")
    print("  python src/experiments/metric_comparison.py")
    print("Results are written to data/processed/metric_comparison.json")
    print("Analysis: docs/02_Vector_Indexing.md")


if __name__ == "__main__":
    main()
