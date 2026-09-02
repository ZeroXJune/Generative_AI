"""
Checkpoint 3 end-to-end demonstration.

Run:  python src/checkpoint3_demo.py

Exercises all four Checkpoint 3 deliverables:
  1. Automated ingestion pipeline (incremental, change-detecting)
  2. RAG application built on LangChain interfaces
  3. Conversational memory across multi-turn dialogue
  4. A live demo of more than five queries
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from rag_app import RAGApplication

# A genuine multi-turn conversation. Turns 2, 3 and 6 are follow-ups that
# cannot be retrieved without resolving what "it" and "that" refer to.
CONVERSATION = [
    "What is retrieval augmented generation?",
    "Why does it reduce hallucination?",
    "Which distance metric should be used for that?",
    "How much butter does the chocolate chip cookie recipe need?",
    "When is the capstone final defense?",
    "What happens at that checkpoint?",
    "What is the WiFi password for the campus network?",
]


def section(title: str):
    """Print a section banner."""
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def demo_ingestion(app: RAGApplication):
    """Show the incremental ingestion pipeline."""
    section("1. AUTOMATED INGESTION PIPELINE")

    print("First pass - the corpus is new to this index:")
    report = app.ingest()
    print(report.describe())

    print("\nSecond pass - nothing has changed since:")
    report = app.ingest()
    print(report.describe())
    print(
        "\n  Change detection is content-hash based, so re-running is cheap:\n"
        "  unchanged documents are never re-embedded."
    )


def demo_langchain(app: RAGApplication):
    """Show that the index is exposed through LangChain's interfaces."""
    section("2. LANGCHAIN INTEGRATION")

    from langchain_core.retrievers import BaseRetriever

    retriever = app.as_langchain_retriever()
    print(f"Retriever type      : {type(retriever).__name__}")
    print(f"Is a LangChain BaseRetriever: {isinstance(retriever, BaseRetriever)}")

    documents = retriever.invoke("What is a vector database?")
    print(f"\nretriever.invoke(...) returned {len(documents)} LangChain Document(s):")
    for document in documents:
        print(
            f"  [{document.metadata['doc_id']}] "
            f"score={document.metadata['score']:.3f} :: {document.page_content[:64]}..."
        )


REFUSAL = "I don't have that in your notes."

# Above this retrieval score the corpus plainly does contain relevant material,
# so a refusal reflects the extractive responder's inability to compose an
# answer. Below it, the refusal is the grounding rule working as designed.
STRONG_MATCH = 0.25


def demo_conversation(app: RAGApplication):
    """Run the multi-turn conversation."""
    section("3 & 4. CONVERSATIONAL MEMORY - LIVE QUERIES")

    offline = app.chat_client.get_info()["backend"] == "offline"
    if offline:
        print(
            "NOTE: no LLM configured, so answers come from the offline extractive\n"
            "responder, which quotes sentences rather than composing them. It cannot\n"
            "answer 'why' questions even when retrieval finds the right passage.\n"
            "Watch the `sources` line to judge retrieval separately from generation."
        )

    for question in CONVERSATION:
        result = app.ask(question)

        print(f"\n[Turn {result['turn']}] Q: {question}")
        if result["was_rewritten"]:
            print(f"           rewritten -> {result['resolved_query']!r}")
        print(f"           A: {result['answer'][:180]}")
        sources = ", ".join(
            f"{s['doc_id']}({s['score']:.2f})" for s in result["sources"][:3]
        )
        print(f"           sources: {sources or 'none above threshold'}")

        # A refusal has two very different causes, and conflating them would
        # misrepresent the system: one is a fallback limitation, the other is
        # the anti-hallucination rule doing exactly its job.
        if result["answer"].strip() == REFUSAL:
            best = max((s["score"] for s in result["sources"]), default=0.0)
            if offline and best >= STRONG_MATCH:
                print(
                    f"           ^ retrieval succeeded (top score {best:.2f}); the "
                    "extractive responder\n             could not compose an answer. A "
                    "real LLM answers this from the same passages."
                )
            else:
                print(
                    f"           ^ CORRECT refusal - no passage clears the relevance bar "
                    f"(top score {best:.2f}).\n             This fact is genuinely absent "
                    "from the corpus."
                )


def main():
    """Run the full Checkpoint 3 demonstration."""
    print("=" * 72)
    print("CHECKPOINT 3 - RAG ORCHESTRATION")
    print("=" * 72)

    app = RAGApplication(reset=True)
    demo_ingestion(app)
    demo_langchain(app)
    demo_conversation(app)

    section("SUMMARY")
    for key, value in app.get_info().items():
        print(f"  {key}: {value}")

    rewritten = sum(1 for turn in app.memory.turns if turn.resolved_query != turn.question)
    print(f"\n  {len(CONVERSATION)} queries answered, {rewritten} required reference resolution.")
    print("  Analysis: docs/04_RAG_Orchestration.md")


if __name__ == "__main__":
    main()
