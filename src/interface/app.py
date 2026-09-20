"""
Personal Assistant AI - web interface.

Checkpoint 4 deliverable. The earlier version read chunks.jsonl directly and
fell back to random mock vectors, bypassing everything built in Checkpoints 2
and 3. This one drives the real application: the Chroma index, conversational
memory, the grounded prompt, and the deadline reminder engine.

Run locally:  streamlit run src/interface/app.py
In Docker:    docker compose up --build
"""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag_app import RAGApplication
from schedule.extractor import extract_from_corpus
from schedule.reminders import ReminderEngine
from schedule.store import load_deadlines

REFUSAL = "I don't have that in your notes."

# Colour per urgency band for the deadline panel.
URGENCY_STYLE = {
    "overdue": ("🔴", "error"),
    "today": ("🟠", "error"),
    "urgent": ("🟠", "warning"),
    "soon": ("🟡", "warning"),
    "upcoming": ("🟢", "info"),
}

SUGGESTED_QUESTIONS = [
    "What is retrieval augmented generation?",
    "When is the capstone final defense?",
    "Which distance metric should a vector database use?",
    "How much butter does the cookie recipe need?",
]

st.set_page_config(
    page_title="Personal Assistant AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner="Loading index and models...")
def get_app() -> RAGApplication:
    """
    Build the RAG application once per session.

    Cached as a resource because the embedder and Chroma client are expensive
    to construct and safe to share across reruns.

    Returns:
        A ready RAGApplication with its index up to date
    """
    app = RAGApplication()
    app.ingest()
    return app


@st.cache_resource(show_spinner=False)
def get_reminder_engine() -> ReminderEngine:
    """
    Build the reminder engine, scanning the corpus if no store exists yet.

    Returns:
        A ReminderEngine over the available deadlines
    """
    deadlines = load_deadlines() or extract_from_corpus()
    return ReminderEngine(deadlines)


def render_sidebar(app: RAGApplication, horizon_days: int) -> int:
    """
    Render the sidebar: system status, corpus stats, and settings.

    Args:
        app: The running application
        horizon_days: Current reminder horizon

    Returns:
        The reminder horizon chosen by the user
    """
    info = app.get_info()

    with st.sidebar:
        st.header("📊 Index")
        st.metric("Documents", info["indexed_documents"])
        st.metric("Chunks", info["indexed_chunks"])

        st.divider()
        st.header("⚙️ System")

        # Report what is actually running, never what was requested. A fallback
        # run must never look like a real one.
        if info["using_real_model"]:
            st.success(f"Embeddings: {info['embedder']}")
        else:
            st.warning(f"Embeddings: {info['embedder']} (fallback)")
            st.caption(
                "The sentence-transformers model could not be loaded, so a "
                "deterministic lexical embedder is in use. Retrieval works but "
                "cannot match synonyms."
            )

        backend = info["llm_backend"]
        if backend == "offline":
            st.warning("LLM: offline responder")
            st.caption(
                "No API key or local server configured. Answers are extracted "
                "from your notes rather than generated. Set OPENAI_API_KEY, or "
                "OPENAI_BASE_URL for a local Ollama server."
            )
        else:
            st.success(f"LLM: {backend}")

        st.caption(f"Distance metric: {info['distance_space']}")

        st.divider()
        st.header("⏰ Reminders")
        horizon = st.slider("Look ahead (days)", 1, 90, horizon_days)

        st.divider()
        if st.button("Clear conversation", use_container_width=True):
            app.reset_conversation()
            st.session_state.messages = []
            st.rerun()

    return horizon


def render_deadline_panel(engine: ReminderEngine, horizon_days: int) -> None:
    """
    Render the upcoming-deadline panel.

    Args:
        engine: The reminder engine to read from
        horizon_days: How far ahead to show
    """
    st.subheader("⏰ Upcoming Deadlines")

    overdue = engine.overdue()
    upcoming = engine.upcoming(horizon_days)
    following = engine.next_deadline()

    col1, col2, col3 = st.columns(3)
    col1.metric("Overdue", len(overdue))
    col2.metric(f"Next {horizon_days} days", len(upcoming))
    col3.metric(
        "Next deadline",
        following.deadline.date.isoformat() if following else "—",
        f"in {following.days_until} days" if following else None,
    )

    if overdue:
        with st.expander(f"🔴 Overdue ({len(overdue)})"):
            for reminder in overdue:
                st.error(
                    f"**{reminder.deadline.title}** — {abs(reminder.days_until)} "
                    f"day(s) overdue ({reminder.deadline.date.isoformat()}) "
                    f"· `{reminder.deadline.source_doc}`"
                )

    if not upcoming:
        st.success(f"Nothing due in the next {horizon_days} days.")
        return

    for reminder in upcoming:
        icon, level = URGENCY_STYLE.get(reminder.urgency, ("🟢", "info"))
        when = (
            "TODAY"
            if reminder.days_until == 0
            else "tomorrow"
            if reminder.days_until == 1
            else f"in {reminder.days_until} days"
        )
        at_time = f" at {reminder.deadline.time}" if reminder.deadline.time else ""
        getattr(st, level)(
            f"{icon} **{reminder.deadline.title}** — {when} "
            f"({reminder.deadline.date.isoformat()}{at_time}) "
            f"· `{reminder.deadline.source_doc}`"
        )


def render_answer(result: dict) -> None:
    """
    Render one answer with its provenance.

    Args:
        result: The dictionary returned by RAGApplication.ask
    """
    st.markdown(result["answer"])

    if result["was_rewritten"]:
        st.caption(
            f"🔄 Resolved from conversation: _{result['resolved_query']}_ — "
            "a follow-up has no topical content on its own, so it is rewritten "
            "before retrieval."
        )

    if not result["sources"]:
        st.caption("No passage cleared the relevance threshold.")
        return

    with st.expander(f"📎 {len(result['sources'])} source(s)"):
        for source in result["sources"]:
            st.write(f"`{source['doc_id']}` — score {source['score']:.3f}")
        st.caption(f"Prompt: grounded_qa · {result['llm']['backend']} backend")


def main():
    """Run the web interface."""
    st.title("🤖 Personal Assistant AI")
    st.caption("Ask questions about your notes. Answers are grounded in your own documents.")

    app = get_app()
    horizon = render_sidebar(app, st.session_state.get("horizon", 14))
    st.session_state.horizon = horizon

    render_deadline_panel(get_reminder_engine(), horizon)
    st.divider()

    st.subheader("💬 Ask a question")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if not st.session_state.messages:
        st.caption("Try one of these:")
        columns = st.columns(len(SUGGESTED_QUESTIONS))
        for column, suggestion in zip(columns, SUGGESTED_QUESTIONS):
            if column.button(suggestion, use_container_width=True):
                st.session_state.pending = suggestion
                st.rerun()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant" and "result" in message:
                render_answer(message["result"])
            else:
                st.markdown(message["content"])

    question = st.chat_input("What would you like to know?") or st.session_state.pop(
        "pending", None
    )

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching your notes..."):
                result = app.ask(question)
            render_answer(result)

        st.session_state.messages.append(
            {"role": "assistant", "content": result["answer"], "result": result}
        )

    st.divider()
    st.caption(
        "Personal Assistant AI · Chroma vector index · conversational memory · "
        "grounded prompts with citations"
    )


if __name__ == "__main__":
    main()
