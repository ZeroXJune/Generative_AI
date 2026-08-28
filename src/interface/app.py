"""
Personal Assistant AI - Web Interface
Streamlit app for testing RAG pipeline locally
"""

import streamlit as st
import json
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from embeddings.embedding_generator import EmbeddingGenerator
    USE_REAL_EMBEDDINGS = True
except ImportError:
    from embeddings.embedding_generator_mock import MockEmbeddingGenerator
    USE_REAL_EMBEDDINGS = False

from schedule.extractor import extract_from_corpus
from schedule.reminders import ReminderEngine
from schedule.store import load_deadlines

# Colour per urgency band, used by the deadline panel.
URGENCY_STYLE = {
    "overdue": ("🔴", "error"),
    "today": ("🟠", "error"),
    "urgent": ("🟠", "warning"),
    "soon": ("🟡", "warning"),
    "upcoming": ("🟢", "info"),
}


def cosine_similarity_simple(a: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between vector a and vectors in B."""
    a = np.array(a)
    B = np.array(B)

    # Normalize
    a_norm = a / (np.linalg.norm(a) + 1e-8)
    B_norm = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-8)

    # Compute similarities
    return np.dot(B_norm, a_norm)


# Page config
st.set_page_config(
    page_title="Personal Assistant AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        max-width: 1200px;
    }
    .stChatMessage {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_chunks():
    """Load processed chunks from JSONL file."""
    chunks_file = Path(__file__).parent.parent.parent / "data" / "processed" / "chunks.jsonl"

    if not chunks_file.exists():
        return None

    chunks = []
    with open(chunks_file, "r") as f:
        for line in f:
            chunks.append(json.loads(line))

    return chunks


@st.cache_resource
def get_embedder():
    """Get embedding generator."""
    if USE_REAL_EMBEDDINGS:
        return EmbeddingGenerator()
    else:
        return MockEmbeddingGenerator()


def get_reminder_engine() -> ReminderEngine:
    """
    Build a ReminderEngine from the stored deadlines.

    Falls back to scanning data/raw when the store has not been built yet, so
    the panel works before anyone runs the CLI.

    Returns:
        A ReminderEngine over the available deadlines
    """
    deadlines = load_deadlines()
    if not deadlines:
        deadlines = extract_from_corpus()
    return ReminderEngine(deadlines)


def render_deadline_panel(engine: ReminderEngine, horizon_days: int):
    """
    Render the upcoming-deadline panel.

    Args:
        engine: The reminder engine to read from
        horizon_days: How far ahead to show
    """
    st.subheader("⏰ Upcoming Deadlines")

    overdue = engine.overdue()
    upcoming = engine.upcoming(horizon_days)

    col1, col2, col3 = st.columns(3)
    col1.metric("Overdue", len(overdue))
    col2.metric(f"Next {horizon_days} days", len(upcoming))

    following = engine.next_deadline()
    col3.metric(
        "Next deadline",
        following.deadline.date.isoformat() if following else "—",
        f"in {following.days_until} days" if following else None,
    )

    if overdue:
        with st.expander(f"🔴 Overdue ({len(overdue)})", expanded=False):
            for reminder in overdue:
                st.error(
                    f"**{reminder.deadline.title}** — "
                    f"{abs(reminder.days_until)} day(s) overdue "
                    f"({reminder.deadline.date.isoformat()}) "
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
        message = (
            f"{icon} **{reminder.deadline.title}** — {when} "
            f"({reminder.deadline.date.isoformat()}{at_time}) "
            f"· `{reminder.deadline.source_doc}`"
        )
        getattr(st, level)(message)


def find_similar_chunks(query: str, chunks: list, top_k: int = 3) -> list:
    """Find most similar chunks to query."""
    embedder = get_embedder()

    # Embed query
    query_embedding = embedder.embed_texts([query])[0]

    # Get chunk embeddings
    chunk_embeddings = np.array([chunk["embedding"] for chunk in chunks])

    # Compute similarities
    similarities = cosine_similarity_simple(query_embedding, chunk_embeddings)

    # Get top-k indices
    top_indices = np.argsort(similarities)[::-1][:top_k]

    # Return chunks with scores
    results = []
    for idx in top_indices:
        results.append({
            "chunk": chunks[idx],
            "similarity": float(similarities[idx])
        })

    return results


def main():
    """Main app."""

    # Header
    st.title("🤖 Personal Assistant AI")
    st.markdown("*Ask questions about your documents*")

    # Load chunks
    chunks = load_chunks()

    if chunks is None:
        st.error("❌ No processed data found. Please run the pipeline first:")
        st.code("python src/pipeline.py", language="bash")
        return

    # Sidebar info
    with st.sidebar:
        st.header("📊 Dataset Info")
        st.metric("Documents Indexed", len(set(c["doc_id"] for c in chunks)))
        st.metric("Total Chunks", len(chunks))
        st.metric("Total Tokens", sum(c["token_count"] for c in chunks))

        embedding_mode = "🔴 Real Embeddings" if USE_REAL_EMBEDDINGS else "🟡 Mock Embeddings"
        st.caption(f"Mode: {embedding_mode}")

        st.divider()

        st.header("📚 Documents")
        docs = sorted(set(c["doc_id"] for c in chunks))
        for doc in docs[:10]:  # Show first 10
            st.text(f"📄 {doc}")
        if len(docs) > 10:
            st.text(f"... and {len(docs)-10} more")

        st.divider()

        st.header("⚙️ Settings")
        top_k = st.slider("Number of results", 1, 10, 3)
        similarity_threshold = st.slider("Min similarity", 0.0, 1.0, 0.3)

        st.divider()

        st.header("⏰ Reminders")
        horizon_days = st.slider("Look ahead (days)", 1, 90, 14)

    # Main content
    st.divider()

    # Deadline reminders - the schedule-tracking objective from the proposal
    engine = get_reminder_engine()
    render_deadline_panel(engine, horizon_days)

    st.divider()

    # Chat interface
    st.subheader("💬 Ask a Question")

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Input
    query = st.chat_input("What would you like to know about your documents?")

    if query:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": query})

        with st.chat_message("user"):
            st.markdown(query)

        # Find similar chunks
        results = find_similar_chunks(query, chunks, top_k=top_k)

        # Filter by threshold
        results = [r for r in results if r["similarity"] >= similarity_threshold]

        if not results:
            response = f"❌ No relevant documents found (threshold: {similarity_threshold})"
            with st.chat_message("assistant"):
                st.markdown(response)
        else:
            # Build response
            response = f"Found **{len(results)}** relevant chunk(s):\n\n"

            for i, result in enumerate(results, 1):
                chunk = result["chunk"]
                similarity = result["similarity"]

                response += f"### Result {i} (Similarity: {similarity:.1%})\n"
                response += f"📄 **Source**: {chunk['doc_id']}\n"
                response += f"📍 **Chunk**: {chunk['chunk_index']}\n"
                response += f"📏 **Tokens**: {chunk['token_count']}\n\n"
                response += f"**Content**:\n{chunk['text'][:300]}...\n\n"
                response += "---\n\n"

            response += "💡 **Note**: Results based on semantic similarity. For best results, ask specific questions about your documents."

            st.session_state.messages.append({"role": "assistant", "content": response})

            with st.chat_message("assistant"):
                st.markdown(response)

                # Show detailed view
                with st.expander("📊 Detailed Similarity Analysis"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Similarity Scores")
                        for i, result in enumerate(results, 1):
                            st.write(f"Result {i}: **{result['similarity']:.1%}**")

                    with col2:
                        st.subheader("Full Chunks")
                        for i, result in enumerate(results, 1):
                            with st.expander(f"Chunk {i}"):
                                st.text(result['chunk']['text'])

    st.divider()

    # Footer
    st.markdown("""
    ---
    **Personal Assistant AI** | Semantic search + deadline reminders
    Built with [Streamlit](https://streamlit.io) | Powered by Semantic Search
    """)


if __name__ == "__main__":
    main()
