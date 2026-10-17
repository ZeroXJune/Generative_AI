"""
Environment diagnostic.

Run:  python scripts/doctor.py

Answers one question the rest of the project cannot answer for you: is this
machine actually running the real system, or silently falling back?

Every component here degrades gracefully rather than crashing - a missing
embedding model becomes a lexical embedder, a missing API key becomes an
offline responder. That is good for robustness and bad for confidence, because
a degraded run looks like a working one. This script makes the difference
explicit before you rely on a result or record a demo.

Exit code is 0 when the system is fully operational, 1 when it runs degraded,
and 2 when something is genuinely broken.
"""

import importlib
import os
import platform
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

OK, WARN, FAIL = "[ OK ]", "[WARN]", "[FAIL]"

# Packages the code imports, and whether the system works without them.
REQUIRED = {
    "chromadb": "vector store - the index cannot be built without it",
    "numpy": "vector maths",
}
OPTIONAL = {
    "sentence_transformers": "real MiniLM embeddings (falls back to lexical TF-IDF)",
    "openai": "Chat Completion client (falls back to offline extractive answers)",
    "streamlit": "web interface (CLI still works)",
    "langchain_core": "LangChain integration (Checkpoint 3)",
    "dotenv": "reading credentials from .env",
}


class Report:
    """Collects findings and decides an overall verdict."""

    def __init__(self):
        self.failures = []
        self.warnings = []

    def check(self, status: str, label: str, detail: str = "") -> None:
        """
        Record and print one finding.

        Args:
            status: OK, WARN or FAIL
            label: Short description of what was checked
            detail: Optional explanation shown indented beneath
        """
        print(f"  {status} {label}")
        if detail:
            print(f"         {detail}")
        if status == FAIL:
            self.failures.append(label)
        elif status == WARN:
            self.warnings.append(label)

    def exit_code(self) -> int:
        """Return 0 fully operational, 1 degraded, 2 broken."""
        if self.failures:
            return 2
        return 1 if self.warnings else 0


def section(title: str) -> None:
    """Print a section heading."""
    print(f"\n{title}\n" + "-" * len(title))


def check_python(report: Report) -> None:
    """Check the interpreter version and whether it is isolated."""
    section("Python")
    major, minor = sys.version_info[:2]
    version = f"{major}.{minor}.{sys.version_info[2]}"

    if (major, minor) < (3, 10):
        report.check(FAIL, f"Python {version}", "Too old. Install 3.11 or 3.12.")
    elif (major, minor) <= (3, 12):
        report.check(OK, f"Python {version}")
    else:
        report.check(
            WARN,
            f"Python {version}",
            "Newer than the pinned dependencies support. Use "
            "requirements-core.txt, or install Python 3.12.",
        )

    print(f"         interpreter: {sys.executable}")

    # A venv or conda env sets one of these; a bare system Python sets neither.
    in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    in_conda = bool(os.environ.get("CONDA_PREFIX"))
    if in_venv or in_conda:
        report.check(OK, f"Isolated environment ({'conda' if in_conda else 'venv'})")
    else:
        report.check(
            WARN,
            "No virtual environment detected",
            "Installing into a system Python risks version conflicts.",
        )


def check_packages(report: Report) -> None:
    """Check that required and optional packages import."""
    section("Packages")
    for name, why in REQUIRED.items():
        try:
            importlib.import_module(name)
            report.check(OK, name)
        except ImportError:
            report.check(FAIL, name, f"Missing. Needed for: {why}")

    for name, why in OPTIONAL.items():
        try:
            importlib.import_module(name)
            report.check(OK, name)
        except ImportError:
            report.check(WARN, f"{name} not installed", why)


def check_embeddings(report: Report) -> None:
    """Determine which embedder will actually run."""
    section("Embeddings - which model will actually be used?")
    try:
        from build_index import load_embedder

        embedder, is_real = load_embedder()
    except Exception as error:  # noqa: BLE001 - diagnostic must not crash
        report.check(FAIL, "Embedder could not be constructed", str(error)[:120])
        return

    if is_real:
        report.check(
            OK,
            f"Real model: {embedder.model_name}",
            "Semantic embeddings active. Results are representative.",
        )
    else:
        report.check(
            WARN,
            f"Fallback embedder: {embedder.model_name}",
            "Lexical, not semantic - it cannot match synonyms. Install "
            "sentence-transformers and ensure huggingface.co is reachable.",
        )


def check_llm(report: Report) -> None:
    """Determine which chat backend will actually answer."""
    section("LLM - which backend will actually answer?")
    try:
        from llm.chat_client import ChatClient

        info = ChatClient().get_info()
    except Exception as error:  # noqa: BLE001
        report.check(FAIL, "Chat client could not be constructed", str(error)[:120])
        return

    backend = info["backend"]
    if backend == "offline":
        report.check(
            WARN,
            "Offline extractive responder",
            "Answers are quoted from your notes, not generated. Set "
            "OPENAI_API_KEY, or OPENAI_BASE_URL for a local Ollama server.",
        )
    else:
        report.check(OK, f"Live backend: {backend} ({info['model']})")


def check_data(report: Report) -> None:
    """Check the corpus and whether an index exists."""
    section("Data")
    raw = PROJECT_ROOT / "data" / "raw"
    documents = list(raw.glob("*.txt")) if raw.is_dir() else []

    if documents:
        report.check(OK, f"Corpus: {len(documents)} documents in data/raw")
    else:
        report.check(FAIL, "No documents found in data/raw")
        return

    try:
        from retrieval.vector_store import VectorStore

        store = VectorStore(
            collection_name="personal_assistant_cp3",
            persist_directory="data/vector_store_cp3",
        )
        count = store.count()
    except Exception:  # noqa: BLE001
        count = 0

    if count:
        report.check(OK, f"Index built: {count} chunks")
    else:
        report.check(
            WARN,
            "No index yet",
            "Run: python src/build_index.py  (or python src/checkpoint3_demo.py)",
        )


def main() -> int:
    """Run every check and print a verdict."""
    print("=" * 68)
    print("PERSONAL ASSISTANT AI - ENVIRONMENT DOCTOR")
    print("=" * 68)
    print(f"  platform: {platform.platform()}")
    print(f"  project : {PROJECT_ROOT}")

    report = Report()
    check_python(report)
    check_packages(report)
    check_data(report)
    check_embeddings(report)
    check_llm(report)

    section("Verdict")
    code = report.exit_code()
    if code == 0:
        print("  FULLY OPERATIONAL - real embeddings and a live LLM.")
        print("  Results from this machine are representative. Record your demo here.")
    elif code == 1:
        print(f"  DEGRADED but working - {len(report.warnings)} fallback(s) active:")
        for item in report.warnings:
            print(f"    - {item}")
        print("\n  The pipeline runs end to end, but answer quality is NOT")
        print("  representative. Resolve the warnings above before recording a demo")
        print("  or quoting retrieval scores as results.")
    else:
        print(f"  BROKEN - {len(report.failures)} blocking problem(s):")
        for item in report.failures:
            print(f"    - {item}")
        print("\n  Fix these before anything else:")
        print("    python -m pip install -r requirements-core.txt")

    print("=" * 68)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
