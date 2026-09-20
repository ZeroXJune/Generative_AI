"""
Automated ingestion pipeline.

Checkpoint 3 deliverable. Checkpoint 2 rebuilt the entire index on every run,
which is fine for 26 documents and wasteful for anything larger. This pipeline
tracks what it has already seen and processes only what changed.

State lives in a manifest (`data/processed/ingestion_manifest.json`) mapping
each document to a SHA-256 of its contents. On each run the pipeline compares
the corpus against that manifest and classifies every document as new,
modified, unchanged, or deleted -- then touches only the first three.

Chunking uses LangChain's RecursiveCharacterTextSplitter, which splits on
paragraph and sentence boundaries rather than a fixed token count, so a chunk
is less likely to end mid-sentence.
"""

import hashlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from preprocessing.text_cleaner import TextCleaner

DEFAULT_MANIFEST_PATH = "data/processed/ingestion_manifest.json"

# Chunk sizes are in characters here, not whitespace tokens: the LangChain
# splitter measures characters by default. ~600 chars approximates the 120-token
# chunks tuned in Checkpoint 2 at roughly 5 characters per token.
DEFAULT_CHUNK_SIZE = 600
DEFAULT_CHUNK_OVERLAP = 150


@dataclass
class IngestionReport:
    """Summary of what one ingestion run changed."""

    new: List[str] = field(default_factory=list)
    modified: List[str] = field(default_factory=list)
    unchanged: List[str] = field(default_factory=list)
    deleted: List[str] = field(default_factory=list)
    chunks_written: int = 0
    chunks_removed: int = 0

    @property
    def processed(self) -> List[str]:
        """Documents that required embedding this run."""
        return self.new + self.modified

    @property
    def changed(self) -> bool:
        """True when the run altered the index in any way."""
        return bool(self.new or self.modified or self.deleted)

    def summary(self) -> dict:
        """Return a compact record for logging and reporting."""
        return {
            "new": len(self.new),
            "modified": len(self.modified),
            "unchanged": len(self.unchanged),
            "deleted": len(self.deleted),
            "chunks_written": self.chunks_written,
            "chunks_removed": self.chunks_removed,
        }

    def describe(self) -> str:
        """Render a human-readable summary of the run."""
        if not self.changed:
            return f"No changes. {len(self.unchanged)} document(s) already current."

        lines = []
        for label, docs in (
            ("NEW", self.new),
            ("MODIFIED", self.modified),
            ("DELETED", self.deleted),
        ):
            if docs:
                lines.append(f"  {label} ({len(docs)}): {', '.join(sorted(docs)[:6])}")
        lines.append(
            f"  {len(self.unchanged)} unchanged, "
            f"+{self.chunks_written} chunks, -{self.chunks_removed} chunks"
        )
        return "\n".join(lines)


class IngestionPipeline:
    """Incrementally ingest a document corpus into a vector store."""

    def __init__(
        self,
        vector_store,
        embedder,
        raw_dir: str = "data/raw",
        manifest_path: str = DEFAULT_MANIFEST_PATH,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ):
        """
        Configure the pipeline.

        Args:
            vector_store: A VectorStore to write chunks into
            embedder: Any object exposing embed_texts / fit
            raw_dir: Directory of .txt source documents
            manifest_path: Where ingestion state is recorded
            chunk_size: Target chunk width in characters
            chunk_overlap: Overlap between neighbouring chunks, in characters
        """
        self.vector_store = vector_store
        self.embedder = embedder
        self.raw_dir = Path(raw_dir)
        self.manifest_path = Path(manifest_path)
        self.cleaner = TextCleaner()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._splitter = None

    @property
    def splitter(self):
        """
        Lazily build the LangChain text splitter.

        Deferred so that importing this module does not require LangChain;
        only an actual ingestion run does.
        """
        if self._splitter is None:
            from langchain_text_splitters import RecursiveCharacterTextSplitter

            self._splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                # Prefer paragraph breaks, then sentences, then words.
                separators=["\n\n", "\n", ". ", " ", ""],
                length_function=len,
            )
        return self._splitter

    @staticmethod
    def content_hash(text: str) -> str:
        """Return a stable SHA-256 hex digest of a document's contents."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def load_manifest(self) -> Dict[str, str]:
        """
        Read the ingestion manifest.

        Returns:
            Mapping of doc_id to content hash; empty on a first run or if the
            manifest is unreadable
        """
        if not self.manifest_path.is_file():
            return {}
        try:
            data = json.loads(self.manifest_path.read_text())
        except json.JSONDecodeError:
            # A corrupt manifest should trigger a full rebuild, not a crash.
            return {}
        return data.get("documents", {}) if isinstance(data, dict) else {}

    def save_manifest(self, hashes: Dict[str, str]) -> None:
        """
        Persist the ingestion manifest.

        Args:
            hashes: Mapping of doc_id to content hash
        """
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self.manifest_path.write_text(
            json.dumps({"documents": hashes}, indent=2, sort_keys=True)
        )

    def scan_corpus(self) -> Dict[str, Tuple[str, str]]:
        """
        Read every source document.

        Returns:
            Mapping of doc_id to (raw_text, content_hash)
        """
        corpus = {}
        for path in sorted(self.raw_dir.glob("*.txt")):
            text = path.read_text(encoding="utf-8")
            corpus[path.stem] = (text, self.content_hash(text))
        return corpus

    def classify(self, corpus: Dict[str, Tuple[str, str]]) -> IngestionReport:
        """
        Compare the corpus against the manifest AND the live index.

        The manifest alone is not trustworthy: it records what was ingested,
        not what the index still contains. If the collection is reset, dropped,
        or partially deleted, a manifest entry can claim a document is indexed
        when it is not -- and ingestion would skip it, leaving an empty index
        that reports success. Reconciling against the store's actual contents
        makes the pipeline self-healing.

        Args:
            corpus: Output of scan_corpus

        Returns:
            A report naming new, modified, unchanged and deleted documents
        """
        manifest = self.load_manifest()
        report = IngestionReport()

        try:
            indexed = set(self.vector_store.list_documents())
        except Exception:
            # If the store cannot be inspected, fall back to manifest-only
            # classification rather than failing the run.
            indexed = set(manifest)

        for doc_id, (_, digest) in corpus.items():
            if doc_id not in manifest or doc_id not in indexed:
                # Unknown, or known but missing from the index.
                report.new.append(doc_id)
            elif manifest[doc_id] != digest:
                report.modified.append(doc_id)
            else:
                report.unchanged.append(doc_id)

        report.deleted = [
            doc_id
            for doc_id in set(manifest) | indexed
            if doc_id not in corpus
        ]
        return report

    def chunk_document(self, text: str, doc_id: str) -> List[Dict]:
        """
        Clean and split a single document.

        Args:
            text: Raw document text
            doc_id: Document identifier recorded on each chunk

        Returns:
            Chunk dictionaries matching the shape the vector store expects
        """
        cleaned = self.cleaner.clean(text)
        pieces = self.splitter.split_text(cleaned)

        return [
            {
                "text": piece,
                "doc_id": doc_id,
                "chunk_index": index,
                "token_count": len(piece.split()),
            }
            for index, piece in enumerate(pieces)
            if piece.strip()
        ]

    def ingest(self, force: bool = False) -> IngestionReport:
        """
        Run one ingestion pass.

        Args:
            force: Re-ingest every document, ignoring the manifest

        Returns:
            A report describing what changed
        """
        corpus = self.scan_corpus()
        report = self.classify(corpus)

        if force:
            report.new = list(corpus)
            report.modified = []
            report.unchanged = []

        # Removing before adding keeps a modified document from ending up with
        # both its old and new chunks in the index.
        for doc_id in report.modified + report.deleted:
            report.chunks_removed += self.vector_store.delete_document(doc_id)

        pending = report.processed
        if pending:
            chunks = []
            for doc_id in pending:
                chunks.extend(self.chunk_document(corpus[doc_id][0], doc_id))

            if chunks:
                # The lexical embedder derives IDF from the corpus, so it must
                # see every document, not just the changed ones.
                if hasattr(self.embedder, "fit"):
                    all_text = [
                        piece["text"]
                        for doc_id, (text, _) in corpus.items()
                        for piece in self.chunk_document(text, doc_id)
                    ]
                    self.embedder.fit(all_text)

                vectors = self.embedder.embed_texts([c["text"] for c in chunks])
                for chunk, vector in zip(chunks, vectors):
                    chunk["embedding"] = vector.tolist()

                report.chunks_written = self.vector_store.add_chunks(chunks)

        self.save_manifest({doc_id: digest for doc_id, (_, digest) in corpus.items()})
        return report
