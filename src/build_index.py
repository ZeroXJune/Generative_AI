"""
Build the Chroma vector index from the raw corpus.

Run:  python src/build_index.py

Loads every document in data/raw, cleans and chunks it, embeds the chunks,
and writes them into a persistent Chroma collection ready for retrieval.
"""

import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent))

from preprocessing.chunker import DocumentChunker
from preprocessing.text_cleaner import TextCleaner
from retrieval.vector_store import VectorStore

# Retrieval chunk settings, tuned in src/experiments/metric_comparison.py.
# Checkpoint 1 used 512/100 for the embedding pipeline; that is too coarse for
# retrieval, since a single chunk can swallow most of a short note and drag
# unrelated text into the prompt. 120/30 measured best on the evaluation set
# and keeps retrieved context tight enough to cite.
RETRIEVAL_CHUNK_SIZE = 120
RETRIEVAL_CHUNK_OVERLAP = 30


def load_embedder(model_name: str = "all-MiniLM-L6-v2"):
    """
    Load the best available embedder.

    Prefers the real Sentence-Transformers model. Falls back to the
    deterministic lexical embedder when the model cannot be downloaded, so
    the pipeline still runs on a restricted network.

    Args:
        model_name: Sentence-Transformers model identifier

    Returns:
        Tuple of (embedder, is_real_model)
    """
    try:
        from embeddings.embedding_generator import EmbeddingGenerator

        return EmbeddingGenerator(model_name=model_name), True
    except Exception as error:
        print(f"⚠ Could not load '{model_name}': {str(error)[:110]}")
        print("  Falling back to the lexical TF-IDF embedder (offline).")
        from embeddings.embedding_generator_lexical import LexicalEmbeddingGenerator

        return LexicalEmbeddingGenerator(), False


def build_chunks(
    raw_dir: str = "data/raw",
    chunk_size: int = RETRIEVAL_CHUNK_SIZE,
    overlap: int = RETRIEVAL_CHUNK_OVERLAP,
) -> List[Dict]:
    """
    Load, clean, and chunk every document in the raw corpus.

    Args:
        raw_dir: Directory of .txt documents
        chunk_size: Chunk width in whitespace tokens
        overlap: Token overlap between chunks

    Returns:
        Chunk dictionaries ready for embedding
    """
    cleaner = TextCleaner()
    chunker = DocumentChunker(chunk_size=chunk_size, overlap=overlap)

    paths = sorted(Path(raw_dir).glob("*.txt"))
    print(f"Loading {len(paths)} documents from {raw_dir}")

    documents = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        documents.append((cleaner.clean(text), path.stem))

    chunks = chunker.chunk_documents(documents)
    print(f"  Created {len(chunks)} chunks")
    return chunks


def build_index(
    collection_name: str = "personal_assistant",
    persist_directory: str = "data/vector_store",
    space: str = "cosine",
    reset: bool = True,
) -> VectorStore:
    """
    Build and persist the vector index.

    Args:
        collection_name: Chroma collection name
        persist_directory: On-disk index location
        space: Distance metric for the collection
        reset: Rebuild the collection from scratch

    Returns:
        The populated VectorStore
    """
    print("=" * 62)
    print("BUILDING VECTOR INDEX")
    print("=" * 62)

    chunks = build_chunks()
    embedder, is_real = load_embedder()

    print(f"\nEmbedding {len(chunks)} chunks...")
    embedder.embed_chunks(chunks)

    store = VectorStore(
        collection_name=collection_name,
        persist_directory=persist_directory,
        space=space,
        reset=reset,
    )
    indexed = store.add_chunks(chunks)

    print(f"\nIndexed {indexed} chunks")
    for key, value in store.get_stats().items():
        print(f"  {key}: {value}")
    print(f"  embedder: {embedder.model_name} ({'real model' if is_real else 'offline fallback'})")
    print("=" * 62)

    return store


if __name__ == "__main__":
    build_index()
