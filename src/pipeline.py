"""
Main data processing pipeline for Personal Assistant AI capstone.
Demonstrates CILO 1: Full end-to-end data pipeline.
"""

import sys
from pathlib import Path
from typing import List, Tuple
import json

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent))

from preprocessing.text_cleaner import TextCleaner
from preprocessing.chunker import DocumentChunker
from embeddings.embedding_generator import EmbeddingGenerator


class DataPipeline:
    """End-to-end data processing pipeline."""

    def __init__(
        self,
        raw_data_dir: str = "data/raw",
        processed_data_dir: str = "data/processed",
        chunk_size: int = 512,
        overlap: int = 100,
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        """
        Initialize the pipeline.

        Args:
            raw_data_dir: Directory containing raw documents
            processed_data_dir: Directory to save processed data
            chunk_size: Size of document chunks
            overlap: Overlap between chunks
            embedding_model: Name of embedding model to use
        """
        self.raw_dir = Path(raw_data_dir)
        self.processed_dir = Path(processed_data_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        self.cleaner = TextCleaner()
        self.chunker = DocumentChunker(chunk_size=chunk_size, overlap=overlap)
        self.embedder = EmbeddingGenerator(model_name=embedding_model)

        print(f"Pipeline initialized")
        print(f"  Raw data: {self.raw_dir}")
        print(f"  Processed data: {self.processed_dir}")

    def load_documents(self) -> List[Tuple[str, str]]:
        """
        Load all text documents from raw data directory.

        Returns:
            List of (text, filename) tuples
        """
        documents = []
        txt_files = list(self.raw_dir.glob("*.txt"))

        print(f"\nLoading {len(txt_files)} documents...")
        for file_path in txt_files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
                documents.append((text, file_path.stem))
                print(f"  ✓ {file_path.name}")
            except Exception as e:
                print(f"  ✗ {file_path.name}: {e}")

        return documents

    def preprocess_documents(
        self, documents: List[Tuple[str, str]]
    ) -> List[Tuple[str, str]]:
        """
        Clean documents using text cleaner.

        Args:
            documents: List of (text, doc_id) tuples

        Returns:
            List of (cleaned_text, doc_id) tuples
        """
        print(f"\nCleaning {len(documents)} documents...")
        cleaned = []

        for text, doc_id in documents:
            cleaned_text = self.cleaner.clean(text)
            cleaned.append((cleaned_text, doc_id))

        # Save before/after examples
        examples = []
        for (raw, doc_id), (cleaned_text, _) in zip(documents[:3], cleaned[:3]):
            examples.append(
                {
                    "doc_id": doc_id,
                    "before": raw[:150],
                    "after": cleaned_text[:150],
                }
            )

        examples_file = self.processed_dir / "cleaning_examples.json"
        with open(examples_file, "w") as f:
            json.dump(examples, f, indent=2)

        print(f"  ✓ Cleaned all documents")
        print(f"  ✓ Saved examples to {examples_file}")

        return cleaned

    def chunk_documents(
        self, documents: List[Tuple[str, str]]
    ) -> List[dict]:
        """
        Chunk documents for embedding.

        Args:
            documents: List of (text, doc_id) tuples

        Returns:
            List of chunk dictionaries
        """
        print(f"\nChunking {len(documents)} documents...")
        chunks = self.chunker.chunk_documents(documents)

        print(f"  ✓ Created {len(chunks)} chunks")
        print(f"    Average chunk tokens: {sum(c['token_count'] for c in chunks) // len(chunks)}")

        return chunks

    def embed_chunks(self, chunks: List[dict]) -> List[dict]:
        """
        Generate embeddings for chunks.

        Args:
            chunks: List of chunk dictionaries

        Returns:
            List of chunks with embeddings
        """
        print(f"\nGenerating embeddings for {len(chunks)} chunks...")
        chunks_with_embeddings = self.embedder.embed_chunks(chunks)

        # Save embeddings
        embeddings_file = self.processed_dir / "chunks.jsonl"
        self.embedder.save_embeddings(chunks_with_embeddings, str(embeddings_file))

        return chunks_with_embeddings

    def run_pipeline(self) -> dict:
        """
        Execute full pipeline end-to-end.

        Returns:
            Dictionary with pipeline results and statistics
        """
        print("=" * 60)
        print("PERSONAL ASSISTANT AI - DATA PIPELINE")
        print("=" * 60)

        # Step 1: Load
        documents = self.load_documents()
        if not documents:
            print("No documents found!")
            return {}

        # Step 2: Preprocess
        cleaned_docs = self.preprocess_documents(documents)

        # Step 3: Chunk
        chunks = self.chunk_documents(cleaned_docs)

        # Step 4: Embed
        final_chunks = self.embed_chunks(chunks)

        # Summary statistics
        total_tokens = sum(c["token_count"] for c in final_chunks)
        total_dims = len(final_chunks[0]["embedding"]) if final_chunks else 0

        results = {
            "num_documents": len(documents),
            "num_chunks": len(final_chunks),
            "total_tokens": total_tokens,
            "avg_chunk_size": total_tokens // len(final_chunks) if final_chunks else 0,
            "embedding_dimension": total_dims,
            "embedding_model": self.embedder.model_name,
        }

        print("\n" + "=" * 60)
        print("PIPELINE COMPLETE - SUMMARY")
        print("=" * 60)
        print(f"Documents processed: {results['num_documents']}")
        print(f"Chunks created: {results['num_chunks']}")
        print(f"Total tokens: {results['total_tokens']:,}")
        print(f"Avg chunk size: {results['avg_chunk_size']} tokens")
        print(f"Embedding dimension: {results['embedding_dimension']}")
        print(f"Embedding model: {results['embedding_model']}")
        print(f"Output: {self.processed_dir}")
        print("=" * 60)

        return results


def main():
    """Run the data pipeline."""
    pipeline = DataPipeline(
        raw_data_dir="data/raw",
        processed_data_dir="data/processed",
        chunk_size=512,
        overlap=100,
        embedding_model="all-MiniLM-L6-v2",
    )

    results = pipeline.run_pipeline()

    return results


if __name__ == "__main__":
    main()
