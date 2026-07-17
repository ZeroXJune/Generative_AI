"""
Process SQuAD dataset into RAG pipeline format.
Converts SQuAD JSON to chunks with embeddings.
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from preprocessing.text_cleaner import TextCleaner
from preprocessing.chunker import DocumentChunker
from embeddings.embedding_generator_mock import MockEmbeddingGenerator


class SQuADProcessor:
    """Convert SQuAD dataset to RAG pipeline format."""

    def __init__(
        self,
        squad_file: str = "data/raw/squad_v2.0.json",
        output_file: str = "data/processed/chunks.jsonl",
        chunk_size: int = 512,
        overlap: int = 100,
    ):
        self.squad_file = Path(squad_file)
        self.output_file = Path(output_file)
        self.output_file.parent.mkdir(parents=True, exist_ok=True)

        self.cleaner = TextCleaner()
        self.chunker = DocumentChunker(chunk_size=chunk_size, overlap=overlap)
        self.embedder = MockEmbeddingGenerator()

    def load_squad(self) -> List[Tuple[str, str, str]]:
        """
        Load SQuAD and extract (context, article_title, para_index).

        Returns:
            List of (context_text, article_id, para_id) tuples
        """
        with open(self.squad_file, 'r') as f:
            squad = json.load(f)

        documents = []
        context_id = 0
        for article in squad['data']:
            article_title = article['title']
            for para_idx, paragraph in enumerate(article['paragraphs']):
                context = paragraph['context']
                doc_id = f"{article_title.replace(' ', '_').lower()}_para{para_idx}"
                documents.append((context, doc_id, context_id))
                context_id += 1

        return documents

    def process(self) -> Dict:
        """Process SQuAD through full pipeline."""
        print("=" * 60)
        print("SQUAD DATASET PROCESSING")
        print("=" * 60)

        # Step 1: Load
        print("\n📥 Loading SQuAD contexts...")
        documents = self.load_squad()
        print(f"   ✓ Extracted {len(documents)} contexts")

        # Convert to cleaner format
        docs_for_cleaning = [(doc[0], doc[1]) for doc in documents]

        # Step 2: Clean
        print("\n🧹 Cleaning text...")
        cleaned = []
        for text, doc_id in docs_for_cleaning:
            cleaned_text = self.cleaner.clean(text)
            cleaned.append((cleaned_text, doc_id))
        print(f"   ✓ Cleaned {len(cleaned)} documents")

        # Step 3: Chunk
        print("\n✂️  Chunking documents...")
        chunks = self.chunker.chunk_documents(cleaned)
        print(f"   ✓ Created {len(chunks)} chunks")
        print(f"   ✓ Avg chunk size: {sum(c['token_count'] for c in chunks) // len(chunks)} tokens")

        # Step 4: Embed
        print("\n🔢 Generating embeddings...")
        chunks_with_embeddings = self.embedder.embed_chunks(chunks)

        # Save
        self.embedder.save_embeddings(chunks_with_embeddings, str(self.output_file))
        print(f"   ✓ Saved to {self.output_file}")

        # Stats
        total_tokens = sum(c['token_count'] for c in chunks_with_embeddings)
        results = {
            "num_documents": len(documents),
            "num_chunks": len(chunks_with_embeddings),
            "total_tokens": total_tokens,
            "avg_chunk_size": total_tokens // len(chunks_with_embeddings) if chunks_with_embeddings else 0,
            "embedding_dimension": len(chunks_with_embeddings[0]["embedding"]) if chunks_with_embeddings else 0,
        }

        print("\n" + "=" * 60)
        print("PROCESSING COMPLETE")
        print("=" * 60)
        print(f"Contexts processed: {results['num_documents']}")
        print(f"Chunks created: {results['num_chunks']}")
        print(f"Total tokens: {results['total_tokens']:,}")
        print(f"Avg chunk size: {results['avg_chunk_size']} tokens")
        print(f"Embedding dimension: {results['embedding_dimension']}")
        print("=" * 60)

        return results


if __name__ == "__main__":
    processor = SQuADProcessor()
    processor.process()
