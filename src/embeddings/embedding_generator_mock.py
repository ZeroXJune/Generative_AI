"""
Mock embedding generator for environments with restricted network access.
Demonstrates the pipeline without requiring actual model download.
"""

import numpy as np
from typing import List, Dict
import json
from pathlib import Path
import hashlib


class MockEmbeddingGenerator:
    """Generate deterministic embeddings for text without downloading models."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", embedding_dim: int = 384):
        """
        Initialize mock embedding generator.

        Args:
            model_name: Model identifier (for documentation)
            embedding_dim: Dimensionality of embeddings
        """
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        print(f"Mock Embedding Generator initialized")
        print(f"  Model: {model_name}")
        print(f"  Embedding Dimension: {embedding_dim}")
        print(f"  (Using deterministic hash-based embeddings)")

    def embed_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Generate deterministic embeddings for texts.

        Args:
            texts: List of text strings
            batch_size: Batch size (ignored for mock)

        Returns:
            Array of shape (len(texts), embedding_dim)
        """
        embeddings = []

        for text in texts:
            # Create deterministic embedding from text hash
            hash_obj = hashlib.sha256(text.encode())
            seed = int(hash_obj.hexdigest(), 16) % (2**31)

            # Use seed to generate consistent random embedding
            np.random.seed(seed)
            embedding = np.random.normal(0, 1, self.embedding_dim)

            # Normalize to unit vector (like real embeddings)
            embedding = embedding / np.linalg.norm(embedding)
            embeddings.append(embedding)

        return np.array(embeddings)

    def embed_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """
        Embed document chunks with metadata.

        Args:
            chunks: List of chunk dictionaries with 'text' field

        Returns:
            List of chunks with added 'embedding' field
        """
        texts = [chunk["text"] for chunk in chunks]
        embeddings = self.embed_texts(texts)

        for chunk, embedding in zip(chunks, embeddings):
            chunk["embedding"] = embedding.tolist()

        return chunks

    def save_embeddings(self, chunks: List[Dict], output_path: str):
        """
        Save embeddings to JSONL file.

        Args:
            chunks: List of chunks with embeddings
            output_path: Path to save JSONL file
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w") as f:
            for chunk in chunks:
                f.write(json.dumps(chunk) + "\n")

        print(f"Saved {len(chunks)} embeddings to {output_path}")

    def get_model_info(self) -> dict:
        """Get information about the embedding model."""
        return {
            "model_name": self.model_name,
            "embedding_dimension": self.embedding_dim,
            "type": "mock_deterministic",
            "note": "Deterministic hash-based embeddings for demonstration"
        }


if __name__ == "__main__":
    generator = MockEmbeddingGenerator()

    # Example texts
    texts = [
        "Machine learning is a subset of artificial intelligence",
        "Deep learning uses neural networks",
        "Cooking requires mixing ingredients",
    ]

    embeddings = generator.embed_texts(texts)

    print(f"\nEmbeddings shape: {embeddings.shape}")
    print(f"First embedding (first 5 dims): {embeddings[0][:5]}")
    print(f"Embedding norm: {np.linalg.norm(embeddings[0]):.4f}")

    # Test similarity
    from sklearn.metrics.pairwise import cosine_similarity
    similarities = cosine_similarity(embeddings)
    print(f"\nSimilarity between text 0 and 1: {similarities[0, 1]:.4f}")
    print(f"Similarity between text 0 and 2: {similarities[0, 2]:.4f}")
