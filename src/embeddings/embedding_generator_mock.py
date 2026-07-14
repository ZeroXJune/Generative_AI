"""Mock embedding generator for environments with restricted network access."""

import numpy as np
import json
from pathlib import Path
from typing import List, Dict


class MockEmbeddingGenerator:
    """Mock embeddings using pre-computed vectors for testing."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize mock generator."""
        self.model_name = model_name
        self.embedding_dim = 384

    def embed_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Generate mock embeddings based on text hash."""
        embeddings = []
        for text in texts:
            # Deterministic embedding based on text hash
            np.random.seed(hash(text) % (2**32))
            embedding = np.random.randn(self.embedding_dim).astype(np.float32)
            # Normalize
            embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
            embeddings.append(embedding)

        return np.array(embeddings)

    def embed_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """Embed chunks with mock embeddings."""
        texts = [chunk["text"] for chunk in chunks]
        embeddings = self.embed_texts(texts)

        for chunk, embedding in zip(chunks, embeddings):
            chunk["embedding"] = embedding.tolist()

        return chunks

    def save_embeddings(self, chunks: List[Dict], output_path: str):
        """Save embeddings to JSONL file."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w") as f:
            for chunk in chunks:
                f.write(json.dumps(chunk) + "\n")

        print(f"Saved {len(chunks)} embeddings to {output_path}")

    def get_model_info(self) -> dict:
        """Get model information."""
        return {
            "model_name": self.model_name,
            "embedding_dimension": self.embedding_dim,
            "model_size_mb": 0,
            "max_seq_length": 512,
            "type": "mock"
        }
