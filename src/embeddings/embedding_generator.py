import numpy as np
from typing import List, Dict, Tuple
from pathlib import Path
import json


class EmbeddingGenerator:
    """Generate embeddings for text using Sentence-Transformers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize embedding generator with a pre-trained model.

        Args:
            model_name: HuggingFace model identifier
                - all-MiniLM-L6-v2: Fast, 384 dims, ~50M params (recommended)
                - all-mpnet-base-v2: Slower, 768 dims, ~110M params (better quality)
                - all-distilroberta-v1: Medium, 768 dims, ~80M params
        """
        self.model_name = model_name
        self.model = None
        self.embedding_dim = None
        self._load_model()

    def _load_model(self):
        """Load Sentence-Transformers model."""
        try:
            from sentence_transformers import SentenceTransformer

            print(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            test_embedding = self.model.encode("test")
            self.embedding_dim = len(test_embedding)
            print(f"Model loaded. Embedding dimension: {self.embedding_dim}")
        except ImportError:
            raise ImportError(
                "Please install sentence-transformers: pip install sentence-transformers"
            )

    def embed_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Embed a list of texts.

        Args:
            texts: List of text strings to embed
            batch_size: Process texts in batches for efficiency

        Returns:
            NumPy array of shape (len(texts), embedding_dim)
        """
        embeddings = self.model.encode(texts, batch_size=batch_size, show_progress_bar=True)
        return embeddings

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
        """
        Get information about the embedding model.

        Returns:
            Dictionary with model details
        """
        return {
            "model_name": self.model_name,
            "embedding_dimension": self.embedding_dim,
            "model_size_mb": self._estimate_model_size(),
            "max_seq_length": self.model.max_seq_length,
        }

    @staticmethod
    def _estimate_model_size() -> float:
        """Rough estimate of model size in MB based on model name."""
        size_map = {
            "all-MiniLM-L6-v2": 50,
            "all-mpnet-base-v2": 440,
            "all-distilroberta-v1": 250,
            "paraphrase-MiniLM-L6-v2": 50,
        }
        return size_map.get("all-MiniLM-L6-v2", 100)

    def get_embedding_example(self, text: str) -> dict:
        """
        Get example of embedding for a piece of text.

        Args:
            text: Text to embed

        Returns:
            Dictionary with embedding example and statistics
        """
        embedding = self.embed_texts([text])[0]

        return {
            "text": text[:100] + "...",
            "embedding_dim": len(embedding),
            "embedding_sample": embedding[:5].tolist(),  # First 5 dims
            "embedding_norm": float(np.linalg.norm(embedding)),
            "embedding_mean": float(np.mean(embedding)),
            "embedding_std": float(np.std(embedding)),
        }


if __name__ == "__main__":
    # Initialize generator
    generator = EmbeddingGenerator(model_name="all-MiniLM-L6-v2")

    # Get model info
    model_info = generator.get_model_info()
    print("MODEL INFORMATION:")
    print(f"  Model: {model_info['model_name']}")
    print(f"  Embedding Dimension: {model_info['embedding_dimension']}")
    print(f"  Model Size: ~{model_info['model_size_mb']} MB")
    print(f"  Max Sequence Length: {model_info['max_seq_length']}\n")

    # Example 1: Single sentence
    example1_text = (
        "Retrieval-Augmented Generation is a powerful technique for grounding LLM responses."
    )
    example1 = generator.get_embedding_example(example1_text)
    print("EXAMPLE 1 - Single Sentence:")
    print(f"  Text: {example1['text']}")
    print(f"  Embedding Dim: {example1['embedding_dim']}")
    print(f"  Sample (first 5): {example1['embedding_sample']}")
    print(f"  Norm: {example1['embedding_norm']:.4f}")
    print(f"  Mean: {example1['embedding_mean']:.6f}")
    print(f"  Std: {example1['embedding_std']:.6f}\n")

    # Example 2: Multi-sentence paragraph
    example2_text = """
    The capstone project demonstrates mastery of generative AI concepts.
    Students design, build, and deploy a real RAG application.
    The project includes data pipeline, prompt engineering, and deployment.
    """
    example2 = generator.get_embedding_example(example2_text)
    print("EXAMPLE 2 - Multi-sentence Paragraph:")
    print(f"  Text: {example2['text']}")
    print(f"  Embedding Dim: {example2['embedding_dim']}")
    print(f"  Norm: {example2['embedding_norm']:.4f}\n")

    # Example 3: Similarity comparison
    texts_to_compare = [
        "Machine learning is a subset of artificial intelligence",
        "Deep learning uses neural networks for pattern recognition",
        "Cooking requires mixing ingredients and heat",
        "RAG systems retrieve documents before generating responses",
    ]

    embeddings = generator.embed_texts(texts_to_compare)

    print("EXAMPLE 3 - Similarity Between Texts:")
    print("Computing cosine similarities between texts...")
    from sklearn.metrics.pairwise import cosine_similarity

    similarities = cosine_similarity(embeddings)
    print("\nSimilarity Matrix (Cosine):")
    for i, text1 in enumerate(texts_to_compare):
        print(f"\n  {i}: {text1[:50]}")
        for j, text2 in enumerate(texts_to_compare[i+1:], i+1):
            sim = similarities[i, j]
            print(f"     vs {j}: {sim:.4f}")
