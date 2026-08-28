"""
Deterministic lexical embedding generator.

Offline fallback used when the Sentence-Transformers model cannot be
downloaded (restricted network). Unlike the random hash-based mock in
`embedding_generator_mock.py`, these vectors carry genuine lexical signal:
documents that share vocabulary land close together, so retrieval and
distance-metric experiments produce meaningful — not random — results.

Method: TF-IDF weighting over feature-hashed unigrams (the "hashing trick"),
projected into a fixed-width vector. No model download required.

NOTE: These are lexical, not semantic, embeddings. They cannot match
synonyms ("car" vs "automobile"). Prefer `EmbeddingGenerator` (MiniLM)
whenever the model is reachable.
"""

import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List

import numpy as np

TOKEN_PATTERN = re.compile(r"[a-z0-9]+")

# Function words carry no topical signal; without removing them, two unrelated
# documents can appear similar purely because both contain "the" and "before".
STOPWORDS = frozenset("""
a an and are as at be been being but by for from had has have he her his i if in
into is it its of on or she that the their them then there these they this to
was were what when where which who will with would you your our us we do does
did not no nor so than too very can could should may might must shall about
after again all also am any because before below between both down during each
few further here how more most other over own same some such only up out
""".split())

# Ordered longest-first so "generation" strips "ion" rather than "n".
SUFFIXES = (
    "ational", "ization", "iveness", "fulness", "ousness", "ability", "ibility",
    "ation", "ition", "ement", "ments", "ingly", "edly", "ness", "ment", "tion",
    "sion", "ally", "ings", "ized", "ised", "ing", "ers", "ies", "ive", "ion",
    "ial", "ous", "ant", "ent", "est", "ed", "es", "al", "er", "ly", "s", "e",
)

MIN_STEM_LENGTH = 4


def stem(token: str) -> str:
    """
    Strip a single common suffix from a token.

    A deliberately small Porter-style stemmer: it collapses inflections such as
    retrieve/retrieval/retrieved and generation/generates onto a shared stem so
    that lexically related documents actually match.
    """
    for suffix in SUFFIXES:
        if token.endswith(suffix) and len(token) - len(suffix) >= MIN_STEM_LENGTH:
            return token[: -len(suffix)]
    return token


class LexicalEmbeddingGenerator:
    """TF-IDF feature-hashing embeddings, API-compatible with EmbeddingGenerator."""

    def __init__(
        self,
        model_name: str = "lexical-tfidf-hash",
        embedding_dim: int = 384,
        normalize: bool = True,
    ):
        """
        Initialize the lexical embedder.

        Args:
            model_name: Identifier recorded in pipeline metadata
            embedding_dim: Width of the output vectors (matches MiniLM's 384)
            normalize: L2-normalize outputs. Set False to retain magnitude,
                which is required to show how Dot Product differs from Cosine.
        """
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.normalize = normalize
        self.document_frequency: Counter = Counter()
        self.num_documents = 0

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Lowercase, split, drop stopwords, and stem the remaining tokens."""
        return [
            stem(token)
            for token in TOKEN_PATTERN.findall(text.lower())
            if token not in STOPWORDS and len(token) > 1
        ]

    def _hash_token(self, token: str) -> tuple:
        """
        Map a token to a (bucket, sign) pair.

        The signed hashing trick lets colliding tokens cancel rather than
        always reinforce, which reduces collision bias.
        """
        digest = hashlib.md5(token.encode("utf-8")).digest()
        bucket = int.from_bytes(digest[:4], "big") % self.embedding_dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        return bucket, sign

    def fit(self, texts: List[str]) -> "LexicalEmbeddingGenerator":
        """
        Learn document frequencies from a corpus, enabling IDF weighting.

        Args:
            texts: Corpus to fit on

        Returns:
            self, so the call can be chained
        """
        self.document_frequency = Counter()
        self.num_documents = len(texts)

        for text in texts:
            for token in set(self.tokenize(text)):
                self.document_frequency[token] += 1

        return self

    def _idf(self, token: str) -> float:
        """Smoothed inverse document frequency for a token."""
        if self.num_documents == 0:
            return 1.0
        df = self.document_frequency.get(token, 0)
        return math.log((1 + self.num_documents) / (1 + df)) + 1.0

    def embed_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Embed a list of texts.

        Args:
            texts: Text strings to embed
            batch_size: Unused; kept for interface parity with EmbeddingGenerator

        Returns:
            NumPy array of shape (len(texts), embedding_dim)
        """
        if self.num_documents == 0:
            self.fit(texts)

        embeddings = np.zeros((len(texts), self.embedding_dim), dtype=np.float32)

        for row, text in enumerate(texts):
            tokens = self.tokenize(text)
            if not tokens:
                continue

            counts = Counter(tokens)
            max_count = max(counts.values())

            for token, count in counts.items():
                # Sublinear TF damps the effect of a token repeated many times
                tf = 0.5 + 0.5 * (count / max_count)
                bucket, sign = self._hash_token(token)
                embeddings[row, bucket] += sign * tf * self._idf(token)

            if self.normalize:
                norm = np.linalg.norm(embeddings[row])
                if norm > 0:
                    embeddings[row] /= norm

        return embeddings

    def embed_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """
        Embed document chunks, adding an 'embedding' field to each.

        Args:
            chunks: Chunk dictionaries containing a 'text' field

        Returns:
            The same chunks, each with an added 'embedding' list
        """
        texts = [chunk["text"] for chunk in chunks]
        self.fit(texts)
        embeddings = self.embed_texts(texts)

        for chunk, embedding in zip(chunks, embeddings):
            chunk["embedding"] = embedding.tolist()

        return chunks

    def save_embeddings(self, chunks: List[Dict], output_path: str):
        """
        Save embedded chunks to a JSONL file.

        Args:
            chunks: Chunks carrying embeddings
            output_path: Destination JSONL path
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w") as f:
            for chunk in chunks:
                f.write(json.dumps(chunk) + "\n")

        print(f"Saved {len(chunks)} embeddings to {output_path}")

    def get_model_info(self) -> dict:
        """Return details about this embedder for pipeline reporting."""
        return {
            "model_name": self.model_name,
            "embedding_dimension": self.embedding_dim,
            "model_size_mb": 0,
            "max_seq_length": None,
            "type": "lexical-tfidf-hashing",
            "normalized": self.normalize,
            "vocabulary_seen": len(self.document_frequency),
        }


if __name__ == "__main__":
    corpus = [
        "Retrieval-Augmented Generation grounds LLM answers in retrieved documents.",
        "RAG systems retrieve relevant chunks before the model generates a response.",
        "Chocolate chip cookies need butter, brown sugar, and vanilla extract.",
        "Preheat the oven and cream the butter with sugar before adding flour.",
    ]

    embedder = LexicalEmbeddingGenerator()
    vectors = embedder.embed_texts(corpus)

    print("LEXICAL EMBEDDER DEMO")
    print(f"  Shape: {vectors.shape}")
    print(f"  Info:  {embedder.get_model_info()}\n")

    print("Cosine similarity between texts (normalized -> dot product):")
    for i in range(len(corpus)):
        for j in range(i + 1, len(corpus)):
            sim = float(np.dot(vectors[i], vectors[j]))
            print(f"  {i} vs {j}: {sim:+.4f}   ({corpus[i][:38]!r:42} | {corpus[j][:38]!r})")
