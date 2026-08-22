"""
Chroma vector store for the Personal Assistant AI.

Wraps a persistent Chroma collection so embedded chunks can be indexed once
and queried repeatedly. The distance space (cosine / l2 / ip) is selected at
collection-creation time, which is what makes the metric comparison in
`distance_metrics.py` possible.
"""

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# Chroma's distance space is fixed when the collection is created.
SUPPORTED_SPACES = ("cosine", "l2", "ip")

SPACE_DESCRIPTIONS = {
    "cosine": "Cosine distance (1 - cosine similarity); ignores vector magnitude",
    "l2": "Squared Euclidean distance; sensitive to both direction and magnitude",
    "ip": "Inner product distance (1 - dot product); rewards large magnitudes",
}


class VectorStore:
    """Persistent Chroma-backed store for embedded document chunks."""

    def __init__(
        self,
        collection_name: str = "personal_assistant",
        persist_directory: str = "data/vector_store",
        space: str = "cosine",
        reset: bool = False,
    ):
        """
        Open (or create) a Chroma collection.

        Args:
            collection_name: Name of the Chroma collection
            persist_directory: On-disk location for the database
            space: Distance metric - one of 'cosine', 'l2', 'ip'
            reset: Delete an existing collection of the same name first.
                Required when changing `space`, since the metric is baked
                into the collection's index at creation time.

        Raises:
            ValueError: If `space` is not a supported Chroma distance space
            ImportError: If chromadb is not installed
        """
        if space not in SUPPORTED_SPACES:
            raise ValueError(
                f"Unsupported space {space!r}. Expected one of {SUPPORTED_SPACES}."
            )

        try:
            import chromadb
        except ImportError:
            raise ImportError("Please install chromadb: pip install chromadb")

        self.collection_name = collection_name
        self.persist_directory = str(persist_directory)
        self.space = space

        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=self.persist_directory)

        if reset:
            try:
                self.client.delete_collection(collection_name)
            except Exception:
                # Nothing to delete on a first run; not an error.
                pass

        # embedding_function=None: embeddings are supplied by our own pipeline,
        # so Chroma must never try to download a model of its own.
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": space},
            embedding_function=None,
        )

    def add_chunks(self, chunks: List[Dict], batch_size: int = 256) -> int:
        """
        Index chunks that already carry an 'embedding' field.

        Args:
            chunks: Chunk dictionaries from the data pipeline
            batch_size: Number of chunks sent to Chroma per call

        Returns:
            Number of chunks indexed

        Raises:
            KeyError: If a chunk is missing its embedding
        """
        if not chunks:
            return 0

        ids, embeddings, documents, metadatas = [], [], [], []

        for position, chunk in enumerate(chunks):
            if "embedding" not in chunk:
                raise KeyError(
                    f"Chunk {position} has no 'embedding'. Run the pipeline first."
                )

            doc_id = chunk.get("doc_id", "unknown")
            chunk_index = chunk.get("chunk_index", position)

            ids.append(f"{doc_id}::chunk_{chunk_index}")
            embeddings.append(list(chunk["embedding"]))
            documents.append(chunk["text"])
            metadatas.append(
                {
                    "doc_id": doc_id,
                    "chunk_index": int(chunk_index),
                    "token_count": int(chunk.get("token_count", 0)),
                }
            )

        for start in range(0, len(ids), batch_size):
            end = start + batch_size
            self.collection.add(
                ids=ids[start:end],
                embeddings=embeddings[start:end],
                documents=documents[start:end],
                metadatas=metadatas[start:end],
            )

        return len(ids)

    def query(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        where: Optional[Dict] = None,
    ) -> List[Dict]:
        """
        Retrieve the nearest chunks to a query embedding.

        Args:
            query_embedding: Embedded query vector
            top_k: Number of results to return
            where: Optional Chroma metadata filter, e.g. {"doc_id": "notes_week1"}

        Returns:
            Ranked result dictionaries with text, metadata, distance and score
        """
        if hasattr(query_embedding, "tolist"):
            query_embedding = query_embedding.tolist()

        response = self.collection.query(
            query_embeddings=[list(query_embedding)],
            n_results=min(top_k, max(self.count(), 1)),
            where=where,
        )

        results = []
        documents = response.get("documents") or [[]]
        for rank, document in enumerate(documents[0]):
            distance = response["distances"][0][rank]
            metadata = response["metadatas"][0][rank]

            results.append(
                {
                    "rank": rank + 1,
                    "id": response["ids"][0][rank],
                    "text": document,
                    "doc_id": metadata.get("doc_id", "unknown"),
                    "chunk_index": metadata.get("chunk_index", -1),
                    "distance": float(distance),
                    "score": self.distance_to_score(float(distance)),
                }
            )

        return results

    def distance_to_score(self, distance: float) -> float:
        """
        Convert a Chroma distance into a similarity score (higher is better).

        Chroma reports 1 - cosine_similarity for 'cosine', 1 - dot for 'ip',
        and *squared* Euclidean for 'l2'. Only the first two invert cleanly
        into a bounded similarity.

        Args:
            distance: Raw distance returned by Chroma

        Returns:
            Similarity score; for 'l2' this is a monotonic 1/(1+d) transform
        """
        if self.space in ("cosine", "ip"):
            return 1.0 - distance
        return 1.0 / (1.0 + distance)

    def count(self) -> int:
        """Return the number of chunks currently indexed."""
        return self.collection.count()

    def get_stats(self) -> dict:
        """Return a summary of the collection for reporting."""
        return {
            "collection_name": self.collection_name,
            "persist_directory": self.persist_directory,
            "space": self.space,
            "space_description": SPACE_DESCRIPTIONS[self.space],
            "num_chunks": self.count(),
        }


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from embeddings.embedding_generator_lexical import LexicalEmbeddingGenerator

    demo_chunks = [
        {"text": "The capstone project is due on November 14, 2026.", "doc_id": "schedule"},
        {"text": "Checkpoint 2 covers prompt architecture and vector indexing.", "doc_id": "schedule"},
        {"text": "Cream the butter and sugar, then fold in the chocolate chips.", "doc_id": "recipe"},
        {"text": "Chroma stores embeddings and supports similarity search.", "doc_id": "guide"},
    ]
    for index, chunk in enumerate(demo_chunks):
        chunk["chunk_index"] = index
        chunk["token_count"] = len(chunk["text"].split())

    embedder = LexicalEmbeddingGenerator()
    embedder.embed_chunks(demo_chunks)

    store = VectorStore(
        collection_name="demo",
        persist_directory="data/vector_store_demo",
        space="cosine",
        reset=True,
    )
    indexed = store.add_chunks(demo_chunks)
    print(f"Indexed {indexed} chunks")
    print(f"Stats: {store.get_stats()}\n")

    question = "When is the capstone project deadline?"
    hits = store.query(embedder.embed_texts([question])[0], top_k=3)

    print(f"QUERY: {question}")
    for hit in hits:
        print(f"  #{hit['rank']} [{hit['doc_id']}] score={hit['score']:.4f} :: {hit['text']}")
