"""Vector indexing and retrieval for the Personal Assistant AI."""

from .distance_metrics import (
    METRICS,
    compare_metrics,
    cosine_similarity,
    dot_product,
    euclidean_distance,
    rank_by_metric,
)
from .retriever import RAGRetriever
from .vector_store import VectorStore

__all__ = [
    "METRICS",
    "RAGRetriever",
    "VectorStore",
    "compare_metrics",
    "cosine_similarity",
    "dot_product",
    "euclidean_distance",
    "rank_by_metric",
]
