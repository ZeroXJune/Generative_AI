"""
Distance metric comparison for vector retrieval.

Checkpoint 2 requires justifying the similarity metric used by the vector
index. This module implements the three candidate metrics directly in NumPy
and provides an experiment harness that measures how their rankings differ
on the project's own corpus.

Metrics
-------
Cosine similarity : angle between vectors; magnitude-invariant
Euclidean (L2)    : straight-line distance; sensitive to magnitude
Dot product (IP)  : projection; grows with the magnitude of either vector
"""

from typing import Callable, Dict, List, Sequence

import numpy as np


def cosine_similarity(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """
    Cosine similarity between a query vector and each row of a matrix.

    Args:
        query: Vector of shape (dim,)
        matrix: Matrix of shape (n, dim)

    Returns:
        Similarities of shape (n,), in [-1, 1]; higher is more similar
    """
    query_norm = np.linalg.norm(query)
    matrix_norms = np.linalg.norm(matrix, axis=1)
    denominator = query_norm * matrix_norms
    # Guard against zero-length vectors (empty or all-stopword chunks)
    denominator = np.where(denominator == 0, 1e-10, denominator)
    return (matrix @ query) / denominator


def euclidean_distance(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """
    Euclidean (L2) distance between a query vector and each row of a matrix.

    Args:
        query: Vector of shape (dim,)
        matrix: Matrix of shape (n, dim)

    Returns:
        Distances of shape (n,); LOWER is more similar
    """
    return np.linalg.norm(matrix - query, axis=1)


def dot_product(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """
    Inner product between a query vector and each row of a matrix.

    Args:
        query: Vector of shape (dim,)
        matrix: Matrix of shape (n, dim)

    Returns:
        Scores of shape (n,); higher is more similar
    """
    return matrix @ query


# Each metric maps to (function, higher_is_better)
METRICS: Dict[str, tuple] = {
    "cosine": (cosine_similarity, True),
    "euclidean": (euclidean_distance, False),
    "dot_product": (dot_product, True),
}

# How each NumPy metric maps onto a Chroma collection space
CHROMA_SPACE_FOR_METRIC = {
    "cosine": "cosine",
    "euclidean": "l2",
    "dot_product": "ip",
}


def rank_by_metric(
    query: np.ndarray, matrix: np.ndarray, metric: str, top_k: int = 5
) -> List[int]:
    """
    Rank corpus rows against a query under a named metric.

    Args:
        query: Query vector of shape (dim,)
        matrix: Corpus matrix of shape (n, dim)
        metric: One of 'cosine', 'euclidean', 'dot_product'
        top_k: Number of indices to return

    Returns:
        Row indices ordered best-first

    Raises:
        ValueError: If the metric name is unknown
    """
    if metric not in METRICS:
        raise ValueError(f"Unknown metric {metric!r}. Expected one of {list(METRICS)}.")

    score_fn, higher_is_better = METRICS[metric]
    scores = score_fn(query, matrix)
    order = np.argsort(-scores if higher_is_better else scores)
    return order[:top_k].tolist()


def rank_agreement(ranking_a: Sequence[int], ranking_b: Sequence[int]) -> float:
    """
    Fraction of shared items between two rankings (overlap@k).

    Args:
        ranking_a: First ranking of row indices
        ranking_b: Second ranking of row indices

    Returns:
        Overlap in [0, 1]; 1.0 means both rankings selected the same set
    """
    if not ranking_a:
        return 0.0
    return len(set(ranking_a) & set(ranking_b)) / len(ranking_a)


def identical_order(ranking_a: Sequence[int], ranking_b: Sequence[int]) -> bool:
    """Return True when two rankings list the same items in the same order."""
    return list(ranking_a) == list(ranking_b)


def compare_metrics(
    queries: List[str],
    query_vectors: np.ndarray,
    corpus_vectors: np.ndarray,
    corpus_meta: List[Dict],
    top_k: int = 5,
) -> Dict:
    """
    Compare all three metrics across a set of queries.

    Args:
        queries: Query strings, for reporting
        query_vectors: Embedded queries of shape (num_queries, dim)
        corpus_vectors: Embedded corpus of shape (n, dim)
        corpus_meta: Per-row metadata; 'doc_id' and 'token_count' are used
        top_k: Depth at which rankings are compared

    Returns:
        Structured results: per-query rankings, pairwise agreement, and the
        mean length of the chunks each metric retrieves
    """
    pairs = [("cosine", "euclidean"), ("cosine", "dot_product"), ("euclidean", "dot_product")]

    per_query = []
    agreement_totals = {f"{a}_vs_{b}": [] for a, b in pairs}
    order_matches = {f"{a}_vs_{b}": 0 for a, b in pairs}
    retrieved_lengths = {metric: [] for metric in METRICS}

    for position, query_text in enumerate(queries):
        query_vector = query_vectors[position]
        rankings = {
            metric: rank_by_metric(query_vector, corpus_vectors, metric, top_k)
            for metric in METRICS
        }

        for metric, ranking in rankings.items():
            retrieved_lengths[metric].extend(
                corpus_meta[index].get("token_count", 0) for index in ranking
            )

        for metric_a, metric_b in pairs:
            key = f"{metric_a}_vs_{metric_b}"
            agreement_totals[key].append(
                rank_agreement(rankings[metric_a], rankings[metric_b])
            )
            if identical_order(rankings[metric_a], rankings[metric_b]):
                order_matches[key] += 1

        per_query.append(
            {
                "query": query_text,
                "rankings": rankings,
                "top_doc": {
                    metric: corpus_meta[ranking[0]].get("doc_id", "unknown")
                    for metric, ranking in rankings.items()
                    if ranking
                },
            }
        )

    return {
        "top_k": top_k,
        "num_queries": len(queries),
        "per_query": per_query,
        "mean_overlap": {
            key: float(np.mean(values)) for key, values in agreement_totals.items()
        },
        "identical_order_rate": {
            key: matches / len(queries) for key, matches in order_matches.items()
        },
        "mean_retrieved_tokens": {
            metric: float(np.mean(lengths)) if lengths else 0.0
            for metric, lengths in retrieved_lengths.items()
        },
    }


def benchmark_metrics(
    query_vectors: np.ndarray, corpus_vectors: np.ndarray, repeats: int = 50
) -> Dict[str, float]:
    """
    Time each metric over repeated full-corpus scans.

    Args:
        query_vectors: Embedded queries of shape (num_queries, dim)
        corpus_vectors: Embedded corpus of shape (n, dim)
        repeats: Number of passes over all queries

    Returns:
        Mean milliseconds per query for each metric
    """
    import time

    timings = {}
    for metric, (score_fn, _) in METRICS.items():
        start = time.perf_counter()
        for _ in range(repeats):
            for query_vector in query_vectors:
                score_fn(query_vector, corpus_vectors)
        elapsed = time.perf_counter() - start
        total_calls = repeats * len(query_vectors)
        timings[metric] = (elapsed / total_calls) * 1000

    return timings
