"""
Checkpoint 2 experiment: which distance metric should the index use?

Runs the project's own corpus through all three candidate metrics and reports
retrieval quality, ranking agreement, length bias, and speed. Results are
written to data/processed/metric_comparison.json and printed as Markdown
tables for the write-up.

Run:  python src/experiments/metric_comparison.py
"""

import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from build_index import RETRIEVAL_CHUNK_OVERLAP, RETRIEVAL_CHUNK_SIZE
from embeddings.embedding_generator_lexical import LexicalEmbeddingGenerator
from preprocessing.chunker import DocumentChunker
from preprocessing.text_cleaner import TextCleaner
from retrieval.distance_metrics import (
    CHROMA_SPACE_FOR_METRIC,
    METRICS,
    benchmark_metrics,
    compare_metrics,
    rank_by_metric,
)
from retrieval.vector_store import VectorStore

# Ground truth: each query and the document that should be retrieved for it.
EVALUATION_QUERIES: List[Dict[str, str]] = [
    {"query": "What is approximate nearest neighbour search?", "expected": "guide_vector_databases"},
    {"query": "How does retrieval augmented generation ground its answers?", "expected": "notes_rag_concepts"},
    {"query": "How do I generate an SSH key pair for a remote server?", "expected": "howto_ssh_setup"},
    {"query": "Which GPU is recommended for LLM inference?", "expected": "datasheet_gpu_requirements"},
    {"query": "How much butter is needed for chocolate chip cookies?", "expected": "recipe_chocolate_chip_cookies"},
    {"query": "What time does the Generative AI class meet each week?", "expected": "schedule_semester_2026"},
    {"query": "Explain self attention and the role of queries keys and values", "expected": "notes_llm_transformers_lesson3"},
    {"query": "What is RLHF and how is the reward model trained?", "expected": "notes_llm_transformers_lesson3"},
]

TOP_K = 5


def load_corpus(
    raw_dir: str = "data/raw",
    chunk_size: int = RETRIEVAL_CHUNK_SIZE,
    overlap: int = RETRIEVAL_CHUNK_OVERLAP,
) -> List[Dict]:
    """
    Load, clean, and chunk the raw corpus.

    Args:
        raw_dir: Directory of .txt source documents
        chunk_size: Chunk width in whitespace tokens
        overlap: Token overlap between neighbouring chunks

    Returns:
        Chunk dictionaries carrying text and metadata

    Raises:
        FileNotFoundError: If the raw directory contains no .txt files
    """
    cleaner = TextCleaner()
    chunker = DocumentChunker(chunk_size=chunk_size, overlap=overlap)

    paths = sorted(Path(raw_dir).glob("*.txt"))
    if not paths:
        raise FileNotFoundError(f"No .txt documents found in {raw_dir}")

    documents = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        documents.append((cleaner.clean(text), path.stem))

    return chunker.chunk_documents(documents)


def retrieval_quality(
    query_vectors: np.ndarray,
    corpus_vectors: np.ndarray,
    corpus_meta: List[Dict],
    top_k: int = TOP_K,
) -> Dict[str, Dict[str, float]]:
    """
    Score each metric against the ground-truth query set.

    Args:
        query_vectors: Embedded evaluation queries
        corpus_vectors: Embedded corpus chunks
        corpus_meta: Chunk metadata, providing 'doc_id'
        top_k: Retrieval depth

    Returns:
        Per-metric hit-rate@1, hit-rate@k, and mean reciprocal rank
    """
    results = {}

    for metric in METRICS:
        hits_at_1 = 0
        hits_at_k = 0
        reciprocal_ranks = []

        for position, case in enumerate(EVALUATION_QUERIES):
            ranking = rank_by_metric(query_vectors[position], corpus_vectors, metric, top_k)
            retrieved_docs = [corpus_meta[index]["doc_id"] for index in ranking]
            expected = case["expected"]

            if retrieved_docs and retrieved_docs[0] == expected:
                hits_at_1 += 1
            if expected in retrieved_docs:
                hits_at_k += 1
                reciprocal_ranks.append(1.0 / (retrieved_docs.index(expected) + 1))
            else:
                reciprocal_ranks.append(0.0)

        total = len(EVALUATION_QUERIES)
        results[metric] = {
            "hit_rate@1": hits_at_1 / total,
            f"hit_rate@{top_k}": hits_at_k / total,
            "mrr": float(np.mean(reciprocal_ranks)),
        }

    return results


def verify_against_chroma(
    chunks: List[Dict],
    query_vectors: np.ndarray,
    corpus_meta: List[Dict],
    top_k: int = TOP_K,
) -> Dict[str, Dict]:
    """
    Cross-check the NumPy rankings against real Chroma collections.

    Builds one collection per distance space so the comparison reflects the
    production index, not just the reference implementation.

    Args:
        chunks: Chunks carrying embeddings
        query_vectors: Embedded evaluation queries
        corpus_meta: Chunk metadata
        top_k: Retrieval depth

    Returns:
        Per-metric agreement with NumPy plus the Chroma hit rate
    """
    corpus_vectors = np.array([chunk["embedding"] for chunk in chunks], dtype=np.float32)
    verification = {}

    for metric, space in CHROMA_SPACE_FOR_METRIC.items():
        store = VectorStore(
            collection_name=f"metric_{space}",
            persist_directory="data/vector_store_experiments",
            space=space,
            reset=True,
        )
        store.add_chunks(chunks)

        # Chroma ids are built from doc_id + chunk_index, so they map back to
        # the row each hit came from.
        id_to_row = {
            f"{chunk.get('doc_id')}::chunk_{chunk.get('chunk_index', row)}": row
            for row, chunk in enumerate(chunks)
        }
        score_fn, higher_is_better = METRICS[metric]

        exact_matches = []
        score_matches = []
        chroma_hits = 0

        for position, case in enumerate(EVALUATION_QUERIES):
            hits = store.query(query_vectors[position], top_k=top_k)
            chroma_docs = [hit["doc_id"] for hit in hits]
            numpy_rows = rank_by_metric(
                query_vectors[position], corpus_vectors, metric, top_k
            )
            numpy_docs = [corpus_meta[row]["doc_id"] for row in numpy_rows]

            exact_matches.append(1.0 if chroma_docs == numpy_docs else 0.0)

            # Order among equally-scoring chunks is arbitrary and differs
            # between argsort and Chroma's index. Comparing the score
            # sequences instead shows whether the two agree on relevance.
            all_scores = score_fn(query_vectors[position], corpus_vectors)
            chroma_rows = [id_to_row[hit["id"]] for hit in hits]
            score_matches.append(
                1.0
                if np.allclose(all_scores[chroma_rows], all_scores[numpy_rows], atol=1e-6)
                else 0.0
            )

            if case["expected"] in chroma_docs:
                chroma_hits += 1

        verification[metric] = {
            "chroma_space": space,
            "exact_order_match_with_numpy": float(np.mean(exact_matches)),
            "score_sequence_match_with_numpy": float(np.mean(score_matches)),
            f"chroma_hit_rate@{top_k}": chroma_hits / len(EVALUATION_QUERIES),
            "indexed_chunks": store.count(),
        }

    return verification


def markdown_table(headers: List[str], rows: List[List[str]]) -> str:
    """Render a Markdown table from headers and pre-formatted rows."""
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def run_experiment() -> Dict:
    """
    Execute the full metric comparison and persist the results.

    Returns:
        The complete results dictionary
    """
    print("=" * 68)
    print("CHECKPOINT 2 - DISTANCE METRIC COMPARISON")
    print("=" * 68)

    chunks = load_corpus()
    corpus_meta = [
        {"doc_id": chunk["doc_id"], "token_count": chunk["token_count"]} for chunk in chunks
    ]
    queries = [case["query"] for case in EVALUATION_QUERIES]

    print(f"\nCorpus: {len(set(c['doc_id'] for c in chunks))} documents -> {len(chunks)} chunks")
    print(f"Queries: {len(queries)} with known ground-truth documents\n")

    results = {
        "corpus": {
            "num_documents": len(set(c["doc_id"] for c in chunks)),
            "num_chunks": len(chunks),
            "top_k": TOP_K,
        },
        "variants": {},
    }

    # The same corpus is embedded twice: once L2-normalized, once raw. The
    # contrast between the two is the whole point of the experiment.
    for variant, normalize in (("normalized", True), ("unnormalized", False)):
        embedder = LexicalEmbeddingGenerator(normalize=normalize)
        texts = [chunk["text"] for chunk in chunks]
        embedder.fit(texts)

        corpus_vectors = embedder.embed_texts(texts)
        query_vectors = embedder.embed_texts(queries)

        norms = np.linalg.norm(corpus_vectors, axis=1)
        quality = retrieval_quality(query_vectors, corpus_vectors, corpus_meta)
        agreement = compare_metrics(queries, query_vectors, corpus_vectors, corpus_meta, TOP_K)
        timings = benchmark_metrics(query_vectors, corpus_vectors)

        results["variants"][variant] = {
            "vector_norms": {
                "min": float(norms.min()),
                "max": float(norms.max()),
                "mean": float(norms.mean()),
            },
            "quality": quality,
            "agreement": {
                "mean_overlap": agreement["mean_overlap"],
                "identical_order_rate": agreement["identical_order_rate"],
            },
            "mean_retrieved_tokens": agreement["mean_retrieved_tokens"],
            "timing_ms_per_query": timings,
        }

        print(f"--- {variant.upper()} EMBEDDINGS " + "-" * (48 - len(variant)))
        print(
            f"Vector norms: min={norms.min():.3f} max={norms.max():.3f} mean={norms.mean():.3f}\n"
        )
        print(
            markdown_table(
                ["Metric", "Hit@1", f"Hit@{TOP_K}", "MRR", "Avg chunk tokens", "ms/query"],
                [
                    [
                        metric,
                        f"{quality[metric]['hit_rate@1']:.3f}",
                        f"{quality[metric][f'hit_rate@{TOP_K}']:.3f}",
                        f"{quality[metric]['mrr']:.3f}",
                        f"{agreement['mean_retrieved_tokens'][metric]:.1f}",
                        f"{timings[metric]:.4f}",
                    ]
                    for metric in METRICS
                ],
            )
        )
        print("\nRanking agreement (overlap@k / identical order):")
        for pair, overlap in agreement["mean_overlap"].items():
            print(
                f"  {pair:<28} overlap={overlap:.3f}  "
                f"identical_order={agreement['identical_order_rate'][pair]:.3f}"
            )
        print()

    # Cross-check against real Chroma collections using normalized vectors,
    # which is the configuration the production index actually uses.
    print("--- CHROMA CROSS-CHECK " + "-" * 45)
    embedder = LexicalEmbeddingGenerator(normalize=True)
    embedder.embed_chunks(chunks)
    query_vectors = embedder.embed_texts(queries)

    results["chroma_verification"] = verify_against_chroma(chunks, query_vectors, corpus_meta)
    for metric, record in results["chroma_verification"].items():
        print(
            f"  {metric:<12} space={record['chroma_space']:<7} "
            f"exact_order={record['exact_order_match_with_numpy']:.3f}  "
            f"score_seq={record['score_sequence_match_with_numpy']:.3f}  "
            f"hit@{TOP_K}={record[f'chroma_hit_rate@{TOP_K}']:.3f}"
        )

    output_path = Path("data/processed/metric_comparison.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults written to {output_path}")
    print("=" * 68)

    return results


if __name__ == "__main__":
    run_experiment()
