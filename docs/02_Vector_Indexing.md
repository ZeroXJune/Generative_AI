# Checkpoint 2 — Vector Indexing & Distance Metric Comparison

**Project**: Personal Assistant AI — Generative RAG Capstone
**Course**: IS Professional Elective #4 — Generative AI Systems
**Instructor**: Jessie A. Melendres

**Reproduce**: `python src/experiments/metric_comparison.py`
**Raw results**: [`data/processed/metric_comparison.json`](../data/processed/metric_comparison.json)

---

## 1. Vector Database Selection

**Chosen: Chroma** (`src/retrieval/vector_store.py`).

| Database | Scale | Setup | Verdict for this project |
|---|---|---|---|
| **Chroma** | Millions | `pip install chromadb` | **Chosen** — persistent, zero infrastructure, all three metrics built in |
| FAISS | Billions | Library only | No metadata store; citations would need a second database |
| Pinecone | Unlimited | Cloud, paid | Cost and network dependency not justified at 26 documents |
| Weaviate / Milvus | Billions | Docker / Kubernetes | Operational overhead far exceeds the corpus size |

Chroma persists to disk via `PersistentClient`, stores chunk metadata alongside
the vectors (so `[doc_id]` citations come straight from the index), and lets the
distance metric be selected per collection — which is what makes the comparison
below possible on the real index rather than only in NumPy.

The store is constructed with `embedding_function=None` because embeddings are
produced by our own pipeline; Chroma must never silently download a model of
its own and embed with something different from the index.

---

## 2. Experimental Setup

| Parameter | Value |
|---|---|
| Corpus | 26 documents → **150 chunks** |
| Chunking | 120 tokens, 30 overlap (tuned — §5) |
| Embedding dimension | 384 |
| Evaluation queries | 8, each with a known ground-truth document |
| Retrieval depth | k = 5 |

**Metrics measured**: Hit@1 and Hit@5 (was the correct document retrieved),
MRR (mean reciprocal rank), mean token length of retrieved chunks (length
bias), and milliseconds per query. Retrieval figures are deterministic and
reproduce exactly; per-query timings vary by a few percent between runs, so
treat their *ratios*, not their absolute values, as the result.

> **Embedding caveat.** The intended model is `all-MiniLM-L6-v2`, but
> `huggingface.co` is blocked by network policy in the environment where these
> numbers were produced, so the corpus was embedded with the project's offline
> fallback — deterministic TF-IDF over feature-hashed tokens
> (`src/embeddings/embedding_generator_lexical.py`). These are **lexical, not
> semantic** vectors: they cannot match synonyms, which caps absolute retrieval
> quality. The *geometric* findings below (§3, §4) follow from vector algebra
> and hold for any embedding; the absolute Hit/MRR values will improve once the
> real model is reachable. `build_index.py` uses MiniLM automatically whenever
> it can be downloaded.

---

## 3. Results — L2-Normalized Embeddings

All vectors unit length (`min = max = mean = 1.000`).

| Metric | Hit@1 | Hit@5 | MRR | Avg chunk tokens | ms/query |
|---|---|---|---|---|---|
| Cosine | 0.625 | 0.875 | 0.692 | 107.2 | 0.0351 |
| Euclidean | 0.625 | 0.875 | 0.692 | 107.2 | 0.1367 |
| Dot product | 0.625 | 0.875 | 0.692 | 107.2 | **0.0045** |

**Ranking agreement — all three metrics produced byte-identical rankings:**

| Pair | Overlap@5 | Identical order |
|---|---|---|
| Cosine vs Euclidean | 1.000 | 1.000 |
| Cosine vs Dot product | 1.000 | 1.000 |
| Euclidean vs Dot product | 1.000 | 1.000 |

### Why this happens

This is not a coincidence — it is forced by the algebra. For unit vectors
‖a‖ = ‖b‖ = 1:

```
dot(a, b)       = cos(a, b)                     … identical
euclidean(a,b)² = ‖a‖² + ‖b‖² − 2·dot(a,b)
                = 2 − 2·cos(a, b)               … strictly decreasing in cosine
```

Dot product **equals** cosine, and Euclidean distance is a strictly decreasing
function of it. A strictly monotonic transform cannot reorder a ranking, so all
three retrieve the same chunks in the same order. **On normalized embeddings the
choice of metric affects only speed, not quality.**

Dot product is ~8× faster than cosine here (no norm computation) and ~30×
faster than Euclidean (no subtraction across the full matrix).

---

## 4. Results — Unnormalized Embeddings

Vector norms now vary widely: `min = 4.729, max = 28.730, mean = 18.712`.

| Metric | Hit@1 | Hit@5 | MRR | Avg chunk tokens | ms/query |
|---|---|---|---|---|---|
| Cosine | 0.625 | 0.875 | 0.692 | 107.2 | 0.0418 |
| Euclidean | **0.125** | **0.125** | **0.125** | **22.8** | 0.0545 |
| Dot product | 0.625 | 0.750 | 0.688 | 114.1 | **0.0046** |

| Pair | Overlap@5 | Identical order |
|---|---|---|
| Cosine vs Euclidean | 0.025 | 0.000 |
| Cosine vs Dot product | 0.850 | 0.000 |
| Euclidean vs Dot product | 0.000 | 0.000 |

The agreement seen in §3 collapses completely.

### 4.1 Euclidean fails badly (MRR 0.692 → 0.125)

Euclidean distance is dominated by the *magnitude* difference between vectors,
and in TF-IDF space magnitude tracks document length. A short chunk sits near
the origin, close to almost everything. The mean retrieved chunk drops from
**107 tokens to 22.8** — Euclidean is no longer retrieving relevant chunks, it
is retrieving *short* ones. Hit@5 falls to 0.125: one correct answer in eight.

### 4.2 Dot product acquires the opposite bias

Dot product grows with magnitude, so it favours **long** chunks: mean retrieved
length rises to 114.1 tokens and Hit@5 slips from 0.875 to 0.750. The bias is
milder than Euclidean's but points the other way — long chunks accumulate more
term weight and so score higher regardless of relevance.

### 4.3 Cosine is unaffected

Cosine's results are **identical** across §3 and §4, because dividing by
‖a‖·‖b‖ normalizes internally. Cosine is the only metric of the three that is
robust to whether normalization happened upstream.

---

## 5. Chunk Size Tuning

Checkpoint 1 used 512-token chunks for the embedding pipeline. That is too
coarse for retrieval: a single chunk can swallow most of a short note and drag
unrelated text into the prompt. Measured with cosine at k = 5:

| Chunk / overlap | Chunks | Hit@1 | Hit@5 | MRR |
|---|---|---|---|---|
| 512 / 100 | 44 | 0.625 | 0.750 | 0.688 |
| 256 / 60 | 81 | 0.500 | 0.750 | 0.588 |
| 160 / 40 | 114 | 0.625 | 0.750 | 0.688 |
| **120 / 30** | **150** | **0.625** | **0.875** | **0.692** |
| 80 / 20 | 219 | 0.500 | 0.750 | 0.604 |

**120 / 30 was adopted** (`RETRIEVAL_CHUNK_SIZE` in `src/build_index.py`). With
only 8 evaluation queries these differences are one or two queries wide and
should not be over-read; 120/30 was chosen because it is best on both measures
*and* yields context tight enough to cite precisely.

---

## 6. Chroma Cross-Check

The NumPy implementations are a reference; the production index is Chroma. One
collection was built per distance space and queried with the same 8 queries.

| NumPy metric | Chroma space | Exact order match | Score-sequence match | Hit@5 |
|---|---|---|---|---|
| Cosine | `cosine` | 0.875 | **1.000** | 0.875 |
| Euclidean | `l2` | 0.875 | **1.000** | 0.875 |
| Dot product | `ip` | 0.875 | **1.000** | 0.875 |

Exact order matched on 7 of 8 queries. The eighth was **not** an approximation
error: two chunks scored *identically* (0.174032 — `notes_week4` and
`notes_week7`), and NumPy's `argsort` and Chroma's index break ties in
different orders. Comparing the **score sequences** instead gives a perfect
1.000 match, confirming Chroma agrees exactly with brute-force search on
relevance. Ordering among equally-scoring chunks is arbitrary in both.

Two Chroma conventions are worth recording, since both are easy to misread:
- `l2` returns **squared** Euclidean distance, not Euclidean distance.
- `ip` returns `1 − dot(a, b)`, so it is a *distance*, not a similarity.

`VectorStore.distance_to_score()` converts each back to a
higher-is-better score.

---

## 7. Decision

**Cosine similarity, on L2-normalized embeddings, in a Chroma collection
created with `hnsw:space="cosine"`.**

Reasoning:

1. **On normalized vectors all three metrics rank identically** (§3), so the
   decision cannot be made on retrieval quality alone — it must be made on
   robustness and cost.
2. **Cosine is the only metric robust to a normalization mistake** (§4.3). If a
   future embedding model, a migration, or a bug ever emits unnormalized
   vectors, cosine degrades not at all, dot product degrades slightly, and
   Euclidean collapses to MRR 0.125. That safety margin is worth far more than
   the microseconds dot product saves on a 150-chunk corpus.
3. **Dot product is the right optimisation later, not now.** At 0.0045 ms/query
   it is the fastest of the three, and it is *exactly equivalent* to cosine as
   long as vectors stay normalized. If the corpus grows to where that matters,
   switching is a one-line change — but it makes the normalization invariant
   load-bearing, and that trade is not worth making at this scale.
4. **Euclidean is rejected** for text retrieval. It is only safe on normalized
   vectors, where it is merely a slower way of computing cosine.

This matches the recommendation already recorded in the project's own notes
(`data/raw/guide_vector_databases.txt`: *"use cosine similarity with Chroma for
development"*) — but it is now supported by measurements on this corpus rather
than adopted on authority.

---

## 8. Reflection

The result that changed how I think about this system is that **on normalized
embeddings, the metric choice does not matter at all** — the three metrics the
checkpoint asks us to compare are provably the same ranking function. The
interesting question is therefore not "which metric retrieves best?" but "which
metric fails most gracefully when an assumption breaks?"

That reframing is what produced the decision in §7. Cosine was not chosen
because it scored highest — it tied. It was chosen because it is the only one of
the three whose quality does not depend on an invariant maintained somewhere
else in the codebase. The §4 experiment exists precisely to measure the cost of
that invariant failing, and the answer (MRR 0.692 → 0.125 for Euclidean) is
large enough to settle the question.

The second lesson was methodological: the initial Chroma cross-check reported
0.875 agreement, which looked like ANN approximation error. It was actually tie
ordering. Had I written that up without checking, the report would have
contained a plausible, confidently-worded, entirely wrong explanation — the
same failure mode as the hallucination this project's prompts are designed to
prevent.
