# Checkpoint 3 — RAG Orchestration

**Project**: Personal Assistant AI — Generative RAG Capstone
**Course**: IS Professional Elective #4 — Generative AI Systems
**Instructor**: Jessie A. Melendres
**Due**: October 17, 2026

**Reproduce**: `python src/checkpoint3_demo.py`

---

## 1. Deliverables

| Requirement | Implementation |
|---|---|
| Automated ingestion pipeline | `src/ingestion/pipeline.py` |
| RAG application (LangChain) | `src/integrations/langchain_adapters.py`, `src/rag_app.py` |
| Conversational memory | `src/memory/conversation.py` |
| Live demo, 5+ queries | `src/checkpoint3_demo.py` — 7 queries |

---

## 2. Automated Ingestion

Checkpoint 2 rebuilt the whole index on every run. That is acceptable at 26
documents and wasteful beyond it. The pipeline now tracks state and processes
only what changed.

**Change detection** is a SHA-256 of each document's contents, recorded in
`data/processed/ingestion_manifest.json`. Every run classifies each document as
**new**, **modified**, **unchanged**, or **deleted**.

| Run | Action | Result |
|---|---|---|
| 1 | First ingestion | 26 new → 204 chunks written |
| 2 | Nothing changed | 26 unchanged, 0 embedded |
| 3 | One document edited | 1 modified: old chunks removed, new written |
| 4 | Document added | 1 new, 25 unchanged |
| 5 | Document deleted | Its chunks purged from the index |

Modified documents have their old chunks **deleted before** the new ones are
added (`VectorStore.delete_document`). Without that, an edited document would
be served from both its old and new text at once.

### The manifest is not trusted on its own

An early version classified purely from the manifest, and it produced a silent
failure: creating the application with `reset=True` wiped the collection, but
the manifest still claimed all 26 documents were ingested. Every document was
therefore marked "unchanged", nothing was written, and **the pipeline reported
success over an empty index**.

The fix is to reconcile against the live index as well:

```python
if doc_id not in manifest or doc_id not in indexed:
    report.new.append(doc_id)      # unknown, OR known but missing from the index
```

The manifest records what *was ingested*; only the store knows what it *still
contains*. Trusting the former alone is how a pipeline reports success while
doing nothing. This makes ingestion self-healing against a reset, a manual
deletion, or a corrupt manifest.

### Chunking

Ingestion uses LangChain's `RecursiveCharacterTextSplitter`, which splits on
paragraph then sentence boundaries rather than a fixed token count, so chunks
are less likely to end mid-sentence. It measures **characters**, not tokens, so
the Checkpoint 2 setting of 120 tokens maps to roughly 600 characters. The
corpus yields **204 chunks** here versus 150 under the Checkpoint 2 chunker —
the same documents, split on different boundaries.

The Checkpoint 2 metric experiment deliberately still uses the original chunker,
so `docs/02_Vector_Indexing.md` remains reproducible.

---

## 3. LangChain Integration

Checkpoint 3 asks for a RAG application built with LangChain or LlamaIndex. The
approach taken was **not** to rebuild retrieval inside LangChain — that would
discard the measured Checkpoint 2 work — but to expose the existing components
*through* LangChain's interfaces:

| Adapter | Implements | Effect |
|---|---|---|
| `PersonalAssistantEmbeddings` | `langchain_core.embeddings.Embeddings` | Our embedder drives any LangChain vector store |
| `PersonalAssistantRetriever` | `langchain_core.retrievers.BaseRetriever` | Our Chroma index plugs into any LangChain chain |

This is real interop, not a wrapper in name only:

```python
>>> retriever = app.as_langchain_retriever()
>>> isinstance(retriever, BaseRetriever)
True
>>> retriever.invoke("What is a vector database?")
[Document(page_content='vector databases - comprehensive guide...',
          metadata={'doc_id': 'guide_vector_databases', 'score': 0.484, ...})]
```

The payoff is that the cosine-versus-Euclidean analysis, the tuned chunking, the
citation metadata, and the offline fallbacks all still apply. LangChain composes
them rather than replacing them.

**Version note**: written against LangChain 1.3.18 / langchain-core 1.6.1. The
API differs substantially from the 0.1.x era that `requirements.txt` assumes.

---

## 4. Conversational Memory

### The problem

Checkpoint 2 answered each question in isolation. That breaks immediately in
dialogue:

```
User: "What is retrieval augmented generation?"
User: "Why does it reduce hallucination?"     <- "it" carries no topical content
```

Embedding *"Why does it reduce hallucination?"* produces a vector with almost
nothing to match on. Retrieval fails **regardless of the distance metric** — no
amount of Checkpoint 2 tuning helps, because the query itself is empty of topic.

### The fix, and where it runs

Reference resolution happens **before retrieval**, not after. It is a
search-quality fix, not an answer-quality one. This is what the
`query_rewriter` prompt was written for in Checkpoint 2.

Three stages, cheapest first:

1. **`needs_resolution()`** — a set-membership check for referring expressions
   (`it`, `that`, `they`, …). A question without one is passed through
   untouched, so the rewriter can never corrupt an already-standalone query.
2. **LLM rewrite** — the `query_rewriter` prompt, when a live client exists.
   A response over 300 characters is discarded: a rewriter returning an essay
   has misunderstood the task.
3. **Heuristic fallback** — carries the previous query's content words forward.
   Works offline, with no API key.

Measured on the demo conversation:

| Question as asked | Resolved to |
|---|---|
| "Why does it reduce hallucination?" | "Why does it reduce hallucination? **retrieval augmented generation**" |
| "What happens at that checkpoint?" | "What happens at that checkpoint? **capstone final defense**" |
| "How much butter does the cookie recipe need?" | *(unchanged — already standalone)* |

**3 of 7** queries in the demo required resolution.

### Two kinds of memory, kept separate

- **Retrieval memory** — resolving references so search works.
- **Generation memory** — prior turns passed to the model for continuity.

They are separate because they fail differently: a bad rewrite returns the wrong
documents, while excess history merely wastes context. History is trimmed to the
6 most recent turns; older turns are dropped rather than summarised, so the
prompt cannot grow without bound.

`total_turns` counts every exchange, separately from `len(turns)`, which caps at
the trim limit. An earlier version reported the turn number from the list
length, so turn 7 displayed as "turn 6" once trimming began.

---

## 5. Live Demo Results

Seven queries, run against 26 documents / 204 chunks:

| # | Query | Rewritten | Outcome |
|---|---|---|---|
| 1 | What is retrieval augmented generation? | — | Answered |
| 2 | Why does it reduce hallucination? | ✓ | Retrieved correctly (0.32) |
| 3 | Which distance metric should be used for that? | ✓ | Retrieved correctly (0.33) |
| 4 | How much butter does the cookie recipe need? | — | Answered from recipe |
| 5 | When is the capstone final defense? | — | Answered from plan |
| 6 | What happens at that checkpoint? | ✓ | Answered from plan |
| 7 | What is the WiFi password? | — | **Correctly refused** |

### Reading the refusals honestly

Two queries returned *"I don't have that in your notes."* for entirely
different reasons, and the demo distinguishes them rather than lumping them
together:

- **Query 7** — top retrieval score 0.15. Nothing relevant exists; the
  refusal is the grounding rule working as designed.
- **Query 2** — top score 0.32 against `notes_rag_concepts`, the correct
  document. Retrieval *succeeded*; the **offline extractive responder** cannot
  compose a "why" answer, because it quotes sentences rather than writing them.
  A real LLM answers this from the same passages.

Conflating those would misrepresent the system in opposite directions —
claiming a failure where the design worked, and hiding a fallback limitation
behind a virtue. The demo prints which case applies, using the retrieval score
as the discriminator.

---

## 6. Limitations

1. **No LLM was available**, so answers come from the offline extractive
   responder. Retrieval, memory, and ingestion are fully exercised; answer
   *quality* is not representative.
2. **Embeddings are the lexical fallback** — `huggingface.co` is blocked in
   this environment. Retrieval scores will improve with MiniLM.
3. **The heuristic rewriter is keyword-carrying, not semantic.** It appends
   prior content words rather than resolving the reference grammatically. Good
   enough to restore topical signal; the LLM path is better when available.
4. **Memory is per-process.** Conversations are not persisted across restarts.
5. **Ingestion is manual.** It detects changes when run, but nothing runs it
   automatically — a file watcher or scheduled job is deployment work.

---

## 7. Reflection

The instructive failure here was the ingestion pipeline reporting success over
an empty index. Every individual piece was correct: the hashing worked, the
classification logic was right, the manifest was written accurately. The bug
lived in an **assumption connecting two components** — that the manifest still
described the index. Reset the collection and that assumption silently became
false, with no error anywhere.

That is the same shape as the Checkpoint 2 finding about Euclidean distance.
There, retrieval quality depended on an invariant (normalized vectors)
maintained somewhere else; here, ingestion correctness depended on an invariant
(manifest matches index) maintained somewhere else. Both held until they
didn't, and neither announced its failure.

The fix in both cases was the same in spirit: stop depending on the invariant.
Cosine normalizes internally instead of trusting upstream; ingestion checks the
index instead of trusting the manifest. Components that verify their own
assumptions fail loudly, and loud failures are cheap. Silent ones cost you a
checkpoint.
