# Personal Assistant AI — Project Overview

**Student**: Junie Mumar
**Course**: IS Professional Elective #4 — Generative AI Systems
**Instructor**: Jessie A. Melendres
**Capstone Theme**: Personal/Domain Notes Assistant
**Repository**: https://github.com/ZeroXJune/Generative_AI
**Status as of**: August 28, 2026 — Checkpoints 1 and 2 complete

---

## 1. What This Project Is

A Retrieval-Augmented Generation (RAG) system that answers questions about a
user's own documents and warns them about approaching deadlines.

**The problem it solves**: notes, schedules, and reference material end up
scattered across files. Keyword search fails when you ask *"when's my capstone
due?"* but the note says *"final defense."* And no amount of searching tells
you a deadline is three days away unless you go looking.

**The approach**: convert every document into vectors that capture meaning,
so "find relevant notes" becomes "find the nearest vectors." Then feed only
those notes to a language model, with instructions to answer from them alone.

**Current scale**: 26 documents → 150 indexed chunks → 32 Python modules.

---

## 2. How It Works, End to End

```
data/raw/*.txt                 26 source documents
      │
      ▼
  TextCleaner                  strip HTML, URLs, normalise unicode
      │
      ▼
  DocumentChunker              120-token chunks, 30-token overlap
      │
      ▼
  EmbeddingGenerator           each chunk → 384 numbers (MiniLM)
      │
      ▼
  Chroma vector store          cosine similarity index
      │
      ▼
  RAGRetriever                 question → vector → nearest chunks
      │
      ▼
  System prompt + context      "answer ONLY from these passages"
      │
      ▼
  Chat Completion API          the answer, with [doc_id] citations
```

Running alongside, and deliberately independent of the LLM:

```
data/raw/*.txt → DeadlineExtractor → deadlines.json → ReminderEngine → alerts
```

---

## 3. Checkpoint 1 — Foundation & Data Pipeline ✅

**Delivered**: project proposal, a 26-document corpus, text preprocessing with
before/after examples, embedding generation with model justification, and
repository setup.

| Component | File |
|---|---|
| Text cleaning | `src/preprocessing/text_cleaner.py` |
| Chunking | `src/preprocessing/chunker.py` |
| Embeddings | `src/embeddings/embedding_generator.py` |
| Full pipeline | `src/pipeline.py` |

**Embedding model**: `all-MiniLM-L6-v2` — 384 dimensions, ~50 MB, fast enough
to run locally without a GPU. Chosen over `all-mpnet-base-v2` (768-dim, 440 MB)
because the quality difference does not justify 9× the download on a corpus
this size.

**Documentation**: `docs/01_Project_Proposal.md`,
`docs/CHECKPOINT1_REFLECTION.md`, `docs/CHECKPOINT1_COMPLETION.md`

---

## 4. Checkpoint 2 — Prompt Architecture & Vector Indexing ✅

Four graded deliverables, all complete.

### 4.1 Prompt Architecture

Four versioned system prompts, each recording the failure mode it fixes
alongside its text (`src/prompts/system_prompts.py`):

| Prompt | Ver | Task | Failure mode addressed |
|---|---|---|---|
| `grounded_qa` | v2 | Answer from notes | Hallucination |
| `schedule_extractor` | v3 | Deadlines as JSON | Unparseable output |
| `study_summarizer` | v2 | Revision material | Content drift |
| `query_rewriter` | v1 | Rewrite follow-ups | Retrieval collapse in chat |

**Verified behaviour**: asked *"What is the WiFi password for the campus
network?"* — absent from all 26 documents — the system replies exactly
`I don't have that in your notes.` rather than inventing one.

### 4.2 Chat Completion API Integration

`src/llm/chat_client.py` targets the OpenAI-compatible
`/v1/chat/completions` contract rather than a single vendor, so it runs
against a hosted API, a local Ollama server, or an offline responder:

| Backend | Requires | Reported as |
|---|---|---|
| Offline extractive responder | nothing (default) | `offline` |
| Ollama / vLLM / LM Studio | local server | `local` |
| OpenAI / Azure / Groq | paid API key | `openai` |

Every response records which backend served it, so a free local run is never
mistaken for a paid API run in a transcript.

### 4.3 Vector Database

**Chroma**, chosen over FAISS (no metadata store — citations would need a
second database), Pinecone (cost and network dependency unjustified at 26
documents), and Weaviate/Milvus (Docker/Kubernetes overhead far exceeding the
corpus).

### 4.4 Distance Metric Comparison — the headline result

Measured on the project's own corpus with 8 ground-truth queries
(`src/experiments/metric_comparison.py`):

**On L2-normalized embeddings, all three metrics produce identical rankings.**

| Metric | Hit@1 | Hit@5 | MRR | Overlap@5 |
|---|---|---|---|---|
| Cosine | 0.625 | 0.875 | 0.692 | 1.000 |
| Euclidean | 0.625 | 0.875 | 0.692 | 1.000 |
| Dot product | 0.625 | 0.875 | 0.692 | 1.000 |

This is forced by algebra, not coincidence. For unit vectors:

```
dot(a,b)        = cos(a,b)                  … identical
euclidean(a,b)² = 2 − 2·cos(a,b)            … strictly decreasing in cosine
```

A strictly monotonic transform cannot reorder a ranking.

**Remove normalization and the agreement collapses:**

| Metric | Hit@1 | Hit@5 | MRR | Avg chunk tokens |
|---|---|---|---|---|
| Cosine | 0.625 | 0.875 | 0.692 | 107.2 |
| Euclidean | **0.125** | **0.125** | **0.125** | **22.8** |
| Dot product | 0.625 | 0.750 | 0.688 | 114.1 |

Euclidean stops retrieving *relevant* chunks and starts retrieving *short*
ones. Dot product acquires the opposite bias, favouring long chunks. Cosine is
unchanged, because it normalizes internally.

**Decision: cosine — chosen for robustness, not speed.** It tied on quality
and is slower than dot product, but it is the only one of the three whose
results do not depend on an invariant maintained elsewhere in the codebase.

**Documentation**: `docs/02_Prompt_Engineering.md`, `docs/02_Vector_Indexing.md`

---

## 5. Schedule Reminders ✅ (proposal objective, beyond CP2 scope)

The original proposal promised "automatic notifications" and *"100% on-time
alerts"*, but nothing had been built — the `schedule_extractor` prompt existed
with no consumer. This component closes that gap.

**The design rule**: the model finds candidate dates; **Python decides what is
approaching**. Date arithmetic is a documented LLM weakness, and the proposal
promises a guarantee — so the comparison is `deadline − today` in plain Python.
Reminders are exact, reproducible, and work with no API key or network.

| Component | File |
|---|---|
| Date/time parsing (stdlib only) | `src/schedule/date_parser.py` |
| Extraction + filters | `src/schedule/extractor.py` |
| Persistence | `src/schedule/store.py` |
| Urgency banding | `src/schedule/reminders.py` |
| CLI | `src/reminders.py` |
| Streamlit panel | `src/interface/app.py` |

**Result**: 20 genuine commitments extracted from the corpus. Four filters cut
28 raw candidates down to 20 by rejecting holidays, document metadata
(`Date: July 14, 2026` dates the *document*), and dates quoted inside
text-cleaning examples.

**Documentation**: `docs/03_Schedule_Reminders.md` plus a companion explainer.

---

## 6. Engineering Fixes Worth Recording

Several defects surfaced during development that are worth naming, because
each one would have quietly produced misleading results:

| Defect | Why it mattered |
|---|---|
| Pipeline reported `all-MiniLM-L6-v2` while using random mock vectors | A fallback run looked identical to a real run. Now reports the backend that actually ran. |
| `ChatResponse` hardcoded `backend="openai"` | A free local run appeared as a paid API run in transcripts. |
| `model` defaulted in the signature, so `OPENAI_MODEL` was ignored | Every Ollama request would have failed — Ollama has no `gpt-3.5-turbo`. |
| `python-dotenv` declared but `load_dotenv()` never called | A key placed in `.env` was silently ignored. |
| Chunking at 512/100 | Too coarse; a single chunk swallowed most of a short note. Tuned to 120/30 on measured hit rate. |
| Setup docs cloned an unrelated repository | Anyone following them got the wrong project. |

The pattern in most of these: **the system was reporting something other than
what it did.** That is the same failure class the `grounded_qa` prompt exists
to prevent, appearing in the infrastructure rather than the model.

---

## 7. What Runs Today

```bash
python src/build_index.py                     # 26 docs → 150 chunks
python src/checkpoint2_demo.py                # prompts + API + live queries
python src/experiments/metric_comparison.py   # regenerates the results
python src/reminders.py                       # deadline digest
streamlit run src/interface/app.py            # web interface
```

All five verified from a clean checkout. The Streamlit interface was checked
with `streamlit.testing.v1.AppTest`: 0 exceptions.

**Environment**: Python 3.11 with `requirements.txt`. The pinned versions are
from 2023 and need 3.11 — on Python 3.13+ several will try to compile from
source and fail.

---

## 8. Known Limitations (stated, not hidden)

1. **Results were produced with a fallback embedder.** `huggingface.co` was
   blocked by network policy in the build environment, so the numbers above
   come from a deterministic TF-IDF embedder rather than MiniLM. The geometric
   findings in §4.4 follow from vector algebra and hold for any embedding;
   absolute hit rates will improve once the real model is used.
2. **Extraction recall is unmeasured.** All 20 extracted deadlines are correct
   (precision 1.00), but no labelled ground truth exists for what was *missed*.
3. **The hosted LLM path is untested end to end.** The code and HTTP contract
   were verified against a local OpenAI-compatible server; no hosted provider
   call has ever succeeded, because no reachable API key was available.
4. **Reminders are pull-based.** Alerts appear when the CLI runs or the app
   opens. True push delivery needs a scheduled job — deployment work.
5. **Relative dates are ignored.** "Next Friday" is deliberately not resolved,
   because that requires assuming a reference date.

---

## 9. What Comes Next

| Checkpoint | Due | Scope |
|---|---|---|
| **3 — RAG Orchestration** | Oct 17, 2026 | Automated ingestion, LangChain/LlamaIndex, conversational memory, live demo |
| **4 — Deployment & Defense** | Nov 14, 2026 | Fine-tuning analysis (LoRA/QLoRA), Streamlit interface, Docker, final defense |

`query_rewriter` v1 is already in place as groundwork for Checkpoint 3's
conversational memory.

---

## 10. The Through-Line

If there is one idea connecting the work so far, it is that **reliability came
from constraining what the model was allowed to decide**, not from prompting it
more carefully.

- `grounded_qa` became trustworthy when refusal was made an explicit,
  first-class outcome — not when the instructions got more emphatic.
- The reminder engine became guaranteeable when date arithmetic was moved out
  of the model entirely.
- The metric comparison concluded not with "which retrieves best?" — they tied
  — but with "which fails most gracefully when an assumption breaks?"

Each time, the durable answer came from narrowing the model's authority and
verifying the result, rather than from trusting it further.
