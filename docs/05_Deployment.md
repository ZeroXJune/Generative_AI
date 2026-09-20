# Checkpoint 4 — Deployment

**Project**: Personal Assistant AI — Generative RAG Capstone
**Course**: IS Professional Elective #4 — Generative AI Systems
**Instructor**: Jessie A. Melendres

---

## 1. Deliverables

| Requirement | Implementation | Verified |
|---|---|---|
| Containerization | `Dockerfile`, `docker-compose.yml`, `.dockerignore`, `docker/entrypoint.sh` | Structure only — see §5 |
| Web interface | `src/interface/app.py` | Yes, executed |
| Fine-tuning analysis | `docs/06_Fine_Tuning_Analysis.md`, `src/finetuning/` | Yes, calculator runs |
| Deployment evidence | §5 | Partial — build blocked |

---

## 2. Web Interface

The Streamlit app was rewritten for this checkpoint. The previous version was
Checkpoint 1-era: it read `chunks.jsonl` directly, computed brute-force cosine
in NumPy, and fell back to the **random-vector mock embedder** — bypassing the
Chroma index, the tuned chunking, conversational memory, and the LLM client.

It now drives `RAGApplication`, so the interface exercises the real system:

| Feature | Backed by |
|---|---|
| Grounded chat with citations | `grounded_qa` v2 + Chroma retrieval |
| Follow-up questions | `ConversationMemory` — shows the rewritten query |
| Upcoming deadlines panel | `ReminderEngine`, urgency-banded |
| Source inspection | Retrieval scores per answer |
| Honest status | Reports the embedder and LLM backend *actually* running |

That last row is deliberate. The sidebar shows
`Embeddings: lexical-tfidf-hash (fallback)` rather than claiming MiniLM, and
`LLM: offline responder` rather than implying a live model. A demo that hides
its own degradation is worse than one that admits it.

### Verified

Executed with `streamlit.testing.v1.AppTest` — the app is run, not just parsed:

```
exceptions : 0
subheaders : ['⏰ Upcoming Deadlines', '💬 Ask a question']
metrics    : Overdue=10, Next 14 days=4, Next deadline=2026-09-22,
             Documents=26, Chunks=204
warnings   : 'Embeddings: lexical-tfidf-hash (fallback)',
             'LLM: offline responder',
             '**Midterm Exam Period** — in 2 days (2026-09-22)'
```

A question submitted through the chat input returned an answer with sources
and raised no exception.

---

## 3. Container Design

### Two-stage build

The builder stage installs `build-essential` and compiles wheels into a venv;
the runtime stage copies only `/opt/venv`. Compilers never ship in the final
image — smaller, and less attack surface.

### Single source of truth for dependencies

`requirements-core.txt` is the only dependency list. `sentence-transformers` is
filtered out at build time unless requested:

```dockerfile
ARG WITH_LOCAL_EMBEDDINGS=false
RUN if [ "$WITH_LOCAL_EMBEDDINGS" = "true" ]; then \
      pip install -r requirements-core.txt; \
    else \
      grep -v '^sentence-transformers' requirements-core.txt > slim.txt \
      && pip install -r slim.txt; \
    fi
```

A second requirements file would drift from the first. Filtering keeps one list
authoritative.

**The trade-off**: `sentence-transformers` pulls PyTorch, roughly 2.5 GB. The
default image omits it and runs on the lexical embedder; `--build-arg
WITH_LOCAL_EMBEDDINGS=true` produces the full-quality image. Image size is
traded against retrieval quality, explicitly rather than accidentally.

### Security

| Measure | Why |
|---|---|
| `USER appuser` (uid 1000) | A container escape as root lands as root on the host namespace |
| `.env` in `.dockerignore` | Credentials must never enter an image layer, where they persist even if a later layer deletes them |
| Notes mounted `:ro` | The container indexes documents; it has no reason to write them |
| Env vars at runtime | Secrets injected by compose, never baked in |

### Persistence

Named volumes (`assistant_index`, `assistant_processed`) keep the Chroma index
across container replacement, so a restart does not re-embed the corpus. The
entrypoint only ingests when no index is present.

### Entrypoint modes

```bash
docker compose up --build                  # web interface on :8501
docker compose run --rm app reminders      # deadline digest
docker compose run --rm app ingest         # refresh the index
docker compose run --rm app shell          # debugging
```

---

## 4. Running It

```bash
git clone https://github.com/ZeroXJune/Generative_AI.git
cd Generative_AI
docker compose up --build
# open http://localhost:8501
```

Optional, for real model answers:

```bash
export OPENAI_API_KEY="sk-..."       # or OPENAI_BASE_URL for local Ollama
docker compose up --build
```

Full-quality embeddings (adds ~2.5 GB):

```bash
docker build --build-arg WITH_LOCAL_EMBEDDINGS=true -t personal-assistant-ai:full .
```

---

## 5. Deployment Evidence — and What Is Missing

**The image has not been built.** This must be stated plainly rather than
implied by omission.

The Docker daemon runs in the build environment (29.3.1), and
`registry-1.docker.io` is reachable, but the CDN serving image **blobs** is
blocked by network policy:

```
production.cloudfront.docker.com:443
  → gateway answered 403 to CONNECT (policy denial)
```

`docker build` therefore fails while pulling `python:3.11-slim`. This is the
same class of block that prevents downloading MiniLM from `huggingface.co` and
calling `api.openai.com` — an environment restriction, not a defect in the
Dockerfile. Routing around an organisational egress policy was not attempted.

### What *was* verified

| Check | Result |
|---|---|
| `sh -n docker/entrypoint.sh` | Parses |
| Entrypoint mode dispatch | `reminders`, `ingest`, passthrough all route correctly |
| Two build stages present | Yes |
| Runs as non-root | `USER appuser` |
| Healthcheck defined | Streamlit `/_stcore/health` |
| `.env` excluded from context | Yes |
| Web interface executes | 0 exceptions under AppTest |
| Dependencies install | `requirements-core.txt` installs cleanly on Python 3.11 |

### What remains unproven

- The image builds end to end
- The container starts and serves on :8501
- The healthcheck passes against a running container
- Volume persistence across `docker compose down && up`

### Runtime layer verified without the image

The build is blocked, but the *application* the container would run is not.
Each entrypoint mode was executed directly in a Linux / Python 3.11
environment, issuing the same commands `docker/entrypoint.sh` issues:

| Mode | Command | Result |
|---|---|---|
| `ingest` | `RAGApplication().ingest()` | Ran; index current at 26 documents |
| `reminders` | `python src/reminders.py` | Ran; 4 deadlines in the next 14 days |
| `serve` | `streamlit run ... --server.address=0.0.0.0 --server.headless=true` | Bound to :8501 |

The Dockerfile's healthcheck was exercised against that running server:

```
GET /_stcore/health  ->  HTTP 200, body: ok
GET /                ->  HTTP 200
```

This does not prove the image builds. It does prove the healthcheck is correct
and that the process starts under the exact flags the container uses — so if
the build succeeds, the container serves.

### Closing the gap: one command

```bash
./scripts/verify_deployment.sh
```

Run on any machine with unrestricted Docker access. It builds the image,
starts the container, waits for the healthcheck, exercises the CLI
entrypoints, checks the named volumes, tears down, and writes the whole
transcript to `docs/deployment_evidence.txt`.

Exit code 0 means the image builds and the container serves healthy; 1 means
it did not, with the failure in the output. That file is the deployment
evidence this checkpoint asks for.

### Checking the environment first

```bash
python scripts/doctor.py
```

Reports whether real MiniLM embeddings and a live LLM are active, or whether
fallbacks are silently in play. Exit code 0 = fully operational, 1 = degraded
but working, 2 = broken.

This exists because every component here degrades gracefully rather than
crashing — good for robustness, bad for confidence, since a degraded run looks
exactly like a working one. Run it before recording a demo or quoting
retrieval scores as results.

---

## 6. Reflection

The useful lesson here was about honest reporting under constraint. It would
have been easy to write the Dockerfile, note that it "follows best practice,"
and let the checkpoint read as complete. The image has never been built, and a
Dockerfile that has never been built is a hypothesis.

So §5 separates verified from unproven, names the exact blocked host, and gives
the three commands that would settle it. That is less satisfying than a green
checkmark and considerably more useful to anyone who has to run this.

It also matches what the rest of the project does. The pipeline reports the
embedder that actually ran rather than the one requested; the chat client
reports which backend served a call; the demo distinguishes a correct refusal
from a fallback limitation. A container that has not been built belongs in the
same category — state it, don't imply otherwise.
