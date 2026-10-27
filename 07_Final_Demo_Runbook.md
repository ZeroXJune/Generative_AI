# Final Demo Runbook

Step-by-step instructions for the remaining Checkpoint 4 work, in the order to
do them. Every command was checked against the scripts in this repository, but
**steps 4–6 have never been run end to end** — no successful Gemini call, no
real MiniLM run and no Docker build exist yet. Expect the first container build
to surface something; that is why it comes early.

| # | Task | Time | Blocks |
|---|---|---|---|
| 0 | Confirm Checkpoint 3 was submitted | 5 min | — |
| 1 | Install the machine prerequisites | 30 min | everything |
| 2 | Set up Python and the Gemini key | 20 min | 3–6 |
| 3 | Clear stale indexes | 1 min | 4 |
| 4 | Pass `doctor.py` locally | 10 min | 5, 6 |
| 5 | Build the container and capture evidence | 30–60 min | 6 |
| 6 | Record the video demo | 1–2 h | — |

Commands are given for **Windows PowerShell** and **macOS/Linux**. Run them
from the repository root.

---

## 0. Confirm Checkpoint 3 was submitted

It was due **October 17, 2026, 23:59**. If it was not submitted, deal with that
before anything below — none of this replaces it.

---

## 1. Prerequisites

- **Python 3.11 or 3.12.** 3.13 works with `requirements-core.txt`, but 3.11
  matches the container and is the safest choice.
- **Git**, and on Windows **Git Bash** (installed with Git for Windows) — the
  deployment script is a bash script.
- **Docker Desktop**, running. Check: `docker info` must print without error.
- **~6 GB free disk**: PyTorch is ~2.5 GB locally and again inside the image.
- **A Gemini API key** from <https://aistudio.google.com/apikey>.
- **Internet access** to `huggingface.co` (first MiniLM download, ~90 MB) and
  `generativelanguage.googleapis.com` (Gemini).

Get the latest code:

```bash
git checkout claude/personal-assistant-ai-yxy5sf
git pull
```

---

## 2. Python environment and Gemini key

### 2.1 Create a virtual environment

```powershell
# Windows
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
```

```bash
# macOS / Linux
python3.11 -m venv .venv
source .venv/bin/activate
```

### 2.2 Install dependencies

Install CPU-only PyTorch **first**. On Windows the default PyTorch wheel
includes CUDA and is several times larger.

```bash
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-core.txt
```

Use `requirements-core.txt`, not `requirements.txt` — the latter is pinned to
2023 versions and fails to install on Python 3.13.

### 2.3 Create `.env`

```powershell
copy .env.example .env      # Windows
```

```bash
cp .env.example .env        # macOS / Linux
```

Open `.env` and add these lines (or uncomment Option C):

```ini
OPENAI_API_KEY=your-gemini-key
OPENAI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
OPENAI_MODEL=gemini-2.5-flash
WITH_LOCAL_EMBEDDINGS=true
```

- **Set all three `OPENAI_*` lines.** If `OPENAI_MODEL` is missing, Docker
  Compose substitutes `gpt-3.5-turbo`, Gemini rejects it, and the app silently
  falls back to offline answers.
- **Check the model name** against Google's current model list. If
  `gemini-2.5-flash` has been retired, use the current Flash model.
- `.env` is git-ignored and excluded from the Docker image. **Never commit it,
  and never show it on screen while recording.**

---

## 3. Clear stale indexes

The index does not record which embedder built it. If it was ever built with
the TF-IDF fallback, switching to MiniLM re-embeds nothing — the documents have
not changed — so MiniLM queries search TF-IDF vectors. Both are 384-dimensional,
so there is no error, only bad retrieval. Delete the old indexes once, now:

```powershell
# Windows
Remove-Item -Recurse -Force data\vector_store, data\vector_store_cp3 -ErrorAction SilentlyContinue
```

```bash
# macOS / Linux
rm -rf data/vector_store data/vector_store_cp3
```

They are regenerated automatically. Repeat this step any time you switch
between real and fallback embeddings.

---

## 4. Pass the environment doctor locally

```bash
python scripts/doctor.py
```

It makes **one real Gemini call** and loads the real embedding model. The
target verdict:

```
FULLY OPERATIONAL - real embeddings and a live LLM.
```

with these two lines in the report:

```
[ OK ] Real model: all-MiniLM-L6-v2
[ OK ] Live backend: gemini (...)
```

| Doctor says | Cause | Fix |
|---|---|---|
| `Fallback embedder: lexical-tfidf-hash` | sentence-transformers missing, or Hugging Face unreachable | Redo 2.2; check internet access to huggingface.co |
| `Offline extractive responder` | `.env` not found or `OPENAI_API_KEY` unset | Run from the repo root; check the `.env` lines |
| `...but the call failed` + `valid API key` | Wrong or revoked key | Generate a new key |
| `...but the call failed` + model not found / 404 | Retired model name | Change `OPENAI_MODEL` |
| `...but the call failed` + 429 | Free-tier rate limit | Wait a minute and retry |
| `returned an empty answer` | Reasoning model spent its token budget thinking | Use a non-thinking / "lite" model |
| `No index yet` | Expected after step 3 | Step 4.1 builds it |

### 4.1 Smoke-test the app

```bash
python src/checkpoint3_demo.py       # builds the index, answers 7 queries
streamlit run src/interface/app.py   # opens http://localhost:8501
```

In the demo output, answers should now read as **generated prose from Gemini**,
not sentences quoted verbatim from your notes. Turn 7 (the WiFi password)
should still refuse with `I don't have that in your notes.` If Gemini answers
it instead, the grounding prompt is not holding — note that before recording.

---

## 5. Build the container and capture deployment evidence

### 5.1 Start clean

`WITH_LOCAL_EMBEDDINGS=true` in `.env` (step 2.3) makes Compose build the full
image with MiniLM. Remove any old volume so the container does not reuse a
fallback index (same reason as step 3):

```bash
docker compose down -v
```

### 5.2 Run the verification script

```bash
# macOS / Linux, or Git Bash on Windows
./scripts/verify_deployment.sh
```

It builds the image, starts it, waits for the healthcheck, calls the HTTP
health endpoint, runs the reminders entrypoint, and writes everything to
`docs/deployment_evidence.txt`.

- **Pass:** the last block says `RESULT: PASS` and the script exits 0.
- **Fail with `health=unhealthy`:** the first start downloads MiniLM and builds
  the index before Streamlit starts, which can outlast the healthcheck. The
  script keeps the volume, so run it **once more** — the second start skips
  indexing.
- **Fail at `BUILD`:** fix the first error in the output; later errors are
  usually consequences of it.

### 5.3 Run the doctor inside the container

The image does not include `scripts/`, so mount it in for this one command:

```bash
docker compose run --rm -v "$PWD/scripts:/app/scripts:ro" app python scripts/doctor.py
```

(Works as written in PowerShell too.) It must show the same two `[ OK ]`
lines as step 4. Paste the output at the end of `docs/deployment_evidence.txt`
— the script proves the container serves; this proves it serves the **real**
stack.

### 5.4 Commit the evidence

```bash
git add docs/deployment_evidence.txt
git commit -m "Add Checkpoint 4 deployment evidence"
git push
```

Then tick **Deployment evidence** in `CLAUDE.md` → Checkpoint 4.

---

## 6. Record the video demo

### 6.1 Before pressing record

- [ ] `docker compose up -d` and open <http://localhost:8501>
- [ ] Container doctor (5.3) shows both `[ OK ]` lines **today**
- [ ] `.env`, the API key and any terminal history containing it are off screen
- [ ] Ask each demo question once off camera — confirms Gemini is responding
      and warms the model, so the first on-camera answer is not slow
- [ ] Notifications off; browser zoom ~125% so text is readable

### 6.2 Suggested sequence

| Scene | Show | Point to make |
|---|---|---|
| 1 | Doctor output in the container | This run uses real embeddings and a live LLM — not fallbacks |
| 2 | `docker compose ps` showing `healthy` | Deployed as a container, health-checked |
| 3 | A direct question from your notes | Answer is grounded, with sources cited |
| 4 | A follow-up such as *"when is it due?"* | Reference resolution runs **before** retrieval; without it the query has no topic to search for |
| 5 | *"What is the WiFi password for the campus network?"* | The assistant refuses rather than inventing an answer |
| 6 | Reminder panel / `docker compose run --rm app reminders` | Python, not the LLM, computes what is due — exact, and works with no API key |
| 7 | Brief architecture recap | Gemini generates; MiniLM + Chroma retrieve; fine-tuning rejected (`docs/06_Fine_Tuning_Analysis.md`) |

### 6.3 Pacing

Leave a few seconds between questions. On the free tier a rate-limit hit makes
the app retry and then fall back to quoting notes; on camera that looks like a
worse model, not an error. If an answer is suddenly verbatim quotes, stop and
re-take the scene.

### 6.4 Likely defense questions

- **Why Gemini?** Free tier, OpenAI-compatible endpoint, so no client code
  change; swappable by environment variable (Ollama runs fully offline).
- **Where does your data go?** Retrieved note excerpts are sent to Google with
  each question; on the free tier Google may use prompts to improve its
  products. Fine for course notes; for private notes, use the Ollama option.
- **Why not fine-tune?** 38 documents cannot produce a training set, and facts
  in weights cannot be cited or updated. Fine-tune for behaviour, retrieve for
  facts.
- **Why cosine?** On normalized vectors all three metrics rank identically;
  cosine is the only one robust to unnormalized input
  (`docs/02_Vector_Indexing.md`).
