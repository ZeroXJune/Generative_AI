# Final Presentation Script — Video Demo

**Format**: recorded video, 3 presenters, **10 minutes**
**Pace**: ~130 spoken words per minute. Each section below is timed to that;
if a rehearsal runs long, cut the lines marked *(optional)* first.

Replace **Member 1 / 2 / 3** with your names. The split follows the component
ownership in the capstone plan, so each person presents what they can defend
in Q&A:

| Presenter | Owns | Segment | Time |
|---|---|---|---|
| Member 1 | Data, embeddings, vector DB | Problem + data pipeline | 0:00–3:00 |
| Member 2 | Prompts, LLM, RAG, memory | Live Q&A demo | 3:00–6:30 |
| Member 3 | Reminders, UI, deployment, fine-tuning | Reminders + container + verdict | 6:30–9:30 |
| All | — | Limitations + close | 9:30–10:00 |

**Before recording**, complete `docs/07_Final_Demo_Runbook.md` through step 5
and its §6.1 checklist. Every demo answer in this script must be rehearsed on
the real stack (Gemini + MiniLM) — the answers have only been observed on the
offline fallback, so the exact wording you get will differ.

**Never say a number that is not in this script or that you have not measured.**
Retrieval scores in `docs/02_Vector_Indexing.md` were measured with the
lexical (TF-IDF) fallback, not MiniLM — the script labels them that way.

---

## Segment 1 — Member 1: Problem and data pipeline (0:00–3:00)

### 0:00 — Opening

**SCREEN**: Title slide — *Personal Assistant AI: A RAG Notes Assistant*, three
names, course, instructor.

> Good day. We are [names], and this is our capstone for IS Professional
> Elective 4, Generative AI Systems: a Personal Assistant AI.
>
> The problem is one every student has. Our notes, schedules and references are
> spread across dozens of files. Searching them by keyword misses things, and
> asking a general chatbot is worse — it doesn't know our notes, so it guesses,
> and it guesses confidently.
>
> So we built an assistant that answers **only from your own documents**, says
> so when the answer isn't there, and warns you about deadlines before you miss
> them.

### 0:45 — The corpus

**SCREEN**: `data/raw/` folder listing.

> Our knowledge base is 38 documents: course notes, twelve college-subject
> lessons, project plans, schedules, how-to guides — even a recipe, so we can
> show the system handles unrelated topics without mixing them up.

### 1:05 — The pipeline

**SCREEN**: The pipeline diagram from `CLAUDE.md` (Raw → Cleaner → Chunker →
Embeddings → Chroma).

> Every document goes through the same pipeline. First we clean it — strip
> HTML and URLs, normalize Unicode, collapse whitespace. Then we split it into
> overlapping chunks, so a fact that sits on a boundary still lands whole in
> at least one chunk.
>
> Each chunk is turned into a 384-dimensional vector by **all-MiniLM-L6-v2**, a
> Sentence-Transformers model. We chose it because it is small — it runs on a
> laptop CPU — and it captures meaning, not just words: a question about
> "hallucination" can match a passage about "made-up answers".
>
> The vectors go into **Chroma**, a local vector database. Ingestion is
> incremental: each file is fingerprinted with a content hash, so adding one
> note re-embeds one note, not the whole collection.

### 2:05 — The distance-metric experiment

**SCREEN**: Terminal running `python src/experiments/metric_comparison.py`,
scrolled to the "Ranking agreement" block.

> For Checkpoint 2 we compared three distance metrics: cosine, Euclidean and
> dot product. On normalized vectors, all three gave **identical rankings** —
> not similar, identical. That's not luck; it's algebra. For unit-length
> vectors, cosine and dot product are the same number, and Euclidean distance
> is a monotonic function of both.
>
> The difference appears on **unnormalized** vectors. In our lexical baseline
> run, Euclidean collapsed — it favoured very short chunks because short
> vectors sit close together, not because they were relevant. We chose
> **cosine** because it is the only one of the three that ignores vector length
> entirely, so it stays correct even if a future model doesn't normalize.
>
> Member 2 will now show what happens when you ask a question.

*(≈ 395 words ≈ 3:00.)*

---

## Segment 2 — Member 2: The RAG assistant, live (3:00–6:30)

### 3:00 — How an answer is made

**SCREEN**: Streamlit app at `http://localhost:8501`, empty chat.

> Thanks. When you ask a question, three things happen. Your question is
> embedded with the same model, Chroma returns the closest chunks, and those
> chunks are handed to the language model — **Google Gemini** — together with
> a system prompt that says: answer only from this context, and if the
> context doesn't contain the answer, say so.
>
> Gemini is called through an OpenAI-compatible endpoint, so the provider is a
> configuration setting, not code. The same app runs fully offline with a
> local Llama model through Ollama.

### 3:35 — Question 1: a direct question

**TYPE**: `What is retrieval augmented generation?`

> *(while it answers)* This comes from our week-one course notes.

**SCREEN**: Open the **sources** expander under the answer.

> Every answer shows which documents it came from and how closely they matched,
> so you can check the answer instead of trusting it.

### 4:10 — Questions 2 and 3: follow-ups

**TYPE**: `Why does it reduce hallucination?`

> Notice I didn't say "RAG" again — I said "it". That's harder than it looks.

**TYPE**: `Which distance metric should be used for that?`

> Here is the key design decision of Checkpoint 3. A follow-up like "what about
> that?" has no topic in it. Turn it into a vector and it points nowhere —
> no distance metric can rescue a query that is empty of meaning.
>
> So **before** we search, the conversation memory rewrites the question using
> earlier turns. "Which distance metric should be used for that" becomes a
> search that includes "retrieval augmented generation" and "hallucination".
> Resolution first, retrieval second.

### 5:05 — Question 4: a different topic

**TYPE**: `How much butter does the chocolate chip cookie recipe need?`

> A completely different domain — and the assistant switches cleanly to the
> recipe, without dragging in the AI notes from the turns before.

*(optional — cut if long)*
> That's the memory knowing when **not** to carry context forward.

### 5:35 — Question 5: the refusal

**TYPE**: `What is the WiFi password for the campus network?`

**SCREEN**: Answer reads *"I don't have that in your notes."*

> This is the answer we are proudest of. That fact isn't in any of our 38
> documents. A general chatbot would invent something plausible. Ours checks
> whether any retrieved passage clears a relevance threshold, and when none
> does, it refuses.
>
> For a notes assistant, **"I don't know" is a correct answer**, and a
> confident wrong one is the worst possible failure.
>
> Member 3 will show the part that doesn't wait for you to ask.

*(≈ 340 words ≈ 2:40 spoken, plus ~50 s waiting on five answers.)*

---

## Segment 3 — Member 3: Reminders, deployment, fine-tuning (6:30–9:30)

### 6:30 — Deadline reminders

**SCREEN**: The deadline panel in the Streamlit sidebar, or terminal running
`docker compose run --rm app reminders`.

> Thanks. A notes assistant should also tell you what's coming. The reminder
> engine reads the schedules in your notes and sorts every deadline into
> bands: overdue, today, urgent, soon, and upcoming.
>
> One rule matters most here: **the language model finds candidate dates, but
> Python decides what is approaching.** Language models are unreliable at date
> arithmetic — "is October 20th within three days?" is exactly where they slip.
> So the comparison is plain code. The alerts are exact, and they work with no
> API key and no internet at all.

### 7:20 — Deployment

**SCREEN**: Terminal — `docker compose ps` showing the container **healthy**.

> The whole system ships as a Docker container. One command —
> `docker compose up` — builds it, indexes the notes on first start, and serves
> the web interface. A healthcheck reports the container healthy only once the
> app actually responds, and the index lives on a named volume, so it survives
> restarts. Your notes are mounted read-only: the container can index them but
> never change them. And API keys are injected at runtime — they never enter
> the image.

### 8:00 — Proving it's the real stack

**SCREEN**: Doctor output inside the container, both `[ OK ]` lines visible.

> One thing we learned the hard way: when a component fails, our app falls back
> gracefully — and a degraded run **looks identical** to a working one. So we
> built a diagnostic that reports which embedding model and which language
> model are actually live, and makes one real API call to prove it. Everything
> you've just seen ran on real MiniLM embeddings and live Gemini.

### 8:35 — Why we didn't fine-tune

**SCREEN**: The 7B row of the VRAM table from
`python src/finetuning/vram_calculator.py`.

> Checkpoint 4 asked us to evaluate fine-tuning. Our calculator shows full
> fine-tuning of a 7-billion-parameter model needs about 95 GB of GPU memory.
> LoRA cuts that to about 15, and QLoRA to **4.8 GB** — small enough for a free
> Colab GPU. So fine-tuning is feasible.
>
> We still rejected it. Fine-tuning moves knowledge into the model's weights,
> where it can't be cited, checked or updated — and 38 documents can't produce
> a training set anyway. Our rule: **fine-tune for behaviour, retrieve for
> facts.**

*(≈ 335 words ≈ 2:35 spoken; the spare ~25 s absorbs screen switches.)*

---

## Segment 4 — All: Limitations and close (9:30–10:00)

**SCREEN**: Closing slide — three limitations, three names.

**Member 1**:
> Three honest limitations. Our evaluation set is small — 20 test questions.

**Member 2**:
> Retrieved note excerpts are sent to Google with each question. For private
> notes, the offline Ollama mode keeps everything on the machine.

**Member 3**:
> And the free API tier has rate limits. If it's hit, the app falls back to
> quoting your notes rather than failing.

**All / Member 1**:
> Thank you. We're happy to take questions.

*(≈ 70 words. Total ≈ 1,045 spoken words ≈ 8:00, plus answer waits and screen changes ≈ 9:45.)*

---

## Q&A preparation — who answers what

The capstone plan schedules Q&A after the presentation. Each question goes to
the member who owns that component.

### Member 1 — data and retrieval

- **Why MiniLM and not a bigger embedding model?** Runs on a CPU, ~90 MB,
  384 dimensions is enough for 38 documents; a larger model costs latency and
  disk for gains this corpus can't show.
- **Why chunk overlap?** A fact split across a chunk boundary would otherwise
  be retrievable from neither half.
- **Why Chroma, not FAISS or Pinecone?** Local, persistent, no server or cost;
  FAISS is the upgrade path at scale. Pinecone's cost and network dependency
  aren't justified at 38 documents.
- **Your metrics experiment used TF-IDF?** Yes — the measured scores are from
  the lexical baseline. The ranking-identity result is algebraic, so it holds
  for MiniLM as well; the absolute scores are not claimed for MiniLM.

### Member 2 — RAG and LLM

- **How do you stop hallucination?** Grounding prompt + a relevance threshold
  on retrieval + a fixed refusal string; sources shown on every answer.
- **What if Gemini is down?** The app retries, then falls back to an offline
  extractive responder that quotes the notes — degraded, but it never invents.
- **Why Gemini?** Free tier, OpenAI-compatible, so swapping providers is an
  environment variable. Ollama gives a fully offline option.
- **How does the memory decide what "it" means?** It rewrites the follow-up
  using topic words from recent turns before embedding it; the rewritten query
  is what gets searched.

### Member 3 — reminders, deployment, fine-tuning

- **Why not let the LLM compute deadlines?** Date arithmetic is a known LLM
  weakness; code is exact, testable, and needs no API.
- **What's in the container?** Two-stage build on Python 3.11-slim, non-root
  user, healthcheck, named volume for the index, read-only notes mount, keys
  injected at runtime.
- **When *would* you fine-tune?** To change behaviour — tone, output format,
  a domain's style — with ≥1,000 verified examples. Never to store facts.

---

## Rehearsal checklist

- [ ] Each member reads their segment aloud with a timer; adjust to fit
- [ ] Full run-through on the real stack — confirm questions 1–5 behave as the
      script describes. **If an answer differs, change the narration to match
      what the system actually does**, not the other way round
- [ ] Question 4 and the refusal especially: they have only been verified on
      the fallback
- [ ] API key, `.env` and terminal history never on screen
- [ ] Final cut ≤ 10:00
