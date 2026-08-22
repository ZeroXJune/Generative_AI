# Checkpoint 2 — Prompt Engineering

**Project**: Personal Assistant AI — Generative RAG Capstone
**Course**: IS Professional Elective #4 — Generative AI Systems
**Instructor**: Jessie A. Melendres
**Implementation**: [`src/prompts/system_prompts.py`](../src/prompts/system_prompts.py)

---

## 1. Design Principles

Every prompt in this system follows four rules, drawn from Module 1 Lesson 3
(*Introduction to Large Language Models*):

1. **Instruct, don't continue.** Instruction-tuned models answer a stated task;
   base models continue text. Each prompt opens with an explicit role and task
   rather than a fragment the model might try to complete.
2. **Forbid parametric memory.** Hallucination is the dominant failure mode when
   an LLM answers from its own weights. Every retrieval prompt bans outside
   knowledge and supplies an explicit refusal string, so *"I don't know"* is a
   success state rather than something the model avoids.
3. **Enumerate constraints.** LLMs are brittle to wording. Numbered rules
   outperform prose adjectives like "be accurate" or "format nicely".
4. **Budget the context window.** Prompts stay short so token budget goes to
   retrieved evidence, not instructions.

Each prompt is a `SystemPrompt` dataclass carrying its version, temperature,
the failure mode it addresses, and its design notes — the rationale lives next
to the text rather than only in this document.

---

## 2. The Four System Prompts

### 2.1 `grounded_qa` (v2) — core RAG answering

**Temperature**: 0.0 — factual recall should be reproducible, not creative.

**Failure mode addressed.** v1 answered from the model's pretraining knowledge
whenever retrieval returned weak context, producing confident but unsupported
claims about the user's own documents.

| Rule | Purpose |
|---|---|
| Use only the CONTEXT | Isolates the model from parametric memory |
| Exact refusal string | Makes "I don't know" a first-class outcome |
| Cite every `[doc_id]` | Makes answers auditable without reading the corpus |
| Quote dates/numbers verbatim | Prevents silent rounding of deadlines |
| Sentence cap | Keeps answers scannable on mobile |

**Verified behaviour.** Asked *"What is the WiFi password for the campus
network?"* — a fact absent from all 26 documents — the system returns exactly
`I don't have that in your notes.` rather than inventing one.

### 2.2 `schedule_extractor` (v3) — structured output

**Temperature**: 0.0

**Failure mode addressed.** v1 returned prose. v2 returned JSON wrapped in
markdown fences with commentary, so `json.loads()` failed roughly a third of
the time and the notification scheduler crashed.

The v3 fixes:
- The JSON schema is stated **literally**, not described in prose.
- A **one-shot example** fixes the output shape — Lesson 3 notes that few-shot
  examples steer format far more reliably than adjectives.
- `Return [] if none` stops the model inventing a placeholder entry.
- Markdown fences are **explicitly forbidden** — the single most common cause
  of parse failure.

The example deliberately includes an undated reminder ("email Sir Melendres
sometime soon") that must be *excluded*, teaching the rule by demonstration.

### 2.3 `study_summarizer` (v2) — revision material

**Temperature**: 0.3 — summarisation benefits from phrasing freedom, unlike
factual lookup.

**Failure mode addressed.** v1 produced fluent summaries that silently merged
the model's own knowledge with the student's notes, so revising from the
summary taught material the course never covered.

The **Gaps** section is the key fix: it turns missing information into a
visible output rather than something the model quietly fills in.

### 2.4 `query_rewriter` (v1) — conversational retrieval

**Temperature**: 0.0

**Failure mode addressed.** In multi-turn chat, a follow-up such as *"when is
it due?"* embeds to a vector with almost no topical content, so vector search
returns unrelated chunks **regardless of which distance metric is used**.

This is a *search-quality* fix, not an answer-quality fix — it runs **before**
retrieval. It returns the question unchanged when already standalone, so it
cannot corrupt good queries. This prompt is the foundation for the
conversational memory required in Checkpoint 3.

---

## 3. Prompt Summary

| Prompt | Ver | Temp | Task | Primary failure mode addressed |
|---|---|---|---|---|
| `grounded_qa` | v2 | 0.0 | Answer from notes | Hallucination |
| `schedule_extractor` | v3 | 0.0 | Extract deadlines as JSON | Unparseable output |
| `study_summarizer` | v2 | 0.3 | Build revision material | Content drift |
| `query_rewriter` | v1 | 0.0 | Rewrite follow-ups | Retrieval collapse in chat |

---

## 4. Message Assembly

`build_rag_messages()` assembles a Chat Completion payload:

```python
[
  {"role": "system", "content": "<system prompt with placeholders filled>"},
  ...conversation history...,
  {"role": "user", "content": "CONTEXT:\n[doc_id] passage...\n\nQUESTION: ..."}
]
```

Context passages are tagged with their `doc_id` so the model can cite them, and
`format_context()` enforces a character budget — chunks are dropped once the
budget is spent, so a long retrieval set can never overrun the context window.

---

## 5. Reflection

The clearest lesson from this checkpoint is that **most prompt failures are
format failures, not reasoning failures**. The `schedule_extractor` never
struggled to *find* the dates; it struggled to return them in a shape the
scheduler could parse. Two cheap changes — a literal schema and one worked
example — fixed what three rounds of prose instructions could not.

The second lesson is that a refusal path must be designed, not assumed. Until
`grounded_qa` v2 supplied an exact refusal string *and* the retriever filtered
weak matches, the model treated answering as mandatory and filled gaps from its
own knowledge. Making refusal easy is what makes grounding real.
