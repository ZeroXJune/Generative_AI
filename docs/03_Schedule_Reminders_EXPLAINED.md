# Explained: Schedule Tracking & Deadline Reminders

**Companion to**: [`docs/03_Schedule_Reminders.md`](03_Schedule_Reminders.md)

This document walks through **every section** of the official document and
explains what it says, why it says it, and what you should be able to answer
if asked about it during the defense. Read the official document first; read
this one when a section is unclear or when preparing to defend it.

---

## How to read the official document

It has 8 sections, following the standard shape of a technical feature
document:

| § | Section | Question it answers |
|---|---|---|
| 1 | Objective | *Why does this exist?* |
| 2 | Design Principle | *What is the one big decision?* |
| 3 | Architecture | *How does data move through it?* |
| 4 | Components | *What are the parts?* |
| 5 | Usage | *How do I run it?* |
| 6 | Verified Results | *What is the proof it works?* |
| 7 | Limitations | *Where does it fall short?* |
| 8 | Reflection | *What did I learn?* |

Sections 1–2 are the argument. Sections 3–5 are the build. Sections 6–8 are
the evidence and the honest accounting.

---

## § 1 — Objective, explained

**What it contains**: a table with five rows, each quoting a line from your
original project proposal and giving its line number.

**Why it is written that way**: this is *traceability*. Rather than asserting
"this feature was required," it shows the exact sentences in your own proposal
that required it. A marker can verify each claim in seconds.

**The key admission**: the last paragraph states plainly that until now the
system could answer questions *about* a deadline but could not *tell you one
was approaching*, and that the `schedule_extractor` prompt from Checkpoint 2
existed but nothing used it.

> **Defense prep**: If asked "why wasn't this in Checkpoint 1 or 2?" — the
> honest answer is that neither checkpoint's rubric required it. CP1 was the
> data pipeline, CP2 was retrieval and prompts. The feature fell into the gap
> between two rubrics. That is a scoping observation, not an excuse.

---

## § 2 — Design Principle, explained

**This is the most important section in the document.** If you only defend one
section, defend this one.

**The claim**: the LLM is deliberately *not allowed* to decide whether a
deadline is approaching.

**The reasoning chain**:

1. Your Lesson 3 notes list "unreliable arithmetic and formal logic" as a
   known LLM limitation.
2. Deciding if a deadline is near is arithmetic (`deadline − today`).
3. Your proposal promises **100% on-time alerts** — a guarantee.
4. You cannot guarantee something an unreliable component performs.
5. Therefore the arithmetic must happen in Python, not in the model.

**The four-row responsibility table** is the design in miniature:

| Step | Who does it | Plain-language reason |
|---|---|---|
| Find dates in text | Regex or LLM | Recognising language is what models are *good* at |
| Validate the date | Python | "February 30" must be rejected every time |
| Decide "approaching" | Python | Must be exact, must be reproducible |
| Show the alert | CLI / UI | Just presentation |

**The consequence sentence** is the payoff: reminders work with no API key, no
network, and no model. Same input → same alert, always.

> **Defense prep**: A likely challenge is *"Isn't this supposed to be a
> generative AI project? Why is the key feature not using AI?"*
> Answer: the AI does the part AI is good at — reading messy human text and
> recognising that a date is present. What it does not do is the part it is
> measurably bad at. Using a model everywhere would make the system *more*
> AI-branded and *less* reliable. Choosing where **not** to use the model is
> itself an engineering decision about the model's known limitations.

---

## § 3 — Architecture, explained

**What it contains**: an ASCII diagram tracing text from `data/raw/` down to
the two output surfaces, plus a subsection titled "Why extraction runs on RAW
text."

**Reading the diagram**: it is a straight pipeline, not a loop. Each box hands
its output to the next:

```
raw text → extractor → date_parser → store → ReminderEngine → CLI / Streamlit
```

**The subtle part — why RAW text?**

Everywhere else in this project, documents pass through the Checkpoint 1
`TextCleaner` first. This component deliberately does not. Here is why, using
a real line from your corpus:

| Stage | The line |
|---|---|
| Original | `Capstone Checkpoint 2 (Midterm): September 19, 2026, 11:59 PM` |
| After cleaning | `capstone checkpoint 2 midterm september 19 2026 11 59 pm` |

Cleaning collapses newlines (so line structure — which separates one event
from the next — is gone) and strips commas (so `September 19, 2026` loses the
comma the date format relies on). Extraction would become far harder for no
benefit.

> **Defense prep**: If asked "isn't bypassing your own pipeline inconsistent?"
> — the pipeline exists to prepare text for *embedding*, where lowercasing and
> punctuation removal help. Date extraction has the opposite needs: it depends
> on exactly the structure cleaning removes. Different task, different
> preprocessing. The document flags this as "the one place in the system that
> deliberately bypasses the cleaning pipeline" so it reads as a decision, not
> an oversight.

---

## § 4 — Components, explained

### The file table

Six files, each with one job. The separation matters: `date_parser` knows
nothing about deadlines, `reminders` knows nothing about files. Each can be
tested alone.

### § 4.1 — Supported date formats

Six formats, each with a worked example. Two rows deserve attention:

- **Date range** — `November 17-21, 2026` returns the **start** date
  (2026-11-17). For an exam *period*, the date you must be ready by is the
  first one.
- **Rejected by design** — `February 30, 2026` and `sometime next week` both
  return nothing.

**The sentence to remember**: *"a wrong deadline is worse than a missing one,
because the user acts on it."* This is the whole philosophy of the component
in one line. A missed reminder is a gap; a false reminder actively misleads.

### § 4.2 — Extraction filters

Not every date in a document is a commitment. This is the least obvious part
of the feature, and the four filters each fix a real failure observed during
development:

| Filter | The actual bug it fixed |
|---|---|
| Non-event keywords | `Founder's Day` was being reported as a deadline to meet |
| Section inheritance | Items under a `Holidays:` header inherit that meaning even though the word "holiday" is not on their own line |
| Metadata titles | `Date: July 14, 2026` at the top of a note dates the *document*, not a task |
| Quoted examples | `meeting_notes_july14` quotes before/after text-cleaning samples containing real dates — those describe CP1's preprocessing, not commitments |

**The number to quote**: filters took the raw extraction from **28 candidates
down to 20 genuine commitments**. That 8-item difference is the filters
earning their place.

### § 4.3 — Urgency bands

Five bands from `overdue` to `upcoming`. The CLI shows them as text markers
(`!!`, `!`, `>`) because a terminal has no colour guarantee; the web UI shows
them as colours. Same data, two presentations.

---

## § 5 — Usage, explained

Six commands. Five are self-explanatory; one needs justification:

**`--as-of 2026-09-15`** overrides what the system treats as "today."

This exists for *demonstrations*. Without it, running the demo on a different
day produces different output, and a recorded demo cannot be reproduced. With
it, the output is identical forever. It is the same reasoning that made
`ReminderEngine` accept an injectable `today` parameter instead of always
calling `date.today()` internally.

> **Defense prep**: "Isn't overriding the date faking your results?" — no,
> because the flag is documented, optional, and off by default. Section 6
> shows output from *both* the real date and an overridden one.

---

## § 6 — Verified Results, explained

**What it contains**: two real CLI transcripts, one Streamlit verification,
and an assessment against the proposal's success metric.

**Why two transcripts**: the run on the real date (2026-08-28) shows the
system working *now*. The `--as-of 2026-09-15` run demonstrates something the
first cannot — that Checkpoint 2's own deadline is correctly surfaced as
"in 4 days" when the reference date is moved near it. One proves it runs; the
other proves the logic generalises across time.

**The Streamlit line**: verified with `streamlit.testing.v1.AppTest`, which
actually executes the app and reports rendered values — not just "the file has
no syntax errors." `0 exceptions` plus the three rendered metric values is
real evidence.

**The success-metric paragraph — read this carefully.** It says the metric is
met *"for extracted deadlines"*, and that the residual risk "sits entirely in
extraction recall." That qualifier is deliberate and important:

- Alerting is **provably** 100% reliable — it is subtraction.
- Whether every deadline in your notes *reached* the store is **not** proven.

Claiming "100% on-time alerts" without that qualifier would overstate the
result. The document states what was actually demonstrated.

---

## § 7 — Limitations, explained

Five honest gaps. In order of how likely they are to be asked about:

1. **Recall is not measured** — 20 found, all 20 correct (precision 1.00), but
   nobody labelled what was *missed*. Precision and recall are different
   things and the document does not conflate them.
2. **Relative dates ignored** — "next Friday" needs a reference date, and the
   document's authoring date is not reliably known. Guessing would violate the
   "wrong is worse than missing" rule from §4.1.
3. **No push delivery** — alerts are *pull-based*: you see them when you run
   the CLI or open the app. True push needs a scheduled job (cron), which is
   deployment work and belongs to Checkpoint 4.
4. **No recurring events** — "every Tuesday 2:00 PM" is not expanded.
5. **LLM path untested end to end** — `extract_with_llm()` is written and its
   output is re-validated by the parser, but no API key was available, so only
   the deterministic path was actually exercised.

> **Defense prep**: Listing limitations is not weakness — an examiner who
> finds an unlisted gap trusts the whole document less. Every item here is one
> you can now answer calmly, because you raised it first. Limitation 3 in
> particular is a natural "future work" answer.

---

## § 8 — Reflection, explained

**The honest story**: the first instinct was to hand the entire task to the
model — read the notes, work out what is due, tell the user. Checking that
design against the proposal's own success metric killed it, because a
guarantee cannot rest on a component that is unreliable at arithmetic.

**The transferable lesson**, stated in the last paragraph: this is the *same*
pattern as the Checkpoint 2 refusal finding. In both cases, reliable behaviour
came from **constraining what the model is allowed to decide**, not from
writing a better prompt.

That connection between two checkpoints is the strongest single point you can
make in a defense — it shows a principle you discovered twice independently,
rather than an isolated implementation detail.

---

## Quick answers to likely questions

| Question | Short answer |
|---|---|
| Where does the AI actually appear? | Recognising dates in messy prose (`extract_with_llm`), and answering questions about schedules via RAG. Not in the date arithmetic. |
| Why 20 deadlines and not 28? | Four filters removed holidays, document metadata, and dates quoted inside cleaning examples. |
| What happens to an unparseable date? | It is dropped. Wrong deadlines are worse than missing ones. |
| Does this need an API key? | No. The reminder path is fully deterministic and offline. |
| How is "approaching" defined? | Five bands: overdue, today, ≤3 days, ≤7 days, ≤14 days. Horizon is configurable via `--days`. |
| Is the 100% metric met? | Yes for alerting on stored deadlines. Extraction recall is unmeasured and stated as such. |
| What is the biggest weakness? | Unmeasured recall — no labelled ground truth for deadlines the extractor missed. |
