# Schedule Tracking & Deadline Reminders

**Project**: Personal Assistant AI — Generative RAG Capstone
**Course**: IS Professional Elective #4 — Generative AI Systems
**Instructor**: Jessie A. Melendres
**Status**: Implemented
**Date**: August 28, 2026

---

## 1. Objective

This component delivers the schedule-awareness objective stated in the
original project proposal (`docs/01_Project_Proposal.md`):

| Proposal reference | Commitment |
|---|---|
| Line 14 | "…while **proactively notifying users of time-sensitive items**?" |
| Line 35 | "**Tracks** schedules and deadlines with automatic notifications" |
| Line 44 | "**Notification component**: Adds practical value beyond typical Q&A bots" |
| Line 114 | Success metric: "Notification reliability: **100% on-time alerts**" |
| Line 155 | Architecture stage: `[Notification Check] → Alert if deadline detected` |

Until now the system could *answer questions about* a deadline but could not
*tell the user one was approaching*. The `schedule_extractor` prompt existed
from Checkpoint 2, but nothing consumed its output. This component closes
that gap.

---

## 2. Design Principle: Dates Are Arithmetic, Not Language

The single most important decision in this component is what the LLM is
**not** allowed to do.

Module 1 Lesson 3 lists "unreliable arithmetic and formal logic" among the
known limitations of Large Language Models. Deciding whether a deadline is
approaching is date arithmetic. If an LLM performed that comparison, the
proposal's "100% on-time alerts" metric would depend on model behaviour, and
could never be guaranteed.

The responsibilities are therefore split:

| Step | Performed by | Why |
|---|---|---|
| Find candidate dates in text | Regex, or optionally an LLM | Language recognition — models are good at this |
| Validate the date is real | `date_parser` (Python) | Deterministic; rejects "February 30" |
| Decide what is "approaching" | `ReminderEngine` (Python) | Arithmetic must be exact and reproducible |
| Present the alert | CLI / Streamlit | Presentation only |

**Consequence**: reminder correctness does not depend on an LLM, an API key,
or a network connection. The same inputs always produce the same alerts.

---

## 3. Architecture

```
data/raw/*.txt  (raw text, NOT cleaned)
        │
        ▼
┌───────────────────────┐
│ extractor.py          │  line-by-line scan
│  · find date          │  · skip holidays / metadata / quoted examples
│  · derive title       │  · classify type (exam/meeting/deadline/task)
└───────────┬───────────┘
            ▼
┌───────────────────────┐
│ date_parser.py        │  "September 19, 2026, 11:59 PM"
│                       │      → date(2026, 9, 19), "23:59"
└───────────┬───────────┘
            ▼
┌───────────────────────┐
│ store.py              │  data/processed/deadlines.json
└───────────┬───────────┘
            ▼
┌───────────────────────┐
│ reminders.py          │  days_until = deadline − today
│ ReminderEngine        │  → overdue / today / urgent / soon / upcoming
└───────────┬───────────┘
            ▼
   ┌────────┴────────┐
   ▼                 ▼
CLI digest      Streamlit panel
```

### Why extraction runs on RAW text

The Checkpoint 1 `TextCleaner` collapses newlines and strips commas. Applied
to `Capstone Checkpoint 2 (Midterm): September 19, 2026, 11:59 PM`, it
destroys both the line structure the title depends on and the comma in the
date. Extraction therefore reads the original files directly. This is the
one place in the system that deliberately bypasses the cleaning pipeline.

---

## 4. Components

| File | Responsibility |
|---|---|
| `src/schedule/date_parser.py` | Parse dates and times; stdlib only |
| `src/schedule/extractor.py` | Find dated commitments; `Deadline` dataclass |
| `src/schedule/store.py` | Persist to `data/processed/deadlines.json` |
| `src/schedule/reminders.py` | `ReminderEngine`; urgency classification |
| `src/reminders.py` | Command-line interface |
| `src/interface/app.py` | Streamlit "Upcoming Deadlines" panel |

### 4.1 Supported date formats

| Format | Example | Result |
|---|---|---|
| Month DD, YYYY | `September 19, 2026` | `2026-09-19` |
| Abbreviated | `Nov 14, 2026` | `2026-11-14` |
| ISO | `2026-10-17` | `2026-10-17` |
| US numeric | `09/19/2026`, `3/5/26` | `2026-09-19`, `2026-03-05` |
| Date range | `November 17-21, 2026` | `2026-11-17` (start) |
| With time | `…, 11:59 PM` | time `23:59` |

Rejected by design: `February 30, 2026` (not a real date) and
`sometime next week` (relative, no stated date). A date that cannot be parsed
with confidence is **dropped, not guessed** — a wrong deadline is worse than a
missing one, because the user acts on it.

### 4.2 Extraction filters

Not every date in a document is a commitment. Four filters apply:

| Filter | Rejects | Example |
|---|---|---|
| Non-event keywords | Calendar facts | `Founder's Day: August 21, 2026` |
| Section inheritance | Entries under a `Holidays:` header | `- National Holiday: September 9, 2026` |
| Metadata titles | Document headers | `Date: July 14, 2026` |
| Quoted examples | Text quoted as a sample | `Raw: "Capstone Checkpoint 1…"` |

These filters reduced the raw extraction from 28 candidates to **20 genuine
commitments** across the 26-document corpus.

### 4.3 Urgency bands

| Band | Condition | CLI marker | UI colour |
|---|---|---|---|
| `overdue` | date < today | `!!` | 🔴 red |
| `today` | date == today | `**` | 🟠 orange |
| `urgent` | ≤ 3 days | `!` | 🟠 orange |
| `soon` | ≤ 7 days | `>` | 🟡 yellow |
| `upcoming` | ≤ 14 days | ` ` | 🟢 green |

---

## 5. Usage

```bash
python src/reminders.py                      # digest, next 14 days
python src/reminders.py --days 30            # widen the horizon
python src/reminders.py --refresh            # re-scan data/raw first
python src/reminders.py --json               # machine-readable output
python src/reminders.py --as-of 2026-09-15   # reproducible demo output
streamlit run src/interface/app.py           # web panel
```

The `--as-of` flag exists so a demonstration produces identical output on any
day. Without it, output naturally changes as the real date moves.

---

## 6. Verified Results

Run on **2026-08-28** against the live 26-document corpus:

```
DEADLINE DIGEST - as of 2026-08-28

OVERDUE (6)
  !! 2026-08-22  6 day(s) OVERDUE     Web Dev Project Phase 1
  !! 2026-08-18  10 day(s) OVERDUE    Prelim Exam Period
  !! 2026-08-15 at 23:59  13 day(s) OVERDUE    Checklist Submission
  !! 2026-08-15 at 23:59  13 day(s) OVERDUE    Capstone Checkpoint 1 (Prelim)
  !! 2026-08-08  20 day(s) OVERDUE    Data Structures Project 1
  !! 2026-07-21  38 day(s) OVERDUE    Next Meeting: (after data collection)

NEXT 14 DAYS (2)
  !  2026-08-29  tomorrow             Data Structures Project 2
  !  2026-08-31  in 3 days            Database Design Assignment
```

Shifting the reference date to 2026-09-15 correctly surfaces Checkpoint 2:

```
NEXT 40 DAYS (7)
  >  2026-09-19 at 23:59  in 4 days   Capstone Checkpoint 2 (Midterm)
  >  2026-09-22           in 7 days   Midterm Exam Period
     2026-09-26           in 11 days  Data Structures Final Project
     ...
```

**Streamlit panel** — verified with `streamlit.testing.v1.AppTest`:
0 exceptions; metrics render `Overdue = 6`, `Next 14 days = 2`,
`Next deadline = 2026-08-29`.

### Against the proposal's success metric

The proposal asks for **100% on-time alerts**. Because the comparison is
`deadline.date - today` in Python, any deadline present in the store is
reported in the correct band, every run. The metric is therefore met *for
extracted deadlines*, and the residual risk sits entirely in extraction
recall — see below.

---

## 7. Limitations

1. **Extraction recall is not measured.** 20 commitments were found and all 20
   were manually confirmed correct (precision 1.00 on this corpus), but no
   labelled ground truth exists for what was *missed*. Recall is unquantified.
2. **Relative dates are ignored.** "Next Friday" and "in two weeks" are
   deliberately not resolved, because doing so requires assuming a reference
   date that may not be the document's authoring date.
3. **No push delivery.** Alerts are pull-based — shown when the CLI is run or
   the app is opened. Email/OS notification would need a scheduled job, which
   is deployment work (Checkpoint 4).
4. **No recurring events.** "Every Tuesday 2:00 PM" is not expanded into
   instances; only fixed dates are tracked.
5. **The LLM path is untested end to end.** `extract_with_llm()` is
   implemented and its output is re-validated, but no API key was available in
   this environment, so only the deterministic path has been exercised.

---

## 8. Reflection

The instructive part of building this was discovering how much of a
"generative AI feature" turned out not to be generative. The proposal frames
notifications as an AI capability, and the instinct was to hand the whole task
to the model: read the notes, work out what is due soon, tell the user.

Measuring against the stated success metric killed that design. "100% on-time
alerts" is a guarantee, and an LLM cannot offer a guarantee about arithmetic
it performs in natural language. Splitting the work — model for *recognition*,
Python for *comparison* — turns an aspiration into a property of the system.

This mirrors the Checkpoint 2 finding about refusal: in both cases the
reliable behaviour came from constraining what the model was allowed to
decide, not from prompting it more carefully.
