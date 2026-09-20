# Checkpoint 4 — Fine-Tuning Analysis (LoRA / QLoRA)

**Project**: Personal Assistant AI — Generative RAG Capstone
**Course**: IS Professional Elective #4 — Generative AI Systems
**Instructor**: Jessie A. Melendres

**Reproduce the numbers**: `python src/finetuning/vram_calculator.py`

---

## 1. The Question

Should this project fine-tune a language model, and if so, with what method on
what hardware?

Two sub-questions, answered in order, because the second only matters if the
first says yes:

1. **Would fine-tuning improve this system?**
2. **Could it be done on hardware a student can reach?**

---

## 2. Why Parameter-Efficient Fine-Tuning Exists

Full fine-tuning updates every weight. The memory cost is not just the weights
— it is dominated by the **optimizer state**. Adam keeps two moment tensors per
trainable parameter, so at fp32 that is 8 bytes per parameter on top of
gradients.

| Term | Cost |
|---|---|
| Weights | params × 2 bytes (fp16) |
| Gradients | trainable × 4 bytes |
| Adam moments | trainable × 8 bytes |
| Activations | grows with batch × sequence × depth |

**LoRA** freezes the base model and trains small low-rank adapters. A weight
update `ΔW` of shape *d × k* is replaced by `B @ A` where `A` is *r × k* and
`B` is *d × r*. With rank 16, roughly 1% of parameters become trainable — so
gradients and optimizer state shrink by ~99%.

**QLoRA** adds one idea: quantise the *frozen* base to 4 bits. Since it is
never updated, precision matters far less. This attacks the one term LoRA
leaves untouched — the weights themselves.

---

## 3. Measured Memory Requirements

Estimates from `src/finetuning/vram_calculator.py` (batch 4, sequence 512,
LoRA rank 16). These are first-order: they ignore gradient checkpointing and
ZeRO sharding, so treat them as a floor.

### Mistral-7B — the realistic candidate

| Strategy | Trainable | Weights | Optimizer | **Total** | Fits on |
|---|---|---|---|---|---|
| Full | 100% | 13.5 GB | 53.9 GB | **94.9 GB** | nothing under 80 GB |
| LoRA | 1.12% | 13.5 GB | 0.6 GB | **14.9 GB** | Colab free (T4) |
| QLoRA | 1.12% | 3.4 GB | 0.6 GB | **4.8 GB** | Colab free (T4) |

Full fine-tuning needs ~95 GB — beyond a single A100 80GB. LoRA cuts that to
14.9 GB, a **6.4×** reduction, purely by removing optimizer state. QLoRA
reaches 4.8 GB, a **20×** reduction overall.

### Scaling across model sizes

| Model | Full | LoRA | QLoRA |
|---|---|---|---|
| Llama-3.2-1B | 16.4 GB | 2.7 GB | 0.9 GB |
| Llama-3.2-3B | 42.2 GB | 6.7 GB | 2.2 GB |
| Mistral-7B | 94.9 GB | 14.9 GB | 4.8 GB |
| Llama-3.1-8B | 105.2 GB | 16.5 GB | 5.3 GB |
| Llama-3.1-70B | 922.1 GB | 141.9 GB | 43.3 GB |

The 70B row is the clearest illustration: full fine-tuning needs **922 GB** —
a multi-node cluster. QLoRA brings it to 43.3 GB, one A100 80GB. That is the
difference between "requires a research lab" and "requires a rented GPU."

**Answer to sub-question 2**: yes, QLoRA makes fine-tuning reachable. A 7B
model trains on a free Colab T4.

---

## 4. Would It Actually Help This Project?

This is the decisive question, and the answer is **no** — for two reasons.

### 4.1 Fine-tuning teaches behaviour, not facts

The distinction matters. Fine-tuning adjusts *how* a model responds: tone,
format, structure, domain vocabulary. It is a poor mechanism for *what* it
knows, because facts baked into weights are:

- **Stale** — a new note means retraining
- **Unciteable** — no `[doc_id]` to point at
- **Unverifiable** — no way to check an answer against a source

This assistant's entire job is factual recall over documents that change. That
is precisely what retrieval does well and fine-tuning does badly. Checkpoint 2
built citations and a refusal path specifically so answers stay auditable;
moving knowledge into weights would discard both.

### 4.2 There is not enough data

Instruction fine-tuning needs thousands of high-quality (prompt, response)
pairs. Module 1 Lesson 3 describes exactly this: human contractors writing
ideal responses across many tasks.

This project has **26 documents**. Even generating several examples per
document yields low hundreds — an order of magnitude short, and self-generated
pairs would teach the model to imitate its own current behaviour rather than
improve it.

**Answer to sub-question 1**: fine-tuning would add cost and staleness while
removing citations, to solve a problem retrieval already solves.

---

## 5. Where Fine-Tuning *Would* Help

Declining it in general does not mean it has no use here. Three cases where it
would be the right tool, none of which this project currently needs:

| Scenario | Why fine-tuning wins |
|---|---|
| **Consistent output format** | `schedule_extractor` v3 needed three prompt revisions to return parseable JSON. Fine-tuning on a few hundred examples would make the format structural rather than instructed. |
| **Domain vocabulary** | If the corpus were medical or legal, a base model may mis-tokenise terminology that retrieval then fails to match. |
| **Shrinking the prompt** | The grounded_qa system prompt costs ~700 characters on every call. Fine-tuned behaviour would cut that recurring cost at scale. |

Of these, **output format** is the strongest candidate — and notably it is a
*behaviour* problem, not a knowledge one. That is the consistent line: fine-tune
for behaviour, retrieve for facts.

---

## 6. The Recommendation

**Do not fine-tune. Keep RAG.** If a future version needed it, the
configuration would be:

| Decision | Choice | Reason |
|---|---|---|
| Method | **QLoRA** | 4.8 GB for 7B — fits a free Colab T4 |
| Base model | Mistral-7B or Llama-3.1-8B | Open weights, commercially usable |
| Rank | 16 | Standard; 8 if memory-bound, 32 for more capacity |
| Target modules | Attention q/k/v/o + MLP | Where adapters are most effective |
| Task | **Structured extraction**, not Q&A | Behaviour, not knowledge |
| Data | ≥1,000 verified pairs | Far beyond what 26 documents yield |

The trigger for revisiting: a corpus large enough to generate real training
data, or a measured failure that prompting cannot fix.

---

## 7. Limitations of This Analysis

1. **No fine-tuning was performed.** No GPU was available, and
   `huggingface.co` is blocked in this environment, so no base model could be
   downloaded. This is analysis and arithmetic, not an experiment.
2. **The estimates are first-order.** Gradient checkpointing, ZeRO, flash
   attention and fused optimizers all shift real usage — generally downward.
   Verify against `nvidia-smi` before renting hardware.
3. **The LoRA parameter approximation is coarse.** Exact adapter counts depend
   on which modules are targeted and the model's specific layer widths.
4. **No quality comparison exists.** The claim that RAG beats fine-tuning
   *for this task* rests on the argument in §4, not on a measured A/B test.
   Running one would need the data §4.2 says does not exist.

---

## 8. Reflection

The tempting answer to "should we fine-tune?" is yes — it is the more advanced
technique, and the checkpoint asks about it. Working through the arithmetic
produced a more useful answer: QLoRA makes it *possible* (4.8 GB, a free
Colab tier), and it is still the wrong tool here.

What settled it was separating **knowledge** from **behaviour**. Fine-tuning
moves information into weights, where it cannot be cited, checked, or updated
without retraining. Every deliberate design choice in this project ran the
other way — citations in Checkpoint 2, a reachable refusal, date arithmetic
kept out of the model in the reminder component. Baking the notes into weights
would undo all of it.

That is the same principle a third time: the system is more reliable when the
model is given less to decide, and when what it produces can be checked against
something outside itself.
