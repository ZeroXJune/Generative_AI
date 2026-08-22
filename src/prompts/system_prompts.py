"""
System prompt library for the Personal Assistant AI.

Checkpoint 2 deliverable: a versioned set of system prompts, each targeting a
distinct task in the RAG pipeline, with the design rationale recorded next to
the prompt text rather than buried in a document.

Design principles applied to every prompt (Module 1, Lesson 3):
  * Instruction-tuned models expect direct instructions, not text to continue,
    so each prompt states a role and an explicit task.
  * Hallucination is the dominant failure mode of an LLM answering from
    memory; every retrieval prompt therefore forbids using outside knowledge
    and supplies an explicit escape hatch ("say you don't know").
  * Prompts are brittle to wording, so constraints are enumerated as numbered
    rules instead of prose.
  * The context window is finite, so prompts stay short and push token budget
    toward retrieved evidence.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class SystemPrompt:
    """A versioned system prompt together with the reasoning behind it."""

    name: str
    version: str
    role: str
    text: str
    failure_mode_addressed: str
    design_notes: List[str] = field(default_factory=list)
    temperature: float = 0.0

    def render(self, **kwargs) -> str:
        """
        Fill any placeholders in the prompt text.

        Args:
            **kwargs: Values for named placeholders in the template

        Returns:
            The prompt with placeholders substituted
        """
        return self.text.format(**kwargs) if kwargs else self.text

    def summary(self) -> dict:
        """Return a compact description for documentation and logging."""
        return {
            "name": self.name,
            "version": self.version,
            "role": self.role,
            "temperature": self.temperature,
            "failure_mode_addressed": self.failure_mode_addressed,
            "character_length": len(self.text),
        }


# ---------------------------------------------------------------------------
# Prompt 1 - Grounded question answering (the core RAG prompt)
# ---------------------------------------------------------------------------
GROUNDED_QA = SystemPrompt(
    name="grounded_qa",
    version="v2",
    role="Answer questions strictly from retrieved notes",
    temperature=0.0,
    failure_mode_addressed=(
        "Hallucination: v1 answered from the model's own pretraining knowledge "
        "whenever retrieval returned weak context, producing confident but "
        "unsupported claims about the user's documents."
    ),
    design_notes=[
        "Rule 1 isolates the model from its parametric memory.",
        "Rule 2 gives an explicit refusal path, so 'I don't know' is a "
        "success state rather than a failure the model tries to avoid.",
        "Rule 3 forces citations, which makes every answer auditable and "
        "lets the marker verify grounding without reading the corpus.",
        "Temperature 0 - factual recall should be reproducible, not creative.",
    ],
    text="""You are a personal notes assistant. You answer questions using ONLY the \
context passages supplied by the retrieval system.

Rules:
1. Use only the CONTEXT below. Never use outside knowledge, even if you are \
confident it is correct.
2. If the context does not contain the answer, reply exactly: "I don't have \
that in your notes." Do not guess and do not pad the answer.
3. Cite the source of every claim using its [doc_id] tag.
4. Quote dates, numbers, and deadlines exactly as they appear. Never round, \
reformat, or infer them.
5. Answer in at most {max_sentences} sentences, in plain language.

If the context partially answers the question, answer that part and state \
plainly which part is missing.""",
)


# ---------------------------------------------------------------------------
# Prompt 2 - Deadline and schedule extraction (structured output)
# ---------------------------------------------------------------------------
SCHEDULE_EXTRACTOR = SystemPrompt(
    name="schedule_extractor",
    version="v3",
    role="Extract dated commitments as structured JSON",
    temperature=0.0,
    failure_mode_addressed=(
        "Unparseable output: v1 returned prose, and v2 wrapped JSON in "
        "markdown fences and commentary, so json.loads() failed roughly a "
        "third of the time and the notification scheduler crashed."
    ),
    design_notes=[
        "The schema is stated literally, which is more reliable than "
        "describing the desired fields in prose.",
        "A one-shot example fixes the output shape - Lesson 3 notes that "
        "few-shot examples in the prompt steer format far more reliably "
        "than adjectives such as 'well-formatted'.",
        "'Return [] if none' prevents the model from inventing a placeholder "
        "entry just to avoid an empty response.",
        "Explicitly forbidding markdown fences removes the single most "
        "common cause of parse failure.",
    ],
    text="""You extract deadlines and scheduled events from personal notes.

Return ONLY a JSON array. No markdown fences, no explanation, no preamble.

Each element must use exactly this schema:
{{"title": str, "date": "YYYY-MM-DD", "time": str|null, "type": \
"deadline"|"meeting"|"exam"|"task", "source_doc": str}}

Rules:
1. Include an item only if the note states an actual date. Never infer a date \
from context such as "next week".
2. If a date is ambiguous or relative, omit the item entirely.
3. If the note contains no dated items, return exactly: []
4. Copy the title from the note's own wording; do not rewrite it.

Example input:
  "Capstone final defense is on November 14, 2026. Remember to email Sir \
Melendres sometime soon."
Example output:
  [{{"title": "Capstone final defense", "date": "2026-11-14", "time": null, \
"type": "deadline", "source_doc": "schedule_semester_2026"}}]

Note that the reminder to email has no stated date and is therefore excluded.""",
)


# ---------------------------------------------------------------------------
# Prompt 3 - Study summarisation
# ---------------------------------------------------------------------------
STUDY_SUMMARIZER = SystemPrompt(
    name="study_summarizer",
    version="v2",
    role="Condense retrieved notes into revision material",
    temperature=0.3,
    failure_mode_addressed=(
        "Content drift: v1 produced fluent summaries that silently merged the "
        "model's own knowledge of a topic with the student's notes, so "
        "revising from the summary taught material the course never covered."
    ),
    design_notes=[
        "Slightly higher temperature (0.3) because summarisation benefits "
        "from some phrasing freedom, unlike factual lookup.",
        "The 'gaps' section turns missing information into a visible output "
        "rather than something the model quietly fills in.",
        "Fixed section headings make summaries comparable across topics and "
        "trivial to render in the Streamlit interface.",
    ],
    text="""You turn a student's own notes into revision material.

Use ONLY the supplied context. If the notes are thin on a point, say so \
rather than completing it from general knowledge.

Produce exactly these sections:
**Key points** - 3 to 6 bullets, each traceable to the context.
**Terms to know** - term: one-line definition, taken from the notes.
**Gaps** - anything the question touches that the notes do not cover. Write \
"None" if the notes are complete.

Keep the whole summary under {max_words} words. Do not add an introduction \
or a closing remark.""",
)


# ---------------------------------------------------------------------------
# Prompt 4 - Conversational query rewriting (enables Checkpoint 3 memory)
# ---------------------------------------------------------------------------
QUERY_REWRITER = SystemPrompt(
    name="query_rewriter",
    version="v1",
    role="Rewrite a follow-up question into a standalone search query",
    temperature=0.0,
    failure_mode_addressed=(
        "Retrieval collapse in multi-turn chat: a follow-up such as 'when is "
        "it due?' embeds to a vector with no topical content, so the vector "
        "search returns unrelated chunks regardless of the metric used."
    ),
    design_notes=[
        "Runs before retrieval, not after - this is a search-quality fix, "
        "not an answer-quality fix.",
        "Returning the question unchanged when it is already standalone "
        "avoids the rewriter corrupting good queries.",
        "Output is the bare query so it can be fed straight to the embedder.",
    ],
    text="""Rewrite the user's follow-up question into a standalone search query.

Rules:
1. Replace every pronoun and implicit reference with the concrete subject \
from the conversation history.
2. Output the rewritten query only - no quotes, no explanation, no preamble.
3. If the question already stands on its own, return it unchanged.
4. Preserve the user's own terminology; do not introduce synonyms.

History:
{history}

Follow-up question: {question}""",
)


PROMPT_LIBRARY: Dict[str, SystemPrompt] = {
    prompt.name: prompt
    for prompt in (GROUNDED_QA, SCHEDULE_EXTRACTOR, STUDY_SUMMARIZER, QUERY_REWRITER)
}


def get_prompt(name: str) -> SystemPrompt:
    """
    Look up a system prompt by name.

    Args:
        name: Prompt name, e.g. 'grounded_qa'

    Returns:
        The matching SystemPrompt

    Raises:
        KeyError: If no prompt with that name is registered
    """
    if name not in PROMPT_LIBRARY:
        raise KeyError(f"Unknown prompt {name!r}. Available: {sorted(PROMPT_LIBRARY)}")
    return PROMPT_LIBRARY[name]


def format_context(retrieved: List[Dict], max_chars: Optional[int] = 4000) -> str:
    """
    Render retrieved chunks into a citable CONTEXT block.

    Args:
        retrieved: Result dictionaries from VectorStore.query
        max_chars: Soft budget for the context block; chunks are dropped once
            the budget is exhausted so the prompt cannot overrun the window

    Returns:
        Newline-separated passages, each tagged with its [doc_id]
    """
    passages = []
    used = 0

    for hit in retrieved:
        passage = f"[{hit['doc_id']}] {hit['text']}"
        if max_chars is not None and used + len(passage) > max_chars:
            break
        passages.append(passage)
        used += len(passage)

    return "\n\n".join(passages) if passages else "(no passages retrieved)"


def build_rag_messages(
    question: str,
    retrieved: List[Dict],
    prompt_name: str = "grounded_qa",
    history: Optional[List[Dict]] = None,
    **prompt_kwargs,
) -> List[Dict[str, str]]:
    """
    Assemble a Chat Completion `messages` payload for a RAG turn.

    Args:
        question: The user's question
        retrieved: Chunks returned by the vector store
        prompt_name: Which system prompt to use
        history: Prior turns as [{'role': ..., 'content': ...}, ...]
        **prompt_kwargs: Values for placeholders in the system prompt

    Returns:
        A messages list ready to send to a Chat Completion endpoint
    """
    prompt = get_prompt(prompt_name)

    defaults = {"max_sentences": 4, "max_words": 200}
    defaults.update(prompt_kwargs)
    # Only pass placeholders this prompt actually declares.
    applicable = {key: value for key, value in defaults.items() if "{" + key + "}" in prompt.text}

    messages = [{"role": "system", "content": prompt.render(**applicable)}]

    if history:
        messages.extend(history)

    messages.append(
        {
            "role": "user",
            "content": f"CONTEXT:\n{format_context(retrieved)}\n\nQUESTION: {question}",
        }
    )

    return messages


if __name__ == "__main__":
    print("SYSTEM PROMPT LIBRARY\n" + "=" * 60)
    for prompt in PROMPT_LIBRARY.values():
        info = prompt.summary()
        print(f"\n{info['name']}  ({info['version']})  temp={info['temperature']}")
        print(f"  Role:    {info['role']}")
        print(f"  Fixes:   {info['failure_mode_addressed'][:88]}...")
        print(f"  Length:  {info['character_length']} chars")

    print("\n" + "=" * 60)
    print("EXAMPLE ASSEMBLED MESSAGES (grounded_qa)\n")
    demo_hits = [
        {"doc_id": "schedule_semester_2026", "text": "Checkpoint 2 is due September 19, 2026."},
        {"doc_id": "project_capstone_plan", "text": "Checkpoint 2 covers vector indexing."},
    ]
    for message in build_rag_messages("When is Checkpoint 2 due?", demo_hits):
        print(f"--- {message['role'].upper()} ---")
        print(message["content"])
        print()
