"""
Conversational memory for multi-turn retrieval.

Checkpoint 3 deliverable. Checkpoint 2 answered each question in isolation,
which breaks the moment a user asks a follow-up:

    User: "What is Checkpoint 3 about?"
    User: "When is it due?"          <- "it" carries no topical content

Embedding "When is it due?" produces a vector with nothing to match on, so
vector search returns unrelated chunks no matter which distance metric is
used. This module fixes that *before* retrieval by rewriting the follow-up
into a standalone query, using the `query_rewriter` prompt written in
Checkpoint 2 for exactly this purpose.

Memory serves two distinct jobs, and they are kept separate:
  * **Retrieval memory** - resolving references so search works.
  * **Generation memory** - prior turns passed to the model for continuity.
"""

import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from embeddings.embedding_generator_lexical import STOPWORDS
from prompts.system_prompts import get_prompt

# Words that signal a question depends on earlier context. A question
# containing one of these and little else cannot be retrieved on its own.
REFERENCE_MARKERS = frozenset(
    {
        "it", "its", "that", "this", "these", "those", "they", "them", "their",
        "he", "she", "him", "her", "his", "hers", "there", "then", "one",
        "same", "above", "former", "latter", "another",
    }
)

DEFAULT_MAX_TURNS = 6


@dataclass
class Turn:
    """A single exchange in the conversation."""

    question: str
    answer: str
    resolved_query: Optional[str] = None
    sources: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))

    def as_messages(self) -> List[Dict[str, str]]:
        """Render this turn as Chat Completion messages."""
        return [
            {"role": "user", "content": self.question},
            {"role": "assistant", "content": self.answer},
        ]


class ConversationMemory:
    """Tracks conversation history and resolves context-dependent questions."""

    def __init__(self, max_turns: int = DEFAULT_MAX_TURNS):
        """
        Args:
            max_turns: How many recent turns to keep. Older turns are dropped
                rather than summarised, so the prompt cannot grow without bound.
        """
        self.max_turns = max_turns
        self.turns: List[Turn] = []
        # Counts every turn ever recorded. `len(self.turns)` cannot serve as a
        # turn number because trimming caps it at max_turns, which would make
        # turn 7 report as turn 6.
        self.total_turns = 0

    def add_turn(
        self,
        question: str,
        answer: str,
        resolved_query: Optional[str] = None,
        sources: Optional[List[str]] = None,
    ) -> Turn:
        """
        Record one completed exchange.

        Args:
            question: What the user asked, as they asked it
            answer: The answer given
            resolved_query: The standalone query actually used for retrieval
            sources: doc_ids cited in the answer

        Returns:
            The stored Turn
        """
        turn = Turn(
            question=question,
            answer=answer,
            resolved_query=resolved_query,
            sources=sources or [],
        )
        self.turns.append(turn)
        self.total_turns += 1
        # Trim from the front so the most recent context always survives.
        if len(self.turns) > self.max_turns:
            self.turns = self.turns[-self.max_turns :]
        return turn

    def as_messages(self, max_turns: Optional[int] = None) -> List[Dict[str, str]]:
        """
        Render recent history as Chat Completion messages.

        Args:
            max_turns: Override how many turns to include

        Returns:
            Alternating user/assistant messages, oldest first
        """
        turns = self.turns[-(max_turns or self.max_turns) :]
        messages = []
        for turn in turns:
            messages.extend(turn.as_messages())
        return messages

    def as_transcript(self, max_turns: Optional[int] = None) -> str:
        """
        Render recent history as plain text, for the rewriter prompt.

        Args:
            max_turns: Override how many turns to include

        Returns:
            A newline-separated transcript, or a placeholder when empty
        """
        turns = self.turns[-(max_turns or self.max_turns) :]
        if not turns:
            return "(no previous turns)"
        return "\n".join(f"User: {t.question}\nAssistant: {t.answer}" for t in turns)

    @staticmethod
    def needs_resolution(question: str) -> bool:
        """
        Decide whether a question depends on earlier context.

        A cheap check that runs before any LLM call: if the question contains
        no referring expression, rewriting it can only introduce errors.

        Args:
            question: The user's question

        Returns:
            True when the question contains a referring expression
        """
        words = {word.strip("?.,!'\"").lower() for word in question.split()}
        return bool(words & REFERENCE_MARKERS)

    def resolve(self, question: str, chat_client=None) -> str:
        """
        Rewrite a follow-up question into a standalone search query.

        Falls back to the original question whenever rewriting is unnecessary
        or unavailable, so this can never make retrieval worse than not having
        memory at all.

        Args:
            question: The user's question, as asked
            chat_client: A ChatClient; when omitted or offline, a deterministic
                keyword-carry heuristic is used instead

        Returns:
            A query suitable for embedding
        """
        if not self.turns or not self.needs_resolution(question):
            return question

        if chat_client is not None and getattr(chat_client, "is_live", False):
            rewritten = self._resolve_with_llm(question, chat_client)
            if rewritten:
                return rewritten

        return self._resolve_heuristically(question)

    def _resolve_with_llm(self, question: str, chat_client) -> Optional[str]:
        """
        Rewrite using the `query_rewriter` prompt.

        Args:
            question: The follow-up question
            chat_client: A live ChatClient

        Returns:
            The rewritten query, or None if the model returned nothing usable
        """
        prompt = get_prompt("query_rewriter")
        rendered = prompt.render(history=self.as_transcript(), question=question)

        response = chat_client.complete(
            [{"role": "user", "content": rendered}], temperature=prompt.temperature
        )
        candidate = (response.content or "").strip().strip('"')

        # A rewriter that returns an essay has misunderstood the task; ignore it
        # rather than embedding a paragraph.
        if not candidate or len(candidate) > 300:
            return None
        return candidate

    def _resolve_heuristically(self, question: str) -> str:
        """
        Resolve references without an LLM by carrying forward prior keywords.

        Appends the distinctive terms from the most recent question, which is
        enough to give the embedder topical content to match on.

        Args:
            question: The follow-up question

        Returns:
            The question with prior context appended
        """
        previous = self.turns[-1].resolved_query or self.turns[-1].question

        current_words = {w.strip("?.,!'\"").lower() for w in question.split()}
        carried = []

        for word in previous.split():
            bare = word.strip("?.,!'\"").lower()
            if not bare or bare in current_words or bare in REFERENCE_MARKERS:
                continue
            # Function words add no topical signal, but short numbers do -
            # "Checkpoint 3" is meaningless once the 3 is dropped.
            if bare in STOPWORDS or (len(bare) <= 3 and not bare.isdigit()):
                continue
            carried.append(word.strip("?.,!'\""))

        return f"{question} {' '.join(carried)}".strip() if carried else question

    def clear(self) -> None:
        """Forget the entire conversation, including the turn count."""
        self.turns = []
        self.total_turns = 0

    def summary(self) -> dict:
        """Return a compact description of the memory's state."""
        return {
            "turns_stored": len(self.turns),
            "total_turns": self.total_turns,
            "max_turns": self.max_turns,
            "last_question": self.turns[-1].question if self.turns else None,
        }
