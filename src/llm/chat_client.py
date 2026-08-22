"""
Chat Completion API integration.

Checkpoint 2 deliverable: a working call against a Chat Completion endpoint.

The client targets any OpenAI-compatible /chat/completions endpoint (OpenAI,
Azure OpenAI, Together, Groq, or a local vLLM / Ollama server) by way of the
`openai` SDK. When no API key is configured - as in a marking environment
with no credentials or no outbound network - it degrades to a deterministic
offline responder so the full RAG chain still runs end to end.

The offline responder is extractive, not generative: it selects sentences
from the supplied context. It exists to exercise the pipeline, not to stand
in for an LLM's answer quality.
"""

import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

# src/ is not an installed package; make sibling modules importable whether
# this file is run directly or imported from the pipeline.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+")

# Reused so the offline responder scores overlap on content words only.
# Matching on function words such as "the" makes every sentence look relevant
# and defeats the refusal path the grounded_qa prompt depends on.
from embeddings.embedding_generator_lexical import STOPWORDS


@dataclass
class ChatResponse:
    """Normalised result of a chat completion call."""

    content: str
    model: str
    backend: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_seconds: float = 0.0
    finish_reason: str = "stop"

    @property
    def total_tokens(self) -> int:
        """Total tokens billed for this call."""
        return self.prompt_tokens + self.completion_tokens

    def summary(self) -> dict:
        """Return a compact record for logging and reporting."""
        return {
            "backend": self.backend,
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "latency_seconds": round(self.latency_seconds, 3),
            "finish_reason": self.finish_reason,
        }


class ChatClient:
    """Chat Completion client with an offline fallback."""

    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 30.0,
        max_retries: int = 2,
        force_offline: bool = False,
    ):
        """
        Configure the client.

        Args:
            model: Model identifier passed to the endpoint
            api_key: API key; falls back to the OPENAI_API_KEY env var
            base_url: Override for OpenAI-compatible providers; falls back to
                the OPENAI_BASE_URL env var
            timeout: Per-request timeout in seconds
            max_retries: Retries on transient API errors
            force_offline: Skip the network entirely and use the offline
                responder, useful for deterministic tests
        """
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
        self.client = None
        self.backend = "offline"

        if force_offline or not self.api_key:
            return

        try:
            from openai import OpenAI

            kwargs = {"api_key": self.api_key, "timeout": timeout}
            if self.base_url:
                kwargs["base_url"] = self.base_url

            self.client = OpenAI(**kwargs)
            self.backend = "openai"
        except ImportError:
            # SDK missing; the offline responder keeps the pipeline usable.
            self.client = None

    @property
    def is_live(self) -> bool:
        """True when calls will hit a real Chat Completion endpoint."""
        return self.client is not None

    def complete(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 500,
    ) -> ChatResponse:
        """
        Send a chat completion request.

        Args:
            messages: Chat messages in [{'role': ..., 'content': ...}] form
            temperature: Sampling temperature
            max_tokens: Upper bound on generated tokens

        Returns:
            A ChatResponse from either the live endpoint or the offline responder
        """
        if not self.is_live:
            return self._complete_offline(messages)

        started = time.perf_counter()
        last_error = None

        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                usage = response.usage
                return ChatResponse(
                    content=response.choices[0].message.content or "",
                    model=response.model,
                    backend="openai",
                    prompt_tokens=getattr(usage, "prompt_tokens", 0),
                    completion_tokens=getattr(usage, "completion_tokens", 0),
                    latency_seconds=time.perf_counter() - started,
                    finish_reason=response.choices[0].finish_reason or "stop",
                )
            except Exception as error:
                last_error = error
                if attempt < self.max_retries:
                    # Exponential backoff on rate limits and transient faults.
                    time.sleep(2**attempt)

        print(f"⚠ Chat API call failed after {self.max_retries + 1} attempts: {last_error}")
        print("  Falling back to the offline responder.")
        return self._complete_offline(messages)

    def _complete_offline(self, messages: List[Dict[str, str]]) -> ChatResponse:
        """
        Answer extractively from the CONTEXT block, without a network call.

        Args:
            messages: The same messages that would be sent to the API

        Returns:
            A ChatResponse tagged with backend='offline'
        """
        started = time.perf_counter()

        system_text = next(
            (m["content"] for m in messages if m["role"] == "system"), ""
        )
        user_text = next(
            (m["content"] for m in reversed(messages) if m["role"] == "user"), ""
        )

        context, question = self._split_context(user_text)
        content = self._extract_answer(context, question, system_text)

        return ChatResponse(
            content=content,
            model="offline-extractive",
            backend="offline",
            # Word count is a rough stand-in; no tokeniser is available offline.
            prompt_tokens=len((system_text + user_text).split()),
            completion_tokens=len(content.split()),
            latency_seconds=time.perf_counter() - started,
        )

    @staticmethod
    def _split_context(user_text: str) -> tuple:
        """Separate the CONTEXT block from the QUESTION in a RAG user message."""
        if "QUESTION:" in user_text:
            context_part, question_part = user_text.rsplit("QUESTION:", 1)
            return context_part.replace("CONTEXT:", "").strip(), question_part.strip()
        return "", user_text.strip()

    @staticmethod
    def _split_segments(body: str, max_words: int = 40) -> List[str]:
        """
        Split a passage into quotable segments.

        The Checkpoint 1 cleaner strips most punctuation and collapses
        newlines, so a retrieved chunk often contains no sentence boundary at
        all. Sentence splitting alone would then return the entire chunk as
        one "sentence", so over-long segments are further cut into windows.

        Args:
            body: Passage text
            max_words: Longest segment to emit before splitting

        Returns:
            Segments of at most `max_words` words
        """
        segments = []
        for sentence in SENTENCE_PATTERN.split(body):
            words = sentence.split()
            if not words:
                continue
            for start in range(0, len(words), max_words):
                segments.append(" ".join(words[start : start + max_words]))
        return segments

    @staticmethod
    def _extract_answer(context: str, question: str, system_text: str) -> str:
        """
        Select the context sentences that best overlap the question.

        Args:
            context: The retrieved passages
            question: The user's question
            system_text: System prompt, inspected for the configured refusal string

        Returns:
            An extractive answer, or the prompt's refusal string when nothing matches
        """
        refusal = "I don't have that in your notes."

        if not context or context == "(no passages retrieved)":
            return refusal

        question_terms = {
            term
            for term in TOKEN_PATTERN.findall(question.lower())
            if len(term) > 2 and term not in STOPWORDS
        }

        if not question_terms:
            return refusal

        # A single shared content word is usually coincidence; require two
        # unless the question is itself only one or two words long.
        min_overlap = 2 if len(question_terms) >= 3 else 1

        scored = []
        for line in context.split("\n"):
            line = line.strip()
            if not line:
                continue

            doc_match = re.match(r"^\[([^\]]+)\]\s*(.*)$", line)
            doc_id = doc_match.group(1) if doc_match else "unknown"
            body = doc_match.group(2) if doc_match else line

            for sentence in ChatClient._split_segments(body):
                sentence = sentence.strip()
                if not sentence:
                    continue
                sentence_terms = set(TOKEN_PATTERN.findall(sentence.lower()))
                overlap = len(question_terms & sentence_terms)
                if overlap >= min_overlap:
                    scored.append((overlap, sentence, doc_id))

        if not scored:
            return refusal

        scored.sort(key=lambda item: -item[0])
        best = scored[:2]
        return " ".join(f"{sentence} [{doc_id}]" for _, sentence, doc_id in best)

    def get_info(self) -> dict:
        """Return the client's effective configuration."""
        return {
            "model": self.model,
            "backend": self.backend if self.is_live else "offline",
            "base_url": self.base_url or "https://api.openai.com/v1",
            "live": self.is_live,
            "api_key_configured": bool(self.api_key),
        }


if __name__ == "__main__":
    from prompts.system_prompts import build_rag_messages

    client = ChatClient()
    print("CHAT COMPLETION API DEMO")
    print(f"  Config: {client.get_info()}")
    if not client.is_live:
        print("  No OPENAI_API_KEY set - using the offline extractive responder.\n")

    retrieved = [
        {"doc_id": "schedule_semester_2026", "text": "Checkpoint 2 is due on September 19, 2026. It covers prompt architecture and vector indexing."},
        {"doc_id": "project_capstone_plan", "text": "The final defense takes place on November 14, 2026."},
    ]

    for question in [
        "When is Checkpoint 2 due?",
        "What is the instructor's home address?",
    ]:
        messages = build_rag_messages(question, retrieved)
        response = client.complete(messages, temperature=0.0)
        print(f"Q: {question}")
        print(f"A: {response.content}")
        print(f"   {response.summary()}\n")
