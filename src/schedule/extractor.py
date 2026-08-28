"""
Extract dated commitments from documents.

Two extraction paths share one output type:

  * `extract_from_text`  - deterministic, regex + date parser, no network.
    This is the default. Date handling is arithmetic, not language, so it
    does not need an LLM and must not depend on one being reachable.

  * `extract_with_llm`   - uses the `schedule_extractor` v3 prompt for prose
    that the regex path cannot structure ("the review moved to the 21st").
    Every date it returns is re-validated by the same parser, so a
    hallucinated date is dropped rather than trusted.
"""

import json
import re
import sys
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from schedule.date_parser import (
    ISO_DATE,
    TEXTUAL_DATE,
    TIME_PATTERN,
    US_DATE,
    parse_date,
    parse_time,
)

# Keyword -> event type, checked in order; first match wins.
TYPE_KEYWORDS = (
    ("exam", ("exam", "midterm", "prelim", "final exam", "quiz", "defense")),
    ("meeting", ("meeting", "office hours", "consultation", "standup", "sync")),
    ("deadline", ("due", "deadline", "submission", "submit", "checkpoint", "deliverable")),
    ("task", ("task", "todo", "action item", "reminder")),
)

# Lines that contain a date but are not commitments.
NON_EVENT_MARKERS = (
    "holiday",
    "sem start",
    "semester start",
    "no classes",
    "break",
    "suspension",
)

# Section headers whose entries are calendar facts, not things to act on.
NON_EVENT_SECTIONS = ("holiday", "suspension", "no classes", "academic calendar")

# Titles that are document metadata rather than a commitment. A line such as
# "Date: July 14, 2026" at the top of a note dates the document itself.
METADATA_TITLES = frozenset(
    {
        "date", "dates", "raw", "cleaned", "created", "updated", "last updated",
        "version", "author", "prepared by", "generated", "as of", "today",
        "sprint period", "period",
    }
)

# Prefixes marking text quoted as an example rather than stated as fact.
# meeting_notes_july14 quotes before/after cleaning samples that contain real
# dates but describe Checkpoint 1's preprocessing, not a commitment.
QUOTED_EXAMPLE_PREFIXES = ("raw:", "cleaned:", "before:", "after:", "example:", "e.g.")

# Titles that name a deadline without identifying it; the document name is
# more informative than the bare word.
GENERIC_TITLES = frozenset({"due", "deadline", "submission", "submit", "deliverable"})


@dataclass
class Deadline:
    """A single dated commitment extracted from a document."""

    title: str
    date: date
    type: str = "deadline"
    time: Optional[str] = None
    source_doc: str = "unknown"

    def to_dict(self) -> dict:
        """Serialise to a JSON-safe dictionary."""
        record = asdict(self)
        record["date"] = self.date.isoformat()
        return record

    @classmethod
    def from_dict(cls, record: Dict) -> "Deadline":
        """
        Rebuild a Deadline from its serialised form.

        Args:
            record: Dictionary produced by `to_dict`

        Returns:
            The reconstructed Deadline

        Raises:
            ValueError: If the stored date is not a valid ISO date
        """
        parsed = date.fromisoformat(record["date"])
        return cls(
            title=record["title"],
            date=parsed,
            type=record.get("type", "deadline"),
            time=record.get("time"),
            source_doc=record.get("source_doc", "unknown"),
        )


def classify(text: str) -> str:
    """
    Infer an event type from the wording of a line.

    Args:
        text: The line the date was found in

    Returns:
        One of 'exam', 'meeting', 'deadline', 'task'
    """
    lowered = text.lower()
    for event_type, keywords in TYPE_KEYWORDS:
        if any(keyword in lowered for keyword in keywords):
            return event_type
    return "deadline"


def _strip_date(line: str) -> str:
    """Remove every date and time expression, leaving the title text."""
    for pattern in (TEXTUAL_DATE, ISO_DATE, US_DATE, TIME_PATTERN):
        line = pattern.sub("", line)
    return line


def _is_section_header(line: str) -> bool:
    """True for a bare 'Holidays:'-style header carrying no date of its own."""
    return line.endswith(":") and parse_date(line) is None


def _rebalance(title: str) -> str:
    """
    Drop a parenthetical left dangling by date removal.

    "Capstone Checkpoint 1 (Prelim): September 19, 2026" loses its closing
    bracket with the date, so the trailing fragment is cut rather than left
    as "Capstone Checkpoint 1 (Prelim".
    """
    if title.count("(") > title.count(")"):
        title = title[: title.rindex("(")]
    return title.strip(" -–•\t:,")


def _humanize(doc_id: str) -> str:
    """Turn a document stem such as 'checklist_submission' into a title."""
    return doc_id.replace("_", " ").strip().title()


def extract_from_text(text: str, source_doc: str = "unknown") -> List[Deadline]:
    """
    Extract dated commitments from raw document text, line by line.

    Runs on RAW text, not cleaned text: the Checkpoint 1 cleaner collapses
    newlines and strips commas, which destroys the line structure and the
    "Month DD, YYYY" form this relies on.

    Args:
        text: Raw document text
        source_doc: Document identifier recorded on each result

    Returns:
        Deadlines found, in document order
    """
    deadlines = []
    section = ""

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        if _is_section_header(line):
            section = line.lower()
            continue

        event_date = parse_date(line)
        if event_date is None:
            continue

        lowered = line.lower()
        if any(marker in lowered for marker in NON_EVENT_MARKERS):
            continue
        if lowered.startswith(QUOTED_EXAMPLE_PREFIXES):
            continue
        # Entries listed under e.g. "Holidays:" inherit that section's meaning.
        if any(marker in section for marker in NON_EVENT_SECTIONS):
            continue

        # The title is whatever remains once date, time and list punctuation go.
        title = _strip_date(line).strip(" -–•\t:,")
        title = _rebalance(re.sub(r"\s{2,}", " ", title).strip(" -–•\t:,"))
        if not title or title.lower() in METADATA_TITLES:
            continue
        if title.lower() in GENERIC_TITLES:
            title = _humanize(source_doc)

        deadlines.append(
            Deadline(
                title=title,
                date=event_date,
                type=classify(line),
                time=parse_time(line),
                source_doc=source_doc,
            )
        )

    return deadlines


def extract_from_corpus(raw_dir: str = "data/raw") -> List[Deadline]:
    """
    Extract dated commitments from every document in the corpus.

    Args:
        raw_dir: Directory of .txt source documents

    Returns:
        All deadlines found, sorted by date
    """
    deadlines = []
    for path in sorted(Path(raw_dir).glob("*.txt")):
        deadlines.extend(
            extract_from_text(path.read_text(encoding="utf-8"), source_doc=path.stem)
        )
    return sorted(deadlines, key=lambda item: item.date)


def extract_with_llm(text: str, source_doc: str, chat_client) -> List[Deadline]:
    """
    Extract deadlines using the `schedule_extractor` prompt.

    Every returned date is re-parsed by `date_parser`, so a malformed or
    hallucinated date is discarded rather than stored.

    Args:
        text: Document text to analyse
        source_doc: Document identifier recorded on each result
        chat_client: A ChatClient instance

    Returns:
        Validated deadlines; empty if the model returned nothing usable
    """
    from prompts.system_prompts import get_prompt

    prompt = get_prompt("schedule_extractor")
    response = chat_client.complete(
        [
            {"role": "system", "content": prompt.render()},
            {"role": "user", "content": text},
        ],
        temperature=prompt.temperature,
    )

    try:
        payload = json.loads(response.content)
    except (json.JSONDecodeError, TypeError):
        # v3 forbids markdown fences, but a weaker model may still emit them.
        cleaned = re.sub(r"^```(?:json)?|```$", "", (response.content or "").strip())
        try:
            payload = json.loads(cleaned)
        except (json.JSONDecodeError, TypeError):
            return []

    if not isinstance(payload, list):
        return []

    deadlines = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        event_date = parse_date(str(item.get("date", "")))
        title = str(item.get("title", "")).strip()
        if event_date is None or not title:
            continue
        deadlines.append(
            Deadline(
                title=title,
                date=event_date,
                type=str(item.get("type", "deadline")),
                time=item.get("time"),
                source_doc=str(item.get("source_doc") or source_doc),
            )
        )

    return deadlines


if __name__ == "__main__":
    found = extract_from_corpus()
    print(f"Extracted {len(found)} dated commitments\n")
    print(f"{'date':<12} {'type':<9} {'source':<28} title")
    print("-" * 92)
    for item in found:
        print(
            f"{item.date.isoformat():<12} {item.type:<9} "
            f"{item.source_doc[:26]:<28} {item.title[:34]}"
        )
