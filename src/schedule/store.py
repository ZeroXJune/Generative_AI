"""
Persistence for extracted deadlines.

Keeps the extraction step separate from the reminder step: documents are
scanned once and the results are stored, so a reminder check does not have to
re-read and re-parse the whole corpus.
"""

import json
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from schedule.extractor import Deadline

DEFAULT_STORE_PATH = "data/processed/deadlines.json"


def save_deadlines(deadlines: List[Deadline], path: str = DEFAULT_STORE_PATH) -> int:
    """
    Write deadlines to disk as JSON.

    Args:
        deadlines: Deadlines to persist
        path: Destination file

    Returns:
        Number of deadlines written
    """
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    records = [item.to_dict() for item in sorted(deadlines, key=lambda d: d.date)]
    output.write_text(json.dumps(records, indent=2))
    return len(records)


def load_deadlines(path: str = DEFAULT_STORE_PATH) -> List[Deadline]:
    """
    Read deadlines from disk.

    Malformed records are skipped rather than aborting the load, so one bad
    entry cannot suppress every reminder.

    Args:
        path: Source file

    Returns:
        Stored deadlines sorted by date; empty if the file does not exist
    """
    source = Path(path)
    if not source.exists():
        return []

    try:
        records = json.loads(source.read_text())
    except json.JSONDecodeError:
        return []

    deadlines = []
    for record in records if isinstance(records, list) else []:
        try:
            deadlines.append(Deadline.from_dict(record))
        except (KeyError, ValueError, TypeError):
            continue

    return sorted(deadlines, key=lambda item: item.date)
