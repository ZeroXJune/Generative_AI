"""
Approaching-deadline detection.

This is deliberately plain Python, not an LLM call. Comparing dates is
arithmetic, and unreliable arithmetic is a documented LLM failure mode
(Module 1, Lesson 3). The model's only job in this feature is *extracting*
candidate dates; deciding what counts as "approaching" happens here, where
the result is exact and reproducible.
"""

import sys
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from schedule.extractor import Deadline

# (label, inclusive upper bound in days, display marker)
URGENCY_BANDS = (
    ("overdue", -1, "!!"),
    ("today", 0, "**"),
    ("urgent", 3, "! "),
    ("soon", 7, "> "),
    ("upcoming", 14, "  "),
)

DEFAULT_HORIZON_DAYS = 14


@dataclass
class Reminder:
    """A deadline paired with how near it is."""

    deadline: Deadline
    days_until: int
    urgency: str
    marker: str

    def describe(self) -> str:
        """Render a one-line human-readable reminder."""
        if self.days_until < 0:
            when = f"{abs(self.days_until)} day(s) OVERDUE"
        elif self.days_until == 0:
            when = "TODAY"
        elif self.days_until == 1:
            when = "tomorrow"
        else:
            when = f"in {self.days_until} days"

        at_time = f" at {self.deadline.time}" if self.deadline.time else ""
        return (
            f"{self.marker} {self.deadline.date.isoformat()}{at_time}  "
            f"{when:<20} {self.deadline.title}  [{self.deadline.source_doc}]"
        )


def classify_urgency(days_until: int) -> tuple:
    """
    Map a day count onto an urgency band.

    Args:
        days_until: Days from today until the deadline; negative if past

    Returns:
        Tuple of (urgency label, display marker)
    """
    for label, bound, marker in URGENCY_BANDS:
        if label == "overdue":
            if days_until < 0:
                return label, marker
        elif days_until <= bound:
            return label, marker
    return "future", "  "


class ReminderEngine:
    """Answers 'what is coming up?' over a set of deadlines."""

    def __init__(self, deadlines: List[Deadline], today: Optional[date] = None):
        """
        Args:
            deadlines: The deadlines to reason over
            today: Reference date; defaults to the real current date.
                Injectable so tests and demos are reproducible.
        """
        self.deadlines = sorted(deadlines, key=lambda item: item.date)
        self.today = today or date.today()

    def days_until(self, deadline: Deadline) -> int:
        """Return days from the reference date to a deadline (negative if past)."""
        return (deadline.date - self.today).days

    def upcoming(self, within_days: int = DEFAULT_HORIZON_DAYS) -> List[Reminder]:
        """
        Deadlines falling between today and the horizon.

        Args:
            within_days: How far ahead to look

        Returns:
            Reminders sorted soonest-first
        """
        reminders = []
        for deadline in self.deadlines:
            delta = self.days_until(deadline)
            if 0 <= delta <= within_days:
                urgency, marker = classify_urgency(delta)
                reminders.append(Reminder(deadline, delta, urgency, marker))
        return reminders

    def overdue(self) -> List[Reminder]:
        """Deadlines whose date has already passed, most recent first."""
        reminders = []
        for deadline in self.deadlines:
            delta = self.days_until(deadline)
            if delta < 0:
                urgency, marker = classify_urgency(delta)
                reminders.append(Reminder(deadline, delta, urgency, marker))
        return sorted(reminders, key=lambda item: item.days_until, reverse=True)

    def next_deadline(self) -> Optional[Reminder]:
        """The single nearest deadline that has not yet passed."""
        future = self.upcoming(within_days=3650)
        return future[0] if future else None

    def group_by_urgency(
        self, within_days: int = DEFAULT_HORIZON_DAYS
    ) -> Dict[str, List[Reminder]]:
        """
        Bucket upcoming reminders by urgency band.

        Args:
            within_days: How far ahead to look

        Returns:
            Mapping of urgency label to its reminders, in escalation order
        """
        grouped: Dict[str, List[Reminder]] = {}
        for reminder in self.upcoming(within_days):
            grouped.setdefault(reminder.urgency, []).append(reminder)
        return grouped

    def digest(self, within_days: int = DEFAULT_HORIZON_DAYS) -> str:
        """
        Render a full text digest of overdue and upcoming items.

        Args:
            within_days: How far ahead to look

        Returns:
            A printable multi-line digest
        """
        lines = [
            "=" * 78,
            f"DEADLINE DIGEST - as of {self.today.isoformat()}",
            "=" * 78,
        ]

        overdue = self.overdue()
        if overdue:
            lines.append(f"\nOVERDUE ({len(overdue)})")
            lines.extend(f"  {reminder.describe()}" for reminder in overdue)

        upcoming = self.upcoming(within_days)
        lines.append(f"\nNEXT {within_days} DAYS ({len(upcoming)})")
        if upcoming:
            lines.extend(f"  {reminder.describe()}" for reminder in upcoming)
        else:
            lines.append("  Nothing due in this window.")

        following = self.next_deadline()
        if not upcoming and following:
            lines.append(
                f"\n  Next after that: {following.deadline.title} "
                f"on {following.deadline.date.isoformat()} "
                f"(in {following.days_until} days)"
            )

        lines.append("=" * 78)
        return "\n".join(lines)


if __name__ == "__main__":
    from schedule.extractor import extract_from_corpus

    engine = ReminderEngine(extract_from_corpus())
    print(engine.digest())
