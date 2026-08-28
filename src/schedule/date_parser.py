"""
Date parsing for deadline tracking.

Deliberately dependency-free (stdlib `re` + `datetime` only) and deliberately
strict: a date this module cannot parse with confidence is rejected rather
than guessed at. A wrong deadline is worse than a missing one, because the
user acts on it.

Handles the formats that occur in the project corpus:
    September 19, 2026        Month DD, YYYY
    september 19 2026         (case-insensitive, comma optional)
    Nov 14, 2026              abbreviated month
    2026-09-19                ISO
    09/19/2026, 9/19/26       US numeric
    November 17-21, 2026      date range (start date is used)
"""

import re
from datetime import date, datetime
from typing import Optional, Tuple

MONTHS = {
    "january": 1, "jan": 1,
    "february": 2, "feb": 2,
    "march": 3, "mar": 3,
    "april": 4, "apr": 4,
    "may": 5,
    "june": 6, "jun": 6,
    "july": 7, "jul": 7,
    "august": 8, "aug": 8,
    "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10,
    "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}

MONTH_NAMES = "|".join(sorted(MONTHS, key=len, reverse=True))

# "September 19, 2026" and "November 17-21, 2026" (range -> first day)
TEXTUAL_DATE = re.compile(
    rf"\b({MONTH_NAMES})\.?\s+(\d{{1,2}})(?:\s*[-–]\s*\d{{1,2}})?(?:st|nd|rd|th)?,?\s+(\d{{4}})\b",
    re.IGNORECASE,
)
ISO_DATE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
US_DATE = re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{2,4})\b")

# "11:59 PM", "2:00 pm", "14:30"
TIME_PATTERN = re.compile(r"\b(\d{1,2}):(\d{2})\s*([ap]\.?m\.?)?\b", re.IGNORECASE)


def _safe_date(year: int, month: int, day: int) -> Optional[date]:
    """
    Build a date, returning None instead of raising on invalid input.

    Args:
        year: Four-digit year
        month: Month number
        day: Day of month

    Returns:
        The date, or None if the combination is not a real calendar date
        (e.g. February 30)
    """
    try:
        return date(year, month, day)
    except ValueError:
        return None


def parse_date(text: str) -> Optional[date]:
    """
    Extract the first parseable date from a string.

    Args:
        text: Text that may contain a date

    Returns:
        The first date found, or None if the text contains none

    Examples:
        >>> parse_date("Capstone Checkpoint 2 (Midterm): September 19, 2026")
        datetime.date(2026, 9, 19)
        >>> parse_date("sometime next week") is None
        True
    """
    if not text:
        return None

    match = TEXTUAL_DATE.search(text)
    if match:
        month_name, day, year = match.groups()
        month = MONTHS.get(month_name.lower().rstrip("."))
        if month:
            return _safe_date(int(year), month, int(day))

    match = ISO_DATE.search(text)
    if match:
        year, month, day = (int(group) for group in match.groups())
        return _safe_date(year, month, day)

    match = US_DATE.search(text)
    if match:
        month, day, year = (int(group) for group in match.groups())
        # Two-digit years are read as 2000-2099; this corpus is entirely
        # future-dated coursework, so a 19xx reading is never correct.
        if year < 100:
            year += 2000
        return _safe_date(year, month, day)

    return None


def parse_time(text: str) -> Optional[str]:
    """
    Extract the first time-of-day from a string, normalised to HH:MM.

    Args:
        text: Text that may contain a time

    Returns:
        A 24-hour "HH:MM" string, or None if no time is present

    Examples:
        >>> parse_time("due September 19, 2026, 11:59 PM")
        '23:59'
    """
    if not text:
        return None

    match = TIME_PATTERN.search(text)
    if not match:
        return None

    hour, minute, meridiem = int(match.group(1)), int(match.group(2)), match.group(3)

    if minute > 59:
        return None

    if meridiem:
        meridiem = meridiem.lower().replace(".", "")
        if hour > 12 or hour < 1:
            return None
        if meridiem == "pm" and hour != 12:
            hour += 12
        elif meridiem == "am" and hour == 12:
            hour = 0
    elif hour > 23:
        return None

    return f"{hour:02d}:{minute:02d}"


def parse_datetime(text: str) -> Tuple[Optional[date], Optional[str]]:
    """
    Extract both a date and a time from a string.

    Args:
        text: Text that may contain a date and time

    Returns:
        Tuple of (date or None, "HH:MM" string or None)
    """
    return parse_date(text), parse_time(text)


if __name__ == "__main__":
    samples = [
        "Capstone Checkpoint 2 (Midterm): September 19, 2026, 11:59 PM",
        "Final Exam Period: November 17-21, 2026",
        "Sem Start: July 14, 2026",
        "due 2026-10-17",
        "submit by 09/19/2026",
        "meeting on 3/5/26 at 2:00 pm",
        "Founder's Day: August 21, 2026",
        "remember to email the instructor sometime soon",
        "February 30, 2026 is not a real date",
    ]

    print(f"{'input':<58} {'date':<12} time")
    print("-" * 82)
    for sample in samples:
        parsed_date, parsed_time = parse_datetime(sample)
        print(f"{sample[:56]:<58} {str(parsed_date or '-'):<12} {parsed_time or '-'}")
