"""
Deadline reminder CLI.

Run:
    python src/reminders.py                  # digest for the next 14 days
    python src/reminders.py --days 30        # widen the horizon
    python src/reminders.py --refresh        # re-scan data/raw, then report
    python src/reminders.py --json           # machine-readable output
    python src/reminders.py --as-of 2026-09-15   # reproducible demo output

Implements the "Tracks schedules and deadlines with automatic notifications"
objective from docs/01_Project_Proposal.md.
"""

import argparse
import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from schedule.extractor import extract_from_corpus
from schedule.reminders import DEFAULT_HORIZON_DAYS, ReminderEngine
from schedule.store import DEFAULT_STORE_PATH, load_deadlines, save_deadlines


def build_parser() -> argparse.ArgumentParser:
    """Construct the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Report approaching deadlines from your notes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--days",
        type=int,
        default=DEFAULT_HORIZON_DAYS,
        help=f"How many days ahead to look (default: {DEFAULT_HORIZON_DAYS})",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Re-scan data/raw for deadlines before reporting",
    )
    parser.add_argument(
        "--as-of",
        metavar="YYYY-MM-DD",
        help="Treat this date as today; makes demo output reproducible",
    )
    parser.add_argument(
        "--json", action="store_true", help="Emit JSON instead of a text digest"
    )
    parser.add_argument(
        "--store",
        default=DEFAULT_STORE_PATH,
        help=f"Deadline store location (default: {DEFAULT_STORE_PATH})",
    )
    return parser


def main(argv=None) -> int:
    """
    Run the reminder CLI.

    Args:
        argv: Argument list; defaults to sys.argv

    Returns:
        Process exit code: 0 normally, 1 if --as-of is not a valid date
    """
    args = build_parser().parse_args(argv)

    today = date.today()
    if args.as_of:
        try:
            today = date.fromisoformat(args.as_of)
        except ValueError:
            print(f"error: --as-of must be YYYY-MM-DD, got {args.as_of!r}")
            return 1

    deadlines = load_deadlines(args.store)

    # Rebuild on request, and automatically when the store is empty so the
    # first run does something useful instead of reporting nothing.
    if args.refresh or not deadlines:
        deadlines = extract_from_corpus()
        written = save_deadlines(deadlines, args.store)
        if not args.json:
            print(f"Scanned data/raw -> {written} dated commitments saved to {args.store}\n")

    engine = ReminderEngine(deadlines, today=today)

    if args.json:
        payload = {
            "as_of": today.isoformat(),
            "horizon_days": args.days,
            "overdue": [
                {**reminder.deadline.to_dict(), "days_until": reminder.days_until}
                for reminder in engine.overdue()
            ],
            "upcoming": [
                {
                    **reminder.deadline.to_dict(),
                    "days_until": reminder.days_until,
                    "urgency": reminder.urgency,
                }
                for reminder in engine.upcoming(args.days)
            ],
        }
        print(json.dumps(payload, indent=2))
    else:
        print(engine.digest(args.days))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
