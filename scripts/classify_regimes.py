"""Backfill regime classifications over historical dates.

Classifies each Friday (weekly granularity — regimes don't change daily)
from --start through today, storing results in regime_classifications.

Idempotent: re-runs replace existing rows for the same
(classification_date, model_version).

Usage:
    python scripts/classify_regimes.py
    python scripts/classify_regimes.py --start 2020-01-01
"""

from __future__ import annotations

import json
import sys
from datetime import date, datetime, timedelta

import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

from alpha_engine.core.logging import configure_logging, get_logger
from alpha_engine.db import get_connection, init_schema
from alpha_engine.regime import classify, extract_features

console = Console()
log = get_logger(__name__)
app = typer.Typer(add_completion=False, help=__doc__)


def _fridays_between(start: date, end: date) -> list[date]:
    """All Fridays inclusive in the range."""
    # Find first Friday >= start
    days_until_fri = (4 - start.weekday()) % 7
    first = start + timedelta(days=days_until_fri)
    out = []
    d = first
    while d <= end:
        out.append(d)
        d += timedelta(days=7)
    return out


@app.command()
def classify_history(
    start: str = typer.Option(
        "2020-01-01",
        help="Start date (YYYY-MM-DD). Backfill covers this through today.",
    ),
    log_level: str = typer.Option("WARNING", "--log-level"),
) -> None:
    configure_logging(level=log_level)
    init_schema()

    start_date = datetime.strptime(start, "%Y-%m-%d").date()
    end_date = date.today()
    fridays = _fridays_between(start_date, end_date)

    console.print(
        f"[bold cyan]Classifying {len(fridays)} weekly dates "
        f"({start_date} -> {end_date})[/]"
    )

    inserted = 0
    skipped = 0
    regime_counts: dict[str, int] = {}
    prior_regime = None  # threaded across the loop for VIX hysteresis

    with (
        get_connection() as con,
        Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            console=console,
        ) as progress,
    ):
        task = progress.add_task("classifying", total=len(fridays))
        for d in fridays:
            try:
                features = extract_features(con, d)
                assessment = classify(features, prior_regime=prior_regime)
                prior_regime = assessment.regime

                # Delete-then-insert (PK is (date, model_version))
                con.execute(
                    "DELETE FROM regime_classifications "
                    "WHERE classification_date = ? AND model_version = ?",
                    [d, assessment.model_version],
                )
                con.execute(
                    "INSERT INTO regime_classifications "
                    "(classification_date, regime, confidence, features_json, model_version) "
                    "VALUES (?, ?, ?, ?, ?)",
                    [
                        d,
                        assessment.regime.value,
                        assessment.confidence,
                        json.dumps(
                            {
                                "reasoning": assessment.reasoning,
                                "features": assessment.features_snapshot,
                            }
                        ),
                        assessment.model_version,
                    ],
                )
                inserted += 1
                regime_counts[assessment.regime.value] = (
                    regime_counts.get(assessment.regime.value, 0) + 1
                )
            except Exception as exc:
                log.error("classify_failed", date=str(d), error=str(exc))
                skipped += 1
            progress.advance(task)

    console.print(f"\n[bold green]Classified {inserted} dates[/] "
                  f"({skipped} skipped)")
    console.print("\nRegime distribution:")
    for regime, count in sorted(regime_counts.items(), key=lambda x: -x[1]):
        pct = count / inserted * 100 if inserted else 0
        console.print(f"  {regime:25s}  {count:4d}  ({pct:5.1f}%)")


if __name__ == "__main__":
    app()
