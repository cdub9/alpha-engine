"""Seed the calendar_events table.

Inserts:
  - Market-wide scheduled events (FOMC, OpEx, quad witching, jobs, CPI)
    for a date range (default: previous year through next 2 years)
  - Per-ticker earnings calendar for every universe symbol via yfinance

Usage:
    python scripts/seed_calendars.py
    python scripts/seed_calendars.py --skip-earnings    # market events only
    python scripts/seed_calendars.py --years 3
"""

from __future__ import annotations

import sys
from datetime import date, timedelta

import typer
from rich.console import Console
from rich.table import Table

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

from alpha_engine.calendars import (
    compute_market_calendar_features,
    fetch_earnings_calendar,
    seed_market_calendar,
)
from alpha_engine.calendars.earnings import upsert_earnings_events
from alpha_engine.core.config import get_settings
from alpha_engine.core.logging import configure_logging
from alpha_engine.db import get_connection, init_schema

console = Console()
app = typer.Typer(add_completion=False, help=__doc__)


@app.command()
def seed(
    years: int = typer.Option(3, help="Years of forward calendar to seed"),
    skip_earnings: bool = typer.Option(False, "--skip-earnings"),
    log_level: str = typer.Option("INFO", "--log-level"),
) -> None:
    configure_logging(level=log_level)
    settings = get_settings()
    init_schema()

    today = date.today()
    start = date(today.year - 1, 1, 1)
    end = date(today.year + years, 12, 31)

    console.print(
        f"[bold cyan]Seeding market calendar {start} -> {end}[/]"
    )
    with get_connection() as con:
        result = seed_market_calendar(con, start, end)

    table = Table(title="Market calendar events seeded")
    table.add_column("Kind")
    table.add_column("Count", justify="right")
    table.add_row("FOMC", str(result.fomc))
    table.add_row("OpEx", str(result.opex))
    table.add_row("Quad witching", str(result.quad_witching))
    table.add_row("Jobs report", str(result.jobs))
    table.add_row("CPI", str(result.cpi))
    table.add_row("[bold]Total[/]", f"[bold]{result.total}[/]")
    console.print(table)

    if not skip_earnings:
        symbols = sorted(
            {
                spec.symbol
                for items in settings.universe.universes.values()
                for spec in items
                # Skip pure ETFs - they don't have earnings
                if spec.type.value in ("equity",)
            }
        )
        if symbols:
            console.print(
                f"\n[bold]Fetching earnings calendar for {len(symbols)} tickers[/]"
            )
            events = fetch_earnings_calendar(symbols)
            with get_connection() as con:
                n = upsert_earnings_events(con, events)
            console.print(f"  earnings events stored: [green]{n}[/]")
        else:
            console.print("[yellow]No equity tickers in universe; skipping earnings[/]")

    # ----- Demo: what does today's calendar context look like? -----
    console.print(f"\n[bold]Calendar features as of today ({today}):[/]")
    with get_connection(read_only=True) as con:
        feats = compute_market_calendar_features(con, today)

    feat_table = Table(show_header=False, box=None)
    feat_table.add_column("Feature", style="cyan")
    feat_table.add_column("Value")
    for k, v in feats.to_dict().items():
        if k in ("as_of", "symbol"):
            continue
        if v is None:
            continue
        feat_table.add_row(k, str(v))
    console.print(feat_table)


if __name__ == "__main__":
    app()
