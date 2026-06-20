"""Backfill historical market and macro data.

Pulls:
  - Every FRED series listed in settings.yaml
  - Daily bars for every symbol in the universe (default: 5 years)

Idempotent: re-running upserts. Use --since to limit to recent data after the
initial load (e.g. --since 7 for the last week).

Usage:
    python scripts/backfill.py                 # full 5-year backfill
    python scripts/backfill.py --since 30      # last 30 days only
    python scripts/backfill.py --skip-fred     # bars only
    python scripts/backfill.py --skip-bars     # macro only
"""

from __future__ import annotations

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

# Force UTF-8 stdout on Windows for Rich box-drawing characters
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

from alpha_engine.core.config import get_settings
from alpha_engine.core.logging import configure_logging, get_logger
from alpha_engine.core.types import Instrument
from alpha_engine.data import (
    FredClient,
    YFinanceProvider,
    upsert_instruments,
    upsert_macro_observations,
    upsert_market_bars,
)
from alpha_engine.db import get_connection, init_schema

log = get_logger(__name__)
console = Console()

app = typer.Typer(add_completion=False, help=__doc__)


@app.command()
def backfill(
    start: str = typer.Option(
        "",
        "--start",
        help="Absolute start date YYYY-MM-DD. Overrides --since if set.",
    ),
    since: int = typer.Option(
        0,
        "--since",
        help="Days of history to fetch. 0 = use settings.default_history_days. Ignored if --start is set.",
    ),
    skip_fred: bool = typer.Option(False, "--skip-fred", help="Skip FRED macro pull"),
    skip_bars: bool = typer.Option(False, "--skip-bars", help="Skip equity bars pull"),
    log_level: str = typer.Option("INFO", "--log-level"),
) -> None:
    configure_logging(level=log_level)
    settings = get_settings()

    if start:
        start_date = datetime.strptime(start, "%Y-%m-%d").date()
        days = (date.today() - start_date).days
    else:
        days = since or settings.data.default_history_days
        start_date = date.today() - timedelta(days=days)
    start = start_date  # rename for downstream code below
    console.print(f"[bold cyan]Backfill from {start} ({days} days)[/]")

    init_schema()

    # --- Seed instrument universe -----------------------------------------
    instruments = [
        Instrument(
            symbol=spec.symbol,
            name=spec.name,
            instrument_type=spec.type,
            sector=spec.sector,
        )
        for items in settings.universe.universes.values()
        for spec in items
    ]
    with get_connection() as con:
        n = upsert_instruments(con, instruments)
    console.print(f"  instruments upserted: {n}")

    symbols = [i.symbol for i in instruments]

    # --- Bars -------------------------------------------------------------
    if not skip_bars:
        console.print(f"\n[bold]Fetching bars for {len(symbols)} symbols[/]")
        yf = YFinanceProvider()
        # yfinance handles multi-ticker in one call efficiently, but with
        # threads disabled it serializes internally. Chunking gives us
        # per-chunk progress + isolates any single-symbol failure.
        chunk_size = 10
        total_bars = 0
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("fetching bars", total=len(symbols))
            for i in range(0, len(symbols), chunk_size):
                chunk = symbols[i : i + chunk_size]
                try:
                    bars = list(yf.fetch(chunk, start=start))
                    with get_connection() as con:
                        upsert_market_bars(con, bars)
                    total_bars += len(bars)
                except Exception as exc:
                    log.error("bar_chunk_failed", chunk=chunk, error=str(exc))
                progress.update(task, advance=len(chunk))
        console.print(f"  total bars stored: [green]{total_bars}[/]")
    else:
        console.print("[yellow]Skipping bars[/]")

    # --- FRED macro -------------------------------------------------------
    if not skip_fred:
        if not settings.fred_api_key:
            console.print("[yellow]FRED_API_KEY not set; skipping macro[/]")
        else:
            console.print(
                f"\n[bold]Fetching {len(settings.fred_series)} FRED series[/]"
            )
            total_obs = 0
            with (
                FredClient() as fred,
                Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TextColumn("{task.fields[series]}"),
                    TimeElapsedColumn(),
                    console=console,
                ) as progress,
            ):
                task = progress.add_task(
                    "fred", total=len(settings.fred_series), series=""
                )
                for spec in settings.fred_series:
                    progress.update(task, series=spec.id)
                    try:
                        obs = list(fred.fetch(spec.id, observation_start=start))
                        with get_connection() as con:
                            upsert_macro_observations(con, obs)
                        total_obs += len(obs)
                    except Exception as exc:
                        log.error(
                            "fred_series_failed", series=spec.id, error=str(exc)
                        )
                    progress.advance(task)
            console.print(f"  total observations: [green]{total_obs}[/]")
    else:
        console.print("[yellow]Skipping FRED[/]")

    # --- Summary ----------------------------------------------------------
    with get_connection(read_only=True) as con:
        bar_count = con.execute("SELECT COUNT(*) FROM market_bars").fetchone()[0]
        macro_count = con.execute("SELECT COUNT(*) FROM macro_series").fetchone()[0]
        symbol_count = con.execute(
            "SELECT COUNT(DISTINCT symbol) FROM market_bars"
        ).fetchone()[0]
        series_count = con.execute(
            "SELECT COUNT(DISTINCT series_id) FROM macro_series"
        ).fetchone()[0]
        latest_bar = con.execute(
            "SELECT MAX(bar_date) FROM market_bars"
        ).fetchone()[0]

    console.print()
    console.print(f"[bold green]Backfill complete[/]")
    console.print(f"  bars:     {bar_count:>9,} rows / {symbol_count} symbols")
    console.print(f"  macro:    {macro_count:>9,} rows / {series_count} series")
    console.print(f"  latest bar date: {latest_bar}")


if __name__ == "__main__":
    app()
