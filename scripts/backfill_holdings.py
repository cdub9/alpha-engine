"""Backfill price bars for the real book's names (incl. non-universe holdings).

The universe backfill (backfill.py) only covers instruments in the config.
The real book holds names outside it (ETFs like IVV/VTI/VUG, plus ALAB/ARCC/
ROIV/SIMO, ...) that the book digest can't evaluate without price history.
This pulls bars for the held names so run_book_digest can form a view on the
whole portfolio.

Two modes:
  python scripts/backfill_holdings.py            # full 5y history for held
                                                 # names that have NO bars yet
  python scripts/backfill_holdings.py --since 7  # last 7 days for ALL held
                                                 # names (daily refresh)
"""

from __future__ import annotations

import json
import sys
from datetime import date, timedelta
from pathlib import Path

import typer
from rich.console import Console

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

from alpha_engine.core.config import get_settings
from alpha_engine.core.logging import configure_logging, get_logger
from alpha_engine.data import YFinanceProvider, upsert_market_bars
from alpha_engine.db import get_connection, init_schema

log = get_logger(__name__)
console = Console()
app = typer.Typer(add_completion=False, help=__doc__)

_HOLDINGS = Path(__file__).resolve().parent.parent / "data" / "real_holdings.json"


@app.command()
def main(
    since: int = typer.Option(
        0, "--since",
        help="Days of history. 0 = full history for held names with NO bars; "
             ">0 = last N days for ALL held names.",
    ),
    log_level: str = typer.Option("WARNING", "--log-level"),
) -> None:
    configure_logging(level=log_level)
    init_schema()
    settings = get_settings()

    if not _HOLDINGS.exists():
        console.print("[yellow]No holdings snapshot — nothing to backfill.[/]")
        raise typer.Exit(0)
    held = sorted({h["symbol"].upper() for h in
                   json.loads(_HOLDINGS.read_text(encoding="utf-8")).get("holdings", [])})

    with get_connection(read_only=True) as con:
        have_bars = {r[0] for r in con.execute(
            "SELECT DISTINCT symbol FROM market_bars"
        ).fetchall()}

    if since > 0:
        symbols = held                                  # refresh everything, recent
        start = date.today() - timedelta(days=since)
    else:
        symbols = [s for s in held if s not in have_bars]   # only missing, full
        start = date.today() - timedelta(days=settings.data.default_history_days)

    if not symbols:
        console.print("[green]All held names already have bars — nothing to do.[/]")
        raise typer.Exit(0)

    console.print(f"[bold cyan]Backfilling {len(symbols)} held names from {start}[/]")
    console.print(f"[dim]{', '.join(symbols)}[/]")

    yf = YFinanceProvider()
    total = 0
    failed: list[str] = []
    for i in range(0, len(symbols), 10):
        chunk = symbols[i:i + 10]
        try:
            bars = list(yf.fetch(chunk, start=start))
            with get_connection() as con:
                upsert_market_bars(con, bars)
            total += len(bars)
        except Exception as exc:
            log.error("holdings_bar_chunk_failed", chunk=chunk, error=str(exc))
            failed.extend(chunk)

    console.print(f"  stored [green]{total}[/] bars.")
    # Report which held names still have no bars (yfinance gaps).
    with get_connection(read_only=True) as con:
        have = {r[0] for r in con.execute("SELECT DISTINCT symbol FROM market_bars").fetchall()}
    still_missing = [s for s in held if s not in have]
    if still_missing:
        console.print(f"[yellow]Still no bars (yfinance had no data): "
                      f"{', '.join(still_missing)}[/]")


if __name__ == "__main__":
    app()
