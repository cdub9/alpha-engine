"""Refresh the earnings calendar for the names that actually matter.

The nightly seed only covers universe *equities*, so the real brokerage
book's names (SNDK, STX, NBIS, COHR, MRVL, ...) never got earnings dates —
which left the Action Center's earnings-trim guard half-blind. This pulls
fresh earnings dates for the union of:
  - symbols in data/real_holdings.json (the real book), and
  - universe equities,
and upserts them into calendar_events. ETFs return no earnings data from
yfinance and are skipped silently.

Free (yfinance), but ~1-2s per symbol and rate-limited, so it's a
standalone refresh rather than a nightly step. Weekly is plenty — earnings
dates barely move.

Usage:
    python scripts/refresh_earnings.py                 # holdings + universe
    python scripts/refresh_earnings.py --only-holdings # just the real book (fast)
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

from alpha_engine.calendars.earnings import (
    fetch_earnings_calendar,
    upsert_earnings_events,
)
from alpha_engine.core.config import get_settings
from alpha_engine.core.logging import configure_logging
from alpha_engine.db import get_connection, init_schema

console = Console()
app = typer.Typer(add_completion=False, help=__doc__)

_HOLDINGS = Path(__file__).resolve().parent.parent / "data" / "real_holdings.json"


def _holdings_symbols() -> list[str]:
    if not _HOLDINGS.exists():
        return []
    try:
        snap = json.loads(_HOLDINGS.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    return [h["symbol"].upper() for h in snap.get("holdings", [])]


def _universe_equities() -> list[str]:
    s = get_settings()
    return sorted({
        spec.symbol
        for items in s.universe.universes.values()
        for spec in items
        if spec.type.value == "equity"
    })


@app.command()
def main(
    only_holdings: bool = typer.Option(False, "--only-holdings"),
    log_level: str = typer.Option("WARNING", "--log-level"),
) -> None:
    configure_logging(level=log_level)
    init_schema()

    held = _holdings_symbols()
    symbols = sorted(set(held) if only_holdings else set(held) | set(_universe_equities()))
    if not symbols:
        console.print("[yellow]No symbols to refresh (no holdings snapshot, "
                      "no universe equities).[/]")
        raise typer.Exit(1)

    console.print(f"[bold cyan]Fetching earnings for {len(symbols)} symbols[/] "
                  f"({len(held)} from holdings)…")
    events = fetch_earnings_calendar(symbols)
    with get_connection() as con:
        n = upsert_earnings_events(con, events)
    console.print(f"  upserted [green]{n}[/] earnings events.")

    # Show upcoming earnings for HELD names — the reason we ran this.
    today = date.today()
    if held:
        ph = ",".join(["?"] * len(held))
        with get_connection(read_only=True) as con:
            rows = con.execute(
                f"""
                SELECT symbol, MIN(event_date) AS d
                FROM calendar_events
                WHERE kind = 'earnings' AND symbol IN ({ph}) AND event_date >= ?
                GROUP BY symbol ORDER BY 2
                """,
                [*held, today],
            ).fetchall()
        table = Table(title="Upcoming earnings for held names")
        table.add_column("Symbol")
        table.add_column("Next earnings")
        table.add_column("Days away", justify="right")
        for sym, d in rows[:40]:
            table.add_row(sym, str(d), str((d - today).days))
        console.print(table)
        if not rows:
            console.print("[yellow]No upcoming earnings found for held names "
                          "(yfinance coverage gaps, or none scheduled).[/]")


if __name__ == "__main__":
    app()
