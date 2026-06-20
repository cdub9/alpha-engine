"""Resume the 2007 backfill for symbols still missing deep history.

The full backfill (backfill.py --start 2007) is idempotent but re-fetches
everything; this one only pulls symbols whose earliest bar is after a
cutoff (default 2010-01-01), so an interrupted run resumes cheaply. Pure
yfinance — free. Safe to re-run; it just re-checks coverage each time.

    python scripts/backfill_resume.py
    python scripts/backfill_resume.py --cutoff 2010-01-01 --start 2007-01-01
"""

from __future__ import annotations

import sys
from datetime import date, datetime

import typer
from rich.console import Console

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

from alpha_engine.core.logging import configure_logging, get_logger
from alpha_engine.data import YFinanceProvider, upsert_market_bars
from alpha_engine.db import get_connection, init_schema

console = Console()
log = get_logger(__name__)
app = typer.Typer(add_completion=False, help=__doc__)


@app.command()
def main(
    start: str = typer.Option("2007-01-01", help="Fetch history from this date."),
    cutoff: str = typer.Option(
        "2010-01-01",
        help="A symbol is 'shallow' (needs refetch) if its earliest bar is after this.",
    ),
    chunk_size: int = typer.Option(8, help="Symbols per yfinance call."),
    log_level: str = typer.Option("WARNING", "--log-level"),
) -> None:
    configure_logging(level=log_level)
    init_schema()
    start_d = datetime.strptime(start, "%Y-%m-%d").date()
    cutoff_d = datetime.strptime(cutoff, "%Y-%m-%d").date()

    with get_connection(read_only=True) as con:
        shallow = [
            r[0]
            for r in con.execute(
                """
                SELECT i.symbol
                FROM instruments i JOIN market_bars mb ON mb.symbol = i.symbol
                WHERE i.active
                GROUP BY i.symbol
                HAVING MIN(mb.bar_date) > ?
                ORDER BY i.symbol
                """,
                [cutoff_d],
            ).fetchall()
        ]

    if not shallow:
        console.print("[green]No shallow symbols — every active instrument already has deep history.[/]")
        return

    console.print(
        f"[bold cyan]Resuming backfill[/] for {len(shallow)} shallow symbol(s) "
        f"from {start_d} (earliest bar currently after {cutoff_d})."
    )

    yf = YFinanceProvider()
    total = 0
    done = 0
    for i in range(0, len(shallow), chunk_size):
        chunk = shallow[i : i + chunk_size]
        try:
            bars = list(yf.fetch(chunk, start=start_d))
            with get_connection() as con:
                upsert_market_bars(con, bars)
            total += len(bars)
            done += len(chunk)
            console.print(f"  [{done}/{len(shallow)}] {', '.join(chunk)} → {len(bars)} bars")
        except Exception as exc:
            log.error("resume_chunk_failed", chunk=chunk, error=str(exc))
            console.print(f"  [red]chunk failed[/] {chunk}: {exc}")

    console.print(f"\n[bold green]Done.[/] Upserted {total:,} bars across {done} symbols.")

    # Re-report what (if anything) is still shallow.
    with get_connection(read_only=True) as con:
        still = con.execute(
            """
            SELECT i.symbol, MIN(mb.bar_date)
            FROM instruments i JOIN market_bars mb ON mb.symbol = i.symbol
            WHERE i.active GROUP BY i.symbol HAVING MIN(mb.bar_date) > ?
            ORDER BY MIN(mb.bar_date) DESC, i.symbol
            """,
            [cutoff_d],
        ).fetchall()
    if still:
        console.print(
            f"[yellow]{len(still)} still start after {cutoff_d} (likely real inception, "
            f"not a gap):[/] " + ", ".join(f"{s}({d})" for s, d in still[:20])
        )


if __name__ == "__main__":
    app()
