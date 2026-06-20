"""Backfill entry-timing counterfactuals onto already-scored trades, then
report how much the next-close vs next-open latency has cost historically.

Completely free — pure local recompute on bars already in the DB. The
forward pipeline records both entry prices automatically going forward;
this one-off fills the gap for trades scored before that shipped, so the
dashboard's "Execution timing" panel has data immediately instead of
waiting weeks for new trades to mature.

For each scored trade missing alt data:
  - entry bar = the trade's entry date (the first session after the digest)
  - the trade's stored `price` is its actual-style entry; we look up the
    OTHER style's price on that same bar (next_open -> need the close,
    next_close -> need the adjusted open)
  - the exit price is recovered exactly from the stored return:
        exit = entry * (1 + direction_sign * return_pct)
  - alt_entry_return_pct = direction_sign * (exit - alt_entry_price) / alt

Idempotent: only fills rows where alt data is still NULL.

Usage:
    python scripts/measure_entry_timing.py
    python scripts/measure_entry_timing.py --dry-run
"""

from __future__ import annotations

import sys

import typer
from rich.console import Console
from rich.table import Table

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

from alpha_engine.core.logging import configure_logging
from alpha_engine.db import get_connection, init_schema

console = Console()
app = typer.Typer(add_completion=False, help=__doc__)

_DIRECTION_SIGN = {"buy": 1, "add": 1, "hold": 1, "sell": -1, "exit": -1, "reduce": -1}


def _adjusted_open(con, symbol, bar_date):
    row = con.execute(
        "SELECT open, close, adj_close FROM market_bars WHERE symbol = ? AND bar_date = ?",
        [symbol, bar_date],
    ).fetchone()
    if not row:
        return None
    open_, close_, adj_close = row
    if open_ and close_ and float(close_) > 0:
        return float(open_) * float(adj_close) / float(close_)
    return float(adj_close)


def _adjusted_close(con, symbol, bar_date):
    row = con.execute(
        "SELECT adj_close FROM market_bars WHERE symbol = ? AND bar_date = ?",
        [symbol, bar_date],
    ).fetchone()
    return float(row[0]) if row else None


@app.command()
def main(
    dry_run: bool = typer.Option(False, "--dry-run", help="Compute + report, write nothing."),
    log_level: str = typer.Option("WARNING", "--log-level"),
) -> None:
    configure_logging(level=log_level)
    init_schema()

    with get_connection() as con:
        rows = con.execute(
            """
            SELECT t.id, t.symbol, t.direction, t.entry_style, t.price,
                   t.placed_at::DATE AS entry_date, o.return_pct, o.notes
            FROM trades t
            JOIN trade_outcomes o ON o.trade_id = t.id
            WHERE o.alt_entry_return_pct IS NULL
              AND t.alt_entry_price IS NULL
              AND (o.notes IS NULL OR o.notes NOT LIKE '%no exit%')
            """
        ).fetchall()

        filled = 0
        skipped = 0
        for tid, symbol, direction, entry_style, price, entry_date, return_pct, _notes in rows:
            ds = _DIRECTION_SIGN.get((direction or "").lower(), 0)
            if ds == 0 or price is None or float(price) <= 0:
                skipped += 1
                continue
            # The OTHER style's entry price on the same bar.
            if entry_style == "next_open":
                alt = _adjusted_close(con, symbol, entry_date)   # legacy close
            else:
                alt = _adjusted_open(con, symbol, entry_date)    # the better open
            if alt is None or alt <= 0:
                skipped += 1
                continue
            # Recover the exact exit price the scorer used.
            exit_price = float(price) * (1.0 + ds * float(return_pct))
            alt_return = ds * (exit_price - alt) / alt
            if not dry_run:
                con.execute("UPDATE trades SET alt_entry_price = ? WHERE id = ?", [alt, tid])
                con.execute(
                    "UPDATE trade_outcomes SET alt_entry_return_pct = ? WHERE trade_id = ?",
                    [alt_return, tid],
                )
            filled += 1

        # Report the aggregate gap from the freshly-complete data.
        summary = con.execute(
            """
            SELECT t.entry_style,
                   COUNT(*) AS n,
                   AVG(CASE WHEN t.entry_style='next_open' THEN o.return_pct
                            ELSE o.alt_entry_return_pct END) AS avg_open,
                   AVG(CASE WHEN t.entry_style='next_open' THEN o.alt_entry_return_pct
                            ELSE o.return_pct END) AS avg_close
            FROM trades t JOIN trade_outcomes o ON o.trade_id = t.id
            WHERE o.alt_entry_return_pct IS NOT NULL
              AND (o.notes IS NULL OR o.notes NOT LIKE '%no exit%')
            GROUP BY 1
            """
        ).fetchall()

    console.print(
        f"[green]{'Would fill' if dry_run else 'Filled'} {filled}[/] trades "
        f"({skipped} skipped for missing data)."
    )
    table = Table(title="Entry-timing comparison (next-open vs next-close, same exit)")
    table.add_column("Cohort")
    table.add_column("Scored", justify="right")
    table.add_column("Avg next-OPEN ret", justify="right")
    table.add_column("Avg next-CLOSE ret", justify="right")
    table.add_column("Gap (open − close)", justify="right")
    tot_n = 0
    for style, n, avg_open, avg_close in summary:
        gap = (avg_open or 0) - (avg_close or 0)
        tot_n += n
        table.add_row(style, str(n), f"{avg_open:+.2%}", f"{avg_close:+.2%}", f"{gap:+.3%}")
    console.print(table)
    if tot_n:
        console.print(
            "\n[bold]Gap > 0 means entering at the next OPEN beats waiting for the "
            "next close[/] — i.e. the latency was costing us that much per trade, "
            "on average, over the same holding window."
        )


if __name__ == "__main__":
    app()
