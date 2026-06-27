"""Remove duplicate paper-trading signals (and their trades/outcomes).

Re-runs during development (manual `run-day` invocations, the v1->v3 backfill)
occasionally persisted the same (date, channel, symbol, direction, model)
suggestion twice under different signal IDs, which then opened duplicate
paper trades. This collapses each such group to ONE signal.

Why dedup at the SIGNAL level, not just the trade: deleting only the
duplicate trade would leave an orphaned signal that the next nightly run
re-opens — recreating the duplicate. Removing the losing signal (and its
trade + outcome) makes it stick.

Keeper per group: the signal whose trade has already been SCORED (preserve
the realized outcome) if any, else the lowest signal id. Same-cohort only —
groups are keyed on model_version, so v1 and v3 are never merged.

Dry-run by default; pass --apply to actually delete. Back up the DB first.

    python scripts/dedup_paper_trades.py            # show what would change
    python scripts/dedup_paper_trades.py --apply    # do it
"""

from __future__ import annotations

import sys

import typer
from rich.console import Console
from rich.table import Table

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

from alpha_engine.db import get_connection

console = Console()
app = typer.Typer(add_completion=False, help=__doc__)


@app.command()
def main(apply: bool = typer.Option(False, "--apply", help="Actually delete (default: dry run)")) -> None:
    with get_connection() as con:
        # Duplicate = >1 paper_filled trade with the same channel + symbol +
        # direction + ENTRY DATE. Keyed on the trade (not the signal) because
        # re-runs sometimes stamp the duplicate signals on different
        # generated dates, so a signal-date key misses them.
        rows = con.execute(
            """
            WITH dup AS (
                SELECT channel, symbol, LOWER(direction) AS dir, placed_at::DATE AS d
                FROM trades WHERE status = 'paper_filled'
                GROUP BY 1, 2, 3, 4
                HAVING COUNT(*) > 1
            )
            SELECT t.id AS trade_id, t.source_signal_id, t.placed_at::DATE AS d,
                   t.channel, t.symbol, LOWER(t.direction) AS dir, s.model_version,
                   CASE WHEN o.trade_id IS NOT NULL THEN 1 ELSE 0 END AS scored
            FROM trades t
            JOIN dup ON dup.channel = t.channel AND dup.symbol = t.symbol
                    AND dup.dir = LOWER(t.direction) AND dup.d = t.placed_at::DATE
            LEFT JOIN signals s ON s.id = t.source_signal_id
            LEFT JOIN trade_outcomes o ON o.trade_id = t.id
            WHERE t.status = 'paper_filled'
            -- keeper first: scored trade preferred, then lowest id
            ORDER BY d, t.channel, t.symbol, dir, scored DESC, t.id
            """
        ).fetchall()

        if not rows:
            console.print("[green]No duplicate trades found — nothing to do.[/]")
            return

        # columns: trade_id, source_signal_id, d, channel, symbol, dir, mv, scored
        groups: dict = {}
        for r in rows:
            key = (r[2], r[3], r[4], r[5])  # date, channel, symbol, dir
            groups.setdefault(key, []).append(r)

        loser_trades: list[int] = []
        loser_signals: list[int] = []
        table = Table(title="Duplicate trade groups", header_style="bold")
        for c in ("Entry date", "Channel", "Symbol", "Dir", "Cohort", "Keep trade", "Drop trades"):
            table.add_column(c)
        for key, members in groups.items():
            keeper, losers = members[0], members[1:]
            for L in losers:
                loser_trades.append(L[0])
                if L[1] is not None:
                    loser_signals.append(L[1])
            d, channel, symbol, dir_ = key
            table.add_row(
                str(d), channel[:12], symbol, dir_, keeper[6] or "?",
                f"{keeper[0]}{' (scored)' if keeper[7] else ''}",
                ", ".join(str(L[0]) for L in losers),
            )
        console.print(table)

        n_oc = 0
        if loser_trades:
            ph = ",".join(["?"] * len(loser_trades))
            n_oc = con.execute(
                f"SELECT COUNT(*) FROM trade_outcomes WHERE trade_id IN ({ph})",
                loser_trades,
            ).fetchone()[0]

        console.print(
            f"\n[bold]{len(groups)}[/] duplicate group(s): would remove "
            f"[red]{len(loser_trades)}[/] trades, [red]{n_oc}[/] outcomes, and "
            f"[red]{len(loser_signals)}[/] orphaned source signals "
            f"(keeping one trade per group)."
        )

        if not apply:
            console.print("\n[yellow]Dry run.[/] Re-run with [bold]--apply[/] to delete.")
            return

        con.execute("BEGIN")
        if loser_trades:
            ph = ",".join(["?"] * len(loser_trades))
            con.execute(f"DELETE FROM trade_outcomes WHERE trade_id IN ({ph})", loser_trades)
            con.execute(f"DELETE FROM trades WHERE id IN ({ph})", loser_trades)
        if loser_signals:
            phs = ",".join(["?"] * len(loser_signals))
            con.execute(f"DELETE FROM signals WHERE id IN ({phs})", loser_signals)
        con.execute("COMMIT")
        console.print(
            f"\n[green]Applied.[/] Removed {len(loser_trades)} duplicate trades, "
            f"{n_oc} outcomes, {len(loser_signals)} source signals."
        )


if __name__ == "__main__":
    app()
