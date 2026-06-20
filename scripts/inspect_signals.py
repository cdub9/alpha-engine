"""Inspect signals in the database."""

from __future__ import annotations

import sys

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

from rich.console import Console
from rich.table import Table

from alpha_engine.db import get_connection

console = Console()

with get_connection(read_only=True) as con:
    total = con.execute("SELECT COUNT(*) FROM signals").fetchone()[0]
    console.print(f"[bold]Total signals in DB: {total}[/]\n")

    for channel in ("steady_alpha", "aggressive_growth"):
        rows = con.execute(
            """
            SELECT symbol, direction, conviction, target_weight, time_horizon_days,
                   stop_loss_pct, model_version, generated_at, rationale
            FROM signals
            WHERE channel = ?
            ORDER BY conviction DESC, generated_at DESC
            """,
            [channel],
        ).fetchall()
        table = Table(title=f"Channel: {channel} ({len(rows)} signals)")
        table.add_column("Symbol")
        table.add_column("Dir", justify="center")
        table.add_column("Conv", justify="right")
        table.add_column("Wt", justify="right")
        table.add_column("Horiz", justify="right")
        table.add_column("Stop", justify="right")
        table.add_column("Rationale (excerpt)")
        for r in rows:
            sym, direction, conv, wt, horiz, stop, mv, gen, rationale = r
            table.add_row(
                sym,
                direction,
                f"{conv:.1f}",
                f"{wt:.1%}" if wt else "—",
                f"{horiz}d" if horiz else "—",
                f"{stop:.1%}" if stop else "—",
                (rationale or "")[:60] + ("..." if len(rationale or "") > 60 else ""),
            )
        console.print(table)
        console.print()
