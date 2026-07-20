"""Log the Action Center's opportunity ideas + show their track record.

The Phase 3 learning loop: this records today's add/trim ideas (so they can
be scored forward once ~21 trading days pass) and prints how the ideas
recorded so far have actually done vs SPY. Free — pure local compute.

Run after the book digest so it captures the freshest ideas:
    python scripts/record_recommendations.py

Intended to run daily (wired into the morning routine).
"""

from __future__ import annotations

import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from alpha_engine.analysis.reco_tracker import (
    record_recommendations,
    score_recommendations,
)
from alpha_engine.core.logging import configure_logging
from alpha_engine.db import get_connection, init_schema
from dashboard import queries as q

console = Console()
app = typer.Typer(add_completion=False, help=__doc__)


@app.command()
def main(log_level: str = typer.Option("WARNING", "--log-level")) -> None:
    configure_logging(level=log_level)
    init_schema()

    data = q.portfolio_action_center()
    if data is None:
        console.print("[yellow]No holdings snapshot — nothing to record.[/]")
        raise typer.Exit(0)
    opp = data.get("opportunity") or {"adds": [], "trims": []}

    with get_connection() as con:
        # Record against SPY's latest close — the most recent session every
        # liquid name shares — so the forward scorer has an aligned entry bar
        # (a symbol backfilled to a later date shouldn't push the reco date
        # past the names being recommended).
        as_of = con.execute(
            "SELECT MAX(bar_date) FROM market_bars WHERE symbol = 'SPY'"
        ).fetchone()[0]
        n = record_recommendations(con, as_of, opp)
        score = score_recommendations(con)

    console.print(f"[green]Recorded {n} opportunity idea(s) as of {as_of}[/] "
                  f"({len(opp['adds'])} add, {len(opp['trims'])} trim).")

    console.print(
        f"\n[bold]Recommendation track record[/] (vs {score['benchmark']}, "
        f"{score['horizon']}-day forward):"
    )
    if score["n_matured"] == 0:
        console.print(
            f"  [dim]{score['n_total']} idea(s) logged, 0 matured yet — first "
            f"scores in ~{score['horizon']} trading days. Accumulating.[/]"
        )
        return

    table = Table()
    table.add_column("Kind"); table.add_column("Matured", justify="right")
    table.add_column("Hit rate", justify="right"); table.add_column("Avg alpha", justify="right")
    for kind in ("add", "trim"):
        k = score["by_kind"][kind]
        if k["n"]:
            table.add_row(kind, str(k["n"]), f"{k['hit_rate']:.0%}", f"{k['avg_alpha']:+.1%}")
    console.print(table)
    console.print(f"  overall hit rate: {score['overall_hit_rate']:.0%} "
                  f"({score['n_matured']} matured, {score['n_pending']} pending)")


if __name__ == "__main__":
    app()
