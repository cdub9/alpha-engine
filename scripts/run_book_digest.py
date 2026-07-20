"""Run the LLM digest on a universe that INCLUDES the real holdings.

The nightly digest evaluates the paper universe. This points the same
holistic engine (regime + macro + GDELT + technicals + feedback) at the
user's ACTUAL book by expanding the universe to include every held name
that has price history, so the digest forms a view on the real portfolio
and can surface buy/hold/trim ideas for names the paper universe ignores.

Tagged with a distinct model_version ('...-book') and cached so the Action
Center's holistic overlay picks it up — WITHOUT persisting signals into the
paper cohort (so the paper track record / forward-validation stays clean).

THIS COSTS MONEY — one primary Opus call (~$0.16-0.25 depending on how many
held names get added to the universe). Free otherwise.

    python scripts/run_book_digest.py
    python scripts/run_book_digest.py --primary-model claude-sonnet-4-6
"""

from __future__ import annotations

import json
import sys
from datetime import date, datetime
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

from alpha_engine.backtest.llm_advisor import (
    DEFAULT_PRIMARY_MODEL,
    config_hash,
    model_version_for,
    store_cached_output,
)
from alpha_engine.core.config import get_settings
from alpha_engine.core.logging import configure_logging
from alpha_engine.db import get_connection, init_schema
from alpha_engine.llm.prompts import SYSTEM_PROMPT

console = Console()
app = typer.Typer(add_completion=False, help=__doc__)

_HOLDINGS = Path(__file__).resolve().parent.parent / "data" / "real_holdings.json"


def _held_symbols() -> list[str]:
    if not _HOLDINGS.exists():
        return []
    snap = json.loads(_HOLDINGS.read_text(encoding="utf-8"))
    return [h["symbol"].upper() for h in snap.get("holdings", [])]


@app.command()
def main(
    as_of: str = typer.Option("", help="YYYY-MM-DD; defaults to today"),
    primary_model: str = typer.Option(DEFAULT_PRIMARY_MODEL, "--primary-model"),
    effort: str = typer.Option("high"),
    log_level: str = typer.Option("WARNING", "--log-level"),
) -> None:
    configure_logging(level=log_level)
    settings = get_settings()
    if not settings.anthropic_api_key:
        console.print("[red]ANTHROPIC_API_KEY not set; cannot generate.[/]")
        raise typer.Exit(1)
    init_schema()
    target = datetime.strptime(as_of, "%Y-%m-%d").date() if as_of else date.today()

    held = _held_symbols()
    with get_connection(read_only=True) as con:
        active = [r[0] for r in con.execute(
            "SELECT symbol FROM instruments WHERE active = TRUE ORDER BY symbol"
        ).fetchall()]
        have_bars = {r[0] for r in con.execute(
            "SELECT DISTINCT symbol FROM market_bars"
        ).fetchall()}

    held_with_bars = sorted(s for s in held if s in have_bars)
    added = sorted(set(held_with_bars) - set(active))
    universe = sorted(set(active) | set(held_with_bars))
    book_mv = model_version_for(primary_model) + "-book"

    console.print(f"[yellow]Book digest for {target} on {len(universe)} names "
                  f"({len(added)} held names added to the paper universe).[/] "
                  f"Model {primary_model}, cohort [bold]{book_mv}[/]. Costs ~$0.20.")
    if added:
        console.print(f"[dim]Added held names: {', '.join(added)}[/]")
    missing = sorted(set(held) - have_bars)
    if missing:
        console.print(f"[dim]Held names with no bars (skipped — mostly ETFs): "
                      f"{', '.join(missing)}[/]")

    from alpha_engine.llm.digest import run_digest

    run = run_digest(
        as_of=target,
        universe=universe,
        enable_dissent=False,
        persist=False,                 # keep the paper cohort clean
        primary_model=primary_model,
        model_version=book_mv,
        effort=effort,
    )

    cfg_hash = config_hash(SYSTEM_PROMPT, universe)
    with get_connection() as con:
        store_cached_output(
            con, as_of=target, model_version=book_mv, cfg_hash=cfg_hash,
            output=run.final_output, universe=universe,
            input_tokens=run.primary_response.input_tokens,
            output_tokens=run.primary_response.output_tokens,
            cost_usd=run.total_cost_usd,
        )
    console.print(f"[green]Book digest cached. Cost: ${run.total_cost_usd:.4f}[/]")

    # Show the digest's view on HELD names specifically.
    out = run.final_output
    held_set = set(held)
    table = Table(title="Digest view on your holdings")
    table.add_column("Symbol"); table.add_column("Channel")
    table.add_column("Dir"); table.add_column("Conv", justify="right")
    table.add_column("Rationale (excerpt)", max_width=70)
    rows = 0
    for ch_key, ch in (("channel_a_suggestions", "steady"),
                       ("channel_b_suggestions", "aggressive")):
        for s in out.get(ch_key, []):
            sym = (s.get("symbol") or "").upper()
            if sym in held_set:
                table.add_row(sym, ch, s.get("direction", ""),
                              f"{s.get('conviction', 0):.1f}",
                              (s.get("rationale", "") or "")[:120])
                rows += 1
    if rows:
        console.print(table)
    else:
        console.print("[dim]The digest made no explicit buy/add/trim call on a held "
                      "name today (i.e. hold everything). Market context is still cached.[/]")


if __name__ == "__main__":
    app()
