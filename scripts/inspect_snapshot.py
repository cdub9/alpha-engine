"""Build and print today's snapshot (the user message sent to the LLM)
without making an API call. Useful for validating context.py output and
estimating prompt size before paying for a real digest.

Usage:
    python scripts/inspect_snapshot.py
    python scripts/inspect_snapshot.py --as-of 2026-05-20
"""

from __future__ import annotations

import sys
from datetime import date, datetime

import typer
from rich.console import Console

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

from alpha_engine.db import get_connection
from alpha_engine.llm import build_snapshot
from alpha_engine.llm.prompts import SYSTEM_PROMPT, USER_MESSAGE_TEMPLATE

console = Console()
app = typer.Typer(add_completion=False, help=__doc__)


@app.command()
def main(as_of: str = typer.Option("", help="YYYY-MM-DD")) -> None:
    target = (
        datetime.strptime(as_of, "%Y-%m-%d").date() if as_of else date.today()
    )

    with get_connection(read_only=True) as con:
        symbols = [
            r[0] for r in con.execute(
                "SELECT symbol FROM instruments WHERE active = TRUE ORDER BY symbol"
            ).fetchall()
        ]
        snap = build_snapshot(con, universe=symbols, as_of=target)

    user_msg = USER_MESSAGE_TEMPLATE.format(snapshot_markdown=snap.markdown)

    console.rule(f"[bold cyan]Snapshot for {snap.as_of}[/]")
    console.print(snap.markdown)
    console.rule("[bold cyan]End snapshot[/]")

    # Rough token estimate: 1 token ~= 4 chars for English
    sys_chars = len(SYSTEM_PROMPT)
    user_chars = len(user_msg)
    sys_tokens = sys_chars // 4
    user_tokens = user_chars // 4

    console.print()
    console.print(f"[bold]Size estimates[/] (rough — 4 chars/token):")
    console.print(f"  system prompt: {sys_chars:,} chars ≈ {sys_tokens:,} tokens (CACHED)")
    console.print(f"  user message:  {user_chars:,} chars ≈ {user_tokens:,} tokens (per-call)")
    console.print(f"  universe size: {len(snap.universe)} symbols")
    console.print(f"  notable events: {snap.notable_events or '(none)'}")


if __name__ == "__main__":
    app()
