"""Phase C: pull current S&P 500 list and backfill price bars only.

We backfill `market_bars` for every S&P 500 constituent that isn't
already in our `instruments` table. We deliberately DO NOT insert into
`instruments` — those rows would make the symbol active in the LLM's
universe, which would blow our snapshot context budget (500 tickers per
prompt is too much).

Bars-only backfill means:
  - Backtests can reference any S&P symbol immediately
  - Future code that wants to selectively expose a subset to the LLM
    (e.g. "biggest movers today + earnings this week") has the data ready
  - The current LLM universe stays tight (~103 after Phase B) for prompt
    economy

Source for S&P 500 list: Wikipedia's "List of S&P 500 companies" page.
Idempotent (yfinance bars upsert by (symbol, bar_date)).

Caveat — survivorship bias: this list is *today's* S&P 500. Companies
delisted or removed from the index don't appear. Backtests on individual
names will overstate returns. Use SPY/VOO for clean backtests until we
add point-in-time index membership (separate followup).
"""

from __future__ import annotations

import sys
from datetime import date, timedelta

import pandas as pd
import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

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

# Maintained S&P 500 constituents CSV; updated periodically by the dataset
# project. More reliable than Wikipedia (which blocks default pandas UA).
SP500_CSV = (
    "https://raw.githubusercontent.com/datasets/"
    "s-and-p-500-companies/main/data/constituents.csv"
)


def fetch_sp500_symbols() -> list[str]:
    """Fetch current S&P 500 constituents from the datasets/s-and-p-500-companies CSV."""
    try:
        df = pd.read_csv(SP500_CSV)
    except Exception as e:  # noqa: BLE001
        console.print(f"[red]Failed to fetch S&P 500 CSV:[/] {e}")
        raise SystemExit(1)
    if "Symbol" not in df.columns:
        console.print(f"[red]Unexpected CSV format: cols={list(df.columns)}[/]")
        raise SystemExit(1)
    # yfinance uses dash for class-shares (BRK.B -> BRK-B); CSV uses dots
    symbols = [s.replace(".", "-").upper().strip() for s in df["Symbol"].dropna().tolist()]
    return sorted(set(symbols))


def existing_symbols(con) -> set[str]:
    """Symbols already in our instruments table (the LLM universe)."""
    return {r[0] for r in con.execute("SELECT symbol FROM instruments").fetchall()}


def existing_bars_symbols(con) -> set[str]:
    """Symbols that already have bars in market_bars (any time)."""
    return {r[0] for r in con.execute("SELECT DISTINCT symbol FROM market_bars").fetchall()}


app = typer.Typer(add_completion=False, help=__doc__)


@app.command()
def main(
    years: int = typer.Option(5, help="Years of history to backfill (initial load)"),
    since: int = typer.Option(
        0,
        "--since",
        help=(
            "Incremental mode: fetch last N days for ALL Phase C symbols. "
            "Implies --no-skip-existing-bars. Use this in the daily scheduler "
            "(e.g. --since 7). 0 = use --years (initial/full backfill mode)."
        ),
    ),
    batch_size: int = typer.Option(20, help="Symbols per yfinance call"),
    skip_existing_bars: bool = typer.Option(
        True,
        help="Skip symbols that already have any bars in market_bars. Auto-disabled when --since > 0.",
    ),
    log_level: str = typer.Option("WARNING", "--log-level"),
) -> None:
    """[FREE] Bars-only backfill for current S&P 500 not already in our universe."""
    configure_logging(level=log_level)
    init_schema()

    # --since implies incremental mode: refresh ALL Phase C symbols, short window.
    incremental = since > 0
    if incremental:
        skip_existing_bars = False  # must refresh symbols that already have older bars

    console.print("Fetching S&P 500 list from GitHub CSV...")
    sp500 = fetch_sp500_symbols()
    console.print(f"  → {len(sp500)} symbols on the list")

    with get_connection(read_only=True) as con:
        in_universe = existing_symbols(con)
        have_bars = existing_bars_symbols(con) if skip_existing_bars else set()

    todo = [s for s in sp500 if s not in in_universe and s not in have_bars]
    mode_label = f"incremental ({since}d)" if incremental else f"initial ({years}y)"
    console.print(
        f"  → {len(in_universe)} already in instruments universe (skipped)\n"
        f"  → {len(have_bars - in_universe)} already have bars, skipped (skip_existing_bars={skip_existing_bars})\n"
        f"  → {len(todo)} symbols to refresh  [{mode_label}]"
    )
    if not todo:
        console.print("[green]Nothing to do.[/]")
        return

    provider = YFinanceProvider()
    if incremental:
        start = date.today() - timedelta(days=since)
    else:
        start = date.today() - timedelta(days=365 * years)
    end = date.today()

    total_bars = 0
    failures: list[str] = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Backfilling S&P 500…", total=len(todo))
        for i in range(0, len(todo), batch_size):
            batch = todo[i : i + batch_size]
            try:
                bars = list(provider.fetch(batch, start=start, end=end))
            except Exception as e:  # noqa: BLE001
                log.warning("sp500_batch_failed", batch=batch, error=str(e))
                failures.extend(batch)
                progress.advance(task, len(batch))
                continue
            if bars:
                with get_connection() as con:
                    upsert_market_bars(con, bars)
                total_bars += len(bars)
            progress.advance(task, len(batch))

    console.print(
        f"\n[green]Done.[/] Inserted/updated [bold]{total_bars}[/] bars "
        f"across [bold]{len(todo)}[/] symbols."
    )
    if failures:
        console.print(f"[yellow]Failed batches contained {len(failures)} symbols.[/]")
        console.print("  examples:", ", ".join(failures[:10]))


if __name__ == "__main__":
    app()
