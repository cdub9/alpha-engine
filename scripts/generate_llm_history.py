"""Pre-generate and cache LLM digests for historical rebalance dates.

This is the expensive step (one Opus call per date). Run it in the
background; results are cached in llm_signal_cache so the actual backtest
(scripts/backtest_llm.py) runs instantly and re-runs are free.

>>> IMPORTANT: Historical LLM backtests are training-data contaminated.
    See alpha_engine/backtest/llm_advisor.py module docstring. Results are
    an optimistic upper bound, not a forward estimate.

Usage:
    python scripts/generate_llm_history.py --start 2024-12-01 --frequency monthly
    python scripts/generate_llm_history.py --start 2025-06-01 --frequency weekly
    python scripts/generate_llm_history.py --start 2025-01-01 --with-dissent
"""

from __future__ import annotations

import sys
from datetime import date, datetime, timedelta

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

from alpha_engine.backtest.engine import _rebalance_dates
from alpha_engine.backtest.llm_advisor import (
    DEFAULT_MODEL_VERSION,
    config_hash,
    get_cached_output,
    store_cached_output,
)
from alpha_engine.core.config import get_settings
from alpha_engine.core.logging import configure_logging, get_logger
from alpha_engine.db import get_connection, init_schema
from alpha_engine.llm.digest import run_digest
from alpha_engine.llm.prompts import SYSTEM_PROMPT

console = Console()
log = get_logger(__name__)
app = typer.Typer(add_completion=False, help=__doc__)


def _universe(con) -> list[str]:
    rows = con.execute(
        "SELECT symbol FROM instruments WHERE active = TRUE ORDER BY symbol"
    ).fetchall()
    return [r[0] for r in rows]


def _trading_days(con, start: date, end: date) -> pd.DatetimeIndex:
    rows = con.execute(
        "SELECT DISTINCT bar_date FROM market_bars "
        "WHERE symbol = 'SPY' AND bar_date BETWEEN ? AND ? ORDER BY bar_date",
        [start, end],
    ).fetchall()
    return pd.DatetimeIndex([pd.Timestamp(r[0]) for r in rows])


@app.command()
def generate(
    start: str = typer.Option(..., help="YYYY-MM-DD start of backtest window"),
    end: str = typer.Option("", help="YYYY-MM-DD end; defaults to today"),
    frequency: str = typer.Option(
        "monthly", help="daily|weekly|biweekly|monthly (rebalance cadence)"
    ),
    model_version: str = typer.Option(DEFAULT_MODEL_VERSION),
    with_dissent: bool = typer.Option(
        False,
        "--with-dissent",
        help="Include the dissent overlay. Default off for backtesting raw signal.",
    ),
    effort: str = typer.Option("high", help="Primary call effort level"),
    force: bool = typer.Option(
        False, "--force", help="Regenerate even if a date is already cached"
    ),
    log_level: str = typer.Option("WARNING", "--log-level"),
) -> None:
    configure_logging(level=log_level)
    settings = get_settings()
    if not settings.anthropic_api_key:
        console.print("[red]ANTHROPIC_API_KEY not set.[/]")
        raise typer.Exit(1)

    init_schema()

    end_d = datetime.strptime(end, "%Y-%m-%d").date() if end else date.today()
    start_d = datetime.strptime(start, "%Y-%m-%d").date()

    with get_connection(read_only=True) as con:
        universe = _universe(con)
        # Pull trading days starting a bit before start so the first
        # rebalance lands on/after start_d.
        tdays = _trading_days(con, start_d, end_d)

    if len(tdays) == 0:
        console.print("[red]No trading days in range. Run backfill first.[/]")
        raise typer.Exit(1)

    rebal_timestamps = sorted(_rebalance_dates(tdays, frequency))
    rebal_dates = [ts.date() for ts in rebal_timestamps]
    cfg_hash = config_hash(SYSTEM_PROMPT, universe)

    console.print(
        f"[bold cyan]Generating LLM history[/]\n"
        f"  window:     {start_d} -> {end_d}\n"
        f"  frequency:  {frequency} ({len(rebal_dates)} rebalance dates)\n"
        f"  model:      {model_version}\n"
        f"  dissent:    {with_dissent}\n"
        f"  config_hash: {cfg_hash}\n"
        f"  universe:   {len(universe)} symbols"
    )
    console.print(
        "[yellow]NOTE: historical LLM signals are training-data "
        "contaminated — results are an upper bound, not a forward estimate.[/]"
    )

    total_cost = 0.0
    generated = 0
    skipped = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.fields[d]}"),
        TextColumn("${task.fields[cost]:.2f}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(
            "generating", total=len(rebal_dates), d="", cost=0.0
        )
        for d in rebal_dates:
            progress.update(task, d=str(d))

            if not force:
                with get_connection(read_only=True) as con:
                    if get_cached_output(con, d, model_version, cfg_hash) is not None:
                        skipped += 1
                        progress.advance(task)
                        continue

            try:
                run = run_digest(
                    as_of=d,
                    universe=universe,
                    enable_dissent=with_dissent,
                    persist=False,
                    effort=effort,
                )
                with get_connection() as con:
                    store_cached_output(
                        con,
                        as_of=d,
                        model_version=model_version,
                        cfg_hash=cfg_hash,
                        output=run.final_output,
                        universe=universe,
                        input_tokens=run.primary_response.input_tokens,
                        output_tokens=run.primary_response.output_tokens,
                        cost_usd=run.total_cost_usd,
                    )
                total_cost += run.total_cost_usd
                generated += 1
            except Exception as exc:
                log.error("generate_failed", date=str(d), error=str(exc))

            progress.update(task, cost=total_cost)
            progress.advance(task)

    console.print(
        f"\n[bold green]Done.[/] generated={generated}  skipped(cached)={skipped}  "
        f"total_cost=${total_cost:.2f}"
    )
    console.print(
        f"Now run: [cyan]python scripts/backtest_llm.py "
        f"--start {start_d} --frequency {frequency}[/]"
    )


if __name__ == "__main__":
    app()
