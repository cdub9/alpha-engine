"""Backtest the LLM digest signals against SPY and reference strategies.

Reads cached digests (generate them first with generate_llm_history.py),
runs both channels through the backtest engine, and compares to
buy-and-hold SPY plus the reference advisors.

============================  CRITICAL CAVEAT  =============================
Historical LLM backtests are TRAINING-DATA CONTAMINATED. Opus 4.7's
weights likely encode knowledge of the very dates being tested, so it may
"remember" outcomes (NVDA's run, COVID, SVB, etc.). Results here are an
OPTIMISTIC UPPER BOUND on skill, NOT a forward-looking estimate. The only
clean test is paper trading on dates after the model's training cutoff.
Do not size live capital from these numbers.
===========================================================================

Usage:
    python scripts/backtest_llm.py --start 2024-12-01 --frequency monthly
"""

from __future__ import annotations

import sys
from datetime import date, datetime, timedelta

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

from alpha_engine.backtest import (
    BacktestConfig,
    BuyAndHoldBenchmark,
    EqualWeightUniverse,
    RegimeWithTrendConfirmation,
    SixtyFortyClassic,
    run_backtest,
)
from alpha_engine.backtest.llm_advisor import (
    DEFAULT_MODEL_VERSION,
    LLMChannelAdvisor,
    config_hash,
)
from alpha_engine.core.config import get_settings
from alpha_engine.core.logging import configure_logging
from alpha_engine.core.types import Channel
from alpha_engine.db import get_connection
from alpha_engine.llm.prompts import SYSTEM_PROMPT

console = Console()
app = typer.Typer(add_completion=False, help=__doc__)

DEFAULT_UNIVERSE = ["SPY", "QQQ", "TLT", "GLD", "XLE", "AGG", "IWM", "XLK", "XLF", "XLV"]


def _full_universe(con) -> list[str]:
    rows = con.execute(
        "SELECT symbol FROM instruments WHERE active = TRUE ORDER BY symbol"
    ).fetchall()
    return [r[0] for r in rows]


@app.command()
def main(
    start: str = typer.Option(..., help="YYYY-MM-DD"),
    end: str = typer.Option("", help="YYYY-MM-DD; defaults to today"),
    frequency: str = typer.Option("monthly"),
    capital: float = typer.Option(100_000.0),
    model_version: str = typer.Option(DEFAULT_MODEL_VERSION),
    channel_a_max_pos: float = typer.Option(0.05, help="steady_alpha max position"),
    channel_b_max_pos: float = typer.Option(0.15, help="aggressive_growth max position"),
    log_level: str = typer.Option("WARNING", "--log-level"),
) -> None:
    configure_logging(level=log_level)

    end_d = datetime.strptime(end, "%Y-%m-%d").date() if end else date.today()
    start_d = datetime.strptime(start, "%Y-%m-%d").date()

    console.print(
        Panel(
            "[bold red]TRAINING-DATA CONTAMINATION WARNING[/]\n\n"
            "Opus 4.7's weights likely encode knowledge of the dates being\n"
            "tested. These results are an [bold]optimistic upper bound[/], not a\n"
            "forward estimate. The only clean test is paper trading on dates\n"
            "after the model's training cutoff. Do NOT size live capital here.",
            border_style="red",
        )
    )

    with get_connection(read_only=True) as con:
        full_universe = _full_universe(con)
        # How many digests are cached in range?
        cached = con.execute(
            """
            SELECT COUNT(*) FROM llm_signal_cache
            WHERE model_version = ? AND as_of BETWEEN ? AND ?
            """,
            [model_version, start_d, end_d],
        ).fetchone()[0]

    cfg_hash = config_hash(SYSTEM_PROMPT, full_universe)
    console.print(
        f"Cached digests in range: {cached}  (config_hash {cfg_hash})\n"
    )
    if cached == 0:
        console.print(
            "[red]No cached digests. Run generate_llm_history.py first.[/]"
        )
        raise typer.Exit(1)

    # Staleness tolerance spans roughly one rebalance interval, so the
    # engine's "previous trading day" query still finds that period's digest.
    tolerance = {"daily": 4, "weekly": 10, "biweekly": 18, "monthly": 40}.get(
        frequency, 40
    )

    # LLM channel backtests use the FULL universe (the LLM can pick anything)
    llm_a = LLMChannelAdvisor(
        Channel.STEADY_ALPHA, cfg_hash, model_version, tolerance_days=tolerance
    )
    llm_b = LLMChannelAdvisor(
        Channel.AGGRESSIVE_GROWTH, cfg_hash, model_version, tolerance_days=tolerance
    )

    cfg_a = BacktestConfig(
        start_date=start_d,
        end_date=end_d,
        initial_capital=capital,
        universe=full_universe,
        rebalance_frequency=frequency,  # type: ignore[arg-type]
        max_position_weight=channel_a_max_pos,
        max_leverage=1.0,
    )
    cfg_b = BacktestConfig(
        start_date=start_d,
        end_date=end_d,
        initial_capital=capital,
        universe=full_universe,
        rebalance_frequency=frequency,  # type: ignore[arg-type]
        max_position_weight=channel_b_max_pos,
        max_leverage=1.0,
    )
    # Reference advisors use the standard 10-name universe
    cfg_ref = BacktestConfig(
        start_date=start_d,
        end_date=end_d,
        initial_capital=capital,
        universe=DEFAULT_UNIVERSE,
        rebalance_frequency=frequency,  # type: ignore[arg-type]
    )

    # Survivorship-bias check (LLM channels use full_universe, which
    # includes individual equities subject to the bias)
    from rich.panel import Panel
    from alpha_engine.backtest.warnings import affected_symbols, survivorship_warning_text
    with get_connection(read_only=True) as con:
        affected = affected_symbols(con, full_universe)
    if affected:
        console.print(
            Panel(
                survivorship_warning_text(affected, include_header=False),
                title="⚠️  SURVIVORSHIP BIAS WARNING",
                border_style="yellow",
            )
        )

    runs = []
    console.print("Running backtests...")
    runs.append(("llm_steady_alpha", run_backtest(cfg_a, llm_a)))
    runs.append(("llm_aggressive_growth", run_backtest(cfg_b, llm_b)))
    runs.append(("buy_and_hold_spy", run_backtest(cfg_ref, BuyAndHoldBenchmark())))
    runs.append(("sixty_forty", run_backtest(cfg_ref, SixtyFortyClassic())))
    runs.append(
        ("regime_with_trend", run_backtest(cfg_ref, RegimeWithTrendConfirmation()))
    )

    # Cache hit/miss for the LLM channels
    a_hits, a_miss = llm_a.cache_stats
    b_hits, b_miss = llm_b.cache_stats
    if a_miss or b_miss:
        console.print(
            f"[yellow]LLM cache misses: A={a_miss}, B={b_miss} "
            f"(those rebalances went to cash)[/]"
        )

    table = Table(title=f"LLM signal backtest  {start_d} -> {end_d}  ({frequency})")
    table.add_column("Strategy", style="cyan")
    table.add_column("Total", justify="right")
    table.add_column("Ann.", justify="right")
    table.add_column("Vol", justify="right")
    table.add_column("Sharpe", justify="right")
    table.add_column("MaxDD", justify="right")
    table.add_column("Alpha", justify="right")
    table.add_column("Beta", justify="right")
    table.add_column("Fills", justify="right")

    for label, r in runs:
        m = r.metrics
        assert m is not None
        table.add_row(
            label,
            f"{m.total_return:+.1%}",
            f"{m.annualized_return:+.1%}",
            f"{m.annualized_volatility:.1%}",
            f"{m.sharpe_ratio:.2f}",
            f"{m.max_drawdown:.1%}",
            f"{m.alpha_annualized:+.1%}",
            f"{m.beta:.2f}",
            f"{m.n_fills}",
        )
    console.print(table)

    # Benchmark line
    m0 = runs[0][1].metrics
    assert m0 is not None
    console.print(
        f"\n[dim]SPY over same period: "
        f"total {m0.benchmark_total_return:+.1%}, "
        f"annualized {m0.benchmark_annualized_return:+.1%}[/]"
    )

    console.print(
        Panel(
            "[bold]Reading these results[/]\n\n"
            "• If the LLM channels beat SPY here, that is NECESSARY but not\n"
            "  SUFFICIENT evidence of skill — contamination inflates it.\n"
            "• If they UNDERperform even with contamination, that's a strong\n"
            "  negative signal worth taking seriously.\n"
            "• The channel A vs B contrast (diversified vs concentrated) is\n"
            "  more trustworthy than absolute levels.\n"
            "• Next real test: paper-trade forward from today.",
            border_style="yellow",
        )
    )


if __name__ == "__main__":
    app()
