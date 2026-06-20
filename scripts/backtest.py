"""Run a single backtest and print results.

Usage:
    python scripts/backtest.py buy_and_hold_spy
    python scripts/backtest.py regime_defensive --start 2022-01-01
    python scripts/backtest.py equal_weight --start 2022-01-01 --rebalance weekly
"""

from __future__ import annotations

import sys
from datetime import date, datetime, timedelta

import typer
from rich.console import Console
from rich.table import Table

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

from alpha_engine.backtest import (
    BacktestConfig,
    BacktestResult,
    BuyAndHoldBenchmark,
    EqualWeightUniverse,
    RegimeDefensive,
    RegimeWithTrendConfirmation,
    SixtyFortyClassic,
    run_backtest,
)
from alpha_engine.backtest.warnings import (
    affected_symbols,
    survivorship_warning_text,
)
from alpha_engine.core.logging import configure_logging
from alpha_engine.db import get_connection
from rich.panel import Panel

console = Console()
app = typer.Typer(add_completion=False, help=__doc__)


ADVISORS = {
    "buy_and_hold_spy": lambda: BuyAndHoldBenchmark(),
    "equal_weight": lambda: EqualWeightUniverse(),
    "sixty_forty": lambda: SixtyFortyClassic(),
    "regime_defensive": lambda: RegimeDefensive(),
    "regime_with_trend": lambda: RegimeWithTrendConfirmation(),
}


DEFAULT_UNIVERSE = ["SPY", "QQQ", "TLT", "GLD", "XLE", "AGG", "IWM", "XLK", "XLF", "XLV"]


def print_result(result: BacktestResult) -> None:
    m = result.metrics
    assert m is not None

    summary = Table(title=f"Backtest: {result.advisor_name}", show_header=False)
    summary.add_column("Field", style="cyan")
    summary.add_column("Value", justify="right")
    summary.add_row("Period",
                    f"{result.config.start_date} → {result.config.end_date}")
    summary.add_row("Trading days", f"{m.n_trading_days}")
    summary.add_row("Initial capital", f"${result.config.initial_capital:,.0f}")
    summary.add_row("Final equity", f"${result.final_equity:,.0f}")
    summary.add_row("Fills", f"{m.n_fills}")
    summary.add_row("Total cost", f"${m.total_transaction_cost:,.2f}")
    console.print(summary)

    perf = Table(title="Performance")
    perf.add_column("Metric")
    perf.add_column("Strategy", justify="right")
    perf.add_column("Benchmark", justify="right")
    perf.add_column("Delta", justify="right")
    perf.add_row(
        "Total return",
        f"{m.total_return:+.2%}",
        f"{m.benchmark_total_return:+.2%}",
        f"{(m.total_return - m.benchmark_total_return):+.2%}",
    )
    perf.add_row(
        "Annualized return",
        f"{m.annualized_return:+.2%}",
        f"{m.benchmark_annualized_return:+.2%}",
        f"{(m.annualized_return - m.benchmark_annualized_return):+.2%}",
    )
    perf.add_row("Annualized vol", f"{m.annualized_volatility:.2%}", "—", "—")
    console.print(perf)

    risk = Table(title="Risk / quality")
    risk.add_column("Metric")
    risk.add_column("Value", justify="right")
    risk.add_row("Sharpe ratio", f"{m.sharpe_ratio:.2f}")
    risk.add_row("Sortino ratio", f"{m.sortino_ratio:.2f}")
    risk.add_row("Calmar ratio", f"{m.calmar_ratio:.2f}")
    risk.add_row("Max drawdown", f"{m.max_drawdown:.2%}")
    risk.add_row("Max DD duration (days)", f"{m.max_drawdown_duration_days}")
    risk.add_row("Alpha (annual)", f"{m.alpha_annualized:+.2%}")
    risk.add_row("Beta", f"{m.beta:.2f}")
    risk.add_row("Information ratio", f"{m.information_ratio:.2f}")
    risk.add_row("Win rate (daily)", f"{m.win_rate:.2%}")
    risk.add_row("Profit factor", f"{m.profit_factor:.2f}")
    console.print(risk)


@app.command()
def main(
    advisor_name: str = typer.Argument(..., help=f"One of: {list(ADVISORS)}"),
    start: str = typer.Option(
        "", help="YYYY-MM-DD. Defaults to 3 years before end."
    ),
    end: str = typer.Option("", help="YYYY-MM-DD. Defaults to today."),
    capital: float = typer.Option(100_000.0, help="Initial capital"),
    rebalance: str = typer.Option("weekly", help="daily|weekly|biweekly|monthly"),
    benchmark: str = typer.Option("SPY"),
    log_level: str = typer.Option("WARNING", "--log-level"),
) -> None:
    configure_logging(level=log_level)

    if advisor_name not in ADVISORS:
        console.print(f"[red]Unknown advisor: {advisor_name}[/]")
        console.print(f"Available: {', '.join(ADVISORS)}")
        raise typer.Exit(1)

    end_d = datetime.strptime(end, "%Y-%m-%d").date() if end else date.today()
    start_d = (
        datetime.strptime(start, "%Y-%m-%d").date()
        if start
        else end_d - timedelta(days=365 * 3)
    )

    cfg = BacktestConfig(
        start_date=start_d,
        end_date=end_d,
        initial_capital=capital,
        universe=DEFAULT_UNIVERSE,
        benchmark=benchmark,
        rebalance_frequency=rebalance,  # type: ignore[arg-type]
    )

    # Survivorship-bias check on the backtest universe
    with get_connection(read_only=True) as con:
        affected = affected_symbols(con, cfg.universe)
    if affected:
        console.print(
            Panel(
                survivorship_warning_text(affected, include_header=False),
                title="⚠️  SURVIVORSHIP BIAS WARNING",
                border_style="yellow",
            )
        )

    advisor = ADVISORS[advisor_name]()
    result = run_backtest(cfg, advisor)
    print_result(result)


if __name__ == "__main__":
    app()
