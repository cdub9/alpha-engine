"""Run all reference advisors over the same period and compare.

Useful as both a validation that the backtest engine produces sensible
numbers (buy-and-hold SPY should match real SPY returns) and a baseline
for any future signal to beat.

Usage:
    python scripts/compare_strategies.py
    python scripts/compare_strategies.py --start 2022-01-01
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
    BuyAndHoldBenchmark,
    EqualWeightUniverse,
    RegimeDefensive,
    RegimeWithTrendConfirmation,
    SixtyFortyClassic,
    run_backtest,
)
from alpha_engine.core.logging import configure_logging

console = Console()
app = typer.Typer(add_completion=False, help=__doc__)

DEFAULT_UNIVERSE = ["SPY", "QQQ", "TLT", "GLD", "XLE", "AGG", "IWM", "XLK", "XLF", "XLV"]


@app.command()
def main(
    start: str = typer.Option("", help="YYYY-MM-DD"),
    end: str = typer.Option("", help="YYYY-MM-DD"),
    capital: float = typer.Option(100_000.0),
    rebalance: str = typer.Option("weekly"),
    log_level: str = typer.Option("WARNING", "--log-level"),
) -> None:
    configure_logging(level=log_level)

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
        rebalance_frequency=rebalance,  # type: ignore[arg-type]
    )

    advisors = [
        BuyAndHoldBenchmark(),
        SixtyFortyClassic(),
        EqualWeightUniverse(),
        RegimeDefensive(),
        RegimeWithTrendConfirmation(),
    ]

    console.print(
        f"[bold cyan]Comparing {len(advisors)} strategies "
        f"{start_d} → {end_d} ({cfg.rebalance_frequency} rebalance)[/]"
    )

    results = []
    for adv in advisors:
        console.print(f"  running {adv.name}...")
        try:
            result = run_backtest(cfg, adv)
            results.append(result)
        except Exception as exc:
            console.print(f"  [red]{adv.name} failed: {exc}[/]")

    # Comparison table
    table = Table(title="Strategy comparison")
    table.add_column("Advisor", style="cyan")
    table.add_column("Total ret", justify="right")
    table.add_column("Ann. ret", justify="right")
    table.add_column("Ann. vol", justify="right")
    table.add_column("Sharpe", justify="right")
    table.add_column("Max DD", justify="right")
    table.add_column("Alpha", justify="right")
    table.add_column("Beta", justify="right")
    table.add_column("Fills", justify="right")
    table.add_column("Cost", justify="right")

    for r in results:
        m = r.metrics
        assert m is not None
        table.add_row(
            r.advisor_name,
            f"{m.total_return:+.1%}",
            f"{m.annualized_return:+.1%}",
            f"{m.annualized_volatility:.1%}",
            f"{m.sharpe_ratio:.2f}",
            f"{m.max_drawdown:.1%}",
            f"{m.alpha_annualized:+.1%}",
            f"{m.beta:.2f}",
            f"{m.n_fills}",
            f"${m.total_transaction_cost:,.0f}",
        )
    console.print(table)

    # Benchmark line for context
    if results:
        m0 = results[0].metrics
        assert m0 is not None
        console.print(
            f"\n[dim]Benchmark (SPY) over same period: "
            f"total return {m0.benchmark_total_return:+.1%}, "
            f"annualized {m0.benchmark_annualized_return:+.1%}[/]"
        )


if __name__ == "__main__":
    app()
