"""Walk-forward validation.

Two reports:

  1. Era-stratified: runs each of the 5 reference advisors in each of the
     4 default eras. Shows whether each strategy's edge is consistent
     across regimes or concentrated.

  2. Parameter sweep on RegimeWithTrendConfirmation: rolling 5-year train
     / 2-year test windows, sweeping SMA window across {50, 100, 150,
     200, 250}. For each walk, picks the best SMA on train data, then
     evaluates on the held-out test window. Reports per-walk selections
     and aggregate OOS performance vs the default SMA=200.

Usage:
    python scripts/walkforward.py
    python scripts/walkforward.py --skip-eras
    python scripts/walkforward.py --skip-sweep
"""

from __future__ import annotations

import sys
from datetime import date

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
    DEFAULT_ERAS,
    BacktestConfig,
    BuyAndHoldBenchmark,
    EqualWeightUniverse,
    RegimeDefensive,
    RegimeWithTrendConfirmation,
    SixtyFortyClassic,
    WalkForwardConfig,
    evaluate_by_era,
    run_walks,
    run_walks_with_param_sweep,
)
from alpha_engine.core.logging import configure_logging

console = Console()
app = typer.Typer(add_completion=False, help=__doc__)

DEFAULT_UNIVERSE = ["SPY", "QQQ", "TLT", "GLD", "XLE", "AGG", "IWM", "XLK", "XLF", "XLV"]

ADVISOR_FACTORIES = [
    ("buy_and_hold_spy", BuyAndHoldBenchmark),
    ("sixty_forty", SixtyFortyClassic),
    ("equal_weight", EqualWeightUniverse),
    ("regime_defensive", RegimeDefensive),
    ("regime_with_trend", RegimeWithTrendConfirmation),
]


def run_era_report(base_config: BacktestConfig) -> None:
    console.print()
    console.rule("[bold cyan]Era-stratified performance[/]")
    for era in DEFAULT_ERAS:
        console.print(f"\n[bold]{era.name}[/]  ({era.start} → {era.end})  "
                      f"[dim]{era.description}[/]")
        table = Table(show_header=True, header_style="bold")
        table.add_column("Strategy")
        table.add_column("Total", justify="right")
        table.add_column("Annual", justify="right")
        table.add_column("Vol", justify="right")
        table.add_column("Sharpe", justify="right")
        table.add_column("Max DD", justify="right")
        table.add_column("Alpha", justify="right")
        table.add_column("Beta", justify="right")
        for name, factory in ADVISOR_FACTORIES:
            results = evaluate_by_era(factory, base_config, eras=[era])
            m = results[0].metrics
            table.add_row(
                name,
                f"{m.total_return:+.1%}",
                f"{m.annualized_return:+.1%}",
                f"{m.annualized_volatility:.1%}",
                f"{m.sharpe_ratio:.2f}",
                f"{m.max_drawdown:.1%}",
                f"{m.alpha_annualized:+.1%}",
                f"{m.beta:.2f}",
            )
        console.print(table)


def run_param_sweep(base_config: BacktestConfig) -> None:
    console.print()
    console.rule("[bold cyan]Walk-forward parameter sweep: RegimeWithTrendConfirmation SMA[/]")

    wf_cfg = WalkForwardConfig(
        full_start=date(2008, 1, 1),
        full_end=date(2026, 5, 29),
        train_years=5,
        test_years=2,
        step_years=2,
        backtest_config=base_config,
    )

    console.print(
        f"  train window: {wf_cfg.train_years}y, "
        f"test window: {wf_cfg.test_years}y, "
        f"step: {wf_cfg.step_years}y\n"
        f"  SMA grid: [50, 100, 150, 200, 250]"
    )

    param_grid = {"sma_window": [50, 100, 150, 200, 250]}
    summary = run_walks_with_param_sweep(
        RegimeWithTrendConfirmation,
        param_grid=param_grid,
        config=wf_cfg,
    )

    # Per-walk table
    table = Table(title="Walk-forward results (out-of-sample test windows only)")
    table.add_column("Test window")
    table.add_column("Chosen SMA", justify="right")
    table.add_column("Train score", justify="right")
    table.add_column("Test total", justify="right")
    table.add_column("Test Sharpe", justify="right")
    table.add_column("Test max DD", justify="right")
    table.add_column("Test vs SPY", justify="right")

    for wr in summary.walks:
        m = wr.test_result.metrics
        assert m is not None
        bench_excess = m.total_return - m.benchmark_total_return
        table.add_row(
            wr.walk.label,
            str(wr.chosen_params.get("sma_window", "?")),
            f"{wr.train_score:.3f}" if wr.train_score is not None else "—",
            f"{m.total_return:+.1%}",
            f"{m.sharpe_ratio:.2f}",
            f"{m.max_drawdown:.1%}",
            f"{bench_excess:+.1%}",
        )
    console.print(table)

    # Aggregate
    agg = summary.aggregate_metrics
    console.print()
    console.print(f"[bold]Aggregated OOS (stitched test windows, "
                  f"{wf_cfg.full_start} → {wf_cfg.full_end}):[/]")
    console.print(f"  total return    {agg.total_return:+.1%}  "
                  f"(SPY: {agg.benchmark_total_return:+.1%})")
    console.print(f"  annualized      {agg.annualized_return:+.1%}  "
                  f"(SPY: {agg.benchmark_annualized_return:+.1%})")
    console.print(f"  Sharpe          {agg.sharpe_ratio:.2f}")
    console.print(f"  max drawdown    {agg.max_drawdown:.1%}")
    console.print(f"  alpha (annual)  {agg.alpha_annualized:+.1%}")
    console.print(f"  beta            {agg.beta:.2f}")
    console.print(f"  info ratio      {agg.information_ratio:.2f}")

    # Comparison: how would the default SMA=200 have done?
    console.print()
    console.print("[bold]Comparison: fixed SMA=200 (no parameter selection)[/]")
    fixed_summary = run_walks(
        RegimeWithTrendConfirmation,
        config=wf_cfg,
        advisor_kwargs={"sma_window": 200},
    )
    fa = fixed_summary.aggregate_metrics
    console.print(f"  total return    {fa.total_return:+.1%}")
    console.print(f"  Sharpe          {fa.sharpe_ratio:.2f}")
    console.print(f"  max drawdown    {fa.max_drawdown:.1%}")
    delta_sharpe = agg.sharpe_ratio - fa.sharpe_ratio
    delta_ret = agg.total_return - fa.total_return
    console.print(
        f"\n[bold]Sweep added value:[/] "
        f"Sharpe Δ {delta_sharpe:+.2f}, total return Δ {delta_ret:+.1%}"
    )
    if delta_sharpe > 0.05:
        console.print("[green]→ Sweep meaningfully improves OOS performance[/]")
    elif delta_sharpe > -0.05:
        console.print("[yellow]→ Sweep is roughly even with default — default SMA=200 is fine[/]")
    else:
        console.print("[red]→ Sweep hurts OOS performance — overfitting on train[/]")


@app.command()
def main(
    skip_eras: bool = typer.Option(False, "--skip-eras"),
    skip_sweep: bool = typer.Option(False, "--skip-sweep"),
    log_level: str = typer.Option("WARNING", "--log-level"),
) -> None:
    configure_logging(level=log_level)

    base = BacktestConfig(
        start_date=date(2008, 1, 1),
        end_date=date(2026, 5, 29),
        universe=DEFAULT_UNIVERSE,
        rebalance_frequency="weekly",
    )

    if not skip_eras:
        run_era_report(base)
    if not skip_sweep:
        run_param_sweep(base)


if __name__ == "__main__":
    app()
