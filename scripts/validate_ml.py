"""Walk-forward validation of the ML signal layer.

Two honest tests, both free of training-data contamination (unlike LLM
historical backtests — these strategies only see prices):

  1. DEEP (2008 → present, 19 survivorship-clean ETFs with full history):
     rolling 5y-train / 2y-test walk-forward. Covers the GFC, the 2010s
     bull, COVID, the 2022 bear, and the AI rally. This is the number
     to trust.

  2. BROAD (2022-07 → present, all 45 active ETFs): plain backtest on the
     wider modern universe. Shorter window — context, not proof.

Individual equities are EXCLUDED on purpose: today's universe equities
were picked because they won (survivorship bias), so backtesting them
flatters any strategy. ETFs hold their historical composition.

Writes data/ml_validation.json for the dashboard's ML Signals page.

Usage:
    python scripts/validate_ml.py
    python scripts/validate_ml.py --skip-broad
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

from alpha_engine.backtest import (
    BacktestConfig,
    BuyAndHoldBenchmark,
    EqualWeightUniverse,
    WalkForwardConfig,
    run_backtest,
    run_walks,
)
from alpha_engine.core.logging import configure_logging
from alpha_engine.db import get_connection
from alpha_engine.ml.advisor import MLMomentumAdvisor, XGBMomentumAdvisor

console = Console()
app = typer.Typer(add_completion=False, help=__doc__)

# 19 ETFs with bars back to 2007 — survivorship-clean deep history.
DEEP_ETF_UNIVERSE = [
    "SPY", "QQQ", "DIA", "IWM",
    "XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY",
    "AGG", "TLT", "SHY", "LQD", "HYG", "TIP",
]

OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "ml_validation.json"

ADVISORS = [
    ("ml_momentum", MLMomentumAdvisor),
    ("ml_xgb", XGBMomentumAdvisor),
    ("equal_weight", EqualWeightUniverse),
    ("buy_and_hold_spy", BuyAndHoldBenchmark),
]


def _metrics_dict(m) -> dict:
    return {
        "total_return": m.total_return,
        "annualized_return": m.annualized_return,
        "annualized_volatility": m.annualized_volatility,
        "sharpe": m.sharpe_ratio,
        "max_drawdown": m.max_drawdown,
        "alpha_annualized": m.alpha_annualized,
        "beta": m.beta,
        "information_ratio": m.information_ratio,
        "benchmark_total_return": m.benchmark_total_return,
        "benchmark_annualized_return": m.benchmark_annualized_return,
    }


def _print_table(title: str, rows: list[tuple[str, dict]]) -> None:
    table = Table(title=title, show_header=True, header_style="bold")
    for col in ("Strategy", "Total", "Annual", "Sharpe", "Max DD", "Alpha/yr", "vs SPY"):
        table.add_column(col, justify="right" if col != "Strategy" else "left")
    for name, m in rows:
        table.add_row(
            name,
            f"{m['total_return']:+.1%}",
            f"{m['annualized_return']:+.1%}",
            f"{m['sharpe']:.2f}",
            f"{m['max_drawdown']:.1%}",
            f"{m['alpha_annualized']:+.1%}",
            f"{m['total_return'] - m['benchmark_total_return']:+.1%}",
        )
    console.print(table)


def run_deep(con) -> dict:
    console.rule("[bold cyan]DEEP: 19 ETFs, 2008 → present, walk-forward 5y/2y[/]")
    base = BacktestConfig(
        start_date=date(2008, 1, 1),
        end_date=date(2026, 6, 10),
        universe=DEEP_ETF_UNIVERSE,
        rebalance_frequency="monthly",   # standard cadence for cross-sectional momentum
    )
    wf = WalkForwardConfig(
        full_start=date(2008, 1, 1),
        full_end=date(2026, 6, 10),
        train_years=5, test_years=2, step_years=2,
        backtest_config=base,
    )
    out: dict = {"advisors": {}, "walks": {}}
    for name, factory in ADVISORS:
        summary = run_walks(factory, wf, con=con)
        out["advisors"][name] = _metrics_dict(summary.aggregate_metrics)
        out["walks"][name] = [
            {
                "window": wr.walk.label,
                "total_return": wr.test_result.metrics.total_return,
                "sharpe": wr.test_result.metrics.sharpe_ratio,
                "max_drawdown": wr.test_result.metrics.max_drawdown,
                "vs_benchmark": (
                    wr.test_result.metrics.total_return
                    - wr.test_result.metrics.benchmark_total_return
                ),
            }
            for wr in summary.walks
        ]
        console.print(f"  [green]{name}[/] done")
    _print_table(
        "Aggregated out-of-sample (stitched test windows)",
        [(n, out["advisors"][n]) for n, _ in ADVISORS],
    )
    return out


def run_broad(con) -> dict:
    console.rule("[bold cyan]BROAD: 45 active ETFs, 2022-07 → present (context only)[/]")
    rows = con.execute(
        "SELECT symbol FROM instruments WHERE active "
        "AND instrument_type IN ('etf', 'bond_etf') ORDER BY symbol"
    ).fetchall()
    universe = [r[0] for r in rows]
    cfg = BacktestConfig(
        start_date=date(2022, 7, 1),
        end_date=date(2026, 6, 10),
        universe=universe,
        rebalance_frequency="monthly",
    )
    out: dict = {"advisors": {}, "universe_size": len(universe)}
    for name, factory in ADVISORS:
        res = run_backtest(cfg, factory(), con=con)
        out["advisors"][name] = _metrics_dict(res.metrics)
        console.print(f"  [green]{name}[/] done")
    _print_table(
        "Broad-universe backtest (single window — context, not proof)",
        [(n, out["advisors"][n]) for n, _ in ADVISORS],
    )
    return out


@app.command()
def main(
    skip_broad: bool = typer.Option(False, "--skip-broad"),
    skip_deep: bool = typer.Option(False, "--skip-deep"),
    log_level: str = typer.Option("WARNING", "--log-level"),
) -> None:
    configure_logging(level=log_level)
    result: dict = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "deep_universe": DEEP_ETF_UNIVERSE,
        "notes": (
            "ETF-only universes; individual equities excluded (survivorship "
            "bias). Deep = walk-forward OOS, the trustworthy number. Broad = "
            "single short window, context only. No LLM involvement — these "
            "results have no training-data contamination."
        ),
    }
    with get_connection(read_only=True) as con:
        if not skip_deep:
            result["deep"] = run_deep(con)
        if not skip_broad:
            result["broad"] = run_broad(con)

    OUT_PATH.parent.mkdir(exist_ok=True)
    OUT_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    console.print(f"\n[bold green]Wrote {OUT_PATH}[/]")


if __name__ == "__main__":
    app()
