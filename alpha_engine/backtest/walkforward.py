"""Walk-forward validation.

The key correctness invariant: NEVER use data from the test window to choose
parameters or evaluate decisions in the test window. The advisor must be
fully specified before the test window starts.

Two flavors provided:

  1. run_walks(advisor_factory, config)
     - Re-runs the same advisor on each (train_start, train_end) +
       (test_start, test_end) window. Reports per-walk and aggregate
       test-window metrics. No parameter selection; useful for checking
       performance stability across eras.

  2. run_walks_with_param_sweep(factory, param_grid, config, score)
     - For each walk:
         a) Backtest the advisor with each param combination on the
            training window.
         b) Pick the params that maximize `score` (e.g. Sharpe) on train.
         c) Build a fresh advisor with those params, backtest on the
            test window.
     - Returns: chosen params per walk + aggregated OOS performance.
     - This is the honest answer to "does our parameter choice generalize?"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from itertools import product
from typing import Any, Callable

import duckdb
import pandas as pd

from alpha_engine.backtest.engine import run_backtest
from alpha_engine.backtest.metrics import BacktestMetrics, compute_metrics, fetch_risk_free_rate
from alpha_engine.backtest.types import (
    BacktestConfig,
    BacktestResult,
    Fill,
    SignalAdvisor,
)
from alpha_engine.core.logging import get_logger
from alpha_engine.db import get_connection

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Walk:
    """A single train/test split."""

    train_start: date
    train_end: date
    test_start: date
    test_end: date

    @property
    def label(self) -> str:
        return f"{self.test_start}→{self.test_end}"


@dataclass(frozen=True)
class WalkForwardConfig:
    """Walk-forward setup. Defines how to slice history into rolling
    train/test windows."""

    full_start: date
    full_end: date
    train_years: int = 5
    test_years: int = 2
    step_years: int = 2          # how far to advance between walks

    # Underlying backtest config (will be cloned per walk with adjusted dates)
    backtest_config: BacktestConfig = None  # type: ignore[assignment]


def generate_walks(config: WalkForwardConfig) -> list[Walk]:
    """Yield rolling (train, test) windows covering [full_start, full_end]."""
    walks: list[Walk] = []
    train_start = config.full_start
    while True:
        train_end = _add_years(train_start, config.train_years) - timedelta(days=1)
        test_start = train_end + timedelta(days=1)
        test_end = _add_years(test_start, config.test_years) - timedelta(days=1)
        if test_end > config.full_end:
            # If the final test window would extend past the data, clip it.
            # Skip walks where the clipped window would be too short.
            if test_start >= config.full_end:
                break
            test_end = config.full_end
            walks.append(Walk(train_start, train_end, test_start, test_end))
            break
        walks.append(Walk(train_start, train_end, test_start, test_end))
        train_start = _add_years(train_start, config.step_years)
    return walks


def _add_years(d: date, years: int) -> date:
    """Add N years to a date, handling Feb 29 by clamping."""
    try:
        return d.replace(year=d.year + years)
    except ValueError:
        return d.replace(year=d.year + years, day=28)


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass
class WalkResult:
    walk: Walk
    test_result: BacktestResult
    chosen_params: dict[str, Any] = field(default_factory=dict)
    train_score: float | None = None  # score on training window (if optimized)


@dataclass
class WalkForwardSummary:
    """Aggregated results of a walk-forward run."""

    advisor_name: str
    config: WalkForwardConfig
    walks: list[WalkResult]
    stitched_equity: pd.Series       # test-window equity curves concatenated
    stitched_returns: pd.Series      # test-window daily returns concatenated
    stitched_benchmark: pd.Series    # same for benchmark
    aggregate_metrics: BacktestMetrics


# ---------------------------------------------------------------------------
# Walk runner (no parameter selection)
# ---------------------------------------------------------------------------


AdvisorFactory = Callable[..., SignalAdvisor]


def run_walks(
    advisor_factory: AdvisorFactory,
    config: WalkForwardConfig,
    con: duckdb.DuckDBPyConnection | None = None,
    advisor_kwargs: dict[str, Any] | None = None,
) -> WalkForwardSummary:
    """Run the advisor across all walks. Same params for every walk.

    Useful for measuring stability — does the strategy work in every era,
    or just one?
    """
    owned = con is None
    if owned:
        con = get_connection(read_only=True)

    try:
        advisor_kwargs = advisor_kwargs or {}
        walk_list = generate_walks(config)
        results: list[WalkResult] = []

        for w in walk_list:
            test_cfg = _clone_config(
                config.backtest_config, w.test_start, w.test_end
            )
            advisor = advisor_factory(**advisor_kwargs)
            test_res = run_backtest(test_cfg, advisor, con=con)
            results.append(
                WalkResult(walk=w, test_result=test_res, chosen_params=advisor_kwargs)
            )
            log.info(
                "walk_complete",
                walk=w.label,
                advisor=advisor.name,
                test_return=round(test_res.metrics.total_return, 4) if test_res.metrics else None,
            )

        return _summarize(advisor_factory(**advisor_kwargs).name, config, results)
    finally:
        if owned:
            con.close()


# ---------------------------------------------------------------------------
# Walk runner with parameter sweep
# ---------------------------------------------------------------------------


ScoreFn = Callable[[BacktestMetrics], float]


def _default_score(m: BacktestMetrics) -> float:
    """Maximize Sharpe; tiebreak on lower max drawdown."""
    return m.sharpe_ratio - 0.01 * abs(m.max_drawdown)


def run_walks_with_param_sweep(
    advisor_factory: AdvisorFactory,
    param_grid: dict[str, list[Any]],
    config: WalkForwardConfig,
    score: ScoreFn = _default_score,
    con: duckdb.DuckDBPyConnection | None = None,
) -> WalkForwardSummary:
    """For each walk: sweep `param_grid` on training window, pick best
    by `score`, evaluate that choice on the test window.

    `param_grid` is a dict of param_name -> list of values. The cartesian
    product is searched on each training window.
    """
    owned = con is None
    if owned:
        con = get_connection(read_only=True)

    try:
        walk_list = generate_walks(config)
        results: list[WalkResult] = []

        param_names = list(param_grid.keys())
        param_combos = list(product(*[param_grid[k] for k in param_names]))

        for w in walk_list:
            # Sweep on training window
            best_score = float("-inf")
            best_params: dict[str, Any] = {}
            train_cfg = _clone_config(
                config.backtest_config, w.train_start, w.train_end
            )
            for combo in param_combos:
                params = dict(zip(param_names, combo))
                try:
                    advisor = advisor_factory(**params)
                    train_res = run_backtest(train_cfg, advisor, con=con)
                    assert train_res.metrics is not None
                    s = score(train_res.metrics)
                    if s > best_score:
                        best_score = s
                        best_params = params
                except Exception as exc:
                    log.warning(
                        "param_combo_failed",
                        walk=w.label,
                        params=params,
                        error=str(exc),
                    )

            # Test with the chosen params
            test_cfg = _clone_config(
                config.backtest_config, w.test_start, w.test_end
            )
            test_advisor = advisor_factory(**best_params)
            test_res = run_backtest(test_cfg, test_advisor, con=con)
            results.append(
                WalkResult(
                    walk=w,
                    test_result=test_res,
                    chosen_params=best_params,
                    train_score=best_score,
                )
            )
            log.info(
                "walk_complete_sweep",
                walk=w.label,
                chosen=best_params,
                train_score=round(best_score, 3),
                test_return=round(test_res.metrics.total_return, 4)
                if test_res.metrics
                else None,
            )

        name = advisor_factory(**(results[0].chosen_params if results else {})).name
        return _summarize(name + "_sweep", config, results)
    finally:
        if owned:
            con.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clone_config(
    base: BacktestConfig, start: date, end: date
) -> BacktestConfig:
    """Return a copy of `base` with start/end dates replaced."""
    return BacktestConfig(
        start_date=start,
        end_date=end,
        initial_capital=base.initial_capital,
        universe=base.universe,
        benchmark=base.benchmark,
        rebalance_frequency=base.rebalance_frequency,
        commission_bps=base.commission_bps,
        spread_bps=base.spread_bps,
        slippage_bps=base.slippage_bps,
        max_position_weight=base.max_position_weight,
        max_leverage=base.max_leverage,
        drift_rebalance_threshold=base.drift_rebalance_threshold,
    )


def _summarize(
    advisor_name: str,
    config: WalkForwardConfig,
    results: list[WalkResult],
) -> WalkForwardSummary:
    """Stitch together test-window returns to get a continuous OOS curve."""
    if not results:
        raise ValueError("No walk results to summarize")

    # Concatenate test-period daily returns (each test result starts at
    # initial_capital, so we use returns and re-compound from the starting
    # capital of the first walk)
    all_rets = pd.concat([r.test_result.daily_returns for r in results])
    all_bench = pd.concat([r.test_result.benchmark_returns for r in results])
    # Drop duplicate first-day-of-walk rows (each backtest starts with 0.0)
    all_rets = all_rets[~all_rets.index.duplicated(keep="first")]
    all_bench = all_bench[~all_bench.index.duplicated(keep="first")]

    initial = config.backtest_config.initial_capital
    stitched_equity = (1.0 + all_rets).cumprod() * initial
    stitched_bench = (1.0 + all_bench).cumprod() * initial

    # Build a synthetic BacktestResult for metrics computation
    synthetic = BacktestResult(
        config=config.backtest_config,
        advisor_name=advisor_name,
        equity_curve=stitched_equity,
        benchmark_curve=stitched_bench,
        daily_returns=all_rets,
        benchmark_returns=all_bench,
        holdings=pd.DataFrame(),
        fills=[f for r in results for f in r.test_result.fills],
    )
    # Honest Sharpe: pull DGS3MO mean over the stitched OOS window
    rf = 0.0
    try:
        from alpha_engine.db import get_connection
        with get_connection(read_only=True) as con:
            rf = fetch_risk_free_rate(
                con,
                start=stitched_equity.index[0].date(),
                end=stitched_equity.index[-1].date(),
            )
    except Exception:
        pass  # Keep rf=0.0 on any DB error; degrades to old behavior
    metrics = compute_metrics(synthetic, risk_free_rate=rf)

    return WalkForwardSummary(
        advisor_name=advisor_name,
        config=config,
        walks=results,
        stitched_equity=stitched_equity,
        stitched_returns=all_rets,
        stitched_benchmark=stitched_bench,
        aggregate_metrics=metrics,
    )
