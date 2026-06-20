"""Performance metrics for a backtest result.

Standard finance metrics. Assumes daily-frequency returns. Risk-free rate
defaults to 0 (Sharpe is raw, not excess); pass a non-zero rate to be
more precise.

A note on sample size:
  - Backtests under 1 year (252 days) make Sharpe/Sortino noisy.
  - Annualization assumes returns are stable across the period, which is
    rarely true. Treat all annualized figures as estimates.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from alpha_engine.backtest.types import BacktestResult


TRADING_DAYS_PER_YEAR = 252


@dataclass(frozen=True)
class BacktestMetrics:
    # Returns
    total_return: float
    annualized_return: float
    annualized_volatility: float

    # Risk-adjusted
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float

    # Drawdown
    max_drawdown: float
    max_drawdown_duration_days: int

    # vs Benchmark
    benchmark_total_return: float
    benchmark_annualized_return: float
    alpha_annualized: float
    beta: float
    information_ratio: float

    # Trade quality
    win_rate: float
    profit_factor: float
    n_trading_days: int
    n_fills: int
    total_transaction_cost: float

    def to_dict(self) -> dict:
        return self.__dict__


# ---------------------------------------------------------------------------
# Computation
# ---------------------------------------------------------------------------


def _max_drawdown(equity: pd.Series) -> tuple[float, int]:
    """Returns (max_drawdown_pct, duration_in_days)."""
    running_max = equity.cummax()
    drawdown = (equity / running_max) - 1.0
    max_dd = float(drawdown.min())

    # Duration: longest stretch where equity stayed below a prior peak
    in_dd = drawdown < 0
    if not in_dd.any():
        return 0.0, 0
    # Group consecutive True runs and find the longest
    groups = (in_dd != in_dd.shift()).cumsum()
    run_lengths = in_dd.groupby(groups).sum()
    max_duration = int(run_lengths.max())
    return max_dd, max_duration


def _safe_div(a: float, b: float) -> float:
    if b == 0 or np.isnan(b) or np.isinf(b):
        return 0.0
    return a / b


def fetch_risk_free_rate(
    con,
    start: "date",  # noqa: F821 — forward ref to avoid circular import
    end: "date",  # noqa: F821
    series_id: str = "DGS3MO",
) -> float:
    """Return the average risk-free rate (annualized fraction) over the
    backtest period from FRED data already in `macro_series`.

    DGS3MO is the canonical choice — 3-month T-bill yield. Returns 0.0
    if the series is missing (legacy behavior).
    """
    row = con.execute(
        """
        SELECT AVG(value) FROM macro_series
        WHERE series_id = ? AND obs_date BETWEEN ? AND ?
          AND value IS NOT NULL
        """,
        [series_id, start, end],
    ).fetchone()
    if not row or row[0] is None:
        return 0.0
    return float(row[0]) / 100.0  # FRED returns 4.5 for 4.5%, we want 0.045


def compute_metrics(
    result: BacktestResult, risk_free_rate: float = 0.0
) -> BacktestMetrics:
    """Compute standard performance metrics from a BacktestResult.

    risk_free_rate is annualized (e.g. 0.05 for 5%). Pass
    `fetch_risk_free_rate(con, start, end)` to use FRED's DGS3MO average.
    """
    equity = result.equity_curve
    rets = result.daily_returns
    bench_curve = result.benchmark_curve
    bench_rets = result.benchmark_returns

    n_days = len(rets)
    years = n_days / TRADING_DAYS_PER_YEAR if n_days > 0 else 0.0

    # ----- Returns -----------------------------------------------------
    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1.0) if n_days else 0.0
    if years > 0:
        ann_return = (1.0 + total_return) ** (1.0 / years) - 1.0
    else:
        ann_return = 0.0

    ann_vol = float(rets.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)) if n_days > 1 else 0.0

    # ----- Risk-adjusted ----------------------------------------------
    excess_ann = ann_return - risk_free_rate
    sharpe = _safe_div(excess_ann, ann_vol)

    # Sortino: only downside std
    downside_rets = rets[rets < 0]
    downside_vol = (
        float(downside_rets.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))
        if len(downside_rets) > 1
        else 0.0
    )
    sortino = _safe_div(excess_ann, downside_vol)

    # ----- Drawdown ----------------------------------------------------
    max_dd, max_dd_dur = _max_drawdown(equity)
    calmar = _safe_div(ann_return, abs(max_dd))

    # ----- vs Benchmark ------------------------------------------------
    bench_total = float(bench_curve.iloc[-1] / bench_curve.iloc[0] - 1.0) if n_days else 0.0
    bench_ann = (1.0 + bench_total) ** (1.0 / years) - 1.0 if years > 0 else 0.0

    if n_days > 1 and bench_rets.var() > 0:
        cov = float(np.cov(rets, bench_rets, ddof=1)[0, 1])
        beta = cov / float(bench_rets.var(ddof=1))
        # Annualized alpha via CAPM
        alpha_daily = float(rets.mean() - beta * bench_rets.mean())
        alpha_ann = alpha_daily * TRADING_DAYS_PER_YEAR
        excess_rets = rets - bench_rets
        ir = _safe_div(
            float(excess_rets.mean()) * TRADING_DAYS_PER_YEAR,
            float(excess_rets.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)),
        )
    else:
        beta = 0.0
        alpha_ann = 0.0
        ir = 0.0

    # ----- Trade quality ----------------------------------------------
    pos = rets[rets > 0]
    neg = rets[rets < 0]
    win_rate = float(len(pos) / len(rets)) if len(rets) else 0.0
    gross_gain = float(pos.sum())
    gross_loss = float(abs(neg.sum()))
    profit_factor = _safe_div(gross_gain, gross_loss)

    return BacktestMetrics(
        total_return=total_return,
        annualized_return=ann_return,
        annualized_volatility=ann_vol,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        calmar_ratio=calmar,
        max_drawdown=max_dd,
        max_drawdown_duration_days=max_dd_dur,
        benchmark_total_return=bench_total,
        benchmark_annualized_return=bench_ann,
        alpha_annualized=alpha_ann,
        beta=beta,
        information_ratio=ir,
        win_rate=win_rate,
        profit_factor=profit_factor,
        n_trading_days=n_days,
        n_fills=len(result.fills),
        total_transaction_cost=result.total_cost,
    )
