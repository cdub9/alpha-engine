"""Backtest engine — the main simulation loop.

Conventions to prevent look-ahead bias:
  - On rebalance trade_date T, the advisor is called with as_of = T - 1
    trading day. Advisor may use any data with obs_date <= as_of.
  - The trade executes at T's close price.
  - This bakes in a realistic one-day delay between decision and execution.
"""

from __future__ import annotations

from datetime import date

import duckdb
import pandas as pd

from alpha_engine.backtest.metrics import compute_metrics, fetch_risk_free_rate
from alpha_engine.backtest.portfolio import Portfolio
from alpha_engine.backtest.types import (
    BacktestConfig,
    BacktestResult,
    SignalAdvisor,
)
from alpha_engine.core.logging import get_logger
from alpha_engine.db import get_connection

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Price loading
# ---------------------------------------------------------------------------


def _load_price_matrix(
    con: duckdb.DuckDBPyConnection,
    symbols: list[str],
    start: date,
    end: date,
) -> pd.DataFrame:
    """Wide DataFrame: rows = trading dates, columns = symbols, values =
    adjusted close. Forward-fills small gaps (max 5 days) so an isolated
    missing print doesn't kill the position."""
    if not symbols:
        return pd.DataFrame()
    placeholders = ",".join(["?"] * len(symbols))
    df = con.execute(
        f"SELECT bar_date, symbol, adj_close FROM market_bars "
        f"WHERE symbol IN ({placeholders}) "
        f"AND bar_date BETWEEN ? AND ? "
        f"ORDER BY bar_date",
        [*symbols, start, end],
    ).fetch_df()
    if df.empty:
        return pd.DataFrame()
    df["bar_date"] = pd.to_datetime(df["bar_date"])
    wide = df.pivot(index="bar_date", columns="symbol", values="adj_close")
    return wide.ffill(limit=5)


# ---------------------------------------------------------------------------
# Rebalance scheduling
# ---------------------------------------------------------------------------


def _rebalance_dates(
    trading_days: pd.DatetimeIndex, frequency: str
) -> set[pd.Timestamp]:
    """Pick rebalance dates from the available trading calendar.

    The first trading day is always a rebalance (initial deployment). Then:
      - daily:    every day
      - weekly:   first trading day of each ISO week
      - biweekly: every other weekly rebalance
      - monthly:  first trading day of each month
    """
    if not len(trading_days):
        return set()

    if frequency == "daily":
        return set(trading_days)

    out: set[pd.Timestamp] = {trading_days[0]}

    if frequency == "monthly":
        seen_months: set[tuple[int, int]] = set()
        for ts in trading_days:
            key = (ts.year, ts.month)
            if key not in seen_months:
                out.add(ts)
                seen_months.add(key)
        return out

    if frequency in ("weekly", "biweekly"):
        seen_weeks: set[tuple[int, int]] = set()
        weekly_dates: list[pd.Timestamp] = []
        for ts in trading_days:
            iso = ts.isocalendar()
            key = (iso.year, iso.week)
            if key not in seen_weeks:
                weekly_dates.append(ts)
                seen_weeks.add(key)
        if frequency == "weekly":
            out.update(weekly_dates)
        else:
            out.update(weekly_dates[::2])
        return out

    raise ValueError(f"Unknown rebalance frequency: {frequency}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_backtest(
    config: BacktestConfig,
    advisor: SignalAdvisor,
    con: duckdb.DuckDBPyConnection | None = None,
) -> BacktestResult:
    """Run a backtest. Returns a BacktestResult with equity curve, fills,
    and computed metrics."""
    owned_con = False
    if con is None:
        con = get_connection(read_only=True)
        owned_con = True

    try:
        # Universe must include the benchmark so we can chart it
        all_symbols = sorted(set(config.universe) | {config.benchmark})
        prices = _load_price_matrix(
            con, all_symbols, config.start_date, config.end_date
        )
        if prices.empty:
            raise RuntimeError(
                f"No price data found for {all_symbols} between "
                f"{config.start_date} and {config.end_date}. Run backfill.py first."
            )

        trading_days: pd.DatetimeIndex = prices.index
        rebal = _rebalance_dates(trading_days, config.rebalance_frequency)

        portfolio = Portfolio(config)
        equity: dict[pd.Timestamp, float] = {}
        holdings: dict[pd.Timestamp, dict[str, float]] = {}
        fills_all = []

        prev_trading_day: pd.Timestamp | None = None
        for ts in trading_days:
            day_prices = prices.loc[ts]

            # Rebalance BEFORE marking equity for the day so the day's
            # equity reflects the new holdings at today's close.
            if ts in rebal:
                # Use previous trading day's close as the "as of" for signals
                # (avoids look-ahead). On the first day, use today as a fallback.
                as_of = (prev_trading_day or ts).date()
                try:
                    targets = advisor.target_weights(
                        as_of=as_of, con=con, universe=list(config.universe)
                    )
                except Exception as exc:
                    log.error(
                        "advisor_failed",
                        advisor=advisor.name,
                        date=str(ts.date()),
                        error=str(exc),
                    )
                    targets = {}

                fills = portfolio.rebalance_to(
                    targets, day_prices, ts.date(), reason="rebalance"
                )
                fills_all.extend(fills)

            equity[ts] = portfolio.mark_to_market(day_prices)
            holdings[ts] = portfolio.weights(day_prices)
            prev_trading_day = ts

        equity_series = pd.Series(equity, name="equity")
        bench_prices = prices[config.benchmark].ffill()
        # Normalize benchmark to start at initial_capital
        bench_curve = (bench_prices / bench_prices.iloc[0]) * config.initial_capital
        bench_curve.name = "benchmark"

        daily_ret = equity_series.pct_change().fillna(0.0)
        bench_ret = bench_curve.pct_change().fillna(0.0)

        holdings_df = pd.DataFrame.from_dict(holdings, orient="index").fillna(0.0)

        result = BacktestResult(
            config=config,
            advisor_name=advisor.name,
            equity_curve=equity_series,
            benchmark_curve=bench_curve,
            daily_returns=daily_ret,
            benchmark_returns=bench_ret,
            holdings=holdings_df,
            fills=fills_all,
        )
        # Use FRED DGS3MO (3-month T-bill) average over the backtest window
        # as the risk-free rate. Honest Sharpe — rf=0 was flattering.
        rf = fetch_risk_free_rate(con, config.start_date, config.end_date)
        result.metrics = compute_metrics(result, risk_free_rate=rf)
        log.info(
            "backtest_complete",
            advisor=advisor.name,
            final_equity=round(result.final_equity, 2),
            n_fills=len(fills_all),
            total_cost=round(result.total_cost, 2),
        )
        return result
    finally:
        if owned_con:
            con.close()
