"""SignalAdvisor adapters so the ML layer plugs into the backtest engine
and walk-forward harness exactly like every other strategy.

Both advisors:
  - load prices with bar_date <= as_of only (point-in-time)
  - score the cross-section, take the top `top_n` names, equal-weight them
  - fall back to the benchmark when the cross-section is too thin to rank
    (early windows where few symbols have 252 days of history) — holding
    SPY is the honest "no signal" position, not cash

Portfolio construction is the standard academic one (top quintile,
equal weight, periodic rebalance). The rebalance cadence comes from the
BacktestConfig, not the advisor.
"""

from __future__ import annotations

from datetime import date, timedelta

import duckdb
import pandas as pd

from alpha_engine.backtest.types import SignalAdvisor
from alpha_engine.core.logging import get_logger
from alpha_engine.ml.features import MIN_HISTORY, compute_features
from alpha_engine.ml.model import MomentumComposite, WalkForwardXGB

log = get_logger(__name__)

# Calendar-day lookback that comfortably contains MIN_HISTORY trading days
# (and, for XGB, enough additional history to build a training panel).
_PRICE_LOOKBACK_DAYS = 420
_XGB_LOOKBACK_DAYS = 420 + 5 * 365  # ~5y of training history behind the features


def load_price_panel(
    con: duckdb.DuckDBPyConnection,
    symbols: list[str],
    as_of: date,
    lookback_days: int,
) -> pd.DataFrame:
    """Wide adj_close panel for `symbols` with bar_date <= as_of."""
    if not symbols:
        return pd.DataFrame()
    placeholders = ",".join(["?"] * len(symbols))
    df = con.execute(
        f"SELECT bar_date, symbol, adj_close FROM market_bars "
        f"WHERE symbol IN ({placeholders}) AND bar_date BETWEEN ? AND ? "
        f"ORDER BY bar_date",
        [*symbols, as_of - timedelta(days=lookback_days), as_of],
    ).fetch_df()
    if df.empty:
        return pd.DataFrame()
    df["bar_date"] = pd.to_datetime(df["bar_date"])
    return df.pivot(index="bar_date", columns="symbol", values="adj_close")


class MLMomentumAdvisor(SignalAdvisor):
    """Top-quintile cross-sectional momentum, equal weight."""

    name = "ml_momentum"
    description = (
        "Equal-weight the top-quintile names by blended 12-1/6-1/3-1 "
        "momentum z-score. Zero trained parameters."
    )

    def __init__(self, top_n: int | None = None, benchmark: str = "SPY") -> None:
        self.top_n = top_n      # None = quintile of the rankable cross-section
        self.benchmark = benchmark
        self.model = MomentumComposite()

    def _score(
        self, as_of: date, con: duckdb.DuckDBPyConnection, universe: list[str]
    ) -> pd.Series:
        prices = load_price_panel(con, universe, as_of, _PRICE_LOOKBACK_DAYS)
        if prices.empty or len(prices) < MIN_HISTORY:
            return pd.Series(dtype=float)
        feats = compute_features(prices)
        return self.model.score_cross_section(feats).dropna()

    def target_weights(
        self,
        as_of: date,
        con: duckdb.DuckDBPyConnection,
        universe: list[str],
    ) -> dict[str, float]:
        scores = self._score(as_of, con, universe)
        # Need a real cross-section to rank within. Fewer than 10 rankable
        # names = quintiles are meaningless; hold the benchmark instead.
        if len(scores) < 10:
            return {self.benchmark: 1.0}
        n = self.top_n or max(3, round(len(scores) * 0.20))
        top = scores.nlargest(n)
        w = 1.0 / len(top)
        return {sym: w for sym in top.index}


class XGBMomentumAdvisor(MLMomentumAdvisor):
    """Same construction, but scores come from a walk-forward-trained
    XGBoost model instead of the fixed composite. The model retrains
    itself (quarterly) inside the backtest using only bars <= as_of."""

    name = "ml_xgb"
    description = (
        "Top-quintile by XGBoost-predicted forward-return rank; model "
        "retrains quarterly on point-in-time data inside the backtest."
    )

    def __init__(self, top_n: int | None = None, benchmark: str = "SPY") -> None:
        super().__init__(top_n=top_n, benchmark=benchmark)
        self.model = WalkForwardXGB()

    def _score(
        self, as_of: date, con: duckdb.DuckDBPyConnection, universe: list[str]
    ) -> pd.Series:
        prices = load_price_panel(con, universe, as_of, _XGB_LOOKBACK_DAYS)
        if prices.empty or len(prices) < MIN_HISTORY:
            return pd.Series(dtype=float)
        self.model.maybe_retrain(prices)
        feats = compute_features(prices)
        return self.model.score_cross_section(feats).dropna()
