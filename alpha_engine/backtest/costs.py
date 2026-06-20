"""Transaction cost model.

Costs are modeled as a single per-trade fraction of notional, decomposed into
three components (all in basis points):

  - commission: broker fees (modern retail: 0-1 bps for stocks, more for options)
  - spread:     half the bid-ask spread you cross when taking liquidity
  - slippage:   market impact for larger trades

For a backtest of a small portfolio rebalancing weekly into liquid ETFs and
mega-caps, total cost of ~5-8 bps per trade is realistic. Smaller stocks or
larger sizes should scale this up.

For Phase 1 of the backtester, a flat-bps model is sufficient. Later we can
extend to a size-dependent slippage model (e.g. Almgren-Chriss).
"""

from __future__ import annotations

from alpha_engine.backtest.types import BacktestConfig


def transaction_cost(notional: float, config: BacktestConfig) -> float:
    """Return total transaction cost for a trade of given notional value.

    Notional is the absolute dollar value of the trade. Cost is symmetric
    (same for buys and sells).
    """
    bps = config.total_cost_bps
    return abs(notional) * (bps / 10_000.0)
