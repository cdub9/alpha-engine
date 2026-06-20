"""Backtest type definitions: config, advisor interface, result containers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date
from typing import TYPE_CHECKING, Literal, Optional

import pandas as pd

if TYPE_CHECKING:
    import duckdb

    from alpha_engine.backtest.metrics import BacktestMetrics


RebalanceFrequency = Literal["daily", "weekly", "biweekly", "monthly"]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BacktestConfig:
    """Parameters that fully define a backtest run."""

    start_date: date
    end_date: date
    initial_capital: float = 100_000.0

    # Universe of symbols the advisor may allocate to (excludes benchmark
    # unless the advisor explicitly adds it).
    universe: list[str] = field(default_factory=list)
    benchmark: str = "SPY"

    rebalance_frequency: RebalanceFrequency = "weekly"

    # Transaction cost model (basis points of notional)
    commission_bps: float = 1.0      # broker fees
    spread_bps: float = 3.0          # half-spread
    slippage_bps: float = 2.0        # market impact

    # Risk caps applied AFTER advisor returns target weights.
    # Defaults: no per-position cap (1.0), no leverage (1.0). Channels
    # supply their own caps (see config/channels.yaml — steady_alpha is
    # 0.05, aggressive_growth is 0.15) when running channel-specific
    # backtests.
    max_position_weight: float = 1.0
    max_leverage: float = 1.0

    # Trade-only-on-rebalance vs allow daily rebalances when targets drift
    drift_rebalance_threshold: float = 0.05  # rebalance if any position > 5pp off

    @property
    def total_cost_bps(self) -> float:
        return self.commission_bps + self.spread_bps + self.slippage_bps


# ---------------------------------------------------------------------------
# Advisor interface
# ---------------------------------------------------------------------------


class SignalAdvisor(ABC):
    """A strategy. Given a date and context, returns target portfolio weights.

    Implementations MUST NOT read data with obs_date > `as_of` — the engine
    does not enforce this, the advisor is responsible. (We pass a connection
    directly so advisors can query the DuckDB schema freely; that's the
    tradeoff for flexibility.)

    Returned weights:
      - keys are symbols (must be in config.universe or equal to benchmark)
      - values are fractions of portfolio NAV; positive = long
      - sum should be <= 1.0; any remainder is cash
      - empty dict = go fully to cash
    """

    name: str = "unnamed"
    description: str = ""

    @abstractmethod
    def target_weights(
        self,
        as_of: date,
        con: "duckdb.DuckDBPyConnection",
        universe: list[str],
    ) -> dict[str, float]:
        """Return target weights {symbol: weight}."""


# ---------------------------------------------------------------------------
# Trade record
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Fill:
    """A single executed trade in the backtest."""

    trade_date: date
    symbol: str
    side: Literal["buy", "sell"]
    quantity: float
    price: float
    notional: float            # signed: positive = bought, negative = sold
    cost: float                # total transaction cost (commission+spread+slippage)
    reason: str = ""           # 'rebalance' / 'drift' / 'liquidate'


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass
class BacktestResult:
    """Output of a backtest run."""

    config: BacktestConfig
    advisor_name: str

    # Daily series indexed by trading date
    equity_curve: pd.Series        # portfolio NAV
    benchmark_curve: pd.Series     # benchmark NAV (same starting capital)
    daily_returns: pd.Series       # portfolio returns
    benchmark_returns: pd.Series

    # Holdings snapshots: DataFrame indexed by date, columns = symbols, values = weights
    holdings: pd.DataFrame

    # Trade log
    fills: list[Fill] = field(default_factory=list)

    # Computed at end of run
    metrics: Optional["BacktestMetrics"] = None

    @property
    def final_equity(self) -> float:
        return float(self.equity_curve.iloc[-1])

    @property
    def total_cost(self) -> float:
        return sum(f.cost for f in self.fills)
