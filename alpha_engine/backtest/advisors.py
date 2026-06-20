"""Reference signal advisors used to validate the backtest harness and to
serve as benchmarks for future signals to beat.

  - BuyAndHoldBenchmark        : 100% benchmark, no rebalancing
  - EqualWeightUniverse        : 1/N each on every rebalance
  - SixtyFortyClassic          : 60% SPY / 40% AGG, classic balanced portfolio
  - RegimeDefensive            : Uses the regime classifier; equities in
                                 expansion, Treasuries in late_cycle / recession
                                 (NAIVE — underperforms when macro and price
                                 action disagree)
  - RegimeWithTrendConfirmation: Improved version. Only goes defensive when
                                 the macro regime is bearish AND price action
                                 confirms (SPY below 200-day MA). When the
                                 signals disagree, trust the price.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Optional

import duckdb

from alpha_engine.backtest.types import SignalAdvisor
from alpha_engine.core.types import MarketRegime
from alpha_engine.regime import classify, extract_features, get_prior_regime


# ---------------------------------------------------------------------------
# Trend confirmation helper (point-in-time safe)
# ---------------------------------------------------------------------------


def spy_trend_at(
    con: duckdb.DuckDBPyConnection,
    as_of: date,
    sma_window: int = 200,
    symbol: str = "SPY",
) -> tuple[Optional[float], Optional[float]]:
    """Return (latest_close, N-day SMA) for `symbol` using only bars with
    bar_date <= as_of. Returns (None, None) if insufficient history.

    The SMA is computed over the most recent `sma_window` *trading days*
    available, not calendar days. We pull ~1.5x the window in calendar
    days to be safe, then take the last `sma_window` rows.
    """
    lookback = int(sma_window * 1.5) + 10
    start = as_of - timedelta(days=lookback)
    rows = con.execute(
        "SELECT bar_date, adj_close FROM market_bars "
        "WHERE symbol = ? AND bar_date BETWEEN ? AND ? "
        "ORDER BY bar_date",
        [symbol, start, as_of],
    ).fetchall()
    if len(rows) < sma_window:
        return None, None
    latest = float(rows[-1][1])
    window = rows[-sma_window:]
    sma = sum(float(r[1]) for r in window) / len(window)
    return latest, sma


class BuyAndHoldBenchmark(SignalAdvisor):
    name = "buy_and_hold_spy"
    description = "Always 100% in the benchmark (SPY)"

    def __init__(self, benchmark: str = "SPY") -> None:
        self.benchmark = benchmark

    def target_weights(
        self,
        as_of: date,
        con: duckdb.DuckDBPyConnection,
        universe: list[str],
    ) -> dict[str, float]:
        return {self.benchmark: 1.0}


class EqualWeightUniverse(SignalAdvisor):
    name = "equal_weight"
    description = "Equal weight across every symbol in the universe"

    def target_weights(
        self,
        as_of: date,
        con: duckdb.DuckDBPyConnection,
        universe: list[str],
    ) -> dict[str, float]:
        if not universe:
            return {}
        w = 1.0 / len(universe)
        return {s: w for s in universe}


class SixtyFortyClassic(SignalAdvisor):
    name = "sixty_forty"
    description = "Classic 60% equities (SPY) / 40% bonds (AGG)"

    def __init__(self, equity: str = "SPY", bond: str = "AGG") -> None:
        self.equity = equity
        self.bond = bond

    def target_weights(
        self,
        as_of: date,
        con: duckdb.DuckDBPyConnection,
        universe: list[str],
    ) -> dict[str, float]:
        return {self.equity: 0.60, self.bond: 0.40}


class RegimeDefensive(SignalAdvisor):
    """Uses the rule-based regime classifier to switch between equities
    and Treasuries. NAIVE — underperforms during periods where macro signal
    and price action disagree (e.g. 2022-2024 inversion + AI rally).

      - expansion_low_vol  : 100% SPY
      - expansion_high_vol : 70% SPY / 30% TLT
      - late_cycle         : 30% SPY / 70% TLT
      - recession          : 100% TLT
      - recovery           : 100% SPY (best risk/reward window)
      - unknown            : 60% SPY / 40% TLT (safe default)

    Kept for comparison with RegimeWithTrendConfirmation.
    """

    name = "regime_defensive"
    description = (
        "Switches between SPY and TLT based on the macro regime classifier"
    )

    REGIME_TO_WEIGHTS: dict[MarketRegime, dict[str, float]] = {
        MarketRegime.EXPANSION_LOW_VOL:  {"SPY": 1.00},
        MarketRegime.EXPANSION_HIGH_VOL: {"SPY": 0.70, "TLT": 0.30},
        MarketRegime.LATE_CYCLE:         {"SPY": 0.30, "TLT": 0.70},
        MarketRegime.RECESSION:          {"TLT": 1.00},
        MarketRegime.RECOVERY:           {"SPY": 1.00},
        MarketRegime.UNKNOWN:            {"SPY": 0.60, "TLT": 0.40},
    }

    def target_weights(
        self,
        as_of: date,
        con: duckdb.DuckDBPyConnection,
        universe: list[str],
    ) -> dict[str, float]:
        features = extract_features(con, as_of)
        prior = get_prior_regime(con, before=as_of)
        assessment = classify(features, prior_regime=prior)
        return self.REGIME_TO_WEIGHTS[assessment.regime]


class RegimeWithTrendConfirmation(SignalAdvisor):
    """Improvement on RegimeDefensive: requires BOTH bearish macro AND
    negative trend (SPY < 200-day MA) before flipping defensive.

    The principle: macro signal and price action must agree to justify
    going risk-off. When they disagree, trust the price — the equity
    premium is real and persistent, and getting out of a bull market
    on a macro warning has historically been very costly (e.g. the
    "great inversion" of 2022-2024).

    Decision matrix (rows = macro regime, columns = trend):

                           | Trend UP (SPY ≥ 200d MA) | Trend DOWN
    -----------------------|--------------------------|-----------------
    EXPANSION_LOW_VOL      |   100% SPY               | 90% SPY / 10% TLT
    EXPANSION_HIGH_VOL     |   100% SPY               | 70% SPY / 30% TLT
    LATE_CYCLE             |   85% SPY / 15% TLT      | 30% SPY / 70% TLT
    RECESSION              |   50% SPY / 50% TLT      | 100% TLT
    RECOVERY               |   100% SPY               | 100% SPY (always buy)
    UNKNOWN                |   60% SPY / 40% AGG      | 50% SPY / 50% AGG

    Fallback: if trend cannot be computed (insufficient history),
    behaves like RegimeDefensive.
    """

    name = "regime_with_trend"
    description = (
        "Macro regime classifier + SPY 200-day MA confirmation. Goes "
        "defensive only when both agree."
    )

    # (regime, trend_down) -> weights
    DECISION_MATRIX: dict[tuple[MarketRegime, bool], dict[str, float]] = {
        # Trend UP
        (MarketRegime.EXPANSION_LOW_VOL, False):  {"SPY": 1.00},
        (MarketRegime.EXPANSION_HIGH_VOL, False): {"SPY": 1.00},
        (MarketRegime.LATE_CYCLE, False):         {"SPY": 0.85, "TLT": 0.15},
        (MarketRegime.RECESSION, False):          {"SPY": 0.50, "TLT": 0.50},
        (MarketRegime.RECOVERY, False):           {"SPY": 1.00},
        (MarketRegime.UNKNOWN, False):            {"SPY": 0.60, "AGG": 0.40},
        # Trend DOWN
        (MarketRegime.EXPANSION_LOW_VOL, True):   {"SPY": 0.90, "TLT": 0.10},
        (MarketRegime.EXPANSION_HIGH_VOL, True):  {"SPY": 0.70, "TLT": 0.30},
        (MarketRegime.LATE_CYCLE, True):          {"SPY": 0.30, "TLT": 0.70},
        (MarketRegime.RECESSION, True):           {"TLT": 1.00},
        (MarketRegime.RECOVERY, True):            {"SPY": 1.00},
        (MarketRegime.UNKNOWN, True):             {"SPY": 0.50, "AGG": 0.50},
    }

    def __init__(self, sma_window: int = 200) -> None:
        self.sma_window = sma_window

    def target_weights(
        self,
        as_of: date,
        con: duckdb.DuckDBPyConnection,
        universe: list[str],
    ) -> dict[str, float]:
        features = extract_features(con, as_of)
        prior = get_prior_regime(con, before=as_of)
        assessment = classify(features, prior_regime=prior)

        latest, sma = spy_trend_at(con, as_of, sma_window=self.sma_window)
        if latest is None or sma is None:
            # Fall back to pure-regime behavior if no trend data yet
            return RegimeDefensive.REGIME_TO_WEIGHTS[assessment.regime]

        trend_down = latest < sma
        return self.DECISION_MATRIX[(assessment.regime, trend_down)]
