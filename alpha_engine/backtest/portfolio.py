"""Portfolio: holds positions, computes mark-to-market, executes rebalances.

Stateful object that the engine mutates each trading day. Tracks:
  - cash balance
  - per-symbol share counts
  - cumulative transaction costs paid

Provides:
  - mark_to_market(prices) -> NAV given price snapshot
  - rebalance_to(target_weights, prices, date) -> list[Fill]
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from alpha_engine.backtest.costs import transaction_cost
from alpha_engine.backtest.types import BacktestConfig, Fill


class Portfolio:
    def __init__(self, config: BacktestConfig) -> None:
        self.config = config
        self.cash: float = config.initial_capital
        self.positions: dict[str, float] = {}  # symbol -> shares (fractional OK)
        self.cumulative_cost: float = 0.0

    # ------------------------------------------------------------------
    # Mark-to-market
    # ------------------------------------------------------------------

    def mark_to_market(self, prices: pd.Series) -> float:
        """NAV = cash + sum(shares * price). prices: Series indexed by symbol."""
        equity = self.cash
        for sym, shares in self.positions.items():
            px = prices.get(sym)
            if px is None or pd.isna(px):
                # Stale price for a held position — fall back to last known
                # by keeping position value flat. Logged elsewhere.
                continue
            equity += shares * float(px)
        return equity

    def weights(self, prices: pd.Series) -> dict[str, float]:
        """Current weights as a fraction of NAV. Includes cash."""
        nav = self.mark_to_market(prices)
        if nav <= 0:
            return {"CASH": 1.0}
        out: dict[str, float] = {"CASH": self.cash / nav}
        for sym, shares in self.positions.items():
            px = prices.get(sym)
            if px is None or pd.isna(px):
                continue
            out[sym] = (shares * float(px)) / nav
        return out

    # ------------------------------------------------------------------
    # Rebalance to target weights
    # ------------------------------------------------------------------

    def rebalance_to(
        self,
        targets: dict[str, float],
        prices: pd.Series,
        trade_date: date,
        reason: str = "rebalance",
    ) -> list[Fill]:
        """Trade to reach target weights. Applies risk caps from config.

        Algorithm:
          1. Cap targets at config.max_position_weight per name.
          2. Scale down if sum > config.max_leverage.
          3. Compute current weights at MTM.
          4. For each symbol with non-trivial diff, compute share delta.
          5. Execute trade, updating cash and positions, accumulating costs.
        """
        # Apply per-position cap
        capped = {
            s: max(0.0, min(w, self.config.max_position_weight))
            for s, w in targets.items()
            if w > 0
        }
        # Apply leverage cap
        total = sum(capped.values())
        if total > self.config.max_leverage:
            scale = self.config.max_leverage / total
            capped = {s: w * scale for s, w in capped.items()}

        nav = self.mark_to_market(prices)
        fills: list[Fill] = []

        # Build full symbol set (current + target); we may need to sell
        # positions absent from the target.
        all_syms = set(self.positions.keys()) | set(capped.keys())

        # Process sells first (frees cash for buys)
        ordered: list[tuple[str, float]] = []
        for sym in all_syms:
            target_w = capped.get(sym, 0.0)
            ordered.append((sym, target_w))
        # Sort: sells (target lower than current value) before buys
        def _sort_key(item: tuple[str, float]) -> int:
            sym, target_w = item
            px = prices.get(sym)
            if px is None or pd.isna(px):
                return 1
            current_val = self.positions.get(sym, 0.0) * float(px)
            target_val = target_w * nav
            return 0 if target_val < current_val else 1

        ordered.sort(key=_sort_key)

        for sym, target_w in ordered:
            px = prices.get(sym)
            if px is None or pd.isna(px) or float(px) <= 0:
                continue
            px_f = float(px)

            target_val = target_w * nav
            current_shares = self.positions.get(sym, 0.0)
            current_val = current_shares * px_f
            delta_val = target_val - current_val

            # Skip trivial adjustments (< $1 or < 0.1% of NAV)
            if abs(delta_val) < max(1.0, 0.001 * nav):
                continue

            delta_shares = delta_val / px_f
            cost = transaction_cost(delta_val, self.config)

            # Check we can afford the buy after costs
            if delta_val > 0 and self.cash < delta_val + cost:
                # Scale down to what cash allows
                affordable = max(0.0, self.cash - cost)
                delta_shares = affordable / px_f
                delta_val = delta_shares * px_f
                if delta_shares <= 0:
                    continue
                cost = transaction_cost(delta_val, self.config)

            self.positions[sym] = current_shares + delta_shares
            self.cash -= delta_val + cost
            self.cumulative_cost += cost

            # Drop dust positions (avoid float-noise accumulation)
            if abs(self.positions[sym]) < 1e-9:
                self.positions.pop(sym, None)

            fills.append(
                Fill(
                    trade_date=trade_date,
                    symbol=sym,
                    side="buy" if delta_shares > 0 else "sell",
                    quantity=abs(delta_shares),
                    price=px_f,
                    notional=delta_val,
                    cost=cost,
                    reason=reason,
                )
            )

        return fills
