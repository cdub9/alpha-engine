"""Debug: trace buy_and_hold_spy for the first ~10 rebalances."""

from __future__ import annotations

import sys
from datetime import date

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

from alpha_engine.backtest import BacktestConfig, BuyAndHoldBenchmark, run_backtest
from alpha_engine.core.logging import configure_logging

configure_logging(level="WARNING")

cfg = BacktestConfig(
    start_date=date(2022, 1, 1),
    end_date=date(2022, 4, 1),
    universe=["SPY"],
    rebalance_frequency="weekly",
)

result = run_backtest(cfg, BuyAndHoldBenchmark())

print(f"Final equity: ${result.final_equity:,.2f}")
print(f"Benchmark final: ${result.benchmark_curve.iloc[-1]:,.2f}")
print(f"Number of fills: {len(result.fills)}")
print()
print("First 10 fills:")
for f in result.fills[:10]:
    print(f"  {f.trade_date}  {f.side:4s}  {f.symbol}  qty={f.quantity:.4f}  px=${f.price:.2f}  notional=${f.notional:,.2f}  cost=${f.cost:.2f}")

print()
print("Equity curve first 5 + last 5:")
print(result.equity_curve.head().to_string())
print(result.equity_curve.tail().to_string())
print()
print("Benchmark curve first 5 + last 5:")
print(result.benchmark_curve.head().to_string())
print(result.benchmark_curve.tail().to_string())
print()
print(f"Cumulative cost: ${result.total_cost:.2f}")
