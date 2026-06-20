"""Forward BUY/AVOID performance: maturation cutoff, spread math, the
both-buckets-required rule, and pending-date bookkeeping.

Live ml_signals haven't reached the 21-trading-day horizon yet, so these
synthetic bars are what verify the scorer before real data matures.
"""

from __future__ import annotations

from datetime import date

import duckdb
import pytest

from alpha_engine.ml.forward_eval import (
    _trading_days_forward,
    compute_forward_performance,
)

SCHEMA = """
CREATE TABLE ml_signals (
    signal_date DATE, symbol VARCHAR, model_version VARCHAR,
    action VARCHAR
);
CREATE TABLE market_bars (
    symbol VARCHAR, bar_date DATE, adj_close DOUBLE
);
"""

# Eight consecutive NYSE trading days (Mon 2026-01-05 .. Wed 2026-01-14;
# 01-10/01-11 are a weekend and excluded).
TRADING_DAYS = [
    date(2026, 1, 5), date(2026, 1, 6), date(2026, 1, 7), date(2026, 1, 8),
    date(2026, 1, 9), date(2026, 1, 12), date(2026, 1, 13), date(2026, 1, 14),
]


def make_db():
    con = duckdb.connect(":memory:")
    for stmt in SCHEMA.strip().split(";"):
        if stmt.strip():
            con.execute(stmt)
    return con


def add_bars(con, symbol, prices):
    """prices: dict[date -> adj_close]."""
    for d, px in prices.items():
        con.execute(
            "INSERT INTO market_bars VALUES (?, ?, ?)", [symbol, d, px]
        )


def add_signal(con, signal_date, symbol, action, model="m1"):
    con.execute(
        "INSERT INTO ml_signals VALUES (?, ?, ?, ?)",
        [signal_date, symbol, model, action],
    )


def _full_series(start_px, step):
    """A bar on every TRADING_DAY, geometric-ish; returns dict[date->px]."""
    return {d: start_px * (step ** i) for i, d in enumerate(TRADING_DAYS)}


def test_spread_is_buy_minus_avoid_on_matured_date():
    con = make_db()
    # Entry 01-05, horizon=2 -> exit 01-07 (third trading bar).
    add_bars(con, "WIN", {date(2026, 1, 5): 100.0, date(2026, 1, 6): 105.0,
                          date(2026, 1, 7): 110.0})
    add_bars(con, "LOSE", {date(2026, 1, 5): 100.0, date(2026, 1, 6): 95.0,
                           date(2026, 1, 7): 90.0})
    add_signal(con, date(2026, 1, 5), "WIN", "BUY")
    add_signal(con, date(2026, 1, 5), "LOSE", "AVOID")

    out = compute_forward_performance(con, horizon=2)
    m = out["by_model"]["m1"]
    assert m["n_dates_matured"] == 1
    assert m["n_pending"] == 0
    assert m["mean_buy_ret"] == pytest.approx(0.10)
    assert m["mean_avoid_ret"] == pytest.approx(-0.10)
    assert m["mean_spread"] == pytest.approx(0.20)
    assert m["spread_hit_rate"] == pytest.approx(1.0)
    # all_ret = equal-weight of +10% and -10% = 0; BUY beats it.
    assert m["mean_all_ret"] == pytest.approx(0.0)
    assert m["buy_beats_all_rate"] == pytest.approx(1.0)


def test_immature_date_is_pending_not_scored():
    con = make_db()
    # Only entry + 1 bar exist; horizon=2 needs a bar two rows out -> pending.
    add_bars(con, "WIN", {date(2026, 1, 5): 100.0, date(2026, 1, 6): 105.0})
    add_bars(con, "LOSE", {date(2026, 1, 5): 100.0, date(2026, 1, 6): 95.0})
    add_signal(con, date(2026, 1, 5), "WIN", "BUY")
    add_signal(con, date(2026, 1, 5), "LOSE", "AVOID")

    out = compute_forward_performance(con, horizon=2)
    m = out["by_model"]["m1"]
    assert m["n_dates_total"] == 1
    assert m["n_dates_matured"] == 0
    assert m["n_pending"] == 1
    assert m["mean_spread"] is None
    # 2 trading days after Mon 01-05 is Wed 01-07.
    assert m["next_maturity_date"] == date(2026, 1, 7)


def test_date_needs_both_buckets():
    con = make_db()
    # BUY only on this date -> spread undefined -> not counted.
    add_bars(con, "WIN", {date(2026, 1, 5): 100.0, date(2026, 1, 6): 105.0,
                          date(2026, 1, 7): 110.0})
    add_signal(con, date(2026, 1, 5), "WIN", "BUY")

    out = compute_forward_performance(con, horizon=2)
    m = out["by_model"]["m1"]
    assert m["n_dates_matured"] == 0
    assert m["n_pending"] == 1


def test_multiple_dates_aggregate_and_hit_rate():
    con = make_db()
    # Two matured signal dates, one with positive spread, one negative.
    win = _full_series(100.0, 1.02)    # rises every bar
    lose = _full_series(100.0, 0.98)   # falls every bar
    add_bars(con, "WIN", win)
    add_bars(con, "LOSE", lose)
    # Date A: BUY=WIN, AVOID=LOSE -> positive spread.
    add_signal(con, date(2026, 1, 5), "WIN", "BUY")
    add_signal(con, date(2026, 1, 5), "LOSE", "AVOID")
    # Date B: labels flipped -> negative spread (the "wrong" call).
    add_signal(con, date(2026, 1, 6), "LOSE", "BUY")
    add_signal(con, date(2026, 1, 6), "WIN", "AVOID")

    out = compute_forward_performance(con, horizon=2)
    m = out["by_model"]["m1"]
    assert m["n_dates_matured"] == 2
    assert m["spread_hit_rate"] == pytest.approx(0.5)
    # Spreads are equal and opposite -> mean ~0.
    assert m["mean_spread"] == pytest.approx(0.0, abs=1e-9)


def test_cohorts_are_independent():
    con = make_db()
    add_bars(con, "WIN", {date(2026, 1, 5): 100.0, date(2026, 1, 6): 105.0,
                          date(2026, 1, 7): 110.0})
    add_bars(con, "LOSE", {date(2026, 1, 5): 100.0, date(2026, 1, 6): 95.0,
                           date(2026, 1, 7): 90.0})
    add_signal(con, date(2026, 1, 5), "WIN", "BUY", model="ml-momentum-v1")
    add_signal(con, date(2026, 1, 5), "LOSE", "AVOID", model="ml-momentum-v1")
    add_signal(con, date(2026, 1, 5), "WIN", "AVOID", model="ml-xgb-v1")
    add_signal(con, date(2026, 1, 5), "LOSE", "BUY", model="ml-xgb-v1")

    out = compute_forward_performance(con, horizon=2)
    assert out["by_model"]["ml-momentum-v1"]["mean_spread"] == pytest.approx(0.20)
    assert out["by_model"]["ml-xgb-v1"]["mean_spread"] == pytest.approx(-0.20)


def test_empty_db_returns_no_models():
    con = make_db()
    out = compute_forward_performance(con, horizon=21)
    assert out["horizon"] == 21
    assert out["by_model"] == {}


def test_trading_days_forward_skips_weekend():
    # Fri 2026-01-09 + 1 trading day = Mon 2026-01-12.
    assert _trading_days_forward(date(2026, 1, 9), 1) == date(2026, 1, 12)
