"""Recommendation logging + forward scoring: maturation, add/trim correctness,
benchmark-relative alpha, idempotent same-day recording.
"""

from __future__ import annotations

from datetime import date

import duckdb
import pytest

from alpha_engine.analysis.reco_tracker import (
    record_recommendations,
    score_recommendations,
)

SCHEMA = """
CREATE TABLE book_recommendations (
    as_of DATE, symbol VARCHAR, kind VARCHAR, score DOUBLE, weight DOUBLE,
    signals_json VARCHAR, created_at TIMESTAMP
);
CREATE TABLE market_bars (symbol VARCHAR, bar_date DATE, adj_close DOUBLE);
"""

DAYS = [date(2026, 1, 5), date(2026, 1, 6), date(2026, 1, 7), date(2026, 1, 8)]


def make_db():
    con = duckdb.connect(":memory:")
    for stmt in SCHEMA.strip().split(";"):
        if stmt.strip():
            con.execute(stmt)
    return con


def add_bars(con, symbol, prices):
    for d, px in prices.items():
        con.execute("INSERT INTO market_bars VALUES (?, ?, ?)", [symbol, d, px])


def test_record_is_idempotent():
    con = make_db()
    opp = {"adds": [{"symbol": "META", "score": 1.5, "weight": 0.02, "signals": ["x"]}],
           "trims": []}
    assert record_recommendations(con, DAYS[0], opp) == 1
    record_recommendations(con, DAYS[0], opp)  # same day again
    n = con.execute("SELECT COUNT(*) FROM book_recommendations").fetchone()[0]
    assert n == 1  # replaced, not duplicated


def test_add_correct_when_beats_benchmark():
    con = make_db()
    # WIN +10% vs SPY +2% over horizon=2 -> add is correct, alpha +8%.
    add_bars(con, "WIN", {DAYS[0]: 100, DAYS[1]: 105, DAYS[2]: 110})
    add_bars(con, "SPY", {DAYS[0]: 100, DAYS[1]: 101, DAYS[2]: 102})
    record_recommendations(con, DAYS[0],
                           {"adds": [{"symbol": "WIN", "score": 1.5}], "trims": []})
    s = score_recommendations(con, horizon=2)
    assert s["n_matured"] == 1
    assert s["by_kind"]["add"]["hit_rate"] == 1.0
    assert s["by_kind"]["add"]["avg_alpha"] == pytest.approx(0.10 - 0.02)


def test_trim_correct_when_lags_benchmark():
    con = make_db()
    # LAG -5% vs SPY +2% -> trim is correct (you were right to lighten).
    add_bars(con, "LAG", {DAYS[0]: 100, DAYS[1]: 98, DAYS[2]: 95})
    add_bars(con, "SPY", {DAYS[0]: 100, DAYS[1]: 101, DAYS[2]: 102})
    record_recommendations(con, DAYS[0],
                           {"adds": [], "trims": [{"symbol": "LAG", "score": -1.5}]})
    s = score_recommendations(con, horizon=2)
    assert s["by_kind"]["trim"]["hit_rate"] == 1.0
    assert s["by_kind"]["trim"]["avg_alpha"] == pytest.approx(-0.05 - 0.02)


def test_trim_incorrect_when_it_rallies():
    con = make_db()
    # Trimmed name that then beats SPY -> trim was wrong.
    add_bars(con, "OOPS", {DAYS[0]: 100, DAYS[1]: 106, DAYS[2]: 112})
    add_bars(con, "SPY", {DAYS[0]: 100, DAYS[1]: 101, DAYS[2]: 102})
    record_recommendations(con, DAYS[0],
                           {"adds": [], "trims": [{"symbol": "OOPS", "score": -1.5}]})
    s = score_recommendations(con, horizon=2)
    assert s["by_kind"]["trim"]["hit_rate"] == 0.0


def test_pending_until_horizon_elapses():
    con = make_db()
    # Only entry + 1 bar; horizon 2 -> not matured.
    add_bars(con, "WIN", {DAYS[0]: 100, DAYS[1]: 105})
    add_bars(con, "SPY", {DAYS[0]: 100, DAYS[1]: 101})
    record_recommendations(con, DAYS[0],
                           {"adds": [{"symbol": "WIN", "score": 1.5}], "trims": []})
    s = score_recommendations(con, horizon=2)
    assert s["n_total"] == 1 and s["n_matured"] == 0 and s["n_pending"] == 1
    assert s["overall_hit_rate"] is None


def test_empty_state():
    con = make_db()
    s = score_recommendations(con, horizon=21)
    assert s["n_total"] == 0 and s["overall_hit_rate"] is None
    assert s["by_kind"]["add"]["hit_rate"] is None
