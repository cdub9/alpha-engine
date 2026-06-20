"""v3 feedback-loop behavior comparison: action mix, duplicate-of-holding
share (computable pre-maturity), conviction-calibration slope, and
repeated-miss detection — all keyed by signal cohort (model_version).
"""

from __future__ import annotations

from datetime import date, datetime

import duckdb
import pytest

from alpha_engine.llm.feedback_eval import (
    cohort_label,
    compute_feedback_loop_behavior,
)

SCHEMA = """
CREATE SEQUENCE signals_id_seq START 1;
CREATE TABLE signals (
    id BIGINT PRIMARY KEY DEFAULT nextval('signals_id_seq'),
    generated_at TIMESTAMP, channel VARCHAR, symbol VARCHAR,
    direction VARCHAR, conviction DOUBLE, model_version VARCHAR
);
CREATE SEQUENCE trades_id_seq START 1;
CREATE TABLE trades (
    id BIGINT PRIMARY KEY DEFAULT nextval('trades_id_seq'),
    placed_at TIMESTAMP, channel VARCHAR, symbol VARCHAR,
    side VARCHAR, direction VARCHAR, price DOUBLE, status VARCHAR,
    source_signal_id BIGINT
);
CREATE TABLE trade_outcomes (
    trade_id BIGINT PRIMARY KEY, evaluated_at TIMESTAMP, days_held INTEGER,
    return_pct DOUBLE, alpha DOUBLE
);
"""


def make_db():
    con = duckdb.connect(":memory:")
    for stmt in SCHEMA.strip().split(";"):
        if stmt.strip():
            con.execute(stmt)
    return con


def add_signal(con, model, symbol="AAA", channel="steady_alpha",
               direction="buy", conviction=7.5, generated=date(2026, 2, 1)):
    con.execute(
        "INSERT INTO signals (generated_at, channel, symbol, direction,"
        " conviction, model_version) VALUES (?, ?, ?, ?, ?, ?)",
        [datetime.combine(generated, datetime.min.time()), channel, symbol,
         direction, conviction, model],
    )
    return con.execute("SELECT MAX(id) FROM signals").fetchone()[0]


def add_trade(con, symbol="AAA", channel="steady_alpha", placed=date(2026, 2, 2),
              signal_id=None, direction="buy"):
    con.execute(
        "INSERT INTO trades (placed_at, channel, symbol, side, direction,"
        " price, status, source_signal_id) VALUES (?, ?, ?, 'long', ?, 100.0,"
        " 'paper_filled', ?)",
        [datetime.combine(placed, datetime.min.time()), channel, symbol,
         direction, signal_id],
    )
    return con.execute("SELECT MAX(id) FROM trades").fetchone()[0]


def add_outcome(con, trade_id, days_held=10, return_pct=0.05, alpha=0.02):
    con.execute(
        "INSERT INTO trade_outcomes (trade_id, evaluated_at, days_held,"
        " return_pct, alpha) VALUES (?, ?, ?, ?, ?)",
        [trade_id, datetime(2026, 3, 1), days_held, return_pct, alpha],
    )


# A cutoff well after every trade's logical completion so matured metrics fire.
AS_OF = date(2026, 4, 1)


def test_action_mix_and_actionable_counts():
    con = make_db()
    add_signal(con, "llm-opus-4-7-v3-fb", symbol="AAA", direction="buy")
    add_signal(con, "llm-opus-4-7-v3-fb", symbol="BBB", direction="add")
    add_signal(con, "llm-opus-4-7-v3-fb", symbol="CCC", direction="hold")

    out = compute_feedback_loop_behavior(con, as_of=AS_OF)
    c = out["cohorts"]["llm-opus-4-7-v3-fb"]
    assert c["n_signals"] == 3
    assert c["n_actionable"] == 2  # buy + add, not hold
    assert c["action_mix"] == {"buy": 1, "add": 1, "hold": 1}
    assert c["label"] == "v3 (feedback loop)"


def test_duplicate_of_holding_share():
    con = make_db()
    # Prior holding: a trade entered 02-01 with a 30-day horizon, still open
    # when the new signal is generated on 02-10.
    s_old = add_signal(con, "llm-opus-4-7-v3-fb", symbol="DUP",
                       generated=date(2026, 1, 25))
    t_old = add_trade(con, symbol="DUP", placed=date(2026, 2, 1), signal_id=s_old)
    add_outcome(con, t_old, days_held=30)  # open through ~03-03

    # New actionable signal for the SAME name while still held -> duplicate.
    add_signal(con, "llm-opus-4-7-v3-fb", symbol="DUP", generated=date(2026, 2, 10))
    # And a fresh name the model does not hold -> not a duplicate.
    add_signal(con, "llm-opus-4-7-v3-fb", symbol="NEW", generated=date(2026, 2, 10))
    # An `add` to the held name is intentional book management, not a
    # wasteful re-buy -> excluded from the metric entirely.
    add_signal(con, "llm-opus-4-7-v3-fb", symbol="DUP", direction="add",
               generated=date(2026, 2, 10))

    out = compute_feedback_loop_behavior(con, as_of=AS_OF)
    c = out["cohorts"]["llm-opus-4-7-v3-fb"]
    # 3 `buy` signals total (s_old + DUP-resuggest + NEW); only the DUP
    # re-suggestion overlaps a prior open holding.
    assert c["n_new_buys"] == 3
    assert c["n_dup_buys"] == 1
    assert c["dup_share"] == pytest.approx(1 / 3)


def test_calibration_slope_high_minus_low():
    con = make_db()
    mv = "llm-opus-4-7-v1"
    # 8.0+ bucket: avg alpha +1%
    for i in range(12):
        sid = add_signal(con, mv, symbol=f"H{i}", conviction=8.5)
        tid = add_trade(con, symbol=f"H{i}", signal_id=sid)
        add_outcome(con, tid, return_pct=0.03, alpha=0.01)
    # <7.0 bucket: avg alpha +5% (inverted scale — low conv beats high)
    for i in range(12):
        sid = add_signal(con, mv, symbol=f"L{i}", conviction=6.0)
        tid = add_trade(con, symbol=f"L{i}", signal_id=sid)
        add_outcome(con, tid, return_pct=0.07, alpha=0.05)

    out = compute_feedback_loop_behavior(con, as_of=AS_OF)
    c = out["cohorts"][mv]
    assert c["n_matured"] == 24
    assert c["calibration"]["8.0+"]["avg_alpha"] == pytest.approx(0.01)
    assert c["calibration"]["<7.0"]["avg_alpha"] == pytest.approx(0.05)
    # slope = alpha(8.0+) - alpha(<7.0) = 0.01 - 0.05 = -0.04 (inverted)
    assert c["calib_slope"] == pytest.approx(-0.04)
    assert c["calib_slope_reliable"] is True


def test_unmatured_trades_excluded_from_calibration():
    con = make_db()
    mv = "llm-opus-4-7-v3-fb"
    sid = add_signal(con, mv, symbol="AAA", conviction=8.5,
                     generated=date(2026, 3, 20))
    tid = add_trade(con, symbol="AAA", placed=date(2026, 3, 21), signal_id=sid)
    add_outcome(con, tid, days_held=30)  # completes ~04-20, after AS_OF

    out = compute_feedback_loop_behavior(con, as_of=AS_OF)
    c = out["cohorts"][mv]
    assert c["n_matured"] == 0
    assert c["calibration"] == {}
    assert c["calib_slope"] is None


def test_repeated_miss_detection():
    con = make_db()
    mv = "llm-opus-4-7-v1"
    # MISS traded twice, avg alpha -5% -> flagged.
    for _ in range(2):
        sid = add_signal(con, mv, symbol="MISS")
        tid = add_trade(con, symbol="MISS", signal_id=sid)
        add_outcome(con, tid, return_pct=-0.05, alpha=-0.05)
    # WINS traded twice, positive -> not flagged.
    for _ in range(2):
        sid = add_signal(con, mv, symbol="WINS")
        tid = add_trade(con, symbol="WINS", signal_id=sid)
        add_outcome(con, tid, return_pct=0.06, alpha=0.04)

    out = compute_feedback_loop_behavior(con, as_of=AS_OF)
    c = out["cohorts"][mv]
    misses = c["repeated_misses"]
    assert [m["symbol"] for m in misses] == ["MISS"]
    assert misses[0]["n"] == 2
    assert misses[0]["avg_alpha"] == pytest.approx(-0.05)


def test_cohorts_separated_by_model_version():
    con = make_db()
    add_signal(con, "llm-opus-4-7-v1", symbol="AAA")
    add_signal(con, "llm-opus-4-7-v3-fb", symbol="BBB")
    out = compute_feedback_loop_behavior(con, as_of=AS_OF)
    assert out["order"] == ["llm-opus-4-7-v1", "llm-opus-4-7-v3-fb"]


def test_cohort_label_fallback():
    assert cohort_label("llm-opus-4-7-v1") == "v1 (pre-feedback)"
    assert cohort_label("something-custom") == "something-custom"
