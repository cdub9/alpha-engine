"""Feedback loop sections: point-in-time scoring cutoff, open-position
dedup and MTM direction-adjustment, conviction buckets, symbol lessons,
and empty-state behavior.
"""

from __future__ import annotations

from datetime import date, datetime

import duckdb
import pytest

from alpha_engine.llm.feedback import (
    format_feedback_sections,
    format_open_positions_section,
    format_track_record_section,
)

SCHEMA = """
CREATE SEQUENCE signals_id_seq START 1;
CREATE TABLE signals (
    id BIGINT PRIMARY KEY DEFAULT nextval('signals_id_seq'),
    generated_at TIMESTAMP, channel VARCHAR, symbol VARCHAR,
    instrument_type VARCHAR, direction VARCHAR, conviction DOUBLE,
    target_weight DOUBLE, time_horizon_days INTEGER, stop_loss_pct DOUBLE,
    rationale VARCHAR, counter_argument VARCHAR, features_snapshot_json VARCHAR,
    model_version VARCHAR DEFAULT 'v0', created_at TIMESTAMP
);
CREATE SEQUENCE trades_id_seq START 1;
CREATE TABLE trades (
    id BIGINT PRIMARY KEY DEFAULT nextval('trades_id_seq'),
    placed_at TIMESTAMP, channel VARCHAR, symbol VARCHAR,
    instrument_type VARCHAR, side VARCHAR, direction VARCHAR,
    quantity DOUBLE, price DOUBLE, status VARCHAR,
    source_signal_id BIGINT, fees DOUBLE DEFAULT 0, notes VARCHAR,
    created_at TIMESTAMP
);
CREATE TABLE trade_outcomes (
    trade_id BIGINT PRIMARY KEY, evaluated_at TIMESTAMP, days_held INTEGER,
    return_pct DOUBLE, max_favorable_excursion DOUBLE,
    max_adverse_excursion DOUBLE, benchmark_return_pct DOUBLE,
    alpha DOUBLE, direction_correct BOOLEAN, notes VARCHAR,
    created_at TIMESTAMP
);
CREATE TABLE market_bars (
    symbol VARCHAR, bar_date DATE, open DOUBLE, high DOUBLE, low DOUBLE,
    close DOUBLE, adj_close DOUBLE, volume BIGINT, source VARCHAR,
    ingested_at TIMESTAMP
);
"""


def make_db():
    con = duckdb.connect(":memory:")
    for stmt in SCHEMA.strip().split(";"):
        if stmt.strip():
            con.execute(stmt)
    return con


def add_signal(con, channel="steady_alpha", symbol="AAA", conviction=7.0,
               horizon=30, direction="buy"):
    con.execute(
        "INSERT INTO signals (generated_at, channel, symbol, instrument_type,"
        " direction, conviction, time_horizon_days, rationale)"
        " VALUES (?, ?, ?, 'etf', ?, ?, ?, 'test')",
        [datetime(2026, 1, 2), channel, symbol, direction, conviction, horizon],
    )
    return con.execute("SELECT MAX(id) FROM signals").fetchone()[0]


def add_trade(con, symbol="AAA", channel="steady_alpha", side="long",
              placed=date(2026, 1, 5), price=100.0, signal_id=None,
              direction="buy"):
    con.execute(
        "INSERT INTO trades (placed_at, channel, symbol, instrument_type,"
        " side, direction, quantity, price, status, source_signal_id)"
        " VALUES (?, ?, ?, 'etf', ?, ?, 1.0, ?, 'paper_filled', ?)",
        [datetime.combine(placed, datetime.min.time()), channel, symbol,
         side, direction, price, signal_id],
    )
    return con.execute("SELECT MAX(id) FROM trades").fetchone()[0]


def add_outcome(con, trade_id, days_held=10, return_pct=0.05, alpha=0.02,
                notes=None):
    con.execute(
        "INSERT INTO trade_outcomes (trade_id, evaluated_at, days_held,"
        " return_pct, max_favorable_excursion, max_adverse_excursion,"
        " benchmark_return_pct, alpha, direction_correct, notes)"
        " VALUES (?, ?, ?, ?, 0, 0, ?, ?, ?, ?)",
        [trade_id, datetime(2026, 6, 1), days_held, return_pct,
         return_pct - alpha, alpha, return_pct > 0, notes],
    )


def add_bar(con, symbol, d, px):
    con.execute(
        "INSERT INTO market_bars VALUES (?, ?, ?, ?, ?, ?, ?, 1000, 'test', NULL)",
        [symbol, d, px, px, px, px, px],
    )


class TestPointInTime:
    def test_trade_completing_after_asof_is_open_not_scored(self):
        """A trade scored by the scorer (outcome row exists) but whose
        horizon completes AFTER as_of must appear as OPEN, not in the
        track record — otherwise historical snapshots leak the future."""
        con = make_db()
        sid = add_signal(con, horizon=30)
        tid = add_trade(con, placed=date(2026, 1, 5), signal_id=sid)
        add_outcome(con, tid, days_held=30, return_pct=0.10, alpha=0.05)
        add_bar(con, "AAA", date(2026, 1, 10), 102.0)

        as_of_mid = date(2026, 1, 15)  # horizon completes 2026-02-04
        track = format_track_record_section(con, as_of_mid)
        open_sec = format_open_positions_section(con, as_of_mid)
        assert track == ""           # nothing scored yet at this date
        assert "**AAA**" in open_sec  # still on the book

        as_of_late = date(2026, 3, 1)
        track = format_track_record_section(con, as_of_late)
        open_sec = format_open_positions_section(con, as_of_late)
        assert "+10.0%" in track     # now it's history
        assert open_sec == ""        # and off the book

    def test_mtm_uses_bar_at_asof_not_latest(self):
        con = make_db()
        sid = add_signal(con)
        add_trade(con, placed=date(2026, 1, 5), price=100.0, signal_id=sid)
        add_bar(con, "AAA", date(2026, 1, 10), 110.0)
        add_bar(con, "AAA", date(2026, 2, 10), 200.0)  # future bar
        sec = format_open_positions_section(con, date(2026, 1, 15))
        assert "MTM +10.0%" in sec   # not +100%


class TestOpenPositions:
    def test_dedup_keeps_latest_entry_per_name(self):
        con = make_db()
        sid = add_signal(con)
        add_trade(con, placed=date(2026, 1, 5), price=100.0, signal_id=sid)
        add_trade(con, placed=date(2026, 2, 5), price=120.0, signal_id=sid)
        add_bar(con, "AAA", date(2026, 2, 10), 126.0)
        sec = format_open_positions_section(con, date(2026, 2, 15))
        assert sec.count("**AAA**") == 1
        assert "entry 2026-02-05" in sec
        assert "MTM +5.0%" in sec    # vs the 120 entry, not the 100 one
        assert "1 unique position" in sec

    def test_short_mtm_direction_adjusted(self):
        con = make_db()
        sid = add_signal(con, direction="sell")
        add_trade(con, side="short", direction="sell", placed=date(2026, 1, 5),
                  price=100.0, signal_id=sid)
        add_bar(con, "AAA", date(2026, 1, 10), 90.0)
        sec = format_open_positions_section(con, date(2026, 1, 15))
        assert "SHORT **AAA**" in sec
        assert "MTM +10.0%" in sec   # price fell 10% = short is UP 10%

    def test_past_horizon_flagged(self):
        con = make_db()
        sid = add_signal(con, horizon=10)
        add_trade(con, placed=date(2026, 1, 5), signal_id=sid)
        add_bar(con, "AAA", date(2026, 1, 28), 100.0)
        sec = format_open_positions_section(con, date(2026, 1, 30))
        assert "PAST HORIZON" in sec


class TestTrackRecord:
    def test_conviction_buckets(self):
        con = make_db()
        for conv, ret, alpha in [(8.5, 0.10, 0.05), (8.0, -0.02, -0.04),
                                 (7.5, 0.03, 0.01), (6.0, 0.01, -0.01)]:
            sid = add_signal(con, conviction=conv)
            tid = add_trade(con, placed=date(2026, 1, 5), signal_id=sid)
            add_outcome(con, tid, days_held=10, return_pct=ret, alpha=alpha)
        sec = format_track_record_section(con, date(2026, 3, 1))
        assert "conv 8.0+: 2 scored, 50% win" in sec
        assert "conv 7.0-7.9: 1 scored, 100% win" in sec
        assert "conv <7.0: 1 scored, 100% win" in sec

    def test_stop_out_flagged(self):
        con = make_db()
        sid = add_signal(con)
        tid = add_trade(con, placed=date(2026, 1, 5), signal_id=sid)
        add_outcome(con, tid, days_held=4, return_pct=-0.06, alpha=-0.07,
                    notes="stopped out at -6% on 2026-01-09")
        sec = format_track_record_section(con, date(2026, 2, 1))
        assert "STOPPED OUT" in sec

    def test_repeated_misses_and_hits(self):
        con = make_db()
        for sym, alpha in [("BAD", -0.05), ("BAD", -0.03),
                           ("GUD", 0.06), ("GUD", 0.04),
                           ("MEH", 0.001), ("MEH", -0.001)]:
            sid = add_signal(con, symbol=sym)
            tid = add_trade(con, symbol=sym, placed=date(2026, 1, 5),
                            signal_id=sid)
            add_outcome(con, tid, days_held=10,
                        return_pct=alpha + 0.01, alpha=alpha)
        sec = format_track_record_section(con, date(2026, 2, 1))
        assert "Repeated misses" in sec and "BAD (2 trades" in sec
        assert "Repeated hits" in sec and "GUD (2 trades" in sec
        assert "MEH" not in sec.split("Repeated")[1]  # below threshold


class TestEmptyStates:
    def test_fresh_db_returns_empty(self):
        con = make_db()
        assert format_feedback_sections(con, date(2026, 1, 1)) == ""

    def test_open_only_no_track_record(self):
        con = make_db()
        sid = add_signal(con)
        add_trade(con, placed=date(2026, 1, 5), signal_id=sid)
        add_bar(con, "AAA", date(2026, 1, 10), 105.0)
        out = format_feedback_sections(con, date(2026, 1, 15))
        assert "## Your current open paper positions" in out
        assert "## Your track record" not in out


class TestPromptAndVersion:
    def test_prompt_contains_feedback_principle(self):
        from alpha_engine.llm.prompts import SYSTEM_PROMPT

        # Normalize whitespace — the prompt hard-wraps at ~76 cols, so
        # multi-word phrases can be split across lines.
        flat = " ".join(SYSTEM_PROMPT.split())
        assert "Learn from your own track record" in flat
        assert "Your current open paper positions" in flat

    def test_version_bumped_to_v3(self):
        import inspect

        from alpha_engine.backtest.llm_advisor import DEFAULT_MODEL_VERSION
        from alpha_engine.llm.parser import persist_signals

        assert DEFAULT_MODEL_VERSION == "llm-opus-4-7-v3-fb"
        sig = inspect.signature(persist_signals)
        assert sig.parameters["model_version"].default == "llm-opus-4-7-v3-fb"
