"""Execution-latency fix: next-open entry, adjusted-open math, stored
counterfactual price, entry-day inclusion in the stop walk, and the
direction-adjusted counterfactual return.
"""

from __future__ import annotations

from datetime import date, datetime, timezone

import duckdb
import pytest

from alpha_engine.db.connection import _apply_column_migrations
from alpha_engine.paper.scorer import score_due_paper_trades
from alpha_engine.paper.trader import (
    ENTRY_STYLE,
    _next_entry_prices,
    open_paper_trades_for_date,
)

# Minimal schema covering what the trader + scorer touch.
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
    alpha DOUBLE, direction_correct BOOLEAN, notes VARCHAR, created_at TIMESTAMP
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
    # Exercise the real migration so the new columns exist exactly as in prod.
    _apply_column_migrations(con)
    return con


def add_bar(con, symbol, d, o, h, low, c, adj=None):
    con.execute(
        "INSERT INTO market_bars VALUES (?,?,?,?,?,?,?,1000,'test',NULL)",
        [symbol, d, o, h, low, c, adj if adj is not None else c],
    )


# Signals are generated on the digest date (1/5); the entry bar is 1/6.
DIGEST_DATE = date(2026, 1, 5)
MODEL_VERSION = "test-mv"


def add_signal(con, symbol="AAA", direction="buy", conviction=7.0, horizon=30,
               stop=None):
    con.execute(
        "INSERT INTO signals (generated_at, channel, symbol, instrument_type,"
        " direction, conviction, time_horizon_days, stop_loss_pct, rationale,"
        " model_version)"
        " VALUES (?, 'steady_alpha', ?, 'etf', ?, ?, ?, ?, 'r', ?)",
        [datetime.combine(DIGEST_DATE, datetime.min.time()), symbol, direction,
         conviction, horizon, stop, MODEL_VERSION],
    )
    return con.execute("SELECT MAX(id) FROM signals").fetchone()[0]


class TestNextEntryPrices:
    def test_adjusted_open_math(self):
        con = make_db()
        # open 100, close 110, adj_close 55 (a 2:1 split factor) ->
        # adj_open = 100 * 55/110 = 50
        add_bar(con, "AAA", date(2026, 1, 6), o=100, h=115, low=99, c=110, adj=55)
        res = _next_entry_prices(con, "AAA", date(2026, 1, 5))
        assert res is not None
        entry_date, adj_open, adj_close = res
        assert entry_date == date(2026, 1, 6)
        assert adj_open == pytest.approx(50.0)
        assert adj_close == pytest.approx(55.0)

    def test_picks_first_bar_strictly_after(self):
        con = make_db()
        add_bar(con, "AAA", date(2026, 1, 5), 10, 10, 10, 10)  # == after, skip
        add_bar(con, "AAA", date(2026, 1, 7), 20, 20, 20, 20)  # first strictly after
        res = _next_entry_prices(con, "AAA", date(2026, 1, 5))
        assert res[0] == date(2026, 1, 7)
        assert res[1] == pytest.approx(20.0)

    def test_degenerate_bar_falls_back_to_close(self):
        con = make_db()
        add_bar(con, "AAA", date(2026, 1, 6), o=0, h=0, low=0, c=0, adj=42.0)
        res = _next_entry_prices(con, "AAA", date(2026, 1, 5))
        assert res[1] == pytest.approx(42.0)  # adj_open falls back to adj_close


class TestOpenPaperTrades:
    def test_enters_at_adjusted_open_and_stores_alt_close(self):
        con = make_db()
        sid = add_signal(con, "AAA")
        add_bar(con, "AAA", date(2026, 1, 6), o=100, h=105, low=99, c=110, adj=110)
        open_paper_trades_for_date(con, DIGEST_DATE, model_version=MODEL_VERSION, min_conviction=6.0)
        row = con.execute(
            "SELECT price, alt_entry_price, entry_style FROM trades WHERE source_signal_id = ?",
            [sid],
        ).fetchone()
        assert row[0] == pytest.approx(100.0)   # entered at the open
        assert row[1] == pytest.approx(110.0)   # close stored for measurement
        assert row[2] == ENTRY_STYLE == "next_open"


class TestScorerCounterfactual:
    def _setup_long(self, con, stop=None):
        sid = add_signal(con, "AAA", direction="buy", horizon=5, stop=stop)
        # entry day: open 100, close 110
        add_bar(con, "AAA", date(2026, 1, 6), o=100, h=112, low=95, c=110, adj=110)
        add_bar(con, "SPY", date(2026, 1, 6), 400, 400, 400, 400)
        return sid

    def test_counterfactual_close_entry_return(self):
        con = make_db()
        self._setup_long(con)
        open_paper_trades_for_date(con, DIGEST_DATE, model_version=MODEL_VERSION, min_conviction=6.0)
        # exit bar at horizon end (entry 1/6 + 5d = 1/11)
        add_bar(con, "AAA", date(2026, 1, 12), 120, 120, 120, 120, adj=120)
        add_bar(con, "SPY", date(2026, 1, 12), 410, 410, 410, 410)
        score_due_paper_trades(con, as_of=date(2026, 1, 20))
        o = con.execute(
            "SELECT return_pct, alt_entry_return_pct FROM trade_outcomes"
        ).fetchone()
        # open entry 100 -> 120 = +20%; close entry 110 -> 120 = +9.09%
        assert o[0] == pytest.approx(0.20, abs=1e-6)
        assert o[1] == pytest.approx((120 - 110) / 110, abs=1e-6)
        # gap (open beats close) is positive here
        assert o[0] - o[1] > 0

    def test_short_counterfactual_sign(self):
        con = make_db()
        add_signal(con, "BBB", direction="sell", horizon=5)
        add_bar(con, "BBB", date(2026, 1, 6), o=100, h=101, low=99, c=90, adj=90)
        add_bar(con, "SPY", date(2026, 1, 6), 400, 400, 400, 400)
        open_paper_trades_for_date(con, DIGEST_DATE, model_version=MODEL_VERSION, min_conviction=6.0)
        add_bar(con, "BBB", date(2026, 1, 12), 80, 80, 80, 80, adj=80)
        add_bar(con, "SPY", date(2026, 1, 12), 400, 400, 400, 400)
        score_due_paper_trades(con, as_of=date(2026, 1, 20))
        o = con.execute(
            "SELECT return_pct, alt_entry_return_pct FROM trade_outcomes"
        ).fetchone()
        # short from open 100 -> 80 = +20%; from close 90 -> 80 = +11.1%
        assert o[0] == pytest.approx(0.20, abs=1e-6)
        assert o[1] == pytest.approx((90 - 80) / 90, abs=1e-6)

    def test_entry_day_stop_triggers_for_open_entry(self):
        """A next-open trade is live through the entry day, so a stop hit on
        that day's low must register (it would be missed if the walk started
        the next day, as it does for close entries)."""
        con = make_db()
        # 10% stop. Entry open 100 -> stop level 90. Entry day low = 85 < 90.
        self._setup_long(con, stop=0.10)
        # rewrite the entry bar with a low that pierces the stop
        con.execute("DELETE FROM market_bars WHERE symbol='AAA'")
        add_bar(con, "AAA", date(2026, 1, 6), o=100, h=101, low=85, c=95, adj=95)
        open_paper_trades_for_date(con, DIGEST_DATE, model_version=MODEL_VERSION, min_conviction=6.0)
        score_due_paper_trades(con, as_of=date(2026, 1, 20))
        o = con.execute(
            "SELECT return_pct, notes FROM trade_outcomes"
        ).fetchone()
        assert o[1] is not None and "stop" in o[1].lower()
        assert o[0] == pytest.approx(-0.10, abs=1e-6)  # exited at the -10% stop


class TestMigrationIdempotent:
    def test_migration_runs_twice_safely(self):
        con = make_db()
        _apply_column_migrations(con)  # already applied in make_db; must no-op
        cols = [r[0] for r in con.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_name='trades'"
        ).fetchall()]
        assert cols.count("entry_style") == 1
        assert cols.count("alt_entry_price") == 1
