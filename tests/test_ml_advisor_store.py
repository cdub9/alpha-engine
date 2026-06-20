"""Advisor integration (in-memory DuckDB with synthetic bars) and the
ml_signals persistence round-trip.
"""

from __future__ import annotations

from datetime import date

import duckdb
import numpy as np
import pandas as pd
import pytest

from alpha_engine.ml.advisor import MLMomentumAdvisor, XGBMomentumAdvisor
from alpha_engine.ml.features import compute_features
from alpha_engine.ml.model import MomentumComposite, assign_actions
from alpha_engine.ml.store import (
    latest_ml_signal_date,
    load_ml_signals,
    persist_ml_signals,
)

ML_SIGNALS_DDL = """
CREATE TABLE IF NOT EXISTS ml_signals (
    signal_date   DATE NOT NULL,
    symbol        VARCHAR NOT NULL,
    model_version VARCHAR NOT NULL,
    score         DOUBLE NOT NULL,
    rank          INTEGER NOT NULL,
    n_universe    INTEGER NOT NULL,
    action        VARCHAR NOT NULL,
    mom_12_1      DOUBLE, mom_6_1 DOUBLE, mom_3_1 DOUBLE, rev_1m DOUBLE,
    vol_30d       DOUBLE, dist_50ma DOUBLE, dist_200ma DOUBLE, rsi_14 DOUBLE,
    generated_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (signal_date, symbol, model_version)
);
"""


def seed_bars(con: duckdb.DuckDBPyConnection, n_days: int = 600, n_syms: int = 15) -> pd.DataFrame:
    """Create market_bars with deterministic trends; returns the price panel."""
    con.execute(
        "CREATE TABLE market_bars (symbol VARCHAR, bar_date DATE, open DOUBLE,"
        " high DOUBLE, low DOUBLE, close DOUBLE, adj_close DOUBLE, volume BIGINT,"
        " source VARCHAR, ingested_at TIMESTAMP)"
    )
    rng = np.random.default_rng(11)
    idx = pd.bdate_range("2019-06-02", periods=n_days)
    panel = {}
    rows = []
    for i in range(n_syms):
        drift = (i - n_syms / 2) * 0.0005
        rets = rng.normal(drift, 0.006, n_days)
        prices = 100.0 * np.exp(np.cumsum(rets))
        panel[f"S{i:02d}"] = prices
        for d, p in zip(idx, prices):
            rows.append([f"S{i:02d}", d.date(), p, p, p, p, p, 1000, "test", None])
    # SPY benchmark — flat-ish
    spy = 100.0 * np.exp(np.cumsum(rng.normal(0.0002, 0.005, n_days)))
    panel["SPY"] = spy
    for d, p in zip(idx, spy):
        rows.append(["SPY", d.date(), p, p, p, p, p, 1000, "test", None])
    con.executemany(
        "INSERT INTO market_bars VALUES (?,?,?,?,?,?,?,?,?,?)", rows
    )
    return pd.DataFrame(panel, index=idx)


class TestMLMomentumAdvisor:
    def test_picks_uptrending_names_equal_weight(self):
        con = duckdb.connect(":memory:")
        panel = seed_bars(con)
        as_of = panel.index[-1].date()
        universe = [c for c in panel.columns if c != "SPY"]
        adv = MLMomentumAdvisor()
        weights = adv.target_weights(as_of, con, universe)

        assert weights, "expected non-empty weights"
        assert sum(weights.values()) == pytest.approx(1.0)
        n = len(weights)
        assert all(w == pytest.approx(1.0 / n) for w in weights.values())
        # Top quintile of 15 = 3 names. With noise, exact membership can
        # wobble, but every pick must come from the strong-drift half
        # (S10+) — picking a downtrending name would be a real bug.
        assert n == 3
        assert all(sym >= "S10" for sym in weights)

    def test_thin_cross_section_falls_back_to_benchmark(self):
        con = duckdb.connect(":memory:")
        panel = seed_bars(con, n_syms=4)  # < 10 rankable names
        as_of = panel.index[-1].date()
        universe = [c for c in panel.columns if c != "SPY"]
        weights = MLMomentumAdvisor().target_weights(as_of, con, universe)
        assert weights == {"SPY": 1.0}

    def test_no_data_falls_back_to_benchmark(self):
        con = duckdb.connect(":memory:")
        con.execute("CREATE TABLE market_bars (symbol VARCHAR, bar_date DATE,"
                    " open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE,"
                    " adj_close DOUBLE, volume BIGINT, source VARCHAR,"
                    " ingested_at TIMESTAMP)")
        weights = MLMomentumAdvisor().target_weights(
            date(2024, 1, 5), con, ["AAA", "BBB"]
        )
        assert weights == {"SPY": 1.0}

    def test_point_in_time_ignores_future_bars(self):
        """Weights for an early as_of are identical whether or not later
        bars exist in the table."""
        con1 = duckdb.connect(":memory:")
        panel = seed_bars(con1, n_days=600)
        early_asof = panel.index[400].date()
        universe = [c for c in panel.columns if c != "SPY"]
        w_with_future = MLMomentumAdvisor().target_weights(early_asof, con1, universe)

        con2 = duckdb.connect(":memory:")
        seed_bars(con2, n_days=600)
        con2.execute("DELETE FROM market_bars WHERE bar_date > ?", [early_asof])
        w_without_future = MLMomentumAdvisor().target_weights(early_asof, con2, universe)

        assert w_with_future == w_without_future


class TestXGBAdvisor:
    def test_produces_valid_weights(self):
        con = duckdb.connect(":memory:")
        panel = seed_bars(con, n_days=700)
        as_of = panel.index[-1].date()
        universe = [c for c in panel.columns if c != "SPY"]
        adv = XGBMomentumAdvisor()
        adv.model.min_train_rows = 200  # synthetic panel is small
        weights = adv.target_weights(as_of, con, universe)
        assert weights
        assert sum(weights.values()) <= 1.0 + 1e-9
        assert all(s in universe or s == "SPY" for s in weights)


class TestStore:
    def _prepped(self):
        con = duckdb.connect(":memory:")
        con.execute(ML_SIGNALS_DDL)
        panel = seed_bars(con)
        feats = compute_features(panel.drop(columns=["SPY"]))
        scores = MomentumComposite().score_cross_section(feats)
        actions = assign_actions(scores)
        return con, panel, feats, scores, actions

    def test_round_trip(self):
        con, panel, feats, scores, actions = self._prepped()
        d = panel.index[-1].date()
        n = persist_ml_signals(con, d, scores, actions, feats, "ml-momentum-v1")
        assert n == scores.notna().sum()

        df = load_ml_signals(con, d)
        assert len(df) == n
        # Rank 1 = highest score; ranks are contiguous
        assert df.iloc[0]["score"] == pytest.approx(scores.max())
        assert list(df["rank"]) == list(range(1, n + 1))
        # Action survived
        top_sym = df.iloc[0]["symbol"]
        assert df.iloc[0]["action"] == actions[top_sym] == "BUY"
        assert latest_ml_signal_date(con) == d

    def test_same_day_rerun_replaces(self):
        con, panel, feats, scores, actions = self._prepped()
        d = panel.index[-1].date()
        persist_ml_signals(con, d, scores, actions, feats, "ml-momentum-v1")
        persist_ml_signals(con, d, scores, actions, feats, "ml-momentum-v1")
        count = con.execute(
            "SELECT COUNT(*) FROM ml_signals WHERE signal_date = ?", [d]
        ).fetchone()[0]
        assert count == scores.notna().sum()  # not doubled

    def test_different_model_versions_coexist(self):
        con, panel, feats, scores, actions = self._prepped()
        d = panel.index[-1].date()
        persist_ml_signals(con, d, scores, actions, feats, "ml-momentum-v1")
        persist_ml_signals(con, d, scores, actions, feats, "ml-xgb-v1")
        count = con.execute(
            "SELECT COUNT(DISTINCT model_version) FROM ml_signals WHERE signal_date = ?",
            [d],
        ).fetchone()[0]
        assert count == 2

    def test_empty_scores_writes_nothing(self):
        con = duckdb.connect(":memory:")
        con.execute(ML_SIGNALS_DDL)
        n = persist_ml_signals(
            con, date(2024, 1, 5),
            pd.Series(dtype=float), pd.Series(dtype=object),
            pd.DataFrame(), "ml-momentum-v1",
        )
        assert n == 0
