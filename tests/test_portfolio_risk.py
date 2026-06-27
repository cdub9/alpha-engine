"""Portfolio concentration analysis, ranked actions, and the earnings guard."""

from __future__ import annotations

from datetime import date

import duckdb
import pytest

from alpha_engine.risk.earnings_guard import has_imminent_earnings, upcoming_earnings
from alpha_engine.risk.portfolio import (
    cluster_of,
    concentration_report,
    rank_actions,
)


def _h(**kv):
    return [{"symbol": s, "value": float(v)} for s, v in kv.items()]


# --- classification --------------------------------------------------------

def test_cluster_mapping():
    assert cluster_of("MU") == "semis_ai_hw"
    assert cluster_of("nvda") == "semis_ai_hw"  # case-insensitive
    assert cluster_of("QQQM") == "tech_growth_etf"
    assert cluster_of("IVV") == "broad_index"
    assert cluster_of("TQQQ") == "leveraged"
    assert cluster_of("ZZZZ") == "other"


# --- concentration ---------------------------------------------------------

def test_weights_and_cluster_totals():
    rep = concentration_report(_h(MU=20, NVDA=20, IVV=60))
    assert rep["total_value"] == 100
    # MU + NVDA = semis cluster 40%
    assert rep["clusters"]["semis_ai_hw"]["weight"] == pytest.approx(0.40)
    assert rep["clusters"]["broad_index"]["weight"] == pytest.approx(0.60)
    assert rep["names"][0]["symbol"] in ("IVV",)  # sorted desc by weight


def test_single_stock_breaches_but_index_etf_exempt():
    # MU 12% (single stock) breaches the 5% name cap; IVV 60% does NOT
    # (diversified basket is exempt from the per-name cap).
    rep = concentration_report(_h(MU=12, IVV=60, AAA=28))
    name_breaches = {b["label"] for b in rep["breaches"] if b["kind"] == "name"}
    assert "MU" in name_breaches
    assert "IVV" not in name_breaches


def test_semis_cluster_breach():
    rep = concentration_report(_h(NVDA=10, AMD=10, MU=10, SMH=10, IVV=60))
    # semis = 40% > 20% cap
    cl = [b for b in rep["breaches"] if b["label"] == "semis_ai_hw"]
    assert cl and cl[0]["weight"] == pytest.approx(0.40)
    assert cl[0]["excess_value"] == pytest.approx(0.20 * 100)


def test_tech_total_meta_cluster():
    rep = concentration_report(_h(NVDA=30, QQQM=20, VTV=50))
    # semis 30 + tech ETF 20 = 50% tech_total > 35%
    assert rep["tech_total_weight"] == pytest.approx(0.50)
    assert any(b["label"] == "tech_total" for b in rep["breaches"])


# --- ranked actions --------------------------------------------------------

def test_actions_ranked_critical_first():
    rep = concentration_report(_h(NVDA=15, AMD=15, MU=12, IVV=58))
    actions = rank_actions(rep)
    assert actions  # there are breaches
    # critical actions sort ahead of high/watch
    sev = [a["severity"] for a in actions]
    assert sev == sorted(sev, key=lambda s: {"critical": 0, "high": 1, "watch": 2}[s])
    # MU (>2x name cap) is critical
    mu = [a for a in actions if "MU" in a["title"]]
    assert mu and mu[0]["severity"] == "critical"


def test_earnings_action_added_and_sized():
    rep = concentration_report(_h(IVV=100))
    actions = rank_actions(
        rep,
        upcoming_earnings=[{"symbol": "AMD", "date": "2026-08-04", "value": 5000.0}],
    )
    amd = [a for a in actions if a["icon"] == "calendar-event"]
    assert amd and "AMD" in amd[0]["title"]
    # 5000 / 100 total >= 4% -> high severity
    assert amd[0]["severity"] == "high"


def test_low_cash_hedge_flag():
    rep = concentration_report(_h(NVDA=40, QQQM=20, IVV=40))  # 60% tech
    actions = rank_actions(rep, cash_weight=0.02)
    assert any(a["icon"] == "shield-half" for a in actions)
    # not flagged when cash is healthy
    actions2 = rank_actions(rep, cash_weight=0.20)
    assert not any(a["icon"] == "shield-half" for a in actions2)


# --- earnings guard --------------------------------------------------------

_CAL = """
CREATE SEQUENCE calendar_events_id_seq START 1;
CREATE TABLE calendar_events (
    id BIGINT PRIMARY KEY DEFAULT nextval('calendar_events_id_seq'),
    event_date DATE, kind VARCHAR, symbol VARCHAR
);
"""


def _cal_db():
    con = duckdb.connect(":memory:")
    for stmt in _CAL.strip().split(";"):
        if stmt.strip():
            con.execute(stmt)
    return con


def _add_earn(con, symbol, d):
    con.execute(
        "INSERT INTO calendar_events (event_date, kind, symbol) VALUES (?, 'earnings', ?)",
        [d, symbol],
    )


def test_upcoming_earnings_in_window_only():
    con = _cal_db()
    as_of = date(2026, 6, 1)
    _add_earn(con, "AVGO", date(2026, 6, 3))   # in window
    _add_earn(con, "NVDA", date(2026, 6, 20))  # outside 7d window
    _add_earn(con, "AMD", date(2026, 5, 30))   # in the past
    out = upcoming_earnings(con, ["AVGO", "NVDA", "AMD"], as_of, horizon_days=7,
                            values={"AVGO": 1000.0})
    assert [e["symbol"] for e in out] == ["AVGO"]
    assert out[0]["days_away"] == 2
    assert out[0]["value"] == 1000.0


def test_has_imminent_earnings():
    con = _cal_db()
    _add_earn(con, "AVGO", date(2026, 6, 3))
    assert has_imminent_earnings(con, "AVGO", date(2026, 6, 1), horizon_days=7)
    assert not has_imminent_earnings(con, "AVGO", date(2026, 6, 1), horizon_days=1)
