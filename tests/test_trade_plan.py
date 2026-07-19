"""Trade plan: single-name trims, earnings trims, trend-gated cluster cuts,
share math, and ML annotations."""

from __future__ import annotations

import pytest

from alpha_engine.risk.portfolio import DEFAULT_CAPS, concentration_report
from alpha_engine.risk.trade_plan import build_trade_plan


def _holdings(*items):
    # items: (symbol, value, price)
    return [{"symbol": s, "value": float(v), "price": float(p)} for s, v, p in items]


def test_single_name_trim_order_and_share_math():
    # MU 12% of a 100k book at $1000 -> trim to 5% = sell $7k = 7 shares.
    h = _holdings(("MU", 12000, 1000), ("IVV", 88000, 500))
    rep = concentration_report(h)
    plan = build_trade_plan(h, rep, DEFAULT_CAPS)
    mu = [o for o in plan["orders"] if o["symbol"] == "MU"]
    assert len(mu) == 1
    assert mu[0]["action"] == "SELL"
    assert mu[0]["when"] == "Now"
    assert mu[0]["shares"] == 7  # ceil(7000/1000)
    assert mu[0]["target_weight"] == pytest.approx(0.05)


def test_index_etf_not_trimmed():
    # IVV at 60% is diversified -> no single-name order for it.
    h = _holdings(("IVV", 60000, 500), ("AAA", 40000, 10))
    plan = build_trade_plan(h, concentration_report(h), DEFAULT_CAPS)
    assert not any(o["symbol"] == "IVV" for o in plan["orders"])


def test_cluster_armed_when_trend_intact():
    # 40% semis via SMH (a basket exempt from the single-name cap), trend
    # intact -> cluster cut is ARMED, not active.
    h = _holdings(("SMH", 40000, 500), ("IVV", 60000, 500))
    rep = concentration_report(h)
    plan = build_trade_plan(h, rep, DEFAULT_CAPS, trend={"above": True})
    assert plan["armed"]
    assert all(o["when"].startswith("Armed") for o in plan["armed"])
    assert "ARMED" in plan["cluster_note"]


def test_cluster_active_when_trend_broken():
    h = _holdings(("SMH", 40000, 500), ("IVV", 60000, 500))
    rep = concentration_report(h)
    plan = build_trade_plan(h, rep, DEFAULT_CAPS, trend={"above": False})
    # trend broken -> cluster orders are active (in orders), not armed
    assert not plan["armed"]
    cluster_orders = [o for o in plan["orders"] if o["when"] == "Now"
                      and "cluster" in o["reason"].lower()]
    assert cluster_orders
    assert "ACTIVE" in plan["cluster_note"]


def test_earnings_trim_order():
    # AMD 4% (under the 5% name cap, so no single-name order pre-empts it).
    h = _holdings(("AMD", 4000, 100), ("IVV", 96000, 500))
    rep = concentration_report(h)
    plan = build_trade_plan(
        h, rep, DEFAULT_CAPS,
        earnings=[{"symbol": "AMD", "date": "2026-08-04", "value": 4000.0}],
    )
    amd = [o for o in plan["orders"] if o["symbol"] == "AMD"]
    assert amd and amd[0]["when"] == "Before 2026-08-04"
    assert amd[0]["shares"] == 20  # trim half of 4000 at $100 = $2000 = 20 sh


def test_ml_avoid_prioritized_in_cluster_trims():
    # Six sub-cap semis names (4% each = 24% cluster) so the cut is cluster-
    # level, not single-name. ML says AVOID on AMD -> AMD trimmed first.
    h = _holdings(
        ("NVDA", 4000, 100), ("AMD", 4000, 100), ("TSM", 4000, 100),
        ("MRVL", 4000, 100), ("LRCX", 4000, 100), ("AMAT", 4000, 100),
        ("IVV", 76000, 500),
    )
    rep = concentration_report(h)
    plan = build_trade_plan(
        h, rep, DEFAULT_CAPS, trend={"above": False},
        ml_actions={"AMD": "AVOID", "NVDA": "BUY"},
    )
    cluster = [o for o in plan["orders"] if "cluster" in o["reason"].lower()]
    assert cluster and cluster[0]["symbol"] == "AMD"  # AVOID trimmed first


def test_ml_buy_annotation_on_forced_trim():
    h = _holdings(("MU", 12000, 1000), ("IVV", 88000, 500))
    plan = build_trade_plan(h, concentration_report(h), DEFAULT_CAPS,
                            ml_actions={"MU": "BUY"})
    mu = [o for o in plan["orders"] if o["symbol"] == "MU"][0]
    assert "risk control" in mu["ml_note"]


def test_clean_book_no_orders():
    h = _holdings(("IVV", 50000, 500), ("VXUS", 50000, 50))
    plan = build_trade_plan(h, concentration_report(h), DEFAULT_CAPS)
    assert plan["orders"] == []
    assert plan["summary"]["n_orders"] == 0
