"""Opportunity-idea synthesis: signal scoring, cap-aware add gating."""

from __future__ import annotations

from alpha_engine.analysis.holistic import opportunity_ideas
from alpha_engine.risk.portfolio import concentration_report


def _report(**vals):
    return concentration_report([{"symbol": s, "value": float(v)} for s, v in vals.items()])


def test_trim_idea_when_signals_turn_negative():
    rep = _report(INTC=8000, IVV=92000)
    sig = {"INTC": {"ml_action": "AVOID", "llm_direction": "sell", "dist_200ma": -0.1}}
    out = opportunity_ideas(rep, sig)
    assert [t["symbol"] for t in out["trims"]] == ["INTC"]
    assert out["adds"] == []


def test_add_idea_for_constructive_uncapped_name():
    # LLY: strong signals, small weight, healthcare (not a capped cluster).
    rep = _report(LLY=2000, IVV=98000)
    sig = {"LLY": {"ml_action": "BUY", "llm_direction": "buy", "dist_200ma": 0.08}}
    out = opportunity_ideas(rep, sig)
    assert [a["symbol"] for a in out["adds"]] == ["LLY"]


def test_add_blocked_when_cluster_over_cap():
    # NVDA has great signals but semis are already 40% (> 20% cap) -> no add.
    rep = _report(NVDA=20000, AMD=20000, IVV=60000)
    sig = {"NVDA": {"ml_action": "BUY", "llm_direction": "add", "dist_200ma": 0.15}}
    out = opportunity_ideas(rep, sig)
    assert out["adds"] == []


def test_add_blocked_when_name_at_cap():
    # A single name already at/over the 5% name cap isn't an "add" candidate.
    rep = _report(LLY=6000, IVV=94000)  # LLY 6% > 5% name cap
    sig = {"LLY": {"ml_action": "BUY", "llm_direction": "buy", "dist_200ma": 0.1}}
    out = opportunity_ideas(rep, sig)
    assert out["adds"] == []


def test_mixed_signals_below_threshold_no_idea():
    rep = _report(AAA=3000, IVV=97000)
    # ML BUY (+1) but LLM sell (-1) and flat -> net ~0, no idea either way.
    sig = {"AAA": {"ml_action": "BUY", "llm_direction": "sell"}}
    out = opportunity_ideas(rep, sig)
    assert out["trims"] == [] and out["adds"] == []


def test_reasons_are_surfaced():
    rep = _report(INTC=8000, IVV=92000)
    sig = {"INTC": {"ml_action": "AVOID", "rsi_14": 28.0, "dist_200ma": -0.2}}
    out = opportunity_ideas(rep, sig)
    # AVOID (-1) + below trend (-0.5); oversold RSI (+0.5) -> net -1.0, not <= -1.5
    # so no trim; assert the scorer combined them (no idea raised).
    assert out["trims"] == []
    # Strengthen: drop the oversold offset -> now a trim with listed reasons.
    sig2 = {"INTC": {"ml_action": "AVOID", "dist_200ma": -0.2, "llm_direction": "reduce"}}
    out2 = opportunity_ideas(rep, sig2)
    assert out2["trims"] and "ML ranks it a bottom-quintile AVOID" in out2["trims"][0]["signals"]
