"""Sonnet A/B plumbing: cohort-tag derivation and the single-snapshot
digest-output comparison.
"""

from __future__ import annotations

from alpha_engine.backtest.llm_advisor import (
    DEFAULT_MODEL_VERSION,
    model_version_for,
)
from alpha_engine.llm.compare import compare_outputs


# --- cohort tag derivation -------------------------------------------------

def test_opus_tag_matches_existing_default():
    # The Opus mapping must reproduce the literal the rest of the system
    # (and test_llm_feedback) pins, so existing cohorts don't shift.
    assert model_version_for("claude-opus-4-7") == "llm-opus-4-7-v3-fb"
    assert DEFAULT_MODEL_VERSION == "llm-opus-4-7-v3-fb"


def test_sonnet_gets_its_own_tag():
    assert model_version_for("claude-sonnet-4-6") == "llm-sonnet-4-6-v3-fb"
    # Distinct from Opus -> cohorts can never collide.
    assert model_version_for("claude-sonnet-4-6") != DEFAULT_MODEL_VERSION


def test_custom_prompt_tag_and_non_claude_passthrough():
    assert model_version_for("claude-haiku-4-5", prompt_tag="v4") == "llm-haiku-4-5-v4"
    assert model_version_for("gpt-x") == "llm-gpt-x-v3-fb"


# --- output comparison -----------------------------------------------------

def _out(channel_a=None, channel_b=None):
    return {
        "channel_a_suggestions": channel_a or [],
        "channel_b_suggestions": channel_b or [],
    }


def _sug(symbol, direction="buy", conviction=7.0):
    return {"symbol": symbol, "direction": direction, "conviction": conviction}


def test_identical_outputs_agree_completely():
    a = _out([_sug("AAA"), _sug("BBB")], [_sug("CCC")])
    cmp = compare_outputs(a, a)
    ov = cmp["overall"]
    assert ov["symbol_jaccard"] == 1.0
    assert ov["direction_agreement"] == 1.0
    assert ov["conviction_mae"] == 0.0
    assert ov["n_shared"] == 3


def test_symbol_jaccard_and_unique_picks():
    a = _out([_sug("AAA"), _sug("BBB")])
    b = _out([_sug("BBB"), _sug("CCC")])
    ch = compare_outputs(a, b)["by_channel"]["steady_alpha"]
    # union {AAA,BBB,CCC}, shared {BBB} -> 1/3
    assert ch["symbol_jaccard"] == 1 / 3
    assert ch["only_a"] == ["AAA"]
    assert ch["only_b"] == ["CCC"]


def test_direction_conflict_detected():
    a = _out([_sug("AAA", direction="buy")])
    b = _out([_sug("AAA", direction="hold")])
    ch = compare_outputs(a, b)["by_channel"]["steady_alpha"]
    assert ch["direction_agreement"] == 0.0
    assert ch["n_diff_direction"] == 1
    assert ch["diff_direction"][0]["symbol"] == "AAA"
    assert ch["diff_direction"][0]["a_dir"] == "buy"
    assert ch["diff_direction"][0]["b_dir"] == "hold"


def test_conviction_mae():
    a = _out([_sug("AAA", conviction=8.0), _sug("BBB", conviction=6.0)])
    b = _out([_sug("AAA", conviction=7.0), _sug("BBB", conviction=6.0)])
    ch = compare_outputs(a, b)["by_channel"]["steady_alpha"]
    # |8-7| and |6-6| -> mean 0.5
    assert ch["conviction_mae"] == 0.5
    assert ch["direction_agreement"] == 1.0


def test_overall_combines_both_channels():
    a = _out([_sug("AAA")], [_sug("XXX"), _sug("YYY")])
    b = _out([_sug("AAA")], [_sug("XXX")])
    ov = compare_outputs(a, b)["overall"]
    # channel A shared AAA (union 1); channel B shared XXX (union 2)
    # overall shared=2, union=3 -> 2/3
    assert ov["symbol_jaccard"] == 2 / 3
    assert ov["n_shared"] == 2
    assert ov["direction_agreement"] == 1.0


def test_empty_outputs_are_vacuously_equal():
    ov = compare_outputs(_out(), _out())["overall"]
    assert ov["symbol_jaccard"] == 1.0
    assert ov["n_shared"] == 0
    assert ov["direction_agreement"] is None
    assert ov["conviction_mae"] is None
