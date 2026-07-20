"""Synthesize the app's return-side signals into opportunity ideas.

The risk layer (trade_plan) answers "what must I trim?" with high confidence
— it's deterministic (cap breaches, calendared earnings). This layer answers
the softer "what do the app's signals say about my names?" by combining the
ML rank, the LLM digest's view, and the technicals into a single per-holding
read, then flagging trim/add IDEAS.

Honesty is built into the structure, not bolted on:
  - These are IDEAS, not orders. The app's return-generating signals have
    UNPROVEN forward skill (the forward-validation scorers are still
    accumulating), so this never speaks with the authority the risk layer
    does. The UI must label it as such.
  - ADD ideas are cap-aware: the engine will NEVER suggest adding to a name
    whose cluster already breaches its cap (no "buy more semis" when semis
    are 38% of the book). Return conviction is gated by the risk constraints.

Pure functions over plain dicts. `signals` maps symbol -> a dict of whatever
is known: {ml_action, ml_rank, ml_n, dist_200ma, rsi_14, llm_direction,
llm_conviction}. Missing keys are treated as "no signal," not zero.
"""

from __future__ import annotations

from typing import Any, Optional

from alpha_engine.risk.portfolio import cluster_of

_LLM_LONG = {"buy", "add"}
_LLM_SHORT = {"sell", "exit", "reduce"}

# How strong the combined signal must be to raise an idea.
_IDEA_THRESHOLD = 1.5


def _score_signals(s: dict[str, Any]) -> tuple[float, list[str]]:
    """Combine a holding's signals into a score (+ = constructive, - =
    negative) and a list of the human-readable reasons that fired."""
    score = 0.0
    why: list[str] = []

    ml = s.get("ml_action")
    if ml == "BUY":
        score += 1.0
        why.append("ML ranks it a top-quintile BUY")
    elif ml == "AVOID":
        score -= 1.0
        why.append("ML ranks it a bottom-quintile AVOID")

    llm = (s.get("llm_direction") or "").lower()
    if llm in _LLM_LONG:
        score += 1.0
        why.append(f"LLM digest says {llm}")
    elif llm in _LLM_SHORT:
        score -= 1.0
        why.append(f"LLM digest says {llm}")

    dist = s.get("dist_200ma")
    if dist is not None:
        if dist > 0:
            score += 0.5
            why.append(f"above its 200-day trend ({dist:+.0%})")
        else:
            score -= 0.5
            why.append(f"below its 200-day trend ({dist:+.0%})")

    rsi = s.get("rsi_14")
    if rsi is not None:
        if rsi >= 75:
            score -= 0.5
            why.append(f"overbought (RSI {rsi:.0f})")
        elif rsi <= 30:
            score += 0.5
            why.append(f"oversold (RSI {rsi:.0f})")

    return score, why


def opportunity_ideas(
    report: dict[str, Any],
    signals: dict[str, dict[str, Any]],
    caps: Optional[dict[str, float]] = None,
) -> dict[str, list[dict[str, Any]]]:
    """Return {'trims': [...], 'adds': [...]} — softer, signal-driven ideas
    to sit alongside (not replace) the risk-layer orders.

    A TRIM idea: the app's signals have turned negative on a held name.
    An ADD idea: signals are constructive AND adding wouldn't worsen a
    cluster that's already over its cap, and the name itself is under the
    single-name cap.
    """
    caps = caps or report.get("caps", {})
    clusters = report.get("clusters", {})
    weight = {n["symbol"]: n["weight"] for n in report.get("names", [])}
    name_cap = caps.get("name", 0.05)

    def _cluster_over_cap(sym: str) -> bool:
        c = cluster_of(sym)
        cap_key = c if c in caps else None
        # semis is the cluster we cap by name; tech_total is a meta-cap.
        w = clusters.get(c, {}).get("weight", 0.0)
        if cap_key and w > caps[cap_key]:
            return True
        # Block adds to semis whenever the semis cluster is over its cap.
        if c == "semis_ai_hw" and clusters.get("semis_ai_hw", {}).get("weight", 0.0) > caps.get("semis_ai_hw", 1.0):
            return True
        return False

    trims: list[dict[str, Any]] = []
    adds: list[dict[str, Any]] = []

    for sym, s in signals.items():
        score, why = _score_signals(s)
        if not why:
            continue
        if score <= -_IDEA_THRESHOLD:
            trims.append({
                "symbol": sym, "score": round(score, 1),
                "weight": weight.get(sym),
                "reason": "Signals have turned negative: " + "; ".join(why) + ".",
                "signals": why,
            })
        elif score >= _IDEA_THRESHOLD:
            # ADD only if it doesn't push a capped cluster further, and the
            # name has room under the single-name cap.
            blocked = _cluster_over_cap(sym) or (weight.get(sym, 0.0) >= name_cap)
            if blocked:
                continue
            adds.append({
                "symbol": sym, "score": round(score, 1),
                "weight": weight.get(sym),
                "reason": "Multi-signal support with room to add: " + "; ".join(why) + ".",
                "signals": why,
            })

    trims.sort(key=lambda r: r["score"])          # most-negative first
    adds.sort(key=lambda r: r["score"], reverse=True)  # strongest first
    return {"trims": trims, "adds": adds}
