"""Compare two digest outputs produced from the SAME snapshot.

Used by `scripts/compare_models.py` to A/B a cheaper primary model
(e.g. Sonnet 4.6) against the incumbent (Opus 4.7) on identical input.
Both models see the exact same system prompt + snapshot, so any
difference is the model, not the context.

The single-day comparison answers "do they pick the same things?"
cheaply and immediately. It is NOT the skill verdict — that comes from
the forward cohorts (each model tagged its own model_version, compared in
the dashboard once trades mature). Two models agreeing here means a switch
is low-risk; disagreement means the forward A/B matters more.

Pure functions over plain dicts so the logic is unit-testable without any
API calls.
"""

from __future__ import annotations

from typing import Any

_CHANNELS = (
    ("channel_a_suggestions", "steady_alpha"),
    ("channel_b_suggestions", "aggressive_growth"),
)


def _index(suggestions: list[dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for sug in suggestions or []:
        sym = (sug.get("symbol") or "").upper().strip()
        if sym:
            out[sym] = sug
    return out


def _direction(sug: dict) -> str:
    return (sug.get("direction") or "").lower().strip()


def _conviction(sug: dict) -> float | None:
    try:
        return float(sug.get("conviction"))
    except (TypeError, ValueError):
        return None


def _compare_channel(a_list: list[dict], b_list: list[dict]) -> dict[str, Any]:
    a_map, b_map = _index(a_list), _index(b_list)
    a_syms, b_syms = set(a_map), set(b_map)
    shared = a_syms & b_syms
    union = a_syms | b_syms

    same_dir, diff_dir = [], []
    conv_abs_diffs: list[float] = []
    for sym in sorted(shared):
        a_d, b_d = _direction(a_map[sym]), _direction(b_map[sym])
        a_c, b_c = _conviction(a_map[sym]), _conviction(b_map[sym])
        if a_c is not None and b_c is not None:
            conv_abs_diffs.append(abs(a_c - b_c))
        (same_dir if a_d == b_d else diff_dir).append({
            "symbol": sym, "a_dir": a_d, "b_dir": b_d,
            "a_conv": a_c, "b_conv": b_c,
        })

    return {
        "n_a": len(a_syms),
        "n_b": len(b_syms),
        "n_shared": len(shared),
        "symbol_jaccard": (len(shared) / len(union)) if union else 1.0,
        "n_same_direction": len(same_dir),
        "n_diff_direction": len(diff_dir),
        "direction_agreement": (
            len(same_dir) / len(shared) if shared else None
        ),
        "conviction_mae": (
            sum(conv_abs_diffs) / len(conv_abs_diffs) if conv_abs_diffs else None
        ),
        "diff_direction": diff_dir,
        "only_a": sorted(a_syms - b_syms),
        "only_b": sorted(b_syms - a_syms),
    }


def compare_outputs(a_output: dict, b_output: dict) -> dict[str, Any]:
    """Compare two parsed digest outputs. Returns per-channel and overall
    agreement: symbol overlap (Jaccard), direction agreement on shared
    names, and mean absolute conviction difference.

    Symbol Jaccard ~1.0 + high direction agreement + low conviction MAE
    means the cheaper model is behaving like the incumbent — a safe switch.
    """
    by_channel: dict[str, Any] = {}
    tot_shared = tot_union = tot_same = 0
    all_conv_diffs: list[float] = []

    for key, label in _CHANNELS:
        ch = _compare_channel(
            a_output.get(key, []), b_output.get(key, [])
        )
        by_channel[label] = ch
        a_syms = set(_index(a_output.get(key, [])))
        b_syms = set(_index(b_output.get(key, [])))
        tot_shared += len(a_syms & b_syms)
        tot_union += len(a_syms | b_syms)
        tot_same += ch["n_same_direction"]
        if ch["conviction_mae"] is not None:
            # weight by shared count to recombine into a global mean
            all_conv_diffs.extend([ch["conviction_mae"]] * ch["n_shared"])

    overall = {
        "symbol_jaccard": (tot_shared / tot_union) if tot_union else 1.0,
        "n_shared": tot_shared,
        "direction_agreement": (tot_same / tot_shared) if tot_shared else None,
        "conviction_mae": (
            sum(all_conv_diffs) / len(all_conv_diffs) if all_conv_diffs else None
        ),
    }
    return {"by_channel": by_channel, "overall": overall}
