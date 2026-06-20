"""Did the v3 self-learning feedback loop actually change the model's behavior?

The v3-fb prompt feeds the LLM its own open book and track record every day
(see `alpha_engine/llm/feedback.py`). That only earns its keep if the
*outputs* respond. This module compares signal cohorts by `model_version`
on the four tells the FOLLOWUPS file lists:

  (a) conviction calibration slope — do 8.0+ picks finally beat <7.0 picks?
      (matured trades only; the inverted backfill cohort is the thing v3 is
      meant to fix.)
  (b) duplicate-of-holding share — what fraction of NEW-buy picks (`buy`,
      not `add`) repeat a name the model ALREADY held when it generated them?
      `add` is excluded on purpose: it definitionally targets an existing
      position, so counting it would inflate the metric. v3 sees its open
      book, so wasteful re-buys should fall.
  (c) action mix — counts per direction. (Under the long-only prompt the
      model can't emit exit/reduce, so this is mostly a sanity read on the
      buy/add/hold split, not a discriminator.)
  (d) repeated-miss alpha — names the cohort traded ≥2× that lost to SPY.
      v3 is told to demand a stronger thesis on these.

(a) and (d) need matured outcomes, so a fresh v3 cohort shows them as
pending until ~horizon days of forward trades complete. (b) and (c) are
computable the moment signals exist, which is what makes this useful early.

Point-in-time discipline mirrors feedback.py: a trade counts as matured
only when entry + days_held <= as_of, never when the scorer happened to
run (evaluated_at), so backfilled history doesn't leak future knowledge.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Optional

import duckdb

from alpha_engine.core.logging import get_logger

log = get_logger(__name__)

MISS_ALPHA_THRESHOLD = 0.02   # avg alpha <= -2% over >=2 trades = a repeated miss
MIN_BUCKET_N = 10             # below this, a calibration bucket is noise

# Friendly labels for known cohorts; unknown versions fall back to the raw id.
COHORT_LABELS = {
    "llm-opus-4-7-v1": "v1 (pre-feedback)",
    "llm-opus-4-7-v2-ta": "v2 (technicals)",
    "llm-opus-4-7-v3-fb": "v3 (feedback loop)",
}

_ACTIONABLE = ("buy", "add")


def cohort_label(model_version: str) -> str:
    return COHORT_LABELS.get(model_version, model_version)


def compute_feedback_loop_behavior(
    con: duckdb.DuckDBPyConnection,
    as_of: Optional[date] = None,
) -> dict[str, Any]:
    """Per-cohort behavior metrics, ordered oldest cohort first.

    `as_of` is the maturity cutoff for the outcome-dependent metrics;
    defaults to today (CURRENT_DATE). Returns:
      {
        "as_of": date,
        "order": [model_version, ...],          # sorted; v3-fb tends to sort last
        "cohorts": {
            model_version: {
                "label": str,
                "n_signals": int,
                "n_actionable": int,             # buy + add
                "action_mix": {direction: count},
                "n_new_buys": int,               # `buy` only (the dup denominator)
                "dup_share": float | None,       # (b) computable immediately
                "n_dup_buys": int,
                "n_matured": int,
                "calibration": {bucket: {"n", "avg_alpha", "win_rate"}},
                "calib_slope": float | None,     # (a) alpha(8.0+) − alpha(<7.0)
                "calib_slope_reliable": bool,    # both buckets >= MIN_BUCKET_N
                "repeated_misses": [{"symbol", "n", "avg_alpha"}],  # (d)
            }, ...
        },
      }
    """
    cutoff = as_of  # None -> use CURRENT_DATE in SQL
    matured_clause = (
        "(t.placed_at::DATE + o.days_held * INTERVAL 1 DAY) <= "
        + ("?" if cutoff else "CURRENT_DATE")
    )

    def _p(extra: Optional[list] = None) -> list:
        base = [cutoff] if cutoff else []
        return base + (extra or [])

    cohorts: dict[str, dict[str, Any]] = {}

    def _slot(mv: str) -> dict[str, Any]:
        return cohorts.setdefault(mv, {
            "label": cohort_label(mv),
            "n_signals": 0,
            "n_actionable": 0,
            "action_mix": {},
            "n_new_buys": 0,
            "dup_share": None,
            "n_dup_buys": 0,
            "n_matured": 0,
            "calibration": {},
            "calib_slope": None,
            "calib_slope_reliable": False,
            "repeated_misses": [],
        })

    # (c) action mix + total signal counts ----------------------------------
    for mv, direction, n in con.execute(
        "SELECT model_version, LOWER(direction), COUNT(*) "
        "FROM signals GROUP BY 1, 2"
    ).fetchall():
        c = _slot(mv)
        c["action_mix"][direction] = int(n)
        c["n_signals"] += int(n)
        if direction in _ACTIONABLE:
            c["n_actionable"] += int(n)

    # (b) duplicate-of-holding share ----------------------------------------
    # For each NEW-buy signal, did an earlier-entered paper position in the
    # same channel+symbol exist (and remain unmatured) on the day it was
    # generated? `placed_at::DATE < generated_at` excludes the signal's own
    # resulting trade, so only PRIOR holdings count. `add` is excluded — it
    # is meant to reference an existing position.
    for mv, n_buys, n_dup in con.execute(
        """
        SELECT s.model_version,
               COUNT(*) AS n_new_buys,
               SUM(CASE WHEN EXISTS (
                   SELECT 1 FROM trades t2
                   LEFT JOIN trade_outcomes o2 ON o2.trade_id = t2.id
                   WHERE t2.channel = s.channel
                     AND t2.symbol = s.symbol
                     AND t2.status = 'paper_filled'
                     AND t2.placed_at::DATE < s.generated_at::DATE
                     AND (o2.trade_id IS NULL
                          OR (t2.placed_at::DATE + o2.days_held * INTERVAL 1 DAY)
                             > s.generated_at::DATE)
               ) THEN 1 ELSE 0 END) AS n_dup
        FROM signals s
        WHERE LOWER(s.direction) = 'buy'
        GROUP BY 1
        """
    ).fetchall():
        c = _slot(mv)
        c["n_new_buys"] = int(n_buys or 0)
        c["n_dup_buys"] = int(n_dup or 0)
        if n_buys:
            c["dup_share"] = float(n_dup or 0) / float(n_buys)

    # (a) conviction calibration (matured only) -----------------------------
    for mv, bucket, n, avg_alpha, win in con.execute(
        f"""
        SELECT s.model_version,
               CASE WHEN s.conviction >= 8 THEN '8.0+'
                    WHEN s.conviction >= 7 THEN '7.0-7.9'
                    ELSE '<7.0' END AS bucket,
               COUNT(*) AS n,
               AVG(o.alpha) AS avg_alpha,
               AVG(CASE WHEN o.return_pct > 0 THEN 1.0 ELSE 0.0 END) AS win_rate
        FROM trades t
        JOIN trade_outcomes o ON o.trade_id = t.id
        JOIN signals s ON s.id = t.source_signal_id
        WHERE t.status = 'paper_filled'
          AND s.conviction IS NOT NULL
          AND {matured_clause}
        GROUP BY 1, 2
        """,
        _p(),
    ).fetchall():
        c = _slot(mv)
        c["calibration"][bucket] = {
            "n": int(n),
            "avg_alpha": float(avg_alpha) if avg_alpha is not None else None,
            "win_rate": float(win) if win is not None else None,
        }

    # (matured count per cohort)
    for mv, n in con.execute(
        f"""
        SELECT s.model_version, COUNT(*)
        FROM trades t
        JOIN trade_outcomes o ON o.trade_id = t.id
        JOIN signals s ON s.id = t.source_signal_id
        WHERE t.status = 'paper_filled' AND {matured_clause}
        GROUP BY 1
        """,
        _p(),
    ).fetchall():
        _slot(mv)["n_matured"] = int(n)

    # (d) repeated misses (matured only) ------------------------------------
    for mv, symbol, n, avg_alpha in con.execute(
        f"""
        SELECT s.model_version, t.symbol, COUNT(*) AS n, AVG(o.alpha) AS avg_alpha
        FROM trades t
        JOIN trade_outcomes o ON o.trade_id = t.id
        JOIN signals s ON s.id = t.source_signal_id
        WHERE t.status = 'paper_filled' AND {matured_clause}
        GROUP BY 1, 2
        HAVING COUNT(*) >= 2
        """,
        _p(),
    ).fetchall():
        if avg_alpha is not None and float(avg_alpha) <= -MISS_ALPHA_THRESHOLD:
            _slot(mv)["repeated_misses"].append({
                "symbol": symbol, "n": int(n), "avg_alpha": float(avg_alpha),
            })

    # Derive calibration slope + sort each cohort's misses
    for c in cohorts.values():
        calib = c["calibration"]
        hi = calib.get("8.0+")
        lo = calib.get("<7.0")
        if hi and lo and hi["avg_alpha"] is not None and lo["avg_alpha"] is not None:
            c["calib_slope"] = hi["avg_alpha"] - lo["avg_alpha"]
            c["calib_slope_reliable"] = (
                hi["n"] >= MIN_BUCKET_N and lo["n"] >= MIN_BUCKET_N
            )
        c["repeated_misses"].sort(key=lambda r: r["avg_alpha"])

    return {
        "as_of": cutoff,
        "order": sorted(cohorts),
        "cohorts": cohorts,
    }
