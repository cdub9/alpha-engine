"""Record and score the Action Center's opportunity ideas — the learning loop.

The opportunity layer (holistic.opportunity_ideas) suggests adds/trims from
the app's return-side signals, but those signals have unproven forward skill.
This module is how the app learns whether they work:

  1. record_recommendations() logs each day's add/trim ideas.
  2. score_recommendations() waits `horizon` trading days, then measures each
     idea's forward return vs a benchmark — an ADD "worked" if the name beat
     the benchmark; a TRIM "worked" if it lagged (you were right to lighten).

Point-in-time and forward-by-construction: a recommendation is only scored
once `horizon` trading days of bars exist after it, so the measurement never
peeks at data the idea didn't have. Nothing is learned until real forward
data accumulates — this builds the substrate; the payoff is weeks out.

Only the return-seeking IDEAS are tracked. Deterministic risk trims (cap
breaches, earnings) aren't predictions, so scoring them would be meaningless.
"""

from __future__ import annotations

import json
from datetime import date
from typing import Any, Optional

import duckdb

from alpha_engine.core.logging import get_logger

log = get_logger(__name__)

DEFAULT_HORIZON = 21
DEFAULT_BENCHMARK = "SPY"


def record_recommendations(
    con: duckdb.DuckDBPyConnection,
    as_of: date,
    opportunity: dict[str, list[dict[str, Any]]],
) -> int:
    """Persist the day's opportunity ideas. Idempotent per day (same-day
    re-runs replace). Returns the number of ideas logged."""
    rows: list[list[Any]] = []
    for kind in ("add", "trim"):
        key = "adds" if kind == "add" else "trims"
        for idea in opportunity.get(key, []):
            rows.append([
                as_of, idea["symbol"].upper(), kind,
                float(idea.get("score") or 0.0),
                float(idea["weight"]) if idea.get("weight") is not None else None,
                json.dumps(idea.get("signals", [])),
            ])

    con.execute("DELETE FROM book_recommendations WHERE as_of = ?", [as_of])
    if rows:
        con.executemany(
            "INSERT INTO book_recommendations (as_of, symbol, kind, score, "
            "weight, signals_json) VALUES (?, ?, ?, ?, ?, ?)",
            rows,
        )
    log.info("recommendations_recorded", as_of=str(as_of), n=len(rows))
    return len(rows)


def score_recommendations(
    con: duckdb.DuckDBPyConnection,
    horizon: int = DEFAULT_HORIZON,
    benchmark: str = DEFAULT_BENCHMARK,
) -> dict[str, Any]:
    """Forward-score every matured recommendation vs `benchmark`.

    Returns:
      {
        "horizon": int, "benchmark": str,
        "n_total": int, "n_matured": int, "n_pending": int,
        "by_kind": {
            "add":  {"n", "hit_rate", "avg_alpha"},
            "trim": {"n", "hit_rate", "avg_alpha"},
        },
        "overall_hit_rate": float | None,
        "matured": [{as_of, symbol, kind, fwd_return, benchmark_return,
                     alpha, correct}],
      }
    An ADD is "correct" when the name beat the benchmark over the window; a
    TRIM is "correct" when it lagged. hit_rate/avg_alpha are None until at
    least one idea of that kind has matured.
    """
    # Entry is the first bar on OR AFTER the reco date (a reco made after the
    # close, or on a non-trading day, prices against the next available bar).
    rows = con.execute(
        """
        WITH bars AS (
            SELECT symbol, bar_date, adj_close,
                   ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY bar_date) AS rn
            FROM market_bars
        ),
        entry AS (
            SELECT r.as_of, r.symbol, r.kind, MIN(b.rn) AS entry_rn
            FROM book_recommendations r
            JOIN bars b ON b.symbol = r.symbol AND b.bar_date >= r.as_of
            GROUP BY r.as_of, r.symbol, r.kind
        )
        SELECT e.as_of, e.symbol, e.kind, en.adj_close AS entry_px,
               x.adj_close AS exit_px, en.bar_date AS entry_date
        FROM entry e
        JOIN bars en ON en.symbol = e.symbol AND en.rn = e.entry_rn
        LEFT JOIN bars x ON x.symbol = e.symbol AND x.rn = e.entry_rn + ?
        """,
        [horizon],
    ).fetchall()

    # Benchmark forward return keyed by the benchmark's ENTRY bar date, so we
    # can align each reco to the benchmark window starting the same session.
    bench = con.execute(
        """
        WITH b AS (
            SELECT bar_date, adj_close,
                   ROW_NUMBER() OVER (ORDER BY bar_date) AS rn
            FROM market_bars WHERE symbol = ?
        )
        SELECT e.bar_date, e.adj_close AS entry_px, x.adj_close AS exit_px
        FROM b e LEFT JOIN b x ON x.rn = e.rn + ?
        """,
        [benchmark, horizon],
    ).fetchall()
    bench_fwd = {
        d: (float(xp) / float(ep) - 1.0)
        for d, ep, xp in bench if ep and xp
    }

    matured: list[dict[str, Any]] = []
    n_total = 0
    for as_of, sym, kind, entry_px, exit_px, entry_date in rows:
        n_total += 1
        if exit_px is None or entry_px is None or entry_px <= 0:
            continue
        bench_ret = bench_fwd.get(entry_date)
        if bench_ret is None:
            continue  # benchmark window not matured either
        fwd = float(exit_px) / float(entry_px) - 1.0
        alpha = fwd - bench_ret
        correct = alpha > 0 if kind == "add" else alpha < 0
        matured.append({
            "as_of": as_of, "symbol": sym, "kind": kind,
            "fwd_return": fwd, "benchmark_return": bench_ret,
            "alpha": alpha, "correct": correct,
        })

    def _agg(kind: str) -> dict[str, Any]:
        sub = [m for m in matured if m["kind"] == kind]
        if not sub:
            return {"n": 0, "hit_rate": None, "avg_alpha": None}
        return {
            "n": len(sub),
            "hit_rate": sum(1 for m in sub if m["correct"]) / len(sub),
            "avg_alpha": sum(m["alpha"] for m in sub) / len(sub),
        }

    overall = (
        sum(1 for m in matured if m["correct"]) / len(matured) if matured else None
    )
    return {
        "horizon": horizon,
        "benchmark": benchmark,
        "n_total": n_total,
        "n_matured": len(matured),
        "n_pending": n_total - len(matured),
        "by_kind": {"add": _agg("add"), "trim": _agg("trim")},
        "overall_hit_rate": overall,
        "matured": matured,
    }
