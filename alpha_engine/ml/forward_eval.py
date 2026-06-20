"""Forward (out-of-sample) skill measurement for the daily ML signals.

The walk-forward validator (`scripts/validate_ml.py`) tells us whether the
*strategy* would have worked historically. This module answers the
narrower, fully honest question the FOLLOWUPS file calls out: do the
BUY-bucket names we actually published each day go on to beat the
AVOID-bucket names over the next `horizon` trading days?

Why this is clean:
  - ml_signals rows are stamped with the date they were generated, and the
    only thing we read forward is *price* (market_bars). A signal never
    sees a bar past its signal_date, so the forward return cannot leak
    into the ranking. (The models are price-only anyway — no LLM, no
    training-data contamination.)
  - A signal date only "matures" once `horizon` trading days of bars exist
    after it. Immature dates are excluded, so the spread is never computed
    on a partial window. This is what makes it a forward track record
    rather than an in-sample fit.

The headline metric is the BUY−AVOID spread: a self-financing long/short
read that needs no benchmark. We also report the BUY bucket vs the
equal-weight cross-section ("did our picks beat the average name?").

Pure functions over a DuckDB connection so the dashboard wrapper stays
thin and the logic is unit-testable on synthetic bars.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import duckdb

from alpha_engine.calendars.scheduled import is_trading_day
from alpha_engine.core.logging import get_logger

log = get_logger(__name__)

DEFAULT_HORIZON = 21  # trading days — matches WalkForwardXGB's label horizon


def _trading_days_forward(start: date, n: int) -> date:
    """Return the date `n` trading days after `start` (rule-based NYSE
    calendar, so it works for future dates the DB has no bars for yet)."""
    d = start
    counted = 0
    # Cap the walk so a bad input can never spin forever (~2 calendar
    # years is far more than any sane horizon).
    for _ in range(n * 4 + 10):
        d = d + timedelta(days=1)
        if is_trading_day(d):
            counted += 1
            if counted >= n:
                return d
    return d


def compute_forward_performance(
    con: duckdb.DuckDBPyConnection,
    horizon: int = DEFAULT_HORIZON,
) -> dict[str, Any]:
    """Forward BUY/AVOID performance of every ml_signals cohort.

    Returns:
      {
        "horizon": int,
        "by_model": {
            model_version: {
                "n_dates_total": int,      # signal dates recorded
                "n_dates_matured": int,    # dates with `horizon` days of bars after
                "n_pending": int,
                "next_maturity_date": date | None,   # when the earliest pending date matures
                "mean_spread": float | None,         # avg (buy_ret − avoid_ret) over matured dates
                "spread_hit_rate": float | None,     # share of matured dates with spread > 0
                "mean_buy_ret": float | None,
                "mean_avoid_ret": float | None,
                "mean_all_ret": float | None,        # equal-weight cross-section
                "buy_beats_all_rate": float | None,  # share of dates buy_ret > all_ret
                "per_date": [                         # matured dates only, oldest first
                    {"signal_date", "n_buy", "n_avoid",
                     "buy_ret", "avoid_ret", "all_ret", "spread"}, ...
                ],
            }, ...
        },
      }
    """
    rows = con.execute(
        """
        WITH bars AS (
            SELECT symbol, bar_date, adj_close,
                   ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY bar_date) AS rn
            FROM market_bars
        ),
        entry AS (
            SELECT m.model_version, m.signal_date, m.symbol, m.action,
                   b.rn AS entry_rn, b.adj_close AS entry_px
            FROM ml_signals m
            JOIN bars b ON b.symbol = m.symbol AND b.bar_date = m.signal_date
        )
        SELECT e.model_version, e.signal_date, e.action,
               e.entry_px, x.adj_close AS exit_px
        FROM entry e
        LEFT JOIN bars x
          ON x.symbol = e.symbol AND x.rn = e.entry_rn + ?
        ORDER BY e.model_version, e.signal_date
        """,
        [horizon],
    ).fetchall()

    # model_version -> signal_date -> action -> list[forward_ret]
    # plus a record of every date seen (matured or not).
    matured: dict[str, dict[date, dict[str, list[float]]]] = {}
    all_dates: dict[str, set[date]] = {}

    for model_version, signal_date, action, entry_px, exit_px in rows:
        all_dates.setdefault(model_version, set()).add(signal_date)
        if exit_px is None or entry_px is None or entry_px <= 0:
            continue  # not matured (or unusable entry) — skip the return
        fwd = float(exit_px) / float(entry_px) - 1.0
        by_date = matured.setdefault(model_version, {})
        by_action = by_date.setdefault(signal_date, {})
        by_action.setdefault(action, []).append(fwd)

    by_model: dict[str, Any] = {}
    for model_version in sorted(all_dates):
        dates_total = sorted(all_dates[model_version])
        per_date: list[dict[str, Any]] = []
        matured_dates = matured.get(model_version, {})

        for sd in sorted(matured_dates):
            buckets = matured_dates[sd]
            buys = buckets.get("BUY", [])
            avoids = buckets.get("AVOID", [])
            everything = [r for lst in buckets.values() for r in lst]
            # A date only counts if both ends of the long/short exist —
            # otherwise the spread is undefined.
            if not buys or not avoids:
                continue
            buy_ret = sum(buys) / len(buys)
            avoid_ret = sum(avoids) / len(avoids)
            all_ret = sum(everything) / len(everything) if everything else None
            per_date.append({
                "signal_date": sd,
                "n_buy": len(buys),
                "n_avoid": len(avoids),
                "buy_ret": buy_ret,
                "avoid_ret": avoid_ret,
                "all_ret": all_ret,
                "spread": buy_ret - avoid_ret,
            })

        n_matured = len(per_date)
        n_total = len(dates_total)
        pending = [d for d in dates_total
                   if d not in {r["signal_date"] for r in per_date}]
        next_maturity = (
            _trading_days_forward(min(pending), horizon) if pending else None
        )

        summary: dict[str, Any] = {
            "n_dates_total": n_total,
            "n_dates_matured": n_matured,
            "n_pending": len(pending),
            "next_maturity_date": next_maturity,
            "per_date": per_date,
            "mean_spread": None,
            "spread_hit_rate": None,
            "mean_buy_ret": None,
            "mean_avoid_ret": None,
            "mean_all_ret": None,
            "buy_beats_all_rate": None,
        }
        if n_matured:
            spreads = [r["spread"] for r in per_date]
            summary["mean_spread"] = sum(spreads) / n_matured
            summary["spread_hit_rate"] = sum(1 for s in spreads if s > 0) / n_matured
            summary["mean_buy_ret"] = sum(r["buy_ret"] for r in per_date) / n_matured
            summary["mean_avoid_ret"] = sum(r["avoid_ret"] for r in per_date) / n_matured
            with_all = [r for r in per_date if r["all_ret"] is not None]
            if with_all:
                summary["mean_all_ret"] = sum(r["all_ret"] for r in with_all) / len(with_all)
                summary["buy_beats_all_rate"] = (
                    sum(1 for r in with_all if r["buy_ret"] > r["all_ret"]) / len(with_all)
                )
        by_model[model_version] = summary

    log.info(
        "ml_forward_performance_computed",
        horizon=horizon,
        models={k: v["n_dates_matured"] for k, v in by_model.items()},
    )
    return {"horizon": horizon, "by_model": by_model}
