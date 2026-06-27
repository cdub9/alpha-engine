"""Earnings-blackout guard.

A held name reporting earnings in the next few days carries the single
biggest *avoidable* single-name risk — AVGO −12.6% on June 4 was a
calendared print the system already knew about. This surfaces held names
with imminent earnings so they can be trimmed before the gap, and gives
the paper trader a hook to block opening fresh full-size positions into a
print.

Pure DB reads over the calendar_events table; no network.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any, Optional

import duckdb

DEFAULT_HORIZON_DAYS = 7


def upcoming_earnings(
    con: duckdb.DuckDBPyConnection,
    symbols: list[str],
    as_of: date,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
    values: Optional[dict[str, float]] = None,
) -> list[dict[str, Any]]:
    """Held symbols with an earnings date in (as_of, as_of + horizon_days].

    `values` optionally maps symbol -> dollar exposure so callers can rank
    by how much is at risk. Returns one row per symbol (the soonest date),
    sorted by date then exposure:
      [{symbol, date (ISO str), days_away, value}]
    """
    if not symbols:
        return []
    syms = [s.upper().strip() for s in symbols]
    ph = ",".join(["?"] * len(syms))
    hi = as_of + timedelta(days=horizon_days)
    rows = con.execute(
        f"""
        SELECT symbol, MIN(event_date) AS d
        FROM calendar_events
        WHERE kind = 'earnings'
          AND symbol IN ({ph})
          AND event_date > ? AND event_date <= ?
        GROUP BY symbol
        """,
        [*syms, as_of, hi],
    ).fetchall()

    values = values or {}
    out = [
        {
            "symbol": sym,
            "date": d.isoformat(),
            "days_away": (d - as_of).days,
            "value": float(values.get(sym, 0.0)),
        }
        for sym, d in rows
    ]
    out.sort(key=lambda r: (r["date"], -r["value"]))
    return out


def has_imminent_earnings(
    con: duckdb.DuckDBPyConnection,
    symbol: str,
    as_of: date,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
) -> bool:
    """True if `symbol` reports within the blackout window — the hook the
    paper trader uses to refuse opening fresh full size into a print."""
    return bool(upcoming_earnings(con, [symbol], as_of, horizon_days))
