"""Per-ticker earnings calendar via yfinance.

yfinance exposes earnings dates via Ticker.earnings_dates (a DataFrame of
past and estimated future earnings) and Ticker.calendar (a dict with the
next scheduled date). Coverage varies by ticker; we degrade gracefully.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime
from typing import Iterable

import duckdb
import pandas as pd
import yfinance as yf

from alpha_engine.core.logging import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class EarningsEvent:
    symbol: str
    event_date: date
    estimated: bool                  # True if forecast, False if reported
    eps_estimate: float | None = None
    eps_actual: float | None = None
    surprise_pct: float | None = None


def fetch_earnings_calendar(
    symbols: Iterable[str],
    only_future: bool = False,
) -> list[EarningsEvent]:
    """Pull earnings events for each symbol from yfinance.

    Args:
        symbols: tickers to query.
        only_future: if True, returns only upcoming dates (forecasts).

    Returns one EarningsEvent per (symbol, date). Silently skips tickers
    yfinance returns no earnings data for.
    """
    out: list[EarningsEvent] = []
    today = date.today()

    for sym in symbols:
        sym = sym.upper()
        try:
            tk = yf.Ticker(sym)
            df = tk.earnings_dates  # DataFrame indexed by datetime, may be None
        except Exception as exc:
            log.warning("earnings_fetch_failed", symbol=sym, error=str(exc))
            continue

        if df is None or df.empty:
            log.debug("earnings_no_data", symbol=sym)
            continue

        for ts, row in df.iterrows():
            try:
                ev_date = ts.date() if hasattr(ts, "date") else ts
                if isinstance(ev_date, datetime):
                    ev_date = ev_date.date()
            except Exception:
                continue

            if only_future and ev_date < today:
                continue

            # Column names vary slightly across yfinance versions
            eps_est = _safe_float(
                row.get("EPS Estimate") or row.get("epsestimate")
            )
            eps_act = _safe_float(
                row.get("Reported EPS") or row.get("epsactual")
            )
            surprise = _safe_float(
                row.get("Surprise(%)") or row.get("surprisepct")
            )
            estimated = ev_date >= today or eps_act is None

            out.append(
                EarningsEvent(
                    symbol=sym,
                    event_date=ev_date,
                    estimated=estimated,
                    eps_estimate=eps_est,
                    eps_actual=eps_act,
                    surprise_pct=surprise,
                )
            )

    log.info("fetched_earnings_events", count=len(out), symbols=len(list(symbols)))
    return out


def upsert_earnings_events(
    con: duckdb.DuckDBPyConnection,
    events: Iterable[EarningsEvent],
) -> int:
    """Insert earnings events into calendar_events. Replaces existing rows
    for the same (symbol, kind='earnings', event_date)."""
    rows = list(events)
    if not rows:
        return 0

    # Delete any pre-existing earnings rows for these (symbol, date) pairs
    # so we don't accumulate duplicates on repeated runs.
    pairs = list({(e.symbol, e.event_date) for e in rows})
    con.executemany(
        "DELETE FROM calendar_events WHERE kind = 'earnings' "
        "AND symbol = ? AND event_date = ?",
        pairs,
    )

    con.executemany(
        "INSERT INTO calendar_events (event_date, kind, symbol, description, raw_json) "
        "VALUES (?, 'earnings', ?, ?, ?)",
        [
            (
                e.event_date,
                e.symbol,
                "Estimated earnings" if e.estimated else "Reported earnings",
                json.dumps(
                    {
                        "estimated": e.estimated,
                        "eps_estimate": e.eps_estimate,
                        "eps_actual": e.eps_actual,
                        "surprise_pct": e.surprise_pct,
                    }
                ),
            )
            for e in rows
        ],
    )
    log.info("upserted_earnings_events", count=len(rows))
    return len(rows)


def _safe_float(v) -> float | None:
    if v is None:
        return None
    try:
        f = float(v)
        if pd.isna(f):
            return None
        return f
    except (TypeError, ValueError):
        return None
