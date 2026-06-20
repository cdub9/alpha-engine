"""Calendar feature computer.

Given a date (and optionally a ticker), compute features the signal engine
can use: days_to_fomc, is_opex_week, days_to_next_earnings, seasonality, etc.

All distance fields are CALENDAR days. A trading-day refinement (skipping
weekends/holidays) can be added later — for medium-horizon signals,
calendar days are accurate enough.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, timedelta
from typing import Optional

import duckdb


@dataclass(frozen=True)
class CalendarFeatures:
    """Calendar context for a given date (and optionally a ticker)."""

    as_of: date

    # Market-wide event proximity (signed: + = future, - = past)
    days_to_next_fomc: Optional[int] = None
    days_since_last_fomc: Optional[int] = None
    days_to_next_cpi: Optional[int] = None
    days_to_next_jobs_report: Optional[int] = None
    days_to_next_opex: Optional[int] = None
    days_to_next_quad_witching: Optional[int] = None

    # Boolean flags
    is_fomc_week: bool = False
    is_opex_week: bool = False
    is_quad_witching_week: bool = False
    is_jobs_report_week: bool = False
    is_cpi_week: bool = False

    # Seasonality / time-of-year
    month: int = 0
    day_of_month: int = 0
    day_of_week: int = 0          # 0 = Monday, 6 = Sunday
    is_quarter_end_week: bool = False
    is_month_end_week: bool = False
    is_santa_claus_window: bool = False    # last 5 trading days Dec + first 2 Jan
    is_january_effect_window: bool = False  # first 2 weeks of January
    is_september: bool = False             # historically worst month
    is_summer_doldrums: bool = False       # July-August

    # Per-ticker (filled only when symbol is provided)
    symbol: Optional[str] = None
    days_to_next_earnings: Optional[int] = None
    days_since_last_earnings: Optional[int] = None
    is_earnings_week: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _week_bounds(d: date) -> tuple[date, date]:
    """Monday and Friday of the week containing `d`."""
    monday = d - timedelta(days=d.weekday())
    friday = monday + timedelta(days=4)
    return monday, friday


def _is_quarter_end_week(d: date) -> bool:
    """True if the week containing d includes the last business day of a
    quarter (Mar/Jun/Sep/Dec)."""
    if d.month not in (3, 6, 9, 12):
        return False
    # Last day of the month
    next_month = (d.replace(day=28) + timedelta(days=4)).replace(day=1)
    last_day = next_month - timedelta(days=1)
    monday, friday = _week_bounds(d)
    return monday <= last_day <= friday


def _is_month_end_week(d: date) -> bool:
    next_month = (d.replace(day=28) + timedelta(days=4)).replace(day=1)
    last_day = next_month - timedelta(days=1)
    monday, friday = _week_bounds(d)
    return monday <= last_day <= friday


def _signed_distance(events: list[date], target: date, future: bool = True) -> Optional[int]:
    """Days from target to nearest event. If future=True, search forward;
    else backward. Returns None if no event found."""
    if future:
        candidates = [d for d in events if d >= target]
        if not candidates:
            return None
        return (min(candidates) - target).days
    else:
        candidates = [d for d in events if d <= target]
        if not candidates:
            return None
        return (target - max(candidates)).days


def _in_same_week(target: date, events: list[date]) -> bool:
    monday, friday = _week_bounds(target)
    return any(monday <= e <= friday for e in events)


# ---------------------------------------------------------------------------
# Main entry points
# ---------------------------------------------------------------------------


def compute_market_calendar_features(
    con: duckdb.DuckDBPyConnection,
    as_of: date,
) -> CalendarFeatures:
    """Compute market-wide calendar features (no ticker context)."""
    # Pull a window of events around as_of for efficient queries
    window_start = as_of - timedelta(days=180)
    window_end = as_of + timedelta(days=180)

    rows = con.execute(
        "SELECT event_date, kind FROM calendar_events "
        "WHERE event_date BETWEEN ? AND ? AND symbol IS NULL",
        [window_start, window_end],
    ).fetchall()

    by_kind: dict[str, list[date]] = {}
    for ev_date, kind in rows:
        by_kind.setdefault(kind, []).append(ev_date)

    fomc = sorted(by_kind.get("fomc", []))
    opex = sorted(by_kind.get("opex", []))
    quad = sorted(by_kind.get("opex_quad_witching", []))
    jobs = sorted(by_kind.get("jobs_report", []))
    cpi = sorted(by_kind.get("cpi", []))

    return CalendarFeatures(
        as_of=as_of,
        days_to_next_fomc=_signed_distance(fomc, as_of, future=True),
        days_since_last_fomc=_signed_distance(fomc, as_of, future=False),
        days_to_next_cpi=_signed_distance(cpi, as_of, future=True),
        days_to_next_jobs_report=_signed_distance(jobs, as_of, future=True),
        days_to_next_opex=_signed_distance(opex, as_of, future=True),
        days_to_next_quad_witching=_signed_distance(quad, as_of, future=True),
        is_fomc_week=_in_same_week(as_of, fomc),
        is_opex_week=_in_same_week(as_of, opex),
        is_quad_witching_week=_in_same_week(as_of, quad),
        is_jobs_report_week=_in_same_week(as_of, jobs),
        is_cpi_week=_in_same_week(as_of, cpi),
        month=as_of.month,
        day_of_month=as_of.day,
        day_of_week=as_of.weekday(),
        is_quarter_end_week=_is_quarter_end_week(as_of),
        is_month_end_week=_is_month_end_week(as_of),
        is_santa_claus_window=_is_santa_claus(as_of),
        is_january_effect_window=(as_of.month == 1 and as_of.day <= 14),
        is_september=(as_of.month == 9),
        is_summer_doldrums=(as_of.month in (7, 8)),
    )


def compute_ticker_calendar_features(
    con: duckdb.DuckDBPyConnection,
    symbol: str,
    as_of: date,
) -> CalendarFeatures:
    """Compute calendar features including ticker-specific earnings context."""
    base = compute_market_calendar_features(con, as_of)

    window_start = as_of - timedelta(days=400)
    window_end = as_of + timedelta(days=400)
    rows = con.execute(
        "SELECT event_date FROM calendar_events "
        "WHERE kind = 'earnings' AND symbol = ? "
        "AND event_date BETWEEN ? AND ?",
        [symbol.upper(), window_start, window_end],
    ).fetchall()
    earnings_dates = sorted({r[0] for r in rows})

    days_to_next = _signed_distance(earnings_dates, as_of, future=True)
    days_since_last = _signed_distance(earnings_dates, as_of, future=False)
    is_earnings_week = _in_same_week(as_of, earnings_dates)

    return CalendarFeatures(
        **{
            **base.to_dict(),
            "symbol": symbol.upper(),
            "days_to_next_earnings": days_to_next,
            "days_since_last_earnings": days_since_last,
            "is_earnings_week": is_earnings_week,
        }
    )


def _is_santa_claus(d: date) -> bool:
    """Santa Claus rally window: last 5 trading days of December + first 2
    of January. Approximated with calendar days (Dec 24-31 + Jan 1-3)."""
    if d.month == 12 and d.day >= 24:
        return True
    if d.month == 1 and d.day <= 3:
        return True
    return False
