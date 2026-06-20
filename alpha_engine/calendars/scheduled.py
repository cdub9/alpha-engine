"""Scheduled, market-impacting events: FOMC, OpEx, jobs reports, CPI.

These dates are either officially published years in advance (FOMC) or
derivable from a fixed monthly pattern (OpEx, jobs reports). Releases that
don't follow a fixed pattern (CPI, PPI) use BLS's released calendar where
known and a heuristic otherwise.

To extend coverage past 2027, update FOMC_MEETINGS with the new dates from
https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, timedelta

import duckdb

from alpha_engine.core.logging import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# FOMC meetings (decision announcement = second day of each meeting)
# Source: federalreserve.gov/monetarypolicy/fomccalendars.htm
# ---------------------------------------------------------------------------

FOMC_MEETINGS: list[date] = [
    # 2024 (historical — useful for backtesting)
    date(2024, 1, 31),
    date(2024, 3, 20),
    date(2024, 5, 1),
    date(2024, 6, 12),
    date(2024, 7, 31),
    date(2024, 9, 18),
    date(2024, 11, 7),
    date(2024, 12, 18),
    # 2025
    date(2025, 1, 29),
    date(2025, 3, 19),
    date(2025, 5, 7),
    date(2025, 6, 18),
    date(2025, 7, 30),
    date(2025, 9, 17),
    date(2025, 10, 29),
    date(2025, 12, 10),
    # 2026
    date(2026, 1, 28),
    date(2026, 3, 18),
    date(2026, 4, 29),
    date(2026, 6, 17),
    date(2026, 7, 29),
    date(2026, 9, 16),
    date(2026, 10, 28),
    date(2026, 12, 9),
    # 2027 (placeholders; update when Fed publishes)
    date(2027, 1, 27),
    date(2027, 3, 17),
    date(2027, 4, 28),
    date(2027, 6, 16),
    date(2027, 7, 28),
    date(2027, 9, 15),
    date(2027, 10, 27),
    date(2027, 12, 8),
]


# ---------------------------------------------------------------------------
# NYSE holidays (full-day closures only; early closes are still trading days)
# Source: https://www.nyse.com/markets/hours-calendars
# When a holiday falls on Saturday, observed Friday; on Sunday, observed Monday.
# Juneteenth became a NYSE holiday in 2022.
# ---------------------------------------------------------------------------

NYSE_HOLIDAYS: frozenset[date] = frozenset(
    [
        # 2024
        date(2024, 1, 1),    # New Year's Day
        date(2024, 1, 15),   # MLK Day
        date(2024, 2, 19),   # Presidents' Day
        date(2024, 3, 29),   # Good Friday
        date(2024, 5, 27),   # Memorial Day
        date(2024, 6, 19),   # Juneteenth
        date(2024, 7, 4),    # Independence Day
        date(2024, 9, 2),    # Labor Day
        date(2024, 11, 28),  # Thanksgiving
        date(2024, 12, 25),  # Christmas
        # 2025
        date(2025, 1, 1),
        date(2025, 1, 9),    # Day of mourning - President Carter
        date(2025, 1, 20),   # MLK Day
        date(2025, 2, 17),   # Presidents' Day
        date(2025, 4, 18),   # Good Friday
        date(2025, 5, 26),   # Memorial Day
        date(2025, 6, 19),   # Juneteenth
        date(2025, 7, 4),    # Independence Day
        date(2025, 9, 1),    # Labor Day
        date(2025, 11, 27),  # Thanksgiving
        date(2025, 12, 25),  # Christmas
        # 2026
        date(2026, 1, 1),
        date(2026, 1, 19),   # MLK Day
        date(2026, 2, 16),   # Presidents' Day
        date(2026, 4, 3),    # Good Friday
        date(2026, 5, 25),   # Memorial Day
        date(2026, 6, 19),   # Juneteenth
        date(2026, 7, 3),    # Independence Day observed (Jul 4 Sat)
        date(2026, 9, 7),    # Labor Day
        date(2026, 11, 26),  # Thanksgiving
        date(2026, 12, 25),  # Christmas
        # 2027
        date(2027, 1, 1),
        date(2027, 1, 18),   # MLK Day
        date(2027, 2, 15),   # Presidents' Day
        date(2027, 3, 26),   # Good Friday
        date(2027, 5, 31),   # Memorial Day
        date(2027, 6, 18),   # Juneteenth observed (Jun 19 Sat)
        date(2027, 7, 5),    # Independence Day observed (Jul 4 Sun)
        date(2027, 9, 6),    # Labor Day
        date(2027, 11, 25),  # Thanksgiving
        date(2027, 12, 24),  # Christmas observed (Dec 25 Sat)
        # 2028
        date(2028, 1, 17),   # MLK Day (Jan 1 is Sat → no observed close)
        date(2028, 2, 21),   # Presidents' Day
        date(2028, 4, 14),   # Good Friday
        date(2028, 5, 29),   # Memorial Day
        date(2028, 6, 19),   # Juneteenth
        date(2028, 7, 4),    # Independence Day
        date(2028, 9, 4),    # Labor Day
        date(2028, 11, 23),  # Thanksgiving
        date(2028, 12, 25),  # Christmas
        # 2029
        date(2029, 1, 1),
        date(2029, 1, 15),   # MLK Day
        date(2029, 2, 19),   # Presidents' Day
        date(2029, 3, 30),   # Good Friday
        date(2029, 5, 28),   # Memorial Day
        date(2029, 6, 19),   # Juneteenth
        date(2029, 7, 4),    # Independence Day
        date(2029, 9, 3),    # Labor Day
        date(2029, 11, 22),  # Thanksgiving
        date(2029, 12, 25),  # Christmas
        # 2030
        date(2030, 1, 1),
        date(2030, 1, 21),   # MLK Day
        date(2030, 2, 18),   # Presidents' Day
        date(2030, 4, 19),   # Good Friday
        date(2030, 5, 27),   # Memorial Day
        date(2030, 6, 19),   # Juneteenth
        date(2030, 7, 4),    # Independence Day
        date(2030, 9, 2),    # Labor Day
        date(2030, 11, 28),  # Thanksgiving
        date(2030, 12, 25),  # Christmas
    ]
)


def is_nyse_holiday(d: date) -> bool:
    """True if `d` is a full-day NYSE closure. Early-close days are NOT holidays."""
    return d in NYSE_HOLIDAYS


def is_trading_day(d: date) -> bool:
    """True if `d` is a weekday and not a NYSE holiday."""
    if d.weekday() >= 5:  # Saturday=5, Sunday=6
        return False
    return not is_nyse_holiday(d)


# ---------------------------------------------------------------------------
# Pattern-derived calendars
# ---------------------------------------------------------------------------


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """The nth occurrence of `weekday` (0=Monday) in (year, month)."""
    first = date(year, month, 1)
    offset = (weekday - first.weekday()) % 7
    day = 1 + offset + (n - 1) * 7
    return date(year, month, day)


def opex_dates(start: date, end: date) -> list[date]:
    """Monthly options expiration: third Friday of each month."""
    out: list[date] = []
    y, m = start.year, start.month
    while True:
        d = _nth_weekday(y, m, 4, 3)  # 4 = Friday
        if d > end:
            break
        if d >= start:
            out.append(d)
        m += 1
        if m > 12:
            m = 1
            y += 1
    return out


def quad_witching_dates(start: date, end: date) -> list[date]:
    """Quadruple witching: third Friday of March, June, September, December.
    Coincides with quarterly futures + options expiration; elevated volatility."""
    return [d for d in opex_dates(start, end) if d.month in (3, 6, 9, 12)]


def jobs_report_dates(start: date, end: date) -> list[date]:
    """Non-farm payrolls: first Friday of each month."""
    out: list[date] = []
    y, m = start.year, start.month
    while True:
        d = _nth_weekday(y, m, 4, 1)
        if d > end:
            break
        if d >= start:
            out.append(d)
        m += 1
        if m > 12:
            m = 1
            y += 1
    return out


def cpi_release_dates(start: date, end: date) -> list[date]:
    """CPI release: heuristic = second Tuesday of each month.

    Real BLS calendar varies (sometimes Wed/Thu of week 2). Refine by pulling
    BLS scheduled-release calendar in a later iteration."""
    out: list[date] = []
    y, m = start.year, start.month
    while True:
        d = _nth_weekday(y, m, 1, 2)  # 1 = Tuesday
        if d > end:
            break
        if d >= start:
            out.append(d)
        m += 1
        if m > 12:
            m = 1
            y += 1
    return out


def fomc_meeting_dates(start: date | None = None, end: date | None = None) -> list[date]:
    """Filter the hardcoded FOMC list to a date range."""
    s = start or date.min
    e = end or date.max
    return [d for d in FOMC_MEETINGS if s <= d <= e]


# ---------------------------------------------------------------------------
# Seeder
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SeedResult:
    fomc: int
    opex: int
    quad_witching: int
    jobs: int
    cpi: int

    @property
    def total(self) -> int:
        return self.fomc + self.opex + self.quad_witching + self.jobs + self.cpi


def seed_market_calendar(
    con: duckdb.DuckDBPyConnection,
    start: date,
    end: date,
) -> SeedResult:
    """Insert market-wide calendar events into calendar_events. Idempotent
    (uses delete-then-insert per kind in the date range)."""

    def upsert(kind: str, dates: list[date], description: str) -> int:
        if not dates:
            return 0
        # Delete any existing events of this kind in the range, then re-insert.
        # This keeps the table tidy if scheduled dates ever get corrected.
        con.execute(
            "DELETE FROM calendar_events WHERE kind = ? "
            "AND event_date BETWEEN ? AND ? AND symbol IS NULL",
            [kind, start, end],
        )
        con.executemany(
            "INSERT INTO calendar_events (event_date, kind, symbol, description, raw_json) "
            "VALUES (?, ?, NULL, ?, ?)",
            [(d, kind, description, json.dumps({})) for d in dates],
        )
        return len(dates)

    fomc = upsert("fomc", fomc_meeting_dates(start, end), "FOMC decision announcement")
    opex = upsert("opex", opex_dates(start, end), "Monthly options expiration")
    qw_dates = quad_witching_dates(start, end)
    # Quad witching dates are a subset of opex; tag them additionally
    qw = upsert("opex_quad_witching", qw_dates, "Quadruple witching (quarterly OpEx)")
    jobs = upsert("jobs_report", jobs_report_dates(start, end), "Non-farm payrolls release")
    cpi = upsert("cpi", cpi_release_dates(start, end), "CPI release (heuristic)")

    result = SeedResult(fomc=fomc, opex=opex, quad_witching=qw, jobs=jobs, cpi=cpi)
    log.info(
        "seeded_market_calendar",
        start=str(start),
        end=str(end),
        fomc=result.fomc,
        opex=result.opex,
        quad_witching=result.quad_witching,
        jobs=result.jobs,
        cpi=result.cpi,
    )
    return result
