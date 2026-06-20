"""Quick inspection: upcoming earnings + ticker calendar features for sanity."""

from __future__ import annotations

import sys
from datetime import date

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

from alpha_engine.calendars import compute_ticker_calendar_features
from alpha_engine.db import get_connection

with get_connection(read_only=True) as con:
    upcoming = con.execute(
        "SELECT symbol, event_date FROM calendar_events "
        "WHERE kind='earnings' AND symbol IN ('NVDA','MSFT','AAPL','TSLA') "
        "AND event_date >= CURRENT_DATE ORDER BY event_date LIMIT 10"
    ).fetchall()
    print("Upcoming earnings (next 10):")
    for sym, d in upcoming:
        print(f"  {sym:6s} {d}")

    print()
    for sym in ("NVDA", "MSFT"):
        f = compute_ticker_calendar_features(con, sym, date.today())
        print(f"{sym} calendar features as of {f.as_of}:")
        print(f"  days_to_next_earnings:    {f.days_to_next_earnings}")
        print(f"  days_since_last_earnings: {f.days_since_last_earnings}")
        print(f"  is_earnings_week:         {f.is_earnings_week}")
        print(f"  days_to_next_fomc:        {f.days_to_next_fomc}")
        print(f"  days_to_next_opex:        {f.days_to_next_opex}")
        print()
