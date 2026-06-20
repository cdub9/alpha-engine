"""GDELT 2.0 DOC API client.

GDELT (Global Database of Events, Language, and Tone) monitors world news
in 100+ languages, updated every 15 minutes. Free, no auth required.

We use the DOC API (https://api.gdeltproject.org/api/v2/doc/doc) rather
than the raw events tables because it gives us pre-aggregated daily
timeseries for any query — perfect for "how much is the world talking
about X" signals.

Two modes we care about:
  - TimelineVol:  daily article volume, normalized 0-1
  - TimelineTone: daily average tone, -10 (very negative) to +10

Note on rate limits: GDELT has no documented hard limit but asks to be
polite. We default to a 0.5s sleep between requests in the bulk fetcher.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Iterable, Literal, Optional

import httpx

from alpha_engine.core.config import get_settings
from alpha_engine.core.logging import get_logger

log = get_logger(__name__)

GDELT_BASE = "https://api.gdeltproject.org/api/v2/doc/doc"


GdeltMode = Literal["TimelineVol", "TimelineTone", "TimelineVolRaw"]


@dataclass(frozen=True)
class GdeltDailyPoint:
    """One day of GDELT signal data for a query."""

    signal_date: date
    volume_intensity: Optional[float] = None      # 0-1 normalized
    avg_tone: Optional[float] = None              # -10..+10
    raw_query: Optional[str] = None


class GDELTClient:
    """Polite client for GDELT DOC API."""

    def __init__(self, timeout: float = 30.0) -> None:
        settings = get_settings()
        self._client = httpx.Client(
            timeout=timeout,
            headers={"User-Agent": settings.data.user_agent},
        )

    def __enter__(self) -> "GDELTClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    # ------------------------------------------------------------------
    # Low-level fetch
    # ------------------------------------------------------------------

    def _fetch_timeline(
        self,
        query: str,
        mode: GdeltMode,
        timespan: str = "30d",
        max_retries: int = 3,
        backoff_base: float = 10.0,
    ) -> dict[date, float]:
        """Fetch a timeline. Returns {date: value} dict.

        Retries on 429 (rate limited) with exponential backoff:
        first retry waits 10s, second 20s, third 40s. GDELT has no
        published rate limit but in practice ~1 request per 3-5 seconds
        is safer than the official guidance.
        """
        params = {
            "query": query,
            "mode": mode,
            "timespan": timespan,
            "format": "json",
        }
        resp: Optional[httpx.Response] = None
        for attempt in range(max_retries + 1):
            try:
                resp = self._client.get(GDELT_BASE, params=params)
                if resp.status_code == 429 and attempt < max_retries:
                    wait = backoff_base * (2 ** attempt)
                    log.warning(
                        "gdelt_rate_limited",
                        query=query,
                        mode=mode,
                        attempt=attempt + 1,
                        wait_seconds=wait,
                    )
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                # GDELT sometimes returns HTTP 200 with an EMPTY body for
                # certain query shapes (confirmed 2026-06-07 on the old
                # 4-clause recession query). That is NOT "zero results" — it
                # is a malformed response. Treat an empty/non-JSON 200 as
                # retriable rather than silently yielding no data.
                if resp.text.strip() and resp.headers.get("content-type", "").startswith(
                    ("application/json", "text/json")
                ):
                    break
                if not resp.text.strip() and attempt < max_retries:
                    wait = backoff_base * (2 ** attempt)
                    log.warning(
                        "gdelt_empty_200_body",
                        query=query, mode=mode, attempt=attempt + 1, wait_seconds=wait,
                    )
                    time.sleep(wait)
                    continue
                break
            except httpx.HTTPError as exc:
                # Non-429 HTTP failures: give up immediately
                if resp is None or resp.status_code != 429:
                    log.warning(
                        "gdelt_fetch_failed", query=query, mode=mode, error=str(exc)
                    )
                    return {}
        if resp is None or resp.status_code >= 400:
            return {}

        try:
            data = resp.json()
        except ValueError as exc:
            log.warning("gdelt_invalid_json", query=query, mode=mode, error=str(exc))
            return {}

        out: dict[date, float] = {}
        for series in data.get("timeline", []):
            for point in series.get("data", []):
                date_str = point.get("date", "")
                try:
                    # GDELT timestamps look like "20260423T000000Z"
                    dt = datetime.strptime(date_str, "%Y%m%dT%H%M%SZ").replace(
                        tzinfo=timezone.utc
                    )
                    point_date = dt.date()
                except ValueError:
                    continue
                try:
                    val = float(point.get("value"))
                except (TypeError, ValueError):
                    continue
                out[point_date] = val
        return out

    # ------------------------------------------------------------------
    # Higher-level: pull volume + tone for one query into typed points
    # ------------------------------------------------------------------

    def fetch_signal(
        self,
        signal_name: str,
        query: str,
        timespan: str = "30d",
        polite_sleep: float = 4.0,
    ) -> list[GdeltDailyPoint]:
        """Pull both volume and tone for one query, merge into per-day points.

        Returns sorted-by-date GdeltDailyPoint list. polite_sleep is the
        delay between the two API calls (volume + tone)."""
        log.info("gdelt_fetch_signal", signal=signal_name, query=query, span=timespan)

        vol = self._fetch_timeline(query, "TimelineVol", timespan=timespan)
        time.sleep(polite_sleep)
        tone = self._fetch_timeline(query, "TimelineTone", timespan=timespan)

        all_dates = sorted(set(vol.keys()) | set(tone.keys()))
        return [
            GdeltDailyPoint(
                signal_date=d,
                volume_intensity=vol.get(d),
                avg_tone=tone.get(d),
                raw_query=query,
            )
            for d in all_dates
        ]

    def fetch_signals(
        self,
        signals: Iterable[tuple[str, str]],
        timespan: str = "30d",
        polite_sleep: float = 0.5,
    ) -> dict[str, list[GdeltDailyPoint]]:
        """Pull many (signal_name, query) pairs. Returns dict keyed by signal_name.

        Sleeps polite_sleep seconds between *queries* (in addition to
        between the two calls per query). For 10 signals this is ~15s
        total wall-clock — modest but courteous."""
        out: dict[str, list[GdeltDailyPoint]] = {}
        signal_list = list(signals)
        for i, (name, query) in enumerate(signal_list):
            out[name] = self.fetch_signal(
                signal_name=name,
                query=query,
                timespan=timespan,
                polite_sleep=polite_sleep,
            )
            if i < len(signal_list) - 1:
                time.sleep(polite_sleep)
        return out
