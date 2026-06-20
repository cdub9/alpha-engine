"""FRED (Federal Reserve Economic Data) client.

Free API. Get a key at: https://fred.stlouisfed.org/docs/api/api_key.html
Rate limit: 120 requests/minute.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Iterable, Optional

import httpx

from alpha_engine.core.config import get_settings
from alpha_engine.core.logging import get_logger
from alpha_engine.core.types import MacroObservation
from alpha_engine.data.base import DataProvider

log = get_logger(__name__)

FRED_BASE = "https://api.stlouisfed.org/fred"


class FredClient(DataProvider[MacroObservation]):
    name = "fred"

    def __init__(self, api_key: Optional[str] = None, timeout: float = 30.0) -> None:
        settings = get_settings()
        self.api_key = api_key or settings.fred_api_key
        if not self.api_key:
            raise RuntimeError(
                "FRED_API_KEY not set. Get a free key at "
                "https://fred.stlouisfed.org/docs/api/api_key.html and add it to .env"
            )
        self._client = httpx.Client(
            base_url=FRED_BASE,
            timeout=timeout,
            headers={"User-Agent": settings.data.user_agent},
        )

    def __enter__(self) -> "FredClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    def fetch(
        self,
        series_id: str,
        observation_start: Optional[date] = None,
        observation_end: Optional[date] = None,
    ) -> Iterable[MacroObservation]:
        """Fetch observations for a single FRED series."""
        params: dict = {
            "series_id": series_id,
            "api_key": self.api_key,
            "file_type": "json",
        }
        if observation_start:
            params["observation_start"] = observation_start.isoformat()
        if observation_end:
            params["observation_end"] = observation_end.isoformat()

        log.info("fred_fetch", series=series_id, start=str(observation_start))
        resp = self._client.get("/series/observations", params=params)
        resp.raise_for_status()
        data = resp.json()

        for obs in data.get("observations", []):
            value_raw = obs.get("value", ".")
            value = None if value_raw in (".", "", None) else float(value_raw)
            try:
                obs_date = datetime.strptime(obs["date"], "%Y-%m-%d").date()
            except (KeyError, ValueError):
                continue
            yield MacroObservation(
                series_id=series_id,
                obs_date=obs_date,
                value=value,
                source=self.name,
            )

    def get_series_metadata(self, series_id: str) -> dict:
        """Fetch metadata for a series (name, units, frequency, etc.)."""
        params = {"series_id": series_id, "api_key": self.api_key, "file_type": "json"}
        resp = self._client.get("/series", params=params)
        resp.raise_for_status()
        seriess = resp.json().get("seriess", [])
        return seriess[0] if seriess else {}
