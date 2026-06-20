"""Compute geopolitical features from stored GDELT signals.

For each tracked signal (iran_conflict, oil_disruption, etc.) we compute:
  - Recent volume intensity (mean of last 7 days, in GDELT's 0-1 units)
  - Baseline volume intensity (mean of last 30 days)
  - Volume ratio: recent / baseline (the "elevation" measure)
  - Recent vs baseline tone
  - Categorical intensity label: LOW | NORMAL | ELEVATED | HIGH

The categorical label is what the LLM digest section uses for at-a-glance
read; the underlying numerics are also exposed for ML/feature use.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from enum import Enum
from typing import Optional

import duckdb
import pandas as pd


class SignalIntensity(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    ELEVATED = "elevated"
    HIGH = "high"
    UNKNOWN = "unknown"


def _classify_intensity(ratio: Optional[float]) -> SignalIntensity:
    """Map volume ratio (recent/baseline) to a categorical level."""
    if ratio is None:
        return SignalIntensity.UNKNOWN
    if ratio < 0.7:
        return SignalIntensity.LOW
    if ratio < 1.3:
        return SignalIntensity.NORMAL
    if ratio < 2.0:
        return SignalIntensity.ELEVATED
    return SignalIntensity.HIGH


@dataclass(frozen=True)
class GeopoliticalSignalState:
    """Computed state for one signal as of a given date."""

    signal_name: str
    description: str
    sector_relevance: list[str]
    recent_volume: Optional[float]       # 7-day mean (GDELT 0-1)
    baseline_volume: Optional[float]     # 30-day mean
    volume_ratio: Optional[float]        # recent / baseline
    intensity: SignalIntensity
    recent_tone: Optional[float]         # 7-day mean (-10..+10)
    baseline_tone: Optional[float]       # 30-day mean
    tone_delta: Optional[float]          # recent - baseline (negative=more negative)
    n_days_data: int

    def to_dict(self) -> dict:
        d = asdict(self)
        d["intensity"] = self.intensity.value
        return d


@dataclass(frozen=True)
class GeopoliticalFeatures:
    """All signal states + summary aggregates as of a date."""

    as_of: date
    signals: list[GeopoliticalSignalState] = field(default_factory=list)
    elevated_signal_count: int = 0
    high_intensity_signals: list[str] = field(default_factory=list)
    avg_tone_recent: Optional[float] = None
    avg_tone_baseline: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "as_of": self.as_of.isoformat(),
            "signals": [s.to_dict() for s in self.signals],
            "elevated_signal_count": self.elevated_signal_count,
            "high_intensity_signals": self.high_intensity_signals,
            "avg_tone_recent": self.avg_tone_recent,
            "avg_tone_baseline": self.avg_tone_baseline,
        }


# ---------------------------------------------------------------------------
# Computation
# ---------------------------------------------------------------------------


def _load_signal_history(
    con: duckdb.DuckDBPyConnection,
    signal_name: str,
    end_date: date,
    days_back: int,
) -> pd.DataFrame:
    """Load (signal_date, volume_intensity, avg_tone) for one signal up to
    end_date going back N days. Returns empty DataFrame if nothing."""
    start = end_date - timedelta(days=days_back)
    rows = con.execute(
        """
        SELECT signal_date, volume_intensity, avg_tone
        FROM geopolitical_signals
        WHERE signal_name = ?
          AND signal_date BETWEEN ? AND ?
        ORDER BY signal_date
        """,
        [signal_name, start, end_date],
    ).fetchall()
    if not rows:
        return pd.DataFrame(columns=["signal_date", "vol", "tone"])
    df = pd.DataFrame(
        rows, columns=["signal_date", "vol", "tone"]
    )
    return df


def _mean(series: pd.Series) -> Optional[float]:
    s = series.dropna()
    if s.empty:
        return None
    return float(s.mean())


def compute_geopolitical_features(
    con: duckdb.DuckDBPyConnection,
    as_of: date,
    recent_window_days: int = 7,
    baseline_window_days: int = 30,
) -> GeopoliticalFeatures:
    """Compute features across all tracked signals as of a date.

    Uses GDELT data with signal_date <= as_of (no look-ahead). Signals with
    no data return UNKNOWN intensity and None numerics."""
    # Get the list of distinct signal names we have data for
    signal_rows = con.execute(
        "SELECT DISTINCT signal_name FROM geopolitical_signals ORDER BY signal_name"
    ).fetchall()
    signal_names = [r[0] for r in signal_rows]

    # Get descriptions from config (best-effort; geopolitical may be empty)
    from alpha_engine.core.config import get_settings
    cfg_lookup = {
        s.name: (s.description, s.sector_relevance)
        for s in get_settings().geopolitical.signals
    }

    states: list[GeopoliticalSignalState] = []
    all_recent_tones: list[float] = []
    all_baseline_tones: list[float] = []
    elevated_count = 0
    high_signals: list[str] = []

    for name in signal_names:
        history = _load_signal_history(con, name, as_of, baseline_window_days)
        n_days = len(history)
        if n_days == 0:
            description, sector_rel = cfg_lookup.get(name, ("", []))
            states.append(
                GeopoliticalSignalState(
                    signal_name=name,
                    description=description,
                    sector_relevance=sector_rel,
                    recent_volume=None,
                    baseline_volume=None,
                    volume_ratio=None,
                    intensity=SignalIntensity.UNKNOWN,
                    recent_tone=None,
                    baseline_tone=None,
                    tone_delta=None,
                    n_days_data=0,
                )
            )
            continue

        recent_df = history.tail(recent_window_days)
        recent_vol = _mean(recent_df["vol"])
        baseline_vol = _mean(history["vol"])
        recent_tone = _mean(recent_df["tone"])
        baseline_tone = _mean(history["tone"])

        ratio: Optional[float] = None
        if recent_vol is not None and baseline_vol is not None and baseline_vol > 0:
            ratio = recent_vol / baseline_vol

        intensity = _classify_intensity(ratio)
        if intensity in (SignalIntensity.ELEVATED, SignalIntensity.HIGH):
            elevated_count += 1
        if intensity == SignalIntensity.HIGH:
            high_signals.append(name)

        if recent_tone is not None:
            all_recent_tones.append(recent_tone)
        if baseline_tone is not None:
            all_baseline_tones.append(baseline_tone)

        tone_delta: Optional[float] = None
        if recent_tone is not None and baseline_tone is not None:
            tone_delta = recent_tone - baseline_tone

        description, sector_rel = cfg_lookup.get(name, ("", []))
        states.append(
            GeopoliticalSignalState(
                signal_name=name,
                description=description,
                sector_relevance=sector_rel,
                recent_volume=recent_vol,
                baseline_volume=baseline_vol,
                volume_ratio=ratio,
                intensity=intensity,
                recent_tone=recent_tone,
                baseline_tone=baseline_tone,
                tone_delta=tone_delta,
                n_days_data=n_days,
            )
        )

    return GeopoliticalFeatures(
        as_of=as_of,
        signals=states,
        elevated_signal_count=elevated_count,
        high_intensity_signals=high_signals,
        avg_tone_recent=(
            sum(all_recent_tones) / len(all_recent_tones) if all_recent_tones else None
        ),
        avg_tone_baseline=(
            sum(all_baseline_tones) / len(all_baseline_tones)
            if all_baseline_tones
            else None
        ),
    )
