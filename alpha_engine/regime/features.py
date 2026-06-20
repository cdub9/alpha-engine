"""Macro feature extraction for regime classification.

Pulls the most recent FRED observations as of a given date and computes
derived features the classifier needs: yield curve inversion duration,
VIX moving averages, Sahm Rule trend, unemployment trajectory, Fed funds
historical percentile, CPI year-over-year change.

Different series have different frequencies (daily/weekly/monthly) and
publication lags (CPI: ~2 weeks, UNRATE: ~1 week). We always use the most
recent available observation <= as_of, so we don't accidentally use future
data in historical backfills.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, timedelta
from typing import Optional

import duckdb
import pandas as pd


# FRED series IDs we depend on. Missing series degrade gracefully.
SERIES = {
    "t10y2y":     "T10Y2Y",       # 10Y-2Y Treasury spread (yield curve)
    "dgs10":      "DGS10",        # 10Y Treasury yield
    "dgs2":       "DGS2",         # 2Y Treasury yield
    "vix":        "VIXCLS",       # CBOE Volatility Index
    "sahm":       "SAHMREALTIME", # Sahm Rule recession indicator
    "unrate":     "UNRATE",       # Unemployment rate (monthly)
    "fed_funds":  "DFF",          # Effective Fed Funds Rate
    "cpi":        "CPIAUCSL",     # CPI (monthly)
    "oil":        "DCOILWTICO",   # WTI crude oil
    "dxy_proxy":  "DEXUSEU",      # USD/EUR (dollar strength proxy)
}


@dataclass(frozen=True)
class MacroFeatures:
    """Computed macro features for a given date.

    All fields are Optional[float] — they will be None if the source FRED
    series has no data within a reasonable lookback. The classifier must
    handle missing fields gracefully.
    """

    as_of: date

    # Yield curve
    t10y2y_latest: Optional[float] = None
    t10y2y_avg_30d: Optional[float] = None
    yield_curve_inverted: Optional[bool] = None     # latest < 0
    yield_curve_days_inverted_90d: Optional[int] = None
    yield_curve_inverted_long: Optional[bool] = None  # inverted >180d in past

    # Volatility
    vix_latest: Optional[float] = None
    vix_avg_30d: Optional[float] = None
    vix_avg_90d: Optional[float] = None
    vix_regime_low: Optional[bool] = None    # avg_30d < 16
    vix_regime_high: Optional[bool] = None   # avg_30d > 25

    # Recession signals
    sahm_latest: Optional[float] = None
    sahm_max_6m: Optional[float] = None
    sahm_triggered: Optional[bool] = None        # latest >= 0.5
    sahm_recently_active: Optional[bool] = None  # max_6m >= 0.5

    # Labor market
    unrate_latest: Optional[float] = None
    unrate_3m_avg: Optional[float] = None
    unrate_6m_avg: Optional[float] = None
    unrate_12m_low: Optional[float] = None
    unrate_rising: Optional[bool] = None     # 3m avg > 6m avg
    unrate_falling: Optional[bool] = None    # 3m avg < 6m avg by >0.05

    # Monetary policy
    fed_funds_latest: Optional[float] = None
    fed_funds_percentile_5y: Optional[float] = None  # 0-1
    fed_funds_elevated: Optional[bool] = None        # > median of 5y

    # Inflation
    cpi_latest: Optional[float] = None
    cpi_yoy_pct: Optional[float] = None              # year-over-year change
    cpi_elevated: Optional[bool] = None              # YoY > 3%

    # Auxiliary
    oil_latest: Optional[float] = None
    oil_pct_change_30d: Optional[float] = None
    dxy_proxy_latest: Optional[float] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["as_of"] = self.as_of.isoformat()
        return d


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------


def _load_series(
    con: duckdb.DuckDBPyConnection,
    series_id: str,
    end_date: date,
    days_back: int,
) -> pd.Series:
    """Load a single FRED series ending at end_date, going back N days.
    Returns a pandas Series indexed by date with float values (NaN for
    missing). Empty if series has no data."""
    start = end_date - timedelta(days=days_back)
    rows = con.execute(
        "SELECT obs_date, value FROM macro_series "
        "WHERE series_id = ? AND obs_date BETWEEN ? AND ? "
        "ORDER BY obs_date",
        [series_id, start, end_date],
    ).fetchall()
    if not rows:
        return pd.Series(dtype=float)

    s = pd.Series({d: v for d, v in rows}, dtype=float)
    s.index = pd.to_datetime(s.index)
    return s


def _last(series: pd.Series) -> Optional[float]:
    """Most recent non-null value, or None."""
    s = series.dropna()
    return float(s.iloc[-1]) if not s.empty else None


def _percentile(series: pd.Series, value: float) -> Optional[float]:
    s = series.dropna()
    if s.empty:
        return None
    return float((s <= value).mean())


def _last_window(series: pd.Series, days: int) -> pd.Series:
    """Return the tail of a date-indexed Series covering the last `days`
    days. Replacement for the removed Series.last('Nd') API."""
    s = series.dropna()
    if s.empty:
        return s
    cutoff = s.index.max() - pd.Timedelta(days=days)
    return s.loc[s.index > cutoff]


def extract_features(con: duckdb.DuckDBPyConnection, as_of: date) -> MacroFeatures:
    """Pull all macro features for `as_of`. Reads only data with
    obs_date <= as_of, so this is safe for historical backfills."""
    # Load each series with enough history for derived features
    t10y2y = _load_series(con, SERIES["t10y2y"], as_of, days_back=400)
    vix = _load_series(con, SERIES["vix"], as_of, days_back=180)
    sahm = _load_series(con, SERIES["sahm"], as_of, days_back=400)
    unrate = _load_series(con, SERIES["unrate"], as_of, days_back=500)
    fed_funds = _load_series(con, SERIES["fed_funds"], as_of, days_back=365 * 5 + 30)
    cpi = _load_series(con, SERIES["cpi"], as_of, days_back=500)
    oil = _load_series(con, SERIES["oil"], as_of, days_back=90)
    dxy = _load_series(con, SERIES["dxy_proxy"], as_of, days_back=30)

    # ---- Yield curve --------------------------------------------------
    t10_latest = _last(t10y2y)
    yc_30 = _last_window(t10y2y, 30)
    yc_90 = _last_window(t10y2y, 90)
    yc_365 = _last_window(t10y2y, 365)
    t10_avg_30 = float(yc_30.mean()) if not yc_30.empty else None
    days_inverted_90 = int((yc_90 < 0).sum()) if not yc_90.empty else None
    inverted_long = bool((yc_365 < 0).sum() >= 180) if not yc_365.empty else None

    # ---- VIX ----------------------------------------------------------
    vix_latest = _last(vix)
    vix_30_win = _last_window(vix, 30)
    vix_90_win = _last_window(vix, 90)
    vix_30 = float(vix_30_win.mean()) if not vix_30_win.empty else None
    vix_90 = float(vix_90_win.mean()) if not vix_90_win.empty else None

    # ---- Sahm ---------------------------------------------------------
    sahm_latest = _last(sahm)
    sahm_180 = _last_window(sahm, 180)
    sahm_max_6m = float(sahm_180.max()) if not sahm_180.empty else None

    # ---- Unemployment -------------------------------------------------
    unrate_latest = _last(unrate)
    ur_90 = _last_window(unrate, 90)
    ur_180 = _last_window(unrate, 180)
    ur_365 = _last_window(unrate, 365)
    unrate_3m = float(ur_90.mean()) if not ur_90.empty else None
    unrate_6m = float(ur_180.mean()) if not ur_180.empty else None
    unrate_12m_low = float(ur_365.min()) if not ur_365.empty else None

    # ---- Fed funds ----------------------------------------------------
    ff_latest = _last(fed_funds)
    ff_5y = _last_window(fed_funds, 1825)  # 5 years
    ff_pct = _percentile(ff_5y, ff_latest) if ff_latest is not None else None
    ff_median = float(ff_5y.median()) if not ff_5y.empty else None

    # ---- CPI YoY ------------------------------------------------------
    cpi_latest = _last(cpi)
    cpi_yoy: Optional[float] = None
    cpi_clean = cpi.dropna()
    if cpi_latest is not None and not cpi_clean.empty:
        latest_ts = cpi_clean.index[-1]
        year_ago_target = latest_ts - pd.Timedelta(days=365)
        # Find the obs nearest 1 year ago
        year_ago_idx = cpi_clean.index.get_indexer(
            [year_ago_target], method="nearest"
        )[0]
        if year_ago_idx >= 0:
            cpi_year_ago = float(cpi_clean.iloc[year_ago_idx])
            if cpi_year_ago > 0:
                cpi_yoy = (cpi_latest / cpi_year_ago - 1.0) * 100.0

    # ---- Oil ----------------------------------------------------------
    oil_latest = _last(oil)
    oil_pct_30: Optional[float] = None
    if oil_latest is not None:
        thirty_back = _last_window(oil, 30)
        if len(thirty_back) >= 2:
            old = float(thirty_back.iloc[0])
            if old > 0:
                oil_pct_30 = (oil_latest / old - 1.0) * 100.0

    return MacroFeatures(
        as_of=as_of,
        # Yield curve
        t10y2y_latest=t10_latest,
        t10y2y_avg_30d=t10_avg_30,
        yield_curve_inverted=(t10_latest < 0) if t10_latest is not None else None,
        yield_curve_days_inverted_90d=days_inverted_90,
        yield_curve_inverted_long=inverted_long,
        # VIX
        vix_latest=vix_latest,
        vix_avg_30d=vix_30,
        vix_avg_90d=vix_90,
        vix_regime_low=(vix_30 < 16) if vix_30 is not None else None,
        vix_regime_high=(vix_30 > 25) if vix_30 is not None else None,
        # Sahm
        sahm_latest=sahm_latest,
        sahm_max_6m=sahm_max_6m,
        sahm_triggered=(sahm_latest >= 0.5) if sahm_latest is not None else None,
        sahm_recently_active=(sahm_max_6m >= 0.5) if sahm_max_6m is not None else None,
        # Labor
        unrate_latest=unrate_latest,
        unrate_3m_avg=unrate_3m,
        unrate_6m_avg=unrate_6m,
        unrate_12m_low=unrate_12m_low,
        unrate_rising=(
            unrate_3m > unrate_6m if unrate_3m is not None and unrate_6m is not None else None
        ),
        unrate_falling=(
            (unrate_6m - unrate_3m) > 0.05
            if unrate_3m is not None and unrate_6m is not None
            else None
        ),
        # Fed funds
        fed_funds_latest=ff_latest,
        fed_funds_percentile_5y=ff_pct,
        fed_funds_elevated=(
            ff_latest > ff_median
            if ff_latest is not None and ff_median is not None
            else None
        ),
        # Inflation
        cpi_latest=cpi_latest,
        cpi_yoy_pct=cpi_yoy,
        cpi_elevated=(cpi_yoy > 3.0) if cpi_yoy is not None else None,
        # Auxiliary
        oil_latest=oil_latest,
        oil_pct_change_30d=oil_pct_30,
        dxy_proxy_latest=_last(dxy),
    )
