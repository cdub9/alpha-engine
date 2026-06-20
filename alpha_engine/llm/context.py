"""Build the daily context snapshot the LLM gets as user input.

Assembles everything we know about today's market into a structured
markdown document. Sections:

  1. Date + regime
  2. Macro signals (yield curve, VIX, Sahm, fed funds, CPI YoY)
  3. Calendar context (days to FOMC/CPI/jobs/OpEx)
  4. Cross-asset levels (SPY, QQQ, TLT, GLD, VIX, oil)
  5. Universe price action (per ticker, with 1d/5d/30d % change)
  6. Upcoming earnings (next 30 days)
  7. Notable conditions (earnings this week, FOMC week, etc.)

The snapshot is what changes every day — it should NOT be cached. The
system prompt (rules, channel definitions, output schema) is cached.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional

import duckdb
import pandas as pd

from alpha_engine.calendars import (
    compute_market_calendar_features,
    compute_ticker_calendar_features,
)
from alpha_engine.core.logging import get_logger
from alpha_engine.intel import (
    GeopoliticalFeatures,
    SignalIntensity,
    compute_geopolitical_features,
)
from alpha_engine.regime import REGIME_DESCRIPTIONS, classify, extract_features, get_prior_regime

log = get_logger(__name__)


@dataclass
class DailySnapshot:
    """Structured snapshot of today's market state."""

    as_of: date
    markdown: str                          # what we send to the LLM
    regime_label: str
    regime_confidence: float
    universe: list[str]
    notable_events: list[str]


CROSS_ASSET_SYMBOLS = ["SPY", "QQQ", "IWM", "TLT", "AGG", "GLD", "XLE"]


def _pct_change(prices: pd.Series, days: int) -> Optional[float]:
    """% change from N trading days ago to latest. None if insufficient data."""
    if len(prices) <= days:
        return None
    return float(prices.iloc[-1] / prices.iloc[-1 - days] - 1.0) * 100.0


def _format_price_row(
    symbol: str, prices: pd.Series, name: str = ""
) -> Optional[str]:
    """Format one row of price action: symbol, name, current, 1d, 5d, 30d."""
    if prices.empty:
        return None
    latest = float(prices.iloc[-1])
    d1 = _pct_change(prices, 1)
    d5 = _pct_change(prices, 5)
    d30 = _pct_change(prices, 30)

    fmt = lambda v: f"{v:+.2f}%" if v is not None else "  —  "  # noqa: E731
    name_part = f" ({name})" if name else ""
    return (
        f"- **{symbol}**{name_part}: ${latest:,.2f}  "
        f"1d={fmt(d1)}  5d={fmt(d5)}  30d={fmt(d30)}"
    )


def _load_prices(
    con: duckdb.DuckDBPyConnection, symbol: str, as_of: date, days: int = 60
) -> pd.Series:
    """Most recent N trading days of adj_close for symbol, ending at as_of."""
    start = as_of - timedelta(days=int(days * 1.5))
    rows = con.execute(
        "SELECT bar_date, adj_close FROM market_bars "
        "WHERE symbol = ? AND bar_date BETWEEN ? AND ? "
        "ORDER BY bar_date",
        [symbol, start, as_of],
    ).fetchall()
    if not rows:
        return pd.Series(dtype=float)
    s = pd.Series({d: v for d, v in rows}, dtype=float)
    s.index = pd.to_datetime(s.index)
    return s


def _format_macro_section(features) -> str:
    """Format the macro signals block."""
    f = features
    lines = ["## Macro signals"]

    def fmt(v, fmt_str=".2f", suffix=""):
        return f"{v:{fmt_str}}{suffix}" if v is not None else "—"

    lines.append(
        f"- Yield curve (T10Y2Y): {fmt(f.t10y2y_latest, '.2f')}  "
        f"30d avg {fmt(f.t10y2y_avg_30d, '.2f')}  "
        f"(inverted {fmt(f.yield_curve_days_inverted_90d, 'd')} of last 90 days)"
    )
    lines.append(
        f"- VIX: latest {fmt(f.vix_latest, '.1f')}  "
        f"30d avg {fmt(f.vix_avg_30d, '.1f')}  "
        f"90d avg {fmt(f.vix_avg_90d, '.1f')}"
    )
    lines.append(
        f"- Sahm Rule: latest {fmt(f.sahm_latest, '.2f')}  "
        f"6m max {fmt(f.sahm_max_6m, '.2f')}  "
        f"(triggered = {f.sahm_triggered})"
    )
    lines.append(
        f"- Unemployment: latest {fmt(f.unrate_latest, '.2f')}%  "
        f"3m avg {fmt(f.unrate_3m_avg, '.2f')}%  "
        f"12m low {fmt(f.unrate_12m_low, '.2f')}%  "
        f"trend = {'rising' if f.unrate_rising else 'flat/falling'}"
    )
    lines.append(
        f"- Fed funds: {fmt(f.fed_funds_latest, '.2f')}%  "
        f"(percentile of last 5y: {fmt((f.fed_funds_percentile_5y or 0) * 100, '.0f')}%)"
    )
    lines.append(
        f"- CPI YoY: {fmt(f.cpi_yoy_pct, '.1f')}%  "
        f"(elevated > 3%: {f.cpi_elevated})"
    )
    lines.append(
        f"- WTI Oil: ${fmt(f.oil_latest, '.2f')}  "
        f"30d change: {fmt(f.oil_pct_change_30d, '+.1f')}%"
    )
    return "\n".join(lines)


def _format_calendar_section(cal) -> str:
    """Format calendar context."""
    lines = ["## Calendar context (next event in calendar days)"]
    lines.append(f"- Next FOMC meeting: {cal.days_to_next_fomc} days")
    lines.append(f"- Next CPI release: {cal.days_to_next_cpi} days")
    lines.append(f"- Next jobs report: {cal.days_to_next_jobs_report} days")
    lines.append(
        f"- Next OpEx: {cal.days_to_next_opex} days"
        + ("  (also QUAD WITCHING)" if cal.days_to_next_quad_witching == cal.days_to_next_opex else "")
    )
    lines.append(
        f"- Seasonality: month={cal.month}  "
        f"is_september={cal.is_september}  "
        f"is_summer_doldrums={cal.is_summer_doldrums}  "
        f"is_santa_claus_window={cal.is_santa_claus_window}"
    )

    flags = []
    if cal.is_fomc_week:
        flags.append("FOMC week")
    if cal.is_opex_week:
        flags.append("OpEx week")
    if cal.is_quad_witching_week:
        flags.append("QUAD WITCHING week")
    if cal.is_jobs_report_week:
        flags.append("jobs report week")
    if cal.is_cpi_week:
        flags.append("CPI week")
    if cal.is_quarter_end_week:
        flags.append("quarter-end week")
    if flags:
        lines.append(f"- This week: **{', '.join(flags)}**")
    return "\n".join(lines)


def _format_universe_section(
    con: duckdb.DuckDBPyConnection,
    universe: list[str],
    as_of: date,
    instrument_lookup: dict[str, str],
) -> str:
    """Format per-ticker price action."""
    lines = ["## Universe price action"]
    for sym in universe:
        prices = _load_prices(con, sym, as_of, days=40)
        row = _format_price_row(sym, prices, name=instrument_lookup.get(sym, ""))
        if row:
            lines.append(row)
    return "\n".join(lines)


def _format_cross_asset_section(con, as_of: date) -> str:
    """Format cross-asset levels."""
    lines = ["## Cross-asset levels"]
    for sym in CROSS_ASSET_SYMBOLS:
        prices = _load_prices(con, sym, as_of, days=40)
        row = _format_price_row(sym, prices)
        if row:
            lines.append(row)
    return "\n".join(lines)


def _format_geopolitical_section(features: GeopoliticalFeatures) -> str:
    """Format the geopolitical context block — most-elevated signals first."""
    if not features.signals:
        return "## Geopolitical context\n- No GDELT data ingested yet (run scripts/ingest_gdelt.py)"

    # Rich console treats [bracketed] as markup — use plain labels so they
    # render as-is in both the Rich UI and the raw text sent to the LLM.
    intensity_label = {
        SignalIntensity.HIGH:     "HIGH    ",
        SignalIntensity.ELEVATED: "elevated",
        SignalIntensity.NORMAL:   "normal  ",
        SignalIntensity.LOW:      "low     ",
        SignalIntensity.UNKNOWN:  "no data ",
    }

    # Sort: HIGH first, then ELEVATED, then by ratio descending
    intensity_order = {
        SignalIntensity.HIGH: 0,
        SignalIntensity.ELEVATED: 1,
        SignalIntensity.NORMAL: 2,
        SignalIntensity.LOW: 3,
        SignalIntensity.UNKNOWN: 4,
    }
    sorted_signals = sorted(
        features.signals,
        key=lambda s: (intensity_order[s.intensity], -(s.volume_ratio or 0)),
    )

    lines = ["## Geopolitical context (GDELT, recent 7d vs 30d baseline)"]
    lines.append(
        f"- Elevated/high signals: {features.elevated_signal_count} of "
        f"{len(features.signals)}"
        + (
            f"  ({', '.join(features.high_intensity_signals)} are HIGH)"
            if features.high_intensity_signals
            else ""
        )
    )
    if features.avg_tone_recent is not None and features.avg_tone_baseline is not None:
        delta = features.avg_tone_recent - features.avg_tone_baseline
        lines.append(
            f"- Avg news tone (across all signals): recent {features.avg_tone_recent:.2f}, "
            f"baseline {features.avg_tone_baseline:.2f}, "
            f"delta {delta:+.2f} ({'more negative' if delta < 0 else 'less negative'})"
        )
    lines.append("")
    lines.append("Per-signal status (sorted by intensity):")

    for s in sorted_signals:
        if s.intensity == SignalIntensity.UNKNOWN:
            lines.append(f"- ({intensity_label[s.intensity]}) **{s.signal_name}**: no data")
            continue
        ratio_str = f"{s.volume_ratio:.2f}x" if s.volume_ratio is not None else "—"
        tone_str = (
            f"tone {s.recent_tone:+.1f} (Δ{s.tone_delta:+.1f})"
            if s.recent_tone is not None and s.tone_delta is not None
            else "tone —"
        )
        sectors = (
            f"  (relevant: {', '.join(s.sector_relevance)})"
            if s.sector_relevance
            else ""
        )
        lines.append(
            f"- ({intensity_label[s.intensity]}) **{s.signal_name}**: "
            f"vol {ratio_str} of baseline, {tone_str}{sectors}"
        )
    return "\n".join(lines)


def _format_technicals_section(
    con: duckdb.DuckDBPyConnection,
    universe: list[str],
    as_of: date,
) -> str:
    """Tier-1 per-symbol technicals + Tier-2 universe breadth.

    Decision-ready TA inputs so the model doesn't have to re-derive trend
    state from raw 1d/5d/30d changes. Deliberately narrow (see FOLLOWUPS
    "Add technical analysis features to the snapshot"): distance from
    50/200-day MA, 14-day RSI, 30-day realized vol — the only TA with
    real empirical support. Computed from bars already in the DB; free.

    Reuses the ML layer's feature module (alpha_engine/ml/features.py) —
    one tested implementation, not two.
    """
    from alpha_engine.ml.advisor import _PRICE_LOOKBACK_DAYS, load_price_panel
    from alpha_engine.ml.features import compute_features

    prices = load_price_panel(con, universe, as_of, _PRICE_LOOKBACK_DAYS)
    if prices.empty:
        return ""
    feats = compute_features(prices)
    valid = feats.dropna()
    if valid.empty:
        return ""

    # Tier 2 — universe-wide breadth (broad trend confirmation)
    pct_above_50 = (valid["dist_50ma"] > 0).mean()
    avg_1m_ret = valid["rev_1m"].mean()

    lines = ["## Per-symbol technicals"]
    lines.append(
        f"Breadth: {pct_above_50:.0%} of {len(valid)} names above their own "
        f"50-day MA; avg 1-month return {avg_1m_ret:+.1%}"
    )

    overbought = valid[valid["rsi_14"] >= 75].index.tolist()
    oversold = valid[valid["rsi_14"] <= 25].index.tolist()
    if overbought:
        lines.append(f"RSI >= 75 (stretched): {', '.join(sorted(overbought))}")
    if oversold:
        lines.append(f"RSI <= 25 (washed out): {', '.join(sorted(oversold))}")

    lines.append("")
    for sym in universe:
        if sym not in valid.index:
            continue  # insufficient history — no partial-information rows
        r = valid.loc[sym]
        lines.append(
            f"- **{sym}**: 50MA {r['dist_50ma']:+.1%}  "
            f"200MA {r['dist_200ma']:+.1%}  "
            f"RSI {r['rsi_14']:.0f}  vol {r['vol_30d']:.0%}"
        )
    return "\n".join(lines)


def _format_upcoming_earnings(con, as_of: date, days_ahead: int = 45) -> str:
    """Format upcoming earnings calendar."""
    end = as_of + timedelta(days=days_ahead)
    rows = con.execute(
        "SELECT symbol, event_date FROM calendar_events "
        "WHERE kind = 'earnings' AND event_date BETWEEN ? AND ? "
        "ORDER BY event_date",
        [as_of, end],
    ).fetchall()
    if not rows:
        return "## Upcoming earnings\n- None known in next 45 days"
    lines = ["## Upcoming earnings (next 45 days)"]
    for sym, d in rows:
        days_away = (d - as_of).days
        lines.append(f"- **{sym}**  {d} ({days_away} days)")
    return "\n".join(lines)


def _format_per_ticker_calendar(
    con, universe: list[str], as_of: date
) -> tuple[str, list[str]]:
    """For each equity ticker, note earnings proximity. Returns (section, notable)."""
    notable: list[str] = []
    interesting_lines: list[str] = []
    for sym in universe:
        feat = compute_ticker_calendar_features(con, sym, as_of)
        if feat.days_to_next_earnings is None:
            continue  # likely an ETF
        if feat.is_earnings_week:
            notable.append(f"{sym} reports earnings THIS WEEK")
            interesting_lines.append(
                f"- **{sym}**: earnings in {feat.days_to_next_earnings}d (THIS WEEK)"
            )
        elif feat.days_to_next_earnings <= 10:
            interesting_lines.append(
                f"- **{sym}**: earnings in {feat.days_to_next_earnings}d"
            )
    section = ""
    if interesting_lines:
        section = "## Per-ticker earnings proximity\n" + "\n".join(interesting_lines)
    return section, notable


# ---------------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------------


def build_snapshot(
    con: duckdb.DuckDBPyConnection,
    universe: list[str],
    as_of: Optional[date] = None,
) -> DailySnapshot:
    """Build the full daily snapshot. Returns a DailySnapshot containing the
    markdown the LLM gets, plus metadata."""
    as_of = as_of or date.today()

    # 1. Macro regime — pass prior regime for VIX hysteresis (avoids flicker)
    features = extract_features(con, as_of)
    prior = get_prior_regime(con, before=as_of)
    assessment = classify(features, prior_regime=prior)
    regime = assessment.regime.value
    regime_desc = REGIME_DESCRIPTIONS.get(assessment.regime, "")
    reasoning = " | ".join(assessment.reasoning) if assessment.reasoning else "—"

    # 2. Market calendar
    market_cal = compute_market_calendar_features(con, as_of)

    # 2b. Geopolitical state (GDELT-derived)
    geo_features = compute_geopolitical_features(con, as_of)

    # 3. Instrument names for nicer formatting
    inst_rows = con.execute(
        "SELECT symbol, name FROM instruments"
    ).fetchall()
    instrument_lookup = {sym: name for sym, name in inst_rows}

    # 4. Per-ticker earnings notes
    per_ticker_section, notable_earnings = _format_per_ticker_calendar(
        con, universe, as_of
    )

    # 5. Build the markdown document
    parts = [
        f"# Daily market snapshot — {as_of}",
        "",
        "## Current regime",
        f"- **{regime.upper()}** (confidence {assessment.confidence:.2f})",
        f"- {regime_desc}",
        f"- Classifier reasoning: {reasoning}",
        "",
        _format_macro_section(features),
        "",
        _format_calendar_section(market_cal),
        "",
        _format_geopolitical_section(geo_features),
        "",
        _format_cross_asset_section(con, as_of),
        "",
        _format_universe_section(con, universe, as_of, instrument_lookup),
    ]
    technicals_section = _format_technicals_section(con, universe, as_of)
    if technicals_section:
        parts.extend(["", technicals_section])
    if per_ticker_section:
        parts.extend(["", per_ticker_section])
    parts.extend(["", _format_upcoming_earnings(con, as_of)])

    # Self-learning feedback: the model's open book + scored track record.
    # Empty (and omitted) until paper trades exist and mature.
    from alpha_engine.llm.feedback import format_feedback_sections

    feedback = format_feedback_sections(con, as_of)
    if feedback:
        parts.extend(["", feedback])

    notable: list[str] = list(notable_earnings)
    if market_cal.is_fomc_week:
        notable.append("FOMC meeting this week")
    if market_cal.is_quad_witching_week:
        notable.append("Quad witching this week")
    if market_cal.is_cpi_week:
        notable.append("CPI release this week")
    if market_cal.is_jobs_report_week:
        notable.append("Non-farm payrolls this week")
    for sig in geo_features.high_intensity_signals:
        notable.append(f"Geopolitical HIGH: {sig}")

    markdown = "\n".join(parts)
    log.info(
        "snapshot_built",
        as_of=str(as_of),
        regime=regime,
        universe_size=len(universe),
        markdown_chars=len(markdown),
        notable=notable,
    )
    return DailySnapshot(
        as_of=as_of,
        markdown=markdown,
        regime_label=regime,
        regime_confidence=assessment.confidence,
        universe=universe,
        notable_events=notable,
    )
