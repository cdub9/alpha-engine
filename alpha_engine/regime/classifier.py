"""Rule-based macro regime classifier.

Hierarchy (first match wins; later rules only apply if earlier ones don't):

  1. RECESSION         — Sahm Rule triggered, OR long inversion + rising
                         unemployment + elevated VIX
  2. RECOVERY          — Recently exited recession, unemployment falling
  3. LATE_CYCLE        — Yield curve inverted/flat + elevated Fed funds +
                         (elevated CPI OR unemployment at multi-year lows)
  4. EXPANSION_HIGH_VOL — Otherwise; VIX 30-day avg > 20
  5. EXPANSION_LOW_VOL  — Otherwise (default healthy expansion)
  6. UNKNOWN            — Critical inputs missing

Rationale for the rule set (each cite is the historical pattern being keyed on):
  - Sahm Rule has correctly identified every US recession since 1970 within
    months of its start (real-time recession indicator).
  - 10Y-2Y inversion has preceded every recession since 1955 by 6-18 months
    (false positive in 2022-2024 = key reason inversion alone is not recession).
  - "Late cycle" combines inversion with restrictive monetary policy and
    either tight labor or sticky inflation — the conditions typically present
    just before downturns.
  - Volatility regime is overlaid on cycle position to inform position sizing
    in downstream signals.

Returns a RegimeAssessment with regime, numeric confidence, and a list of
human-readable reasoning strings — never a black-box label. Reasoning is
stored alongside the classification so any future user can ask "why was
2024-09-15 classified as LATE_CYCLE?" and get a real answer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Optional

import duckdb

from alpha_engine.core.types import MarketRegime
from alpha_engine.regime.features import MacroFeatures


# VIX hysteresis thresholds (Tier 2 #4 fix for the 20-boundary flicker).
# Stateless threshold = 20 caused 2-week regime flips when VIX hovered.
# With prior_regime context: require VIX > 22 to flip UP to high_vol,
# < 18 to come DOWN to low_vol. Inside the 18-22 band, we stick.
VIX_FLIP_UP = 22.0     # low_vol → high_vol requires this
VIX_FLIP_DOWN = 18.0   # high_vol → low_vol requires this
VIX_DEFAULT_THRESHOLD = 20.0  # used when no prior expansion regime to be sticky to


REGIME_DESCRIPTIONS = {
    MarketRegime.EXPANSION_LOW_VOL: (
        "Healthy expansion with subdued volatility. Risk-on environment; "
        "growth/momentum signals should be weighted higher."
    ),
    MarketRegime.EXPANSION_HIGH_VOL: (
        "Expansion but with elevated realized volatility. Position sizing "
        "should be reduced; favor higher-quality names."
    ),
    MarketRegime.LATE_CYCLE: (
        "Restrictive monetary policy combined with curve inversion and/or "
        "stretched labor market. Historically precedes recession but with "
        "uncertain timing. Favor defensives over cyclicals."
    ),
    MarketRegime.RECESSION: (
        "Active recession signals (Sahm Rule or composite). Risk-off; "
        "Treasuries and defensive sectors typically outperform. Be wary "
        "of dead-cat bounces in cyclicals."
    ),
    MarketRegime.RECOVERY: (
        "Emerging from recession; unemployment improving. Historically "
        "the best risk-adjusted period for equities. Favor small caps "
        "and cyclicals."
    ),
    MarketRegime.UNKNOWN: (
        "Insufficient data to classify. Default to conservative sizing."
    ),
}


@dataclass(frozen=True)
class RegimeAssessment:
    regime: MarketRegime
    confidence: float                       # 0.0 to 1.0
    reasoning: list[str] = field(default_factory=list)
    features_snapshot: dict = field(default_factory=dict)
    model_version: str = "rule_v1"


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------


def classify(
    features: MacroFeatures,
    prior_regime: MarketRegime | None = None,
) -> RegimeAssessment:
    """Apply the rule hierarchy and return a RegimeAssessment.

    `prior_regime` enables VIX hysteresis on the expansion sub-regimes:
    when the previous classification was EXPANSION_LOW_VOL or
    EXPANSION_HIGH_VOL, we require VIX to cross 22 (going up) or 18
    (coming down) to flip rather than touching 20. This eliminates the
    2-week vol-regime whipsaws when VIX hovers near 20.

    When `prior_regime` is None or was not an expansion sub-regime, the
    classic threshold 20 applies (no hysteresis to apply). This makes
    the function backward-compatible — callers that don't yet pass a
    prior get the original behavior.
    """
    f = features
    snapshot = f.to_dict()

    # ----- Data-availability gate -------------------------------------
    critical = [f.t10y2y_latest, f.vix_avg_30d]
    if any(v is None for v in critical):
        return RegimeAssessment(
            regime=MarketRegime.UNKNOWN,
            confidence=0.0,
            reasoning=["Critical inputs (T10Y2Y, VIX) missing"],
            features_snapshot=snapshot,
        )

    # ----- Recession --------------------------------------------------
    rec = _check_recession(f)
    if rec is not None:
        return RegimeAssessment(
            regime=MarketRegime.RECESSION,
            confidence=rec[0],
            reasoning=rec[1],
            features_snapshot=snapshot,
        )

    # ----- Recovery ---------------------------------------------------
    recov = _check_recovery(f)
    if recov is not None:
        return RegimeAssessment(
            regime=MarketRegime.RECOVERY,
            confidence=recov[0],
            reasoning=recov[1],
            features_snapshot=snapshot,
        )

    # ----- Late cycle -------------------------------------------------
    late = _check_late_cycle(f)
    if late is not None:
        return RegimeAssessment(
            regime=MarketRegime.LATE_CYCLE,
            confidence=late[0],
            reasoning=late[1],
            features_snapshot=snapshot,
        )

    # ----- Expansion (high vs low vol, with hysteresis) ---------------
    # Hysteresis: if last call classified an expansion regime, require
    # VIX to cross 22 (going up) or 18 (coming down) to flip. Without
    # hysteresis the 20-threshold caused 2-week whipsaws when VIX
    # hovered. See FOLLOWUPS "Vol-regime flicker at the VIX 20 boundary."
    vix = f.vix_avg_30d
    hysteresis_applied = False
    if prior_regime == MarketRegime.EXPANSION_LOW_VOL:
        # Was low; need VIX > 22 to flip to high
        high_vol = vix is not None and vix > VIX_FLIP_UP
        hysteresis_applied = True
        threshold_str = f">{VIX_FLIP_UP:.0f} to flip up (was low)"
    elif prior_regime == MarketRegime.EXPANSION_HIGH_VOL:
        # Was high; need VIX < 18 to flip to low
        high_vol = not (vix is not None and vix < VIX_FLIP_DOWN)
        hysteresis_applied = True
        threshold_str = f"<{VIX_FLIP_DOWN:.0f} to flip down (was high)"
    else:
        # No prior expansion state; use the default 20 threshold
        high_vol = f.vix_regime_high or (vix is not None and vix > VIX_DEFAULT_THRESHOLD)
        threshold_str = f">{VIX_DEFAULT_THRESHOLD:.0f}"

    if high_vol:
        reason = f"VIX 30d avg = {vix:.1f} ({threshold_str})"
        if hysteresis_applied:
            reason += " — hysteresis"
        return RegimeAssessment(
            regime=MarketRegime.EXPANSION_HIGH_VOL,
            confidence=_vix_confidence(vix, high=True),
            reasoning=[reason],
            features_snapshot=snapshot,
        )

    vix_str = f"{vix:.1f}" if vix is not None else "n/a"
    reason = f"No recession/late-cycle signals; VIX 30d avg = {vix_str}"
    if hysteresis_applied:
        reason += f" ({threshold_str}) — hysteresis"
    return RegimeAssessment(
        regime=MarketRegime.EXPANSION_LOW_VOL,
        confidence=_vix_confidence(vix, high=False),
        reasoning=[reason],
        features_snapshot=snapshot,
    )


# ---------------------------------------------------------------------------
# Rule helpers (return (confidence, reasoning) on match, None on no-match)
# ---------------------------------------------------------------------------


def _check_recession(f: MacroFeatures) -> tuple[float, list[str]] | None:
    """Recession if Sahm triggered OR composite of long inversion +
    rising unemployment + elevated VIX."""
    reasons: list[str] = []

    # Primary: Sahm Rule
    if f.sahm_triggered:
        # Confidence scales with how far above 0.5 the Sahm value is
        s = f.sahm_latest or 0.5
        conf = min(0.95, 0.7 + (s - 0.5) * 0.5)
        reasons.append(f"Sahm Rule triggered ({s:.2f} >= 0.50)")
        return conf, reasons

    # Composite: requires all three structural conditions
    conditions_met = 0
    if f.yield_curve_inverted_long:
        conditions_met += 1
        reasons.append("Yield curve inverted >180 days in past year")
    if f.unrate_rising:
        conditions_met += 1
        reasons.append(
            f"Unemployment 3m avg ({f.unrate_3m_avg:.2f}) > 6m avg "
            f"({f.unrate_6m_avg:.2f})"
        )
    if f.vix_avg_30d is not None and f.vix_avg_30d > 28:
        conditions_met += 1
        reasons.append(f"VIX 30d avg = {f.vix_avg_30d:.1f} (>28)")

    if conditions_met >= 3:
        return 0.75, reasons

    return None


def _check_recovery(f: MacroFeatures) -> tuple[float, list[str]] | None:
    """Recovery: was recently in recession but is no longer, and labor
    market is improving."""
    if not f.sahm_recently_active:
        return None
    if f.sahm_triggered:
        return None  # still in recession
    if not f.unrate_falling:
        return None

    reasons = [
        f"Sahm Rule recently active (6m max = {f.sahm_max_6m:.2f}) but now "
        f"{f.sahm_latest:.2f}",
        f"Unemployment improving: 3m avg ({f.unrate_3m_avg:.2f}) < 6m avg "
        f"({f.unrate_6m_avg:.2f})",
    ]
    return 0.7, reasons


def _check_late_cycle(f: MacroFeatures) -> tuple[float, list[str]] | None:
    """Late cycle: curve inverted/flat + restrictive Fed + (elevated CPI
    OR unemployment near multi-year low)."""
    if f.t10y2y_avg_30d is None or f.t10y2y_avg_30d > 0.30:
        return None
    if not f.fed_funds_elevated:
        return None

    tight_labor = (
        f.unrate_latest is not None
        and f.unrate_12m_low is not None
        and (f.unrate_latest - f.unrate_12m_low) < 0.30
    )
    sticky_inflation = bool(f.cpi_elevated)

    if not (tight_labor or sticky_inflation):
        return None

    reasons = [
        f"Yield curve flat/inverted (T10Y2Y 30d avg = {f.t10y2y_avg_30d:.2f})",
        f"Fed funds elevated ({f.fed_funds_latest:.2f}, "
        f"{(f.fed_funds_percentile_5y or 0) * 100:.0f}th pct of 5y)",
    ]
    if tight_labor:
        reasons.append(
            f"Tight labor market (UR {f.unrate_latest:.2f} near 12m low "
            f"{f.unrate_12m_low:.2f})"
        )
    if sticky_inflation:
        reasons.append(f"CPI YoY = {f.cpi_yoy_pct:.1f}% (>3%)")

    # Confidence: more confirmations = more confident
    confirmations = sum([tight_labor, sticky_inflation, bool(f.yield_curve_inverted)])
    conf = 0.55 + 0.10 * confirmations
    return min(conf, 0.85), reasons


def get_prior_regime(
    con: duckdb.DuckDBPyConnection,
    before: date,
    model_version: str = "rule_v1",
) -> MarketRegime | None:
    """Look up the most recent persisted regime classification strictly
    before `before`. Returns None if no prior row exists.

    Used by callers that want hysteresis without manually tracking state.
    Live (LLM context, backtest advisor) calls use this; the batch
    classify_regimes.py script tracks state in-loop instead to handle
    fresh-start backfills correctly.
    """
    row = con.execute(
        """
        SELECT regime FROM regime_classifications
        WHERE classification_date < ? AND model_version = ?
        ORDER BY classification_date DESC LIMIT 1
        """,
        [before, model_version],
    ).fetchone()
    if not row:
        return None
    try:
        return MarketRegime(row[0])
    except ValueError:
        return None


def _vix_confidence(vix_30: float | None, high: bool) -> float:
    """Confidence for the expansion regime: stronger when VIX is firmly
    in the expected zone."""
    if vix_30 is None:
        return 0.5
    if high:
        # vix > 20 (high). Stronger when farther above 20.
        return min(0.9, 0.6 + (vix_30 - 20) * 0.02)
    # vix <= 20 (low). Stronger when farther below 16.
    return min(0.9, 0.6 + max(0.0, (16 - vix_30) * 0.04))
