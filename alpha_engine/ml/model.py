"""Scoring models for the ML signal layer.

Two models, deliberately spanning the complexity spectrum:

  MomentumComposite — equal-weight blend of the three momentum z-scores.
      Zero trained parameters, zero tuning surface. This is the robust
      baseline: if XGBoost can't beat it out-of-sample, ship this one.

  WalkForwardXGB — gradient-boosted trees predicting cross-sectionally
      z-scored forward returns. Retrains itself from scratch using only
      data observable at the as-of date (label embargo included), so it
      can sit inside a walk-forward backtest without leaking.

Both expose `.score_cross_section(features) -> pd.Series` (higher = more
attractive) so the advisor and the daily signal script treat them
identically.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd

from alpha_engine.core.logging import get_logger
from alpha_engine.ml.features import (
    FEATURE_COLUMNS,
    compute_features,
    forward_returns,
    zscore_cross_sectional,
)

log = get_logger(__name__)

# Action bucket thresholds — top/bottom quintile, the standard academic
# portfolio construction for cross-sectional signals. Not tuned.
BUY_PERCENTILE = 0.80
AVOID_PERCENTILE = 0.20

ACTION_BUY = "BUY"
ACTION_HOLD = "HOLD"
ACTION_AVOID = "AVOID"


def assign_actions(scores: pd.Series) -> pd.Series:
    """Bucket a cross-section of scores into BUY / HOLD / AVOID.

    Top quintile = BUY, bottom quintile = AVOID, middle = HOLD. NaN scores
    (insufficient history) get no action and are dropped by callers.
    """
    valid = scores.dropna()
    out = pd.Series(index=scores.index, dtype=object)
    if valid.empty:
        return out
    ranks = valid.rank(pct=True)  # 0..1, higher = better score
    out.loc[valid.index] = ACTION_HOLD
    out.loc[ranks[ranks > BUY_PERCENTILE].index] = ACTION_BUY
    out.loc[ranks[ranks <= AVOID_PERCENTILE].index] = ACTION_AVOID
    return out


class MomentumComposite:
    """Equal-weight blend of 12-1, 6-1, and 3-1 momentum z-scores.

    Equal weights are a choice of NO choice — any other weighting implies
    a tuning decision we'd have to validate, and the walk-forward SMA sweep
    already showed that picking parameters on our history destroys value.
    """

    version = "ml-momentum-v1"

    def score_cross_section(self, features: pd.DataFrame) -> pd.Series:
        z = zscore_cross_sectional(features)
        score = z[["mom_12_1", "mom_6_1", "mom_3_1"]].mean(axis=1)
        # Require all three legs — a symbol missing any momentum leg has
        # NaN features across the board anyway (see compute_features).
        score[features[["mom_12_1", "mom_6_1", "mom_3_1"]].isna().any(axis=1)] = np.nan
        return score


@dataclass
class WalkForwardXGB:
    """XGBoost regressor with built-in point-in-time retraining.

    On every call to `maybe_retrain(prices_up_to_asof)`, retrains if the
    model is stale (older than `retrain_every` trading days) using:

      - feature rows sampled every `sample_every` trading days (weekly by
        default — daily rows are 95% redundant and 5x the fit time)
      - labels = forward `horizon`-day returns, z-scored per date so the
        model learns cross-sectional ordering, not market direction
      - an embargo: the last `horizon` days before as-of are excluded
        because their labels would peek past as-of

    Hyperparameters are small-and-conservative defaults. They are NOT to
    be tuned on our history (see FOLLOWUPS: "Parameter sweep on SMA window
    doesn't survive walk-forward").
    """

    horizon: int = 21
    retrain_every: int = 63        # trading days (~quarterly)
    sample_every: int = 5          # weekly feature rows
    min_train_rows: int = 500
    version: str = "ml-xgb-v1"

    _model: Optional[object] = field(default=None, repr=False)
    _last_train_idx: int = -10_000  # index into the price panel at last fit

    def maybe_retrain(self, prices: pd.DataFrame) -> bool:
        """Retrain if stale. `prices` must already be sliced to <= as-of.
        Returns True if a fit happened."""
        n = len(prices)
        if n - self._last_train_idx < self.retrain_every and self._model is not None:
            return False

        X, y = self._build_training_set(prices)
        if len(X) < self.min_train_rows:
            log.info("xgb_skip_train", rows=len(X), needed=self.min_train_rows)
            return False

        from xgboost import XGBRegressor

        model = XGBRegressor(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            n_jobs=4,
            random_state=42,
            verbosity=0,
        )
        model.fit(X, y)
        self._model = model
        self._last_train_idx = n
        log.info("xgb_trained", rows=len(X), as_of=str(prices.index[-1].date()))
        return True

    def _build_training_set(
        self, prices: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Stack (features, z-scored forward return) over sampled past dates.

        The label for date D needs prices through D+horizon, so the latest
        usable D is len(prices) - horizon - 1: an automatic embargo that
        keeps labels strictly inside the observable window.
        """
        fwd = forward_returns(prices, self.horizon)
        rows_X: list[pd.DataFrame] = []
        rows_y: list[pd.Series] = []

        last_usable = len(prices) - self.horizon - 1
        # Walk backward from the most recent usable date so recent regimes
        # are always included regardless of panel length.
        for i in range(last_usable, 260, -self.sample_every):
            feats = compute_features(prices.iloc[: i + 1])
            label = fwd.iloc[i]
            # Cross-sectional z-score of the label: learn ordering, not beta
            mu, sd = label.mean(), label.std()
            if not np.isfinite(sd) or sd == 0:
                continue
            label_z = (label - mu) / sd
            both = feats.join(label_z.rename("y")).dropna()
            if len(both) < 5:
                continue  # cross-section too thin to teach ordering
            rows_X.append(both[FEATURE_COLUMNS])
            rows_y.append(both["y"])

        if not rows_X:
            return pd.DataFrame(columns=FEATURE_COLUMNS), pd.Series(dtype=float)
        return pd.concat(rows_X), pd.concat(rows_y)

    def score_cross_section(self, features: pd.DataFrame) -> pd.Series:
        if self._model is None:
            # Untrained (not enough history yet) — defer to the rule
            # composite so early backtest windows aren't random.
            return MomentumComposite().score_cross_section(features)
        valid = features.dropna()
        out = pd.Series(np.nan, index=features.index)
        if valid.empty:
            return out
        preds = self._model.predict(valid[FEATURE_COLUMNS])
        out.loc[valid.index] = preds
        return out

    def training_date_bounds(
        self, prices: pd.DataFrame
    ) -> tuple[Optional[date], Optional[date]]:
        """The (first, last) feature dates a fit on `prices` would use.
        Exposed for the no-look-ahead unit test."""
        last_usable = len(prices) - self.horizon - 1
        if last_usable <= 260:
            return None, None
        first_idx = last_usable - ((last_usable - 261) // self.sample_every) * self.sample_every
        return prices.index[first_idx].date(), prices.index[last_usable].date()
