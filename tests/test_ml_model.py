"""Scoring models: composite ordering, action buckets, XGB walk-forward
training hygiene (embargo, retrain cadence, fallback behavior).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alpha_engine.ml.features import compute_features
from alpha_engine.ml.model import (
    ACTION_AVOID,
    ACTION_BUY,
    ACTION_HOLD,
    MomentumComposite,
    WalkForwardXGB,
    assign_actions,
)


def trending_panel(n_days: int = 600, n_syms: int = 20, seed: int = 3) -> pd.DataFrame:
    """Panel where symbol i trends at a rate proportional to i, plus noise.
    Higher index = stronger uptrend = should rank higher on momentum."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2019-01-01", periods=n_days)
    data = {}
    for i in range(n_syms):
        drift = (i - n_syms / 2) * 0.0004  # from -0.4bp to +0.4bp daily
        rets = rng.normal(drift, 0.008, n_days)
        data[f"S{i:02d}"] = 100.0 * np.exp(np.cumsum(rets))
    return pd.DataFrame(data, index=idx)


class TestMomentumComposite:
    def test_ranks_follow_trend_strength(self):
        prices = trending_panel()
        feats = compute_features(prices)
        scores = MomentumComposite().score_cross_section(feats)
        # The strongest uptrend should outrank the strongest downtrend
        assert scores["S19"] > scores["S00"]
        # Rank correlation between drift order and score should be strongly positive
        order = scores.rank()
        drift_order = pd.Series(range(20), index=[f"S{i:02d}" for i in range(20)]).rank()
        corr = order.corr(drift_order, method="spearman")
        assert corr > 0.7

    def test_nan_features_get_nan_score(self):
        prices = trending_panel(n_days=600)
        prices["NEWBIE"] = np.nan
        prices.iloc[-50:, prices.columns.get_loc("NEWBIE")] = 100.0  # only 50 days
        feats = compute_features(prices)
        scores = MomentumComposite().score_cross_section(feats)
        assert pd.isna(scores["NEWBIE"])
        assert scores.drop("NEWBIE").notna().all()


class TestAssignActions:
    def test_quintile_buckets(self):
        scores = pd.Series(np.arange(20, dtype=float), index=[f"S{i}" for i in range(20)])
        actions = assign_actions(scores)
        assert (actions.value_counts()[ACTION_BUY]) == 4      # top 20% of 20
        assert (actions.value_counts()[ACTION_AVOID]) == 4    # bottom 20%
        assert (actions.value_counts()[ACTION_HOLD]) == 12
        # Highest score is a BUY, lowest an AVOID
        assert actions["S19"] == ACTION_BUY
        assert actions["S0"] == ACTION_AVOID

    def test_nan_scores_get_no_action(self):
        scores = pd.Series([3.0, np.nan, 1.0], index=["A", "B", "C"])
        actions = assign_actions(scores)
        assert pd.isna(actions["B"])

    def test_all_nan(self):
        actions = assign_actions(pd.Series([np.nan, np.nan], index=["A", "B"]))
        assert actions.isna().all()


class TestWalkForwardXGB:
    def test_embargo_excludes_unknowable_labels(self):
        """Training features must stop `horizon` days before the as-of date —
        otherwise labels would require future prices."""
        prices = trending_panel(n_days=700)
        xgb = WalkForwardXGB()
        first, last = xgb.training_date_bounds(prices)
        as_of = prices.index[-1].date()
        gap_rows = len(prices.loc[str(last):])  # rows from last train date to as-of
        assert last < as_of
        assert gap_rows - 1 >= xgb.horizon  # at least `horizon` rows after last label date

    def test_trains_and_scores(self):
        prices = trending_panel(n_days=700)
        xgb = WalkForwardXGB(min_train_rows=200)
        trained = xgb.maybe_retrain(prices)
        assert trained
        feats = compute_features(prices)
        scores = xgb.score_cross_section(feats)
        assert scores.notna().sum() == 20
        # XGB should also learn that stronger drift = better forward rank
        drift_order = pd.Series(range(20), index=[f"S{i:02d}" for i in range(20)]).rank()
        corr = scores.rank().corr(drift_order, method="spearman")
        assert corr > 0.4

    def test_no_retrain_when_fresh(self):
        prices = trending_panel(n_days=700)
        xgb = WalkForwardXGB(min_train_rows=200)
        assert xgb.maybe_retrain(prices) is True
        # Immediately again: model is fresh, must not refit
        assert xgb.maybe_retrain(prices) is False

    def test_retrains_after_staleness_window(self):
        prices = trending_panel(n_days=800)
        xgb = WalkForwardXGB(min_train_rows=200)
        assert xgb.maybe_retrain(prices.iloc[:700]) is True
        # 63+ new trading days later → stale → refit
        assert xgb.maybe_retrain(prices.iloc[:700 + 64]) is True

    def test_untrained_falls_back_to_composite(self):
        prices = trending_panel(n_days=300)  # too short to build min_train_rows
        xgb = WalkForwardXGB(min_train_rows=100_000)  # force "never trains"
        assert xgb.maybe_retrain(prices) is False
        feats = compute_features(prices)
        scores = xgb.score_cross_section(feats)
        expected = MomentumComposite().score_cross_section(feats)
        pd.testing.assert_series_equal(scores, expected)

    def test_deterministic_given_same_data(self):
        prices = trending_panel(n_days=700)
        s1 = WalkForwardXGB(min_train_rows=200)
        s2 = WalkForwardXGB(min_train_rows=200)
        s1.maybe_retrain(prices)
        s2.maybe_retrain(prices)
        feats = compute_features(prices)
        pd.testing.assert_series_equal(
            s1.score_cross_section(feats), s2.score_cross_section(feats)
        )
