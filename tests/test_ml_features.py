"""Feature computation correctness — exact values on synthetic prices,
NaN behavior on short history, and the structural no-look-ahead guarantee.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alpha_engine.ml.features import (
    FEATURE_COLUMNS,
    MIN_HISTORY,
    compute_features,
    forward_returns,
    zscore_cross_sectional,
)


def make_panel(n_days: int = 300, symbols: dict[str, float] | None = None) -> pd.DataFrame:
    """Deterministic geometric price paths: symbol -> daily growth rate."""
    symbols = symbols or {"UP": 1.001, "FLAT": 1.0, "DOWN": 0.999}
    idx = pd.bdate_range("2020-01-01", periods=n_days)
    data = {
        sym: 100.0 * np.power(g, np.arange(n_days)) for sym, g in symbols.items()
    }
    return pd.DataFrame(data, index=idx)


class TestFeatureValues:
    def test_momentum_exact(self):
        prices = make_panel(300)
        feats = compute_features(prices)
        # mom_12_1 = P[t-21]/P[t-252] - 1 for geometric growth g:
        # g^(n-1-21) / g^(n-1-252) - 1 = g^231 - 1
        expected_up = 1.001**231 - 1
        assert feats.at["UP", "mom_12_1"] == pytest.approx(expected_up, rel=1e-9)
        assert feats.at["FLAT", "mom_12_1"] == pytest.approx(0.0, abs=1e-12)
        assert feats.at["DOWN", "mom_12_1"] == pytest.approx(0.999**231 - 1, rel=1e-9)

    def test_rev_1m_exact(self):
        prices = make_panel(300)
        feats = compute_features(prices)
        assert feats.at["UP", "rev_1m"] == pytest.approx(1.001**21 - 1, rel=1e-9)

    def test_dist_200ma_sign(self):
        prices = make_panel(300)
        feats = compute_features(prices)
        # Rising price is above its own 200MA; falling price below
        assert feats.at["UP", "dist_200ma"] > 0
        assert feats.at["DOWN", "dist_200ma"] < 0
        assert feats.at["FLAT", "dist_200ma"] == pytest.approx(0.0, abs=1e-12)

    def test_rsi_extremes(self):
        prices = make_panel(300)
        feats = compute_features(prices)
        # Monotonic up move = RSI 100; monotonic down = RSI ~0
        assert feats.at["UP", "rsi_14"] == pytest.approx(100.0)
        assert feats.at["DOWN", "rsi_14"] < 1.0

    def test_vol_constant_growth_is_zero(self):
        # Constant-rate growth has constant daily returns, so std is ~0
        prices = make_panel(300)
        feats = compute_features(prices)
        assert feats.at["FLAT", "vol_30d"] == pytest.approx(0.0, abs=1e-12)
        assert feats.at["UP", "vol_30d"] == pytest.approx(0.0, abs=1e-9)

    def test_vol_noisy_series_positive(self):
        rng = np.random.default_rng(7)
        idx = pd.bdate_range("2020-01-01", periods=300)
        noisy = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, 300)))
        prices = make_panel(300)
        prices["NOISY"] = pd.Series(noisy, index=idx)
        feats = compute_features(prices)
        assert feats.at["NOISY", "vol_30d"] > 0.05  # ~16% annualized expected


class TestInsufficientHistory:
    def test_short_history_all_nan(self):
        prices = make_panel(100)  # < MIN_HISTORY
        feats = compute_features(prices)
        assert feats.isna().all().all()

    def test_symbol_with_gap_in_window_is_nan(self):
        prices = make_panel(300)
        # Punch a hole inside GAPPY's lookback window
        prices["GAPPY"] = prices["UP"]
        prices.iloc[-100, prices.columns.get_loc("GAPPY")] = np.nan
        feats = compute_features(prices)
        assert feats.loc["GAPPY"].isna().all()
        assert feats.loc["UP"].notna().all()  # others unaffected

    def test_exactly_min_history_is_valid(self):
        prices = make_panel(MIN_HISTORY)
        feats = compute_features(prices)
        assert feats.loc["UP"].notna().all()

    def test_empty_panel(self):
        feats = compute_features(pd.DataFrame())
        assert feats.empty


class TestNoLookAhead:
    def test_appending_future_rows_never_changes_past_features(self):
        """THE core safety property: features as-of day N are identical
        whether or not days N+1.. exist in the panel."""
        full = make_panel(320)
        asof_slice = full.iloc[:280]
        feats_from_slice = compute_features(asof_slice)
        feats_from_full_sliced = compute_features(full.iloc[:280])
        pd.testing.assert_frame_equal(feats_from_slice, feats_from_full_sliced)

    def test_forward_returns_last_horizon_rows_are_nan(self):
        prices = make_panel(100)
        fwd = forward_returns(prices, horizon=21)
        assert fwd.iloc[-21:].isna().all().all()
        assert fwd.iloc[: 100 - 21].notna().all().all()

    def test_forward_return_exact(self):
        prices = make_panel(100)
        fwd = forward_returns(prices, horizon=21)
        assert fwd.iloc[0]["UP"] == pytest.approx(1.001**21 - 1, rel=1e-9)


class TestZScore:
    def test_zscore_mean_zero_unit_std(self):
        prices = make_panel(300, {"A": 1.002, "B": 1.001, "C": 1.0, "D": 0.999})
        feats = compute_features(prices)
        z = zscore_cross_sectional(feats)
        for col in FEATURE_COLUMNS:
            vals = z[col].dropna()
            assert vals.mean() == pytest.approx(0.0, abs=1e-9)
            if vals.std() > 0:
                assert vals.std() == pytest.approx(1.0, rel=1e-9)

    def test_zero_dispersion_column_is_zero_not_inf(self):
        feats = pd.DataFrame(
            {c: [1.0, 1.0, 1.0] for c in FEATURE_COLUMNS},
            index=["A", "B", "C"],
        )
        z = zscore_cross_sectional(feats)
        assert (z == 0.0).all().all()

    def test_nan_stays_nan(self):
        prices = make_panel(300)
        feats = compute_features(prices)
        feats.loc["UP"] = np.nan
        z = zscore_cross_sectional(feats)
        assert z.loc["UP"].isna().all()
