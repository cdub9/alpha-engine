"""Point-in-time feature computation for the ML signal layer.

Everything here is a pure function of a wide price panel (rows = trading
dates, columns = symbols, values = adjusted close). No DB access, no
wall-clock awareness — which makes look-ahead bugs structurally hard:
features "as of date D" are computed from a panel sliced to rows <= D by
the caller, and the unit tests assert that appending future rows never
changes a past feature value.

Feature choices are conventional academic defaults, deliberately NOT tuned
on our own history (the SMA walk-forward sweep taught us that lesson):

  mom_12_1    12-month momentum skipping the most recent month —
              P[t-21] / P[t-252] - 1. The canonical cross-sectional
              momentum signal (Jegadeesh & Titman). The skip-month avoids
              contamination from short-term reversal.
  mom_6_1     Same construction over 6 months (126 trading days).
  mom_3_1     Same over 3 months (63 trading days).
  rev_1m      Most recent month's return, P[t] / P[t-21] - 1. Short-term
              REVERSAL: high recent returns slightly predict mean-reversion,
              so models should learn a negative loading. Kept separate from
              momentum rather than blended.
  vol_30d     Annualized realized vol of the last 30 daily returns. Risk
              context + the denominator for vol-scaling.
  dist_50ma   (P[t] - SMA50) / SMA50 — medium-term trend location.
  dist_200ma  (P[t] - SMA200) / SMA200 — long-term trend location.
  rsi_14      Wilder's 14-day RSI — the only oscillator with meaningful
              empirical support; everything fancier is noise.

All features need >= 252 prior trading days; symbols with insufficient
history get NaN rows and are excluded from ranking by the caller.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Order matters: store.py persists these columns positionally.
FEATURE_COLUMNS = [
    "mom_12_1",
    "mom_6_1",
    "mom_3_1",
    "rev_1m",
    "vol_30d",
    "dist_50ma",
    "dist_200ma",
    "rsi_14",
]

# Trading-day lookbacks for the momentum legs
_SKIP = 21          # "skip the most recent month"
_LOOKBACKS = {"mom_12_1": 252, "mom_6_1": 126, "mom_3_1": 63}
MIN_HISTORY = 253   # longest lookback + 1 (need P[t-252])


def _rsi_14(prices: pd.DataFrame, window: int = 14) -> pd.Series:
    """Wilder-smoothed RSI for the LAST row of each column of `prices`."""
    delta = prices.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    # Wilder smoothing == EMA with alpha = 1/window
    avg_gain = gain.ewm(alpha=1.0 / window, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    # Work on the last row as Series (symbol-indexed) — masking the full
    # DataFrame with a symbol-indexed condition would align on dates and
    # corrupt every value.
    last = rsi.iloc[-1]
    last_gain, last_loss = avg_gain.iloc[-1], avg_loss.iloc[-1]
    # Straight-up move (no losses): RSI = 100 by convention.
    last = last.where(last_loss != 0.0, 100.0)
    # Perfectly flat (no gains AND no losses): neutral 50, not 100.
    last = last.where(~((last_gain == 0.0) & (last_loss == 0.0)), 50.0)
    return last


def compute_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute the feature row for the LAST date of `prices`.

    `prices`: wide panel of adjusted closes, rows sorted ascending by date,
    already sliced so the last row is the as-of date. Returns a DataFrame
    indexed by symbol with FEATURE_COLUMNS; symbols lacking MIN_HISTORY
    non-NaN observations come back as all-NaN rows.
    """
    if prices.empty:
        return pd.DataFrame(columns=FEATURE_COLUMNS)

    out = pd.DataFrame(index=prices.columns, columns=FEATURE_COLUMNS, dtype=float)
    n = len(prices)
    last = prices.iloc[-1]

    # Momentum legs: P[t-21] / P[t-L] - 1
    for name, lb in _LOOKBACKS.items():
        if n > lb:
            out[name] = prices.iloc[-1 - _SKIP] / prices.iloc[-1 - lb] - 1.0

    if n > _SKIP:
        out["rev_1m"] = last / prices.iloc[-1 - _SKIP] - 1.0

    if n >= 31:
        rets = prices.iloc[-31:].pct_change()
        out["vol_30d"] = rets.std() * np.sqrt(252.0)

    if n >= 50:
        sma50 = prices.iloc[-50:].mean()
        out["dist_50ma"] = (last - sma50) / sma50
    if n >= 200:
        sma200 = prices.iloc[-200:].mean()
        out["dist_200ma"] = (last - sma200) / sma200

    if n >= 15:
        # RSI needs a little history beyond the window for the EMA to settle;
        # use up to 100 days which is plenty for convergence.
        out["rsi_14"] = _rsi_14(prices.iloc[-100:])

    # A symbol is usable only if it has the full longest lookback of real
    # data (no NaN in the window) — otherwise blank the whole row so the
    # ranker drops it instead of ranking on partial information.
    valid_window = prices.iloc[-MIN_HISTORY:] if n >= MIN_HISTORY else None
    if valid_window is None:
        out.loc[:, :] = np.nan
    else:
        insufficient = valid_window.isna().any() | (n < MIN_HISTORY)
        out.loc[insufficient[insufficient].index, :] = np.nan

    return out


def zscore_cross_sectional(features: pd.DataFrame) -> pd.DataFrame:
    """Z-score each feature column across symbols (one date's cross-section).

    Symbols with NaN features stay NaN. Columns with zero dispersion come
    back as 0.0 (every symbol identical = no information, not infinities).
    """
    mu = features.mean()
    sd = features.std()
    z = (features - mu) / sd.replace(0.0, np.nan)
    return z.fillna(0.0).where(features.notna(), np.nan)


def forward_returns(prices: pd.DataFrame, horizon: int = 21) -> pd.DataFrame:
    """Forward `horizon`-trading-day returns for every (date, symbol).

    Label construction for training: row at date D holds the return from
    D to D+horizon. The last `horizon` rows are NaN (future unknown) —
    the dataset builder must drop them, and the no-look-ahead test
    asserts it does.
    """
    return prices.shift(-horizon) / prices - 1.0
