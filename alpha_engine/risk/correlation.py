"""Correlation matrix monitoring.

During market crashes, asset correlations converge toward 1.0 — a
"diversified" portfolio suddenly behaves like a single bet. Tracking the
average pairwise correlation over a rolling window provides an early-warning
signal for regime convergence.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CorrelationReport:
    """Summary of a correlation matrix."""

    matrix: pd.DataFrame
    avg_pairwise: float            # mean of upper-triangle
    max_pairwise: float
    min_pairwise: float
    n_assets: int
    n_observations: int
    regime_warning: bool           # avg_pairwise above threshold
    threshold: float


def rolling_correlation_matrix(
    returns: pd.DataFrame,
    window: int = 60,
) -> pd.DataFrame:
    """Return the correlation matrix for the last `window` observations.

    Args:
        returns: DataFrame with columns = symbols, rows = dates, values = returns.
        window: lookback length in observations (typically trading days).

    Returns:
        Correlation matrix as a DataFrame.
    """
    if returns.shape[0] < window:
        raise ValueError(
            f"Need at least {window} observations, got {returns.shape[0]}"
        )
    tail = returns.tail(window).dropna(how="all", axis=1)
    return tail.corr()


def summarize_correlation(
    corr: pd.DataFrame,
    n_observations: int,
    threshold: float = 0.7,
) -> CorrelationReport:
    """Summarize a correlation matrix into a single report."""
    n = corr.shape[0]
    if n < 2:
        raise ValueError(f"Need at least 2 assets in correlation matrix, got {n}")

    # Upper triangle, excluding diagonal
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    values = corr.values[mask]
    values = values[~np.isnan(values)]

    avg = float(values.mean()) if values.size > 0 else 0.0
    return CorrelationReport(
        matrix=corr,
        avg_pairwise=avg,
        max_pairwise=float(values.max()) if values.size > 0 else 0.0,
        min_pairwise=float(values.min()) if values.size > 0 else 0.0,
        n_assets=n,
        n_observations=n_observations,
        regime_warning=avg >= threshold,
        threshold=threshold,
    )


def returns_from_prices(prices: pd.DataFrame, method: str = "log") -> pd.DataFrame:
    """Convert a wide-format price DataFrame to returns.

    `method`: 'log' for log returns (preferred for risk math) or 'simple' for
    arithmetic returns.
    """
    if method == "log":
        return np.log(prices / prices.shift(1)).dropna(how="all")
    elif method == "simple":
        return prices.pct_change().dropna(how="all")
    else:
        raise ValueError(f"Unknown method: {method}")
