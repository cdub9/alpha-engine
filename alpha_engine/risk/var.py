"""Value at Risk (VaR) and Conditional VaR (CVaR / Expected Shortfall).

Two methods supported:
  - Historical simulation: percentile of empirical return distribution. No
    distributional assumption; respects fat tails. Preferred for finance.
  - Parametric (variance-covariance): assumes normal distribution. Faster
    but understates tail risk in real markets.

CVaR (Expected Shortfall) is the average loss conditional on losses worse
than VaR. It's a coherent risk measure — VaR is not, technically — and
more informative about tail risk.

All returns and outputs are in fractional form (0.01 = 1% loss). To convert
to dollar terms at a portfolio level, multiply by portfolio_value.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class VaRResult:
    """Result of a VaR / CVaR computation."""

    method: str               # 'historical' | 'parametric'
    confidence: float         # e.g. 0.95
    horizon_days: int         # holding period
    var: float                # positive number; fractional loss
    cvar: float               # positive number; fractional loss
    sample_size: int
    portfolio_value: float | None = None

    @property
    def var_dollars(self) -> float | None:
        return self.var * self.portfolio_value if self.portfolio_value else None

    @property
    def cvar_dollars(self) -> float | None:
        return self.cvar * self.portfolio_value if self.portfolio_value else None


def historical_var_cvar(
    returns: np.ndarray,
    confidence: float = 0.95,
    horizon_days: int = 1,
    portfolio_value: float | None = None,
) -> VaRResult:
    """Compute VaR and CVaR via historical simulation.

    Args:
        returns: 1-D array of historical period returns (fractional).
        confidence: e.g. 0.95 for 95% VaR.
        horizon_days: scale 1-day VaR by sqrt(horizon) for multi-day.
        portfolio_value: optional, to compute dollar-denominated VaR.

    Returns:
        VaRResult with positive `var` and `cvar` (loss magnitudes).
    """
    if not 0 < confidence < 1:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    returns = np.asarray(returns, dtype=float)
    returns = returns[~np.isnan(returns)]
    if returns.size < 30:
        raise ValueError(
            f"Need at least 30 observations for stable historical VaR, got {returns.size}"
        )

    alpha = 1.0 - confidence
    # VaR: the alpha-quantile loss. Returns are losses when negative; flip sign.
    var_1d = -np.quantile(returns, alpha)
    # CVaR: mean loss in the worst alpha-tail.
    tail = returns[returns <= np.quantile(returns, alpha)]
    cvar_1d = -tail.mean() if tail.size > 0 else var_1d

    # Scale for multi-day horizon (square-root-of-time, assumes iid)
    scale = np.sqrt(horizon_days)
    var = max(var_1d * scale, 0.0)
    cvar = max(cvar_1d * scale, 0.0)

    return VaRResult(
        method="historical",
        confidence=confidence,
        horizon_days=horizon_days,
        var=var,
        cvar=cvar,
        sample_size=int(returns.size),
        portfolio_value=portfolio_value,
    )


def parametric_var(
    returns: np.ndarray,
    confidence: float = 0.95,
    horizon_days: int = 1,
    portfolio_value: float | None = None,
) -> VaRResult:
    """Compute VaR and CVaR via parametric (variance-covariance) method.

    Assumes returns are normally distributed. Understates tail risk in real
    markets — use historical_var_cvar by default. Useful as a sanity check.
    """
    if not 0 < confidence < 1:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    returns = np.asarray(returns, dtype=float)
    returns = returns[~np.isnan(returns)]
    if returns.size < 30:
        raise ValueError(
            f"Need at least 30 observations for parametric VaR, got {returns.size}"
        )

    mu = float(np.mean(returns))
    sigma = float(np.std(returns, ddof=1))
    z = stats.norm.ppf(1.0 - confidence)  # negative z (left tail)

    # 1-day VaR: -(mu + z*sigma); flip to positive loss
    var_1d = -(mu + z * sigma)

    # Closed-form CVaR for normal: mu - sigma * phi(z)/(1-confidence), flipped
    phi_z = stats.norm.pdf(z)
    cvar_1d = -(mu - sigma * phi_z / (1.0 - confidence))

    scale = np.sqrt(horizon_days)
    return VaRResult(
        method="parametric",
        confidence=confidence,
        horizon_days=horizon_days,
        var=max(var_1d * scale, 0.0),
        cvar=max(cvar_1d * scale, 0.0),
        sample_size=int(returns.size),
        portfolio_value=portfolio_value,
    )
