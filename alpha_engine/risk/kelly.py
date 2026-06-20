"""Kelly Criterion position sizing.

The Kelly formula gives the bet size that maximizes long-run logarithmic
wealth growth. For a discrete bet:

    f* = (bp - q) / b
        where b = net odds received on a win (e.g. 1.0 for even money),
              p = probability of winning,
              q = probability of losing = 1 - p

For continuous returns (closer to financial markets):

    f* = mean_return / variance_of_returns

**In practice, always use fractional Kelly** (typically 0.25–0.5x full
Kelly). Full Kelly is mathematically optimal under perfect information but
catastrophically volatile in practice — small estimation errors in the
edge/variance produce huge sizing errors. Quarter-Kelly is the standard
defensible default.

The returned `KellySize.fraction` is the recommended fraction of the
portfolio to allocate to this position.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class KellySize:
    """A recommended position size from the Kelly Criterion."""

    full_kelly: float       # mathematically optimal fraction
    fraction_applied: float # the fractional-Kelly multiplier used (e.g. 0.25)
    recommended: float      # full_kelly * fraction_applied, clipped to [0, max_size]
    max_size_cap: float     # the cap applied
    capped: bool            # True if recommended hit the cap
    method: str             # 'continuous' | 'discrete'


def kelly_discrete(
    win_probability: float,
    win_payoff: float,
    loss_payoff: float = 1.0,
    fraction: float = 0.25,
    max_size: float = 0.25,
) -> KellySize:
    """Kelly sizing for a discrete bet.

    Args:
        win_probability: probability of winning (0..1).
        win_payoff: amount won per unit bet on a win (b in the formula).
                    e.g. 1.5 means you get back 1.5x your stake plus stake.
        loss_payoff: amount lost per unit bet on a loss. Default 1.0 (lose stake).
        fraction: fractional Kelly multiplier (0.25 = quarter-Kelly).
        max_size: hard cap on the recommended fraction (safety).

    Returns:
        KellySize. `recommended` may be 0 if edge is negative.
    """
    if not 0 <= win_probability <= 1:
        raise ValueError(f"win_probability must be in [0, 1], got {win_probability}")
    if win_payoff <= 0 or loss_payoff <= 0:
        raise ValueError("payoffs must be positive")

    p = win_probability
    q = 1.0 - p
    b = win_payoff / loss_payoff

    full = (b * p - q) / b
    # If edge is negative, don't bet.
    full = max(full, 0.0)

    rec = full * fraction
    capped = rec > max_size
    rec = min(rec, max_size)

    return KellySize(
        full_kelly=full,
        fraction_applied=fraction,
        recommended=rec,
        max_size_cap=max_size,
        capped=capped,
        method="discrete",
    )


def kelly_continuous(
    returns: np.ndarray,
    fraction: float = 0.25,
    max_size: float = 0.25,
) -> KellySize:
    """Kelly sizing for an asset with continuous returns.

    Uses the standard continuous-Kelly approximation:
        f* = mean / variance
    where mean and variance are computed from the empirical return distribution.

    Args:
        returns: 1-D array of historical period returns (fractional).
        fraction: fractional Kelly multiplier.
        max_size: hard cap on recommended fraction.
    """
    returns = np.asarray(returns, dtype=float)
    returns = returns[~np.isnan(returns)]
    if returns.size < 30:
        raise ValueError(
            f"Need at least 30 return observations, got {returns.size}"
        )

    mu = float(np.mean(returns))
    var = float(np.var(returns, ddof=1))

    if var <= 0:
        full = 0.0
    else:
        full = mu / var
    # Don't bet on negative edge.
    full = max(full, 0.0)

    rec = full * fraction
    capped = rec > max_size
    rec = min(rec, max_size)

    return KellySize(
        full_kelly=full,
        fraction_applied=fraction,
        recommended=rec,
        max_size_cap=max_size,
        capped=capped,
        method="continuous",
    )
