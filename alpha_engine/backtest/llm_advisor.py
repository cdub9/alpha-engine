"""Backtest advisor that replays cached LLM digest signals.

================================  READ THIS  ================================
TRAINING-DATA CONTAMINATION WARNING

Backtesting an LLM's signals on HISTORICAL dates is fundamentally
optimistically biased. The market *snapshot* we feed the model is
point-in-time clean (only data with date <= as_of), but the model's
*weights* are not: Opus 4.7's training corpus almost certainly covers
the historical dates being tested. The model may "remember" that NVDA
ripped in 2023-24, that COVID crashed in Feb 2020, that SVB failed in
March 2023, etc.

Therefore: results from this backtest are an UPPER BOUND on skill, not a
forward-looking estimate. A strong historical result does NOT prove the
system will generate alpha live. The only clean test is paper/forward
trading on dates AFTER the model's training cutoff.

Use this backtest to:
  - Sanity-check the plumbing (weights sum sanely, no crashes)
  - Compare channel A vs channel B behavior
  - Spot obvious failure modes (e.g. always 100% cash)

Do NOT use it to:
  - Claim the LLM "beats the market"
  - Size live capital
============================================================================

The advisor reads ONLY from llm_signal_cache — it never makes API calls
during a backtest (that would be slow and non-reproducible). Pre-generate
the cache with scripts/generate_llm_history.py first. On a cache miss the
advisor returns empty weights (cash) and logs a warning.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date
from typing import Any, Optional

import duckdb

from alpha_engine.backtest.types import SignalAdvisor
from alpha_engine.core.logging import get_logger
from alpha_engine.core.types import Channel

log = get_logger(__name__)


# Version history (the string flows into signals.model_version, the digest
# cache key, and the paper trader; old rows keep their tag so cohorts stay
# separable):
#   v1     — original prompt/snapshot
#   v2-ta  — 2026-06-11: "Per-symbol technicals" snapshot section + TA
#            operating principle (cohort is ~empty; v3 shipped same day)
#   v3-fb  — 2026-06-11: self-learning feedback loop — open-positions +
#            track-record snapshot sections + calibration principle
DEFAULT_MODEL_VERSION = "llm-opus-4-7-v3-fb"

# Directions that mean "be in this position" at the given weight
_POSITIVE_DIRECTIONS = {"buy", "add", "hold"}
_REDUCE_DIRECTIONS = {"reduce"}
# sell / exit → weight 0 (omit)

_CHANNEL_KEYS = {
    Channel.STEADY_ALPHA: "channel_a_suggestions",
    Channel.AGGRESSIVE_GROWTH: "channel_b_suggestions",
}


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------


def config_hash(system_prompt: str, universe: list[str], tag: str = "v1") -> str:
    """Stable hash over the things that should invalidate the cache:
    the system prompt text, the universe, and a manual version tag."""
    h = hashlib.sha256()
    h.update(tag.encode())
    h.update(system_prompt.encode())
    h.update(",".join(sorted(universe)).encode())
    return h.hexdigest()[:16]


def get_cached_output(
    con: duckdb.DuckDBPyConnection,
    as_of: date,
    model_version: str,
    cfg_hash: str,
    tolerance_days: int = 0,
) -> Optional[dict[str, Any]]:
    """Return cached parsed digest output for a date, or None on miss.

    If tolerance_days > 0, returns the most recent digest with
    as_of <= the requested date and within `tolerance_days` calendar days
    before it. This matches live behaviour: on any decision date you use
    the latest digest available, not necessarily one stamped exactly that
    day. It also bridges the engine's look-ahead guard (which queries the
    advisor with the *previous trading day*, so an exact match against the
    generated rebalance date would always miss).
    """
    if tolerance_days <= 0:
        row = con.execute(
            """
            SELECT output_json FROM llm_signal_cache
            WHERE as_of = ? AND model_version = ? AND config_hash = ?
            """,
            [as_of, model_version, cfg_hash],
        ).fetchone()
    else:
        from datetime import timedelta

        lower = as_of - timedelta(days=tolerance_days)
        row = con.execute(
            """
            SELECT output_json FROM llm_signal_cache
            WHERE model_version = ? AND config_hash = ?
              AND as_of <= ? AND as_of >= ?
            ORDER BY as_of DESC
            LIMIT 1
            """,
            [model_version, cfg_hash, as_of, lower],
        ).fetchone()
    if not row:
        return None
    try:
        return json.loads(row[0])
    except json.JSONDecodeError:
        log.warning("llm_cache_corrupt", as_of=str(as_of))
        return None


def store_cached_output(
    con: duckdb.DuckDBPyConnection,
    as_of: date,
    model_version: str,
    cfg_hash: str,
    output: dict[str, Any],
    universe: list[str],
    input_tokens: int = 0,
    output_tokens: int = 0,
    cost_usd: float = 0.0,
) -> None:
    """Upsert a digest output into the cache."""
    con.execute(
        """
        INSERT INTO llm_signal_cache
            (as_of, model_version, config_hash, output_json, universe_json,
             input_tokens, output_tokens, cost_usd)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (as_of, model_version, config_hash) DO UPDATE SET
            output_json   = EXCLUDED.output_json,
            universe_json = EXCLUDED.universe_json,
            input_tokens  = EXCLUDED.input_tokens,
            output_tokens = EXCLUDED.output_tokens,
            cost_usd      = EXCLUDED.cost_usd,
            generated_at  = now()
        """,
        [
            as_of,
            model_version,
            cfg_hash,
            json.dumps(output),
            json.dumps(universe),
            input_tokens,
            output_tokens,
            cost_usd,
        ],
    )


# ---------------------------------------------------------------------------
# Weight conversion
# ---------------------------------------------------------------------------


def suggestions_to_weights(
    suggestions: list[dict[str, Any]],
    equal_weight_fallback: bool = True,
) -> dict[str, float]:
    """Convert a channel's suggestion list to target portfolio weights.

    Rules:
      - buy / add / hold  -> target_weight (as given)
      - reduce            -> target_weight * 0.5
      - sell / exit       -> 0 (omitted)
      - missing target_weight + equal_weight_fallback -> equal split of
        remaining budget across weight-less positive names

    The backtest engine applies the channel's max_position_weight and
    max_leverage caps afterward, so we don't enforce them here.
    """
    weights: dict[str, float] = {}
    weightless_positive: list[str] = []

    for s in suggestions:
        symbol = (s.get("symbol") or "").upper().strip()
        if not symbol:
            continue
        direction = (s.get("direction") or "").lower().strip()
        tw = s.get("target_weight")

        if direction in _POSITIVE_DIRECTIONS:
            if tw is not None and tw > 0:
                weights[symbol] = float(tw)
            elif equal_weight_fallback:
                weightless_positive.append(symbol)
        elif direction in _REDUCE_DIRECTIONS:
            if tw is not None and tw > 0:
                weights[symbol] = float(tw) * 0.5
        # sell / exit / unknown -> omit (0 weight)

    # Distribute remaining budget across weightless positive names
    if weightless_positive and equal_weight_fallback:
        used = sum(weights.values())
        remaining = max(0.0, 1.0 - used)
        if remaining > 0:
            per = remaining / len(weightless_positive)
            for sym in weightless_positive:
                weights[sym] = weights.get(sym, 0.0) + per

    return weights


# ---------------------------------------------------------------------------
# Advisor
# ---------------------------------------------------------------------------


class LLMChannelAdvisor(SignalAdvisor):
    """Replays cached LLM digest signals for one channel.

    Reads ONLY from llm_signal_cache. See module docstring for the
    training-data-contamination warning that applies to all results.
    """

    def __init__(
        self,
        channel: Channel,
        config_hash_value: str,
        model_version: str = DEFAULT_MODEL_VERSION,
        tolerance_days: int = 40,
    ) -> None:
        self.channel = channel
        self.config_hash_value = config_hash_value
        self.model_version = model_version
        # How stale a cached digest may be relative to the decision date.
        # The engine queries with the previous trading day, and digests are
        # generated on rebalance dates, so a small tolerance is required even
        # for exact-cadence generation. Set to span one rebalance interval.
        self.tolerance_days = tolerance_days
        self.name = f"llm_{channel.value}"
        self.description = (
            f"Cached LLM digest signals for {channel.value} "
            f"(model={model_version}) — CONTAMINATED, see module docstring"
        )
        self._miss_count = 0
        self._hit_count = 0

    @property
    def cache_stats(self) -> tuple[int, int]:
        """(hits, misses) accumulated during the backtest."""
        return self._hit_count, self._miss_count

    def target_weights(
        self,
        as_of: date,
        con: duckdb.DuckDBPyConnection,
        universe: list[str],
    ) -> dict[str, float]:
        output = get_cached_output(
            con,
            as_of,
            self.model_version,
            self.config_hash_value,
            tolerance_days=self.tolerance_days,
        )
        if output is None:
            self._miss_count += 1
            log.warning(
                "llm_advisor_cache_miss",
                channel=self.channel.value,
                as_of=str(as_of),
            )
            return {}  # cash

        self._hit_count += 1
        key = _CHANNEL_KEYS[self.channel]
        suggestions = output.get(key, [])
        weights = suggestions_to_weights(suggestions)

        # Drop any symbols not in the backtest universe (defense)
        universe_set = {u.upper() for u in universe}
        return {
            sym: w for sym, w in weights.items() if sym in universe_set
        }
