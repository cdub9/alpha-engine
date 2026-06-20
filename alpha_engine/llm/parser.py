"""Parse and persist LLM-generated suggestions.

Validates each suggestion:
  - Symbol must be in the snapshot universe (no fabrication)
  - Direction must be valid (already enforced by JSON Schema, defense-in-depth)
  - Conviction in [0, 10]

Writes valid suggestions to the `signals` table with model_version set so
we can group analytics per LLM model + prompt version. Invalid
suggestions are dropped with a warning — better to drop than to write
garbage downstream signals will act on.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import duckdb

from alpha_engine.core.logging import get_logger
from alpha_engine.core.types import Channel, InstrumentType, SignalDirection

log = get_logger(__name__)


VALID_DIRECTIONS = {d.value for d in SignalDirection}


@dataclass
class PersistResult:
    inserted_a: int
    inserted_b: int
    skipped_a: int
    skipped_b: int

    @property
    def total_inserted(self) -> int:
        return self.inserted_a + self.inserted_b


def _validate_suggestion(
    suggestion: dict[str, Any], universe: set[str]
) -> tuple[bool, str]:
    """Return (is_valid, reason). Reason is empty when valid."""
    symbol = suggestion.get("symbol", "").upper().strip()
    if not symbol:
        return False, "missing symbol"
    if symbol not in universe:
        return False, f"symbol {symbol} not in universe"

    direction = suggestion.get("direction", "").lower().strip()
    if direction not in VALID_DIRECTIONS:
        return False, f"invalid direction: {direction}"

    conviction = suggestion.get("conviction")
    try:
        conviction = float(conviction)
    except (TypeError, ValueError):
        return False, "conviction not numeric"
    if not (0.0 <= conviction <= 10.0):
        return False, f"conviction out of range: {conviction}"

    rationale = suggestion.get("rationale", "")
    if not rationale or len(rationale.strip()) < 10:
        return False, "rationale missing or too short"

    return True, ""


def _instrument_type_for(con: duckdb.DuckDBPyConnection, symbol: str) -> str:
    """Look up an instrument's type; default to equity if unknown."""
    row = con.execute(
        "SELECT instrument_type FROM instruments WHERE symbol = ?",
        [symbol],
    ).fetchone()
    return row[0] if row else InstrumentType.EQUITY.value


def persist_signals(
    con: duckdb.DuckDBPyConnection,
    primary_output: dict[str, Any],
    snapshot_universe: list[str],
    # v2-ta 2026-06-11: technicals snapshot section + TA principle.
    # v3-fb 2026-06-11: feedback loop (open positions + track record in
    # snapshot, calibration principle). Grouping by this column is how
    # pre/post-change performance gets compared (see FOLLOWUPS).
    model_version: str = "llm-opus-4-7-v3-fb",
    generated_at: datetime | None = None,
) -> PersistResult:
    """Write all suggestions to the `signals` table. Returns counts of
    inserted vs skipped per channel.

    Same-day re-run safety: if signals already exist for the same
    (DATE(generated_at), model_version), they are deleted before insert.
    This makes re-runs replace rather than double-count. Note that paper
    trades linked to deleted signals will end up with NULL
    `source_signal_id` joins — in practice this is fine because we only
    re-run on the same day before any trades have been opened, and the
    `open_paper_trades_for_date` query keys on signal IDs that still
    exist.
    """
    ts = generated_at or datetime.now(timezone.utc)
    universe = {s.upper() for s in snapshot_universe}

    # Same-day dedup: clear any prior rows for this (date, model).
    day = ts.date()
    deleted = con.execute(
        """
        DELETE FROM signals
        WHERE DATE(generated_at) = ? AND model_version = ?
        RETURNING id
        """,
        [day, model_version],
    ).fetchall()
    if deleted:
        log.info(
            "signals_same_day_replaced",
            day=str(day),
            model_version=model_version,
            cleared=len(deleted),
        )

    counts = {"a_in": 0, "a_skip": 0, "b_in": 0, "b_skip": 0}
    bundled_market_summary = primary_output.get("market_summary", "")
    bundled_themes = primary_output.get("key_themes", [])
    bundled_risks = primary_output.get("risk_notes", [])

    for channel, key, in_key, skip_key in [
        (Channel.STEADY_ALPHA, "channel_a_suggestions", "a_in", "a_skip"),
        (Channel.AGGRESSIVE_GROWTH, "channel_b_suggestions", "b_in", "b_skip"),
    ]:
        raw_suggestions = list(primary_output.get(key, []))

        # Dedup within a channel by (symbol_upper, direction_lower), keeping
        # the highest-conviction entry. The LLM occasionally emits the same
        # ticker twice in one digest; persisting both double-counts the
        # decision and double-opens paper trades. Log every drop so we can
        # see how often it happens.
        deduped: dict[tuple[str, str], dict[str, Any]] = {}
        for sug in raw_suggestions:
            sym = str(sug.get("symbol", "")).upper().strip()
            direction = str(sug.get("direction", "")).lower().strip()
            if not sym or not direction:
                # Validation later will catch and log; pass through unchanged
                deduped[(sym, direction, id(sug))] = sug  # type: ignore[index]
                continue
            k = (sym, direction)
            existing = deduped.get(k)
            try:
                new_conv = float(sug.get("conviction") or 0.0)
            except (TypeError, ValueError):
                new_conv = 0.0
            if existing is None:
                deduped[k] = sug
                continue
            try:
                old_conv = float(existing.get("conviction") or 0.0)
            except (TypeError, ValueError):
                old_conv = 0.0
            if new_conv > old_conv:
                log.warning(
                    "signal_duplicate_dropped",
                    channel=channel.value,
                    symbol=sym,
                    direction=direction,
                    kept_conviction=new_conv,
                    dropped_conviction=old_conv,
                )
                deduped[k] = sug
            else:
                log.warning(
                    "signal_duplicate_dropped",
                    channel=channel.value,
                    symbol=sym,
                    direction=direction,
                    kept_conviction=old_conv,
                    dropped_conviction=new_conv,
                )

        for sug in deduped.values():
            ok, reason = _validate_suggestion(sug, universe)
            if not ok:
                log.warning(
                    "signal_skipped",
                    channel=channel.value,
                    symbol=sug.get("symbol"),
                    reason=reason,
                )
                counts[skip_key] += 1
                continue

            symbol = sug["symbol"].upper().strip()
            instrument_type = _instrument_type_for(con, symbol)

            features_snapshot = {
                "market_summary": bundled_market_summary,
                "key_themes": bundled_themes,
                "risk_notes": bundled_risks,
                "raw_suggestion": sug,
            }

            con.execute(
                """
                INSERT INTO signals (
                    generated_at, channel, symbol, instrument_type, direction,
                    conviction, target_weight, time_horizon_days, stop_loss_pct,
                    rationale, counter_argument, features_snapshot_json,
                    model_version
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    ts,
                    channel.value,
                    symbol,
                    instrument_type,
                    sug["direction"].lower(),
                    float(sug["conviction"]),
                    sug.get("target_weight"),
                    sug.get("time_horizon_days"),
                    sug.get("stop_loss_pct"),
                    sug["rationale"],
                    sug.get("counter_argument"),
                    json.dumps(features_snapshot),
                    model_version,
                ],
            )
            counts[in_key] += 1

    log.info(
        "signals_persisted",
        a_inserted=counts["a_in"],
        a_skipped=counts["a_skip"],
        b_inserted=counts["b_in"],
        b_skipped=counts["b_skip"],
    )
    return PersistResult(
        inserted_a=counts["a_in"],
        skipped_a=counts["a_skip"],
        inserted_b=counts["b_in"],
        skipped_b=counts["b_skip"],
    )
