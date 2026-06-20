"""Paper trader: convert LLM signals into tracked paper trades.

This module is **completely free** — no API calls. It walks the existing
`signals` table and opens corresponding rows in `trades` with
status='paper_filled', so the downstream scorer can grade outcomes once
each trade's time horizon elapses.

**Direction → trade type:**
  - buy / add  → long, full size (quantity=1.0), side='long'
  - sell / exit → short, full size (quantity=1.0), side='short'
  - reduce     → short, half size (quantity=0.5), side='short' (partial-exit semantics)
  - hold       → skipped (no change in position)

Stored on `trades` as side ∈ {long, short}; the scorer direction-adjusts
returns based on the original `direction` column. Quantity is for
analytics, not capital simulation.

Idempotent: re-running won't duplicate trades, because every paper trade
points back to its source signal id and we skip already-processed ones.

Backfill path: if you have cached digests in `llm_signal_cache` but never
persisted their signals (e.g. because the backtest generation used
`persist=False`), call `backfill_signals_from_cache()` first to populate
the `signals` table with historical generated_at timestamps. Then
`open_paper_trades_for_date()` for each cached date opens the
corresponding paper trades.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from typing import Optional

import duckdb

from alpha_engine.backtest.llm_advisor import (
    DEFAULT_MODEL_VERSION,
    config_hash,
)
from alpha_engine.core.logging import get_logger
from alpha_engine.core.types import InstrumentType, TradeStatus
from alpha_engine.llm.parser import persist_signals
from alpha_engine.llm.prompts import SYSTEM_PROMPT

log = get_logger(__name__)


# Direction → (side, quantity). None = not actionable (skip).
# Held positions are skipped because they represent "no change."
_DIRECTION_TO_TRADE: dict[str, tuple[str, float] | None] = {
    "buy":    ("long", 1.0),
    "add":    ("long", 1.0),
    "sell":   ("short", 1.0),
    "exit":   ("short", 1.0),
    "reduce": ("short", 0.5),
    "hold":   None,
}

DEFAULT_TIME_HORIZON_DAYS = 30
DEFAULT_PAPER_CONVICTION_THRESHOLD = 6.0


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass
class OpenResult:
    """Summary of one `open_paper_trades_for_date` call."""

    digest_date: date
    signals_seen: int = 0
    paper_trades_opened: int = 0
    skipped_non_actionable: int = 0
    skipped_below_conviction: int = 0
    skipped_no_entry_price: int = 0
    skipped_already_opened: int = 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


# Entry-timing convention. The digest is generated the evening of day D
# (after D's close, on D's data), so the earliest HONEST fill is the next
# session's open — a market-on-open order placed overnight. Entering at the
# next close instead (the legacy behavior) throws away a full trading
# session of latency on momentum/news-driven picks for no reason.
ENTRY_STYLE = "next_open"


def _next_entry_prices(
    con: duckdb.DuckDBPyConnection,
    symbol: str,
    after: date,
    max_lookahead_days: int = 7,
) -> Optional[tuple[date, float, float]]:
    """Return (first trading day > after, adjusted_open, adjusted_close) for
    `symbol`, or None if no bar within max_lookahead_days.

    The opening print is put on the same split/dividend-adjusted scale as
    the adj_close series the scorer exits against:
        adj_open = open * adj_close / close
    so an open-entry return measured against an adj_close exit is clean.
    adjusted_close is returned too — it's the legacy 'next_close' entry
    price, stored as alt_entry_price so the scorer can measure how much the
    next-open switch is worth.
    """
    upper = after + timedelta(days=max_lookahead_days)
    row = con.execute(
        """
        SELECT bar_date, open, close, adj_close FROM market_bars
        WHERE symbol = ? AND bar_date > ? AND bar_date <= ?
        ORDER BY bar_date ASC LIMIT 1
        """,
        [symbol, after, upper],
    ).fetchone()
    if not row:
        return None
    bar_date, open_, close_, adj_close = row
    adj_close_f = float(adj_close)
    if open_ and close_ and float(close_) > 0:
        adj_open = float(open_) * adj_close_f / float(close_)
    else:
        # Degenerate bar (missing open/close) — fall back to close entry so
        # we never crash; alt == canonical means zero measured gap for it.
        adj_open = adj_close_f
    return (bar_date, adj_open, adj_close_f)


# ---------------------------------------------------------------------------
# Backfill historical signals from the LLM cache
# ---------------------------------------------------------------------------


def backfill_signals_from_cache(
    con: duckdb.DuckDBPyConnection,
    model_version: str = DEFAULT_MODEL_VERSION,
    cfg_hash: Optional[str] = None,
) -> int:
    """Walk `llm_signal_cache` and persist each cached digest's signals
    into the `signals` table with `generated_at` matching the cached
    `as_of`. Idempotent (skips dates with existing signals at this
    model_version)."""
    if cfg_hash is None:
        # Use the current system prompt + the active universe to derive
        # the hash that matches the cache.
        universe = [
            r[0]
            for r in con.execute(
                "SELECT symbol FROM instruments WHERE active = TRUE ORDER BY symbol"
            ).fetchall()
        ]
        cfg_hash = config_hash(SYSTEM_PROMPT, universe)

    rows = con.execute(
        """
        SELECT as_of, output_json, universe_json
        FROM llm_signal_cache
        WHERE model_version = ? AND config_hash = ?
        ORDER BY as_of
        """,
        [model_version, cfg_hash],
    ).fetchall()

    if not rows:
        log.info("backfill_no_cached_digests", model_version=model_version)
        return 0

    total_persisted = 0
    for as_of, output_json, universe_json in rows:
        # Skip if signals already exist for this exact (date, model_version).
        # Use a range query: DuckDB's DATE() vs Python date parameter binding
        # silently mismatches in some cases, so we compare against the
        # full-day timestamp range instead.
        day_start = datetime.combine(as_of, time(0, 0), tzinfo=timezone.utc)
        day_end = day_start + timedelta(days=1)
        already = con.execute(
            """
            SELECT COUNT(*) FROM signals
            WHERE generated_at >= ? AND generated_at < ?
              AND model_version = ?
            """,
            [day_start, day_end, model_version],
        ).fetchone()[0]
        if already > 0:
            continue

        try:
            output = json.loads(output_json)
            universe = json.loads(universe_json)
        except json.JSONDecodeError:
            log.warning("backfill_corrupt_cache_row", as_of=str(as_of))
            continue

        # Stamp the generated_at at midnight UTC on the cached date
        ts = datetime.combine(as_of, time(0, 0), tzinfo=timezone.utc)
        result = persist_signals(
            con,
            primary_output=output,
            snapshot_universe=universe,
            model_version=model_version,
            generated_at=ts,
        )
        total_persisted += result.total_inserted

    log.info("backfill_signals_complete", total_persisted=total_persisted)
    return total_persisted


# ---------------------------------------------------------------------------
# Open paper trades
# ---------------------------------------------------------------------------


def open_paper_trades_for_date(
    con: duckdb.DuckDBPyConnection,
    digest_date: date,
    model_version: str = DEFAULT_MODEL_VERSION,
    min_conviction: float = DEFAULT_PAPER_CONVICTION_THRESHOLD,
) -> OpenResult:
    """For each signal generated on `digest_date` at this model_version,
    open a paper LONG trade for actionable buy/add directions where
    conviction >= min_conviction. Skips signals already turned into
    trades.

    Entry = adjusted OPEN of the first trading day strictly after
    digest_date (a realistic market-on-open fill; see ENTRY_STYLE). The
    same day's adjusted close is stored as alt_entry_price so the scorer
    can measure what the old next-close timing would have returned.
    Quantity assumes $1 NAV per trade unit (sizing is for analytics, not
    portfolio simulation — the LLM backtest covers that path).
    """
    rows = con.execute(
        """
        SELECT s.id, s.channel, s.symbol, s.instrument_type, s.direction,
               s.conviction, s.target_weight, s.time_horizon_days,
               s.stop_loss_pct, s.rationale
        FROM signals s
        LEFT JOIN trades t ON t.source_signal_id = s.id
        WHERE DATE(s.generated_at) = ?
          AND s.model_version = ?
          AND t.id IS NULL                              -- not already opened
        """,
        [digest_date, model_version],
    ).fetchall()

    result = OpenResult(digest_date=digest_date, signals_seen=len(rows))

    for (
        signal_id,
        channel,
        symbol,
        instrument_type,
        direction,
        conviction,
        target_weight,
        time_horizon_days,
        stop_loss_pct,
        rationale,
    ) in rows:
        direction_l = (direction or "").lower()
        mapping = _DIRECTION_TO_TRADE.get(direction_l)
        if mapping is None:
            result.skipped_non_actionable += 1
            continue
        side, quantity = mapping
        if (conviction or 0) < min_conviction:
            result.skipped_below_conviction += 1
            continue

        entry = _next_entry_prices(con, symbol, digest_date)
        if entry is None:
            log.warning(
                "paper_trade_no_entry_price",
                symbol=symbol,
                digest_date=str(digest_date),
            )
            result.skipped_no_entry_price += 1
            continue
        entry_date, entry_price, alt_entry_price = entry

        # Time component is cosmetic (scoring uses .date()); 13:30 UTC marks
        # the ~9:30 ET open this fill models.
        placed_at = datetime.combine(entry_date, time(13, 30), tzinfo=timezone.utc)
        con.execute(
            """
            INSERT INTO trades
                (placed_at, channel, symbol, instrument_type, side,
                 direction, quantity, price, status, source_signal_id, notes,
                 entry_style, alt_entry_price)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                placed_at,
                channel,
                symbol,
                instrument_type,
                side,
                direction_l,
                quantity,
                entry_price,
                TradeStatus.PAPER_FILLED.value,
                signal_id,
                (rationale or "")[:500],
                ENTRY_STYLE,
                alt_entry_price,
            ],
        )
        result.paper_trades_opened += 1

    log.info(
        "open_paper_trades_complete",
        digest_date=str(digest_date),
        opened=result.paper_trades_opened,
        skipped_non_actionable=result.skipped_non_actionable,
        skipped_below_conviction=result.skipped_below_conviction,
        skipped_no_entry=result.skipped_no_entry_price,
    )
    return result
