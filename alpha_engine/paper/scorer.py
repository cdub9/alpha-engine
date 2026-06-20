"""Paper scorer: grade due paper trades against actual price outcomes.

Completely free — no API calls. Walks open paper trades, finds those
whose planned exit date has passed, computes:
  - return_pct (direction-adjusted, stop-loss-aware)
  - benchmark return (SPY over same window)
  - alpha (trade return - benchmark return)
  - direction_correct
  - max_favorable_excursion (best mid-trade move in trade's favor)
  - max_adverse_excursion (worst mid-trade move against trade)

**Stop-loss modeling**: every signal carries `stop_loss_pct`. We walk the
trade's intraday bars (low for longs, high for shorts). If the stop is
ever hit, the trade exits THAT DAY at the stop price, not at the horizon
end. Without this, a trade that touched -20% intraday and recovered would
score as a winner — that's fantasy P&L.

**MFE/MAE**: same walk yields best and worst marks against entry,
direction-adjusted. Both are stored as % returns from entry.

Writes results to `trade_outcomes`. Idempotent (a trade with an existing
outcome is skipped on re-run).

A paper trade's planned exit is `entry_date + time_horizon_days` (calendar
days). If that date is in the future, the trade stays open. If it's in
the past, we find the closest trading day with a bar on or after the
planned exit and use that adj_close as the exit price.

If we can't find an exit price (e.g. ticker dropped, data gap), we mark
the trade scored with NULLs and direction_correct=False rather than
leaving it open forever.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Optional

import duckdb

from alpha_engine.core.logging import get_logger

log = get_logger(__name__)


DEFAULT_HORIZON_DAYS = 30
BENCHMARK_SYMBOL = "SPY"

# Direction → 1 if long-equivalent, -1 if short-equivalent, 0 otherwise.
# This drives both return sign and stop-loss bar direction (low vs high).
_DIRECTION_SIGN = {
    "buy": 1, "add": 1, "hold": 1,
    "sell": -1, "exit": -1, "reduce": -1,
}


@dataclass
class OutcomeResult:
    """Summary of one `score_due_paper_trades` call."""

    as_of: date
    inspected: int = 0
    scored: int = 0
    stopped_out: int = 0
    still_open: int = 0
    no_exit_price: int = 0
    no_entry_price: int = 0
    already_scored: int = 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _exit_price_on_or_after(
    con: duckdb.DuckDBPyConnection,
    symbol: str,
    target: date,
    max_lookahead_days: int = 14,
) -> Optional[tuple[date, float]]:
    """Return (first trading day with bar >= target, adj_close), or None."""
    upper = target + timedelta(days=max_lookahead_days)
    row = con.execute(
        """
        SELECT bar_date, adj_close FROM market_bars
        WHERE symbol = ? AND bar_date >= ? AND bar_date <= ?
        ORDER BY bar_date ASC LIMIT 1
        """,
        [symbol, target, upper],
    ).fetchone()
    if not row:
        return None
    return (row[0], float(row[1]))


def _price_on_or_before(
    con: duckdb.DuckDBPyConnection,
    symbol: str,
    target: date,
    max_lookback_days: int = 14,
) -> Optional[float]:
    """Return adj_close on first trading day <= target."""
    lower = target - timedelta(days=max_lookback_days)
    row = con.execute(
        """
        SELECT adj_close FROM market_bars
        WHERE symbol = ? AND bar_date <= ? AND bar_date >= ?
        ORDER BY bar_date DESC LIMIT 1
        """,
        [symbol, target, lower],
    ).fetchone()
    return float(row[0]) if row else None


def _walk_trade_window(
    con: duckdb.DuckDBPyConnection,
    symbol: str,
    entry_date: date,
    horizon_end: date,
    entry_price: float,
    direction_sign: int,
    stop_loss_pct: Optional[float],
    include_entry_day: bool = False,
) -> tuple[Optional[tuple[date, float, str]], float, float]:
    """Walk daily bars over the trade window. Returns
    ((stop_exit_date, stop_exit_price, reason) or None, mfe, mae).

    direction_sign: +1 long, -1 short. For longs, stop triggers when bar's
    `low` <= entry × (1 − stop). For shorts, when bar's `high` >= entry ×
    (1 + stop). The exit price is the stop level itself (assumes order
    fills exactly at stop — slightly optimistic vs reality where slippage
    is normal, but a defensible approximation for daily-bar simulation).

    include_entry_day: for next-open entries the position is live through
    the entry day's own session, so its high/low can trigger the stop and
    its close counts toward MFE/MAE. Next-close entries fill at the close,
    leaving no intraday on the entry day, so they start the day after.

    MFE/MAE are direction-adjusted % returns vs entry, computed from
    adj_close. We don't use intra-day high/low for MFE/MAE because
    adj_close is what we mark our open trades against elsewhere and we
    want consistent units.
    """
    if direction_sign == 0 or entry_price <= 0:
        return (None, 0.0, 0.0)

    lower_op = ">=" if include_entry_day else ">"
    bars = con.execute(
        f"""
        SELECT bar_date, high, low, adj_close, close
        FROM market_bars
        WHERE symbol = ? AND bar_date {lower_op} ? AND bar_date <= ?
        ORDER BY bar_date
        """,
        [symbol, entry_date, horizon_end],
    ).fetchall()
    if not bars:
        return (None, 0.0, 0.0)

    # Stop level — note we compare on raw OHLC (high/low) but realize on
    # the stop level as a price. Using adj_close-based stop level
    # introduces split-adjustment skew; for stop modeling purposes,
    # treating entry_price as comparable to today's high/low is the
    # simplest defensible choice given our data shape.
    stop_level: Optional[float] = None
    if stop_loss_pct is not None and stop_loss_pct > 0:
        if direction_sign > 0:
            stop_level = entry_price * (1.0 - stop_loss_pct)
        else:
            stop_level = entry_price * (1.0 + stop_loss_pct)

    mfe = 0.0  # best direction-adjusted return seen
    mae = 0.0  # worst direction-adjusted return seen
    stop_hit: Optional[tuple[date, float, str]] = None

    for bar_date, high, low, adj_close, _close in bars:
        # MFE / MAE on adj_close
        ret = direction_sign * (float(adj_close) - entry_price) / entry_price
        mfe = max(mfe, ret)
        mae = min(mae, ret)

        # Stop check (intraday, using high/low)
        if stop_level is not None and stop_hit is None:
            if direction_sign > 0 and float(low) <= stop_level:
                stop_hit = (bar_date, stop_level, "stop_loss_long")
                break
            if direction_sign < 0 and float(high) >= stop_level:
                stop_hit = (bar_date, stop_level, "stop_loss_short")
                break

    return (stop_hit, mfe, mae)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def score_due_paper_trades(
    con: duckdb.DuckDBPyConnection,
    as_of: Optional[date] = None,
    default_horizon_days: int = DEFAULT_HORIZON_DAYS,
) -> OutcomeResult:
    """Score every paper trade whose planned exit (entry + horizon) has
    elapsed by `as_of` (defaults to today) and has no outcome yet.

    For each trade we walk the entry→exit window to detect stop-outs and
    compute MFE/MAE. Stop-outs exit at the stop level on the day the stop
    was hit; non-stopped trades exit at horizon end."""
    as_of = as_of or date.today()
    result = OutcomeResult(as_of=as_of)

    rows = con.execute(
        """
        SELECT t.id, t.symbol, t.direction, t.price, t.placed_at,
               s.time_horizon_days, s.stop_loss_pct,
               t.entry_style, t.alt_entry_price
        FROM trades t
        LEFT JOIN trade_outcomes o ON o.trade_id = t.id
        LEFT JOIN signals s ON s.id = t.source_signal_id
        WHERE t.status = 'paper_filled'
          AND o.trade_id IS NULL                       -- not yet scored
        """,
    ).fetchall()

    result.inspected = len(rows)

    for (trade_id, symbol, direction, entry_price, placed_at,
         time_horizon_days, stop_loss_pct, entry_style, alt_entry_price) in rows:
        if isinstance(placed_at, datetime):
            entry_date = placed_at.date()
        else:
            entry_date = placed_at  # already a date

        horizon = int(time_horizon_days or default_horizon_days)
        planned_exit = entry_date + timedelta(days=horizon)
        if planned_exit > as_of:
            result.still_open += 1
            continue

        entry_price_f = float(entry_price)
        direction_sign = _DIRECTION_SIGN.get((direction or "").lower(), 0)

        # Walk the window for stop check + MFE/MAE. Open-entry trades are
        # live through the entry day's session, so include it.
        stop_hit, mfe, mae = _walk_trade_window(
            con, symbol, entry_date, planned_exit, entry_price_f,
            direction_sign, float(stop_loss_pct) if stop_loss_pct else None,
            include_entry_day=(entry_style == "next_open"),
        )

        # Determine actual exit
        if stop_hit is not None:
            actual_exit_date, exit_price, exit_reason = stop_hit
            days_held = (actual_exit_date - entry_date).days
            result.stopped_out += 1
            notes = f"stopped out ({exit_reason})"
        else:
            # Normal horizon exit: closest bar >= planned_exit
            exit_ = _exit_price_on_or_after(con, symbol, planned_exit)
            if exit_ is None:
                # Final attempt: use the last available bar
                last_row = con.execute(
                    "SELECT bar_date, adj_close FROM market_bars "
                    "WHERE symbol = ? AND bar_date >= ? ORDER BY bar_date DESC LIMIT 1",
                    [symbol, planned_exit],
                ).fetchone()
                if last_row:
                    exit_ = (last_row[0], float(last_row[1]))
                else:
                    log.warning(
                        "score_no_exit_price",
                        trade_id=trade_id, symbol=symbol,
                        planned_exit=str(planned_exit),
                    )
                    con.execute(
                        """
                        INSERT INTO trade_outcomes
                            (trade_id, evaluated_at, days_held, return_pct,
                             max_favorable_excursion, max_adverse_excursion,
                             benchmark_return_pct, alpha, direction_correct, notes)
                        VALUES (?, ?, 0, 0, 0, 0, 0, 0, FALSE, 'no exit price')
                        """,
                        [trade_id, datetime.now(timezone.utc)],
                    )
                    result.no_exit_price += 1
                    continue
            actual_exit_date, exit_price = exit_
            days_held = (actual_exit_date - entry_date).days
            notes = None

        # Realized return — direction-adjusted
        if direction_sign == 0:
            return_pct = 0.0
        else:
            return_pct = direction_sign * (exit_price - entry_price_f) / entry_price_f

        # SPY benchmark over the same window
        spy_entry = _price_on_or_before(con, BENCHMARK_SYMBOL, entry_date)
        spy_exit = _price_on_or_before(con, BENCHMARK_SYMBOL, actual_exit_date)
        if spy_entry is not None and spy_exit is not None and spy_entry > 0:
            bench_return = (spy_exit - spy_entry) / spy_entry
        else:
            bench_return = 0.0

        alpha = return_pct - bench_return
        direction_correct = return_pct > 0

        # If we stopped out, ensure MAE reflects the stop level at minimum
        if stop_hit is not None:
            mae = min(mae, return_pct)

        # Counterfactual: what the OTHER entry-timing style would have
        # returned over the SAME exit. For next_open trades, alt_entry_price
        # is the next CLOSE (the legacy fill); the difference
        # return_pct - alt_entry_return_pct is the realized value of having
        # entered a session earlier. (For stop-outs this is approximate —
        # a close entry would carry a slightly different stop level — but
        # stop-outs are a minority and the central tendency holds.)
        alt_entry_return_pct: Optional[float] = None
        if alt_entry_price is not None and float(alt_entry_price) > 0 and direction_sign != 0:
            alt = float(alt_entry_price)
            alt_entry_return_pct = direction_sign * (exit_price - alt) / alt

        con.execute(
            """
            INSERT INTO trade_outcomes
                (trade_id, evaluated_at, days_held, return_pct,
                 max_favorable_excursion, max_adverse_excursion,
                 benchmark_return_pct, alpha, direction_correct, notes,
                 alt_entry_return_pct)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                trade_id,
                datetime.now(timezone.utc),
                int(days_held),
                float(return_pct),
                float(mfe),
                float(mae),
                float(bench_return),
                float(alpha),
                bool(direction_correct),
                notes,
                float(alt_entry_return_pct) if alt_entry_return_pct is not None else None,
            ],
        )
        result.scored += 1

    log.info(
        "score_due_paper_trades_complete",
        as_of=str(as_of),
        scored=result.scored,
        stopped_out=result.stopped_out,
        still_open=result.still_open,
        no_exit_price=result.no_exit_price,
    )
    return result
