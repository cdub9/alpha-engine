"""Self-learning feedback sections for the daily snapshot.

The scorer grades every paper trade after its horizon elapses, but until
now the LLM never saw its own results — every digest was generated as if
it had no history. These sections close the loop:

  1. "Your current open paper positions" — the model's actual book, so
     add/reduce/exit/hold refer to real state instead of being stateless
     guesses, and positions nearing horizon or stop get conscious exits.
  2. "Your track record" — conviction-bucket calibration per channel,
     the most recent scored trades (with stop-outs flagged), and symbols
     with repeated hits/misses. This is what lets conviction become
     *calibrated*: if 8+ picks have been losing, the model can see that
     and tighten its own scale.

Everything is computed point-in-time from `as_of`:
  - a trade counts as SCORED only if its logical completion date
    (entry + days_held) is on or before as_of — not when the scorer
    happened to run (evaluated_at), which would leak future knowledge
    into historical snapshot generation;
  - open-position MTM uses the latest bar <= as_of, never today's.

The loop is automatic: nightly run scores yesterday's matured trades →
next snapshot includes them → next digest adjusts. No manual step.
"""

from __future__ import annotations

from datetime import date

import duckdb

from alpha_engine.core.logging import get_logger

log = get_logger(__name__)

# Keep the sections compact: the model needs the signal, not a data dump.
MAX_OPEN_ROWS = 30
MAX_RECENT_SCORED = 12
MAX_SYMBOL_LESSONS = 5
MIN_TRADES_FOR_LESSON = 2
LESSON_ALPHA_THRESHOLD = 0.02  # ±2% avg alpha to count as a repeat hit/miss


def _direction_sign(side: str) -> float:
    return -1.0 if (side or "").lower() == "short" else 1.0


def format_open_positions_section(
    con: duckdb.DuckDBPyConnection, as_of: date
) -> str:
    """The model's current paper book, MTM'd point-in-time."""
    # Dedupe to one row per (channel, symbol, side), keeping the most
    # recent entry. The trades table mixes daily forward runs with the
    # monthly historical backfill, so the same name can be "open" several
    # times — but what the model needs is its current unique exposure,
    # not the bookkeeping history.
    rows = con.execute(
        """
        WITH px AS (
            SELECT symbol, arg_max(adj_close, bar_date) AS cur_px
            FROM market_bars WHERE bar_date <= ? GROUP BY symbol
        ),
        open_trades AS (
            SELECT t.*,
                   ROW_NUMBER() OVER (
                       PARTITION BY t.channel, t.symbol, t.side
                       ORDER BY t.placed_at DESC, t.id DESC
                   ) AS rn
            FROM trades t
            LEFT JOIN trade_outcomes o ON o.trade_id = t.id
            WHERE t.status = 'paper_filled'
              AND t.placed_at::DATE <= ?
              AND (o.trade_id IS NULL
                   OR (t.placed_at::DATE + o.days_held * INTERVAL 1 DAY) > ?)
        )
        SELECT
            t.channel, t.symbol, t.side, t.direction,
            t.placed_at::DATE AS entry_date,
            t.price AS entry_px,
            px.cur_px,
            date_diff('day', t.placed_at::DATE, ?) AS days_held,
            s.time_horizon_days, s.conviction, s.stop_loss_pct
        FROM open_trades t
        LEFT JOIN signals s ON s.id = t.source_signal_id
        LEFT JOIN px ON px.symbol = t.symbol
        WHERE t.rn = 1
        ORDER BY t.channel, t.placed_at
        """,
        [as_of, as_of, as_of, as_of],
    ).fetchall()
    if not rows:
        return ""

    lines = ["## Your current open paper positions"]
    lines.append(
        f"{len(rows)} unique positions (latest entry per name shown). "
        "add/reduce/exit/hold suggestions apply to THIS book. Flag exits "
        "for positions past horizon or near their stop."
    )
    for (channel, sym, side, direction, entry_d, entry_px, cur_px,
         days_held, horizon, conv, stop) in rows[:MAX_OPEN_ROWS]:
        mtm = ""
        if cur_px is not None and entry_px:
            unreal = _direction_sign(side) * (cur_px - entry_px) / entry_px
            mtm = f"MTM {unreal:+.1%}"
        horizon_part = (
            f"{int(days_held)}d held of {int(horizon)}d"
            + (" (PAST HORIZON)" if horizon is not None and days_held > horizon else "")
            if horizon is not None
            else f"{int(days_held)}d held"
        )
        conv_part = f"conv {conv:.1f}" if conv is not None else ""
        side_label = (side or "long").upper()
        lines.append(
            f"- ({channel}) {side_label} **{sym}**: entry {entry_d}, "
            f"{horizon_part}, {mtm} {conv_part}".rstrip()
        )
    # Overflow: the model must still see its FULL book (or it may re-buy
    # names it already holds) — compact ticker list per channel for the rest.
    if len(rows) > MAX_OPEN_ROWS:
        rest = rows[MAX_OPEN_ROWS:]
        by_channel: dict[str, list[str]] = {}
        for r in rest:
            label = f"{r[1]}(s)" if (r[2] or "").lower() == "short" else r[1]
            by_channel.setdefault(r[0], []).append(label)
        for channel, syms in by_channel.items():
            lines.append(
                f"- also open ({channel}, details omitted): {', '.join(syms)}"
            )
    return "\n".join(lines)


def format_track_record_section(
    con: duckdb.DuckDBPyConnection, as_of: date
) -> str:
    """Conviction calibration + recent scored trades + per-symbol lessons."""
    # Only trades whose horizon completed by as_of count as scored —
    # evaluated_at is when the scorer ran, which can postdate completion
    # by months for backfilled history.
    completed = """
        FROM trades t
        JOIN trade_outcomes o ON o.trade_id = t.id
        LEFT JOIN signals s ON s.id = t.source_signal_id
        WHERE t.status = 'paper_filled'
          AND (t.placed_at::DATE + o.days_held * INTERVAL 1 DAY) <= ?
    """

    buckets = con.execute(
        f"""
        SELECT t.channel,
               CASE WHEN s.conviction >= 8 THEN '8.0+'
                    WHEN s.conviction >= 7 THEN '7.0-7.9'
                    ELSE '<7.0' END AS bucket,
               COUNT(*) AS n,
               AVG(CASE WHEN o.return_pct > 0 THEN 1.0 ELSE 0.0 END) AS win_rate,
               AVG(o.alpha) AS avg_alpha
        {completed}
          AND s.conviction IS NOT NULL
        GROUP BY 1, 2 ORDER BY 1, 2 DESC
        """,
        [as_of],
    ).fetchall()
    if not buckets:
        return ""

    lines = ["## Your track record (scored paper trades)"]
    lines.append(
        "Win rate = share of scored trades with positive direction-adjusted "
        "return. Alpha = vs SPY over the same window. Buckets under ~10 "
        "trades are noise — update beliefs gradually."
    )
    cur_channel = None
    for channel, bucket, n, win, alpha in buckets:
        if channel != cur_channel:
            lines.append(f"**{channel}** by conviction:")
            cur_channel = channel
        lines.append(
            f"- conv {bucket}: {n} scored, {win:.0%} win, avg alpha {alpha:+.1%}"
        )

    recent = con.execute(
        f"""
        SELECT (t.placed_at::DATE + o.days_held * INTERVAL 1 DAY)::DATE AS done,
               t.channel, t.direction, t.symbol, s.conviction,
               o.days_held, o.return_pct, o.alpha, o.notes
        {completed}
        ORDER BY done DESC, t.id DESC
        LIMIT {MAX_RECENT_SCORED}
        """,
        [as_of],
    ).fetchall()
    if recent:
        lines.append("")
        lines.append(f"Most recent {len(recent)} scored:")
        for done, channel, direction, sym, conv, days, ret, alpha, notes in recent:
            stop_flag = "  STOPPED OUT" if notes and "stop" in notes.lower() else ""
            conv_part = f" conv {conv:.1f}" if conv is not None else ""
            mark = "✓" if ret > 0 else "✗"
            lines.append(
                f"- {done} ({channel}) {direction.upper()} {sym}{conv_part} "
                f"{days}d → ret {ret:+.1%}, alpha {alpha:+.1%} {mark}{stop_flag}"
            )

    lessons = con.execute(
        f"""
        SELECT t.symbol, COUNT(*) AS n, AVG(o.alpha) AS avg_alpha
        {completed}
        GROUP BY t.symbol
        HAVING COUNT(*) >= {MIN_TRADES_FOR_LESSON}
        """,
        [as_of],
    ).fetchall()
    misses = sorted(
        [r for r in lessons if r[2] <= -LESSON_ALPHA_THRESHOLD], key=lambda r: r[2]
    )[:MAX_SYMBOL_LESSONS]
    hits = sorted(
        [r for r in lessons if r[2] >= LESSON_ALPHA_THRESHOLD], key=lambda r: -r[2]
    )[:MAX_SYMBOL_LESSONS]
    if misses:
        lines.append("")
        lines.append(
            "Repeated misses (your calls on these have lost vs SPY — "
            "demand a stronger thesis before the next one): "
            + ", ".join(f"{s} ({n} trades, avg alpha {a:+.1%})" for s, n, a in misses)
        )
    if hits:
        lines.append(
            "Repeated hits: "
            + ", ".join(f"{s} ({n} trades, avg alpha {a:+.1%})" for s, n, a in hits)
        )
    return "\n".join(lines)


def format_feedback_sections(
    con: duckdb.DuckDBPyConnection, as_of: date
) -> str:
    """Both feedback sections, or "" when there's no history yet (first
    runs, fresh DBs, early backtest dates) — the snapshot simply omits
    them rather than showing empty scaffolding."""
    parts = [
        format_open_positions_section(con, as_of),
        format_track_record_section(con, as_of),
    ]
    parts = [p for p in parts if p]
    if not parts:
        return ""
    out = "\n\n".join(parts)
    log.info("feedback_sections_built", as_of=str(as_of), chars=len(out))
    return out
