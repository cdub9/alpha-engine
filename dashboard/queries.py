"""All DuckDB read queries for the dashboard.

Centralized so view files stay focused on layout. Every function takes a
read-only connection; the caller is responsible for opening/closing it.

We use the project's existing `get_connection(read_only=True)`.
"""

from __future__ import annotations

import json
from datetime import date
from typing import Any, Optional

import pandas as pd

from alpha_engine.db import get_connection


# ---------------------------------------------------------------------------
# Connection helper (cached per-Streamlit-session)
# ---------------------------------------------------------------------------


def _conn():
    """Open a fresh read-only connection. Cheap; DuckDB embedded."""
    return get_connection(read_only=True)


# ---------------------------------------------------------------------------
# Suggestions (latest + by date)
# ---------------------------------------------------------------------------


def latest_digest_date() -> Optional[date]:
    with _conn() as con:
        row = con.execute(
            "SELECT MAX(as_of) FROM llm_signal_cache"
        ).fetchone()
    return row[0] if row and row[0] else None


def available_digest_dates() -> list[date]:
    with _conn() as con:
        rows = con.execute(
            "SELECT as_of FROM llm_signal_cache ORDER BY as_of DESC"
        ).fetchall()
    return [r[0] for r in rows]


def digest_meta(d: date) -> dict[str, Any]:
    with _conn() as con:
        row = con.execute(
            """
            SELECT cost_usd, input_tokens, output_tokens, model_version, generated_at
            FROM llm_signal_cache WHERE as_of = ?
            ORDER BY generated_at DESC LIMIT 1
            """,
            [d],
        ).fetchone()
    if not row:
        return {}
    return {
        "cost_usd": float(row[0] or 0),
        "input_tokens": int(row[1] or 0),
        "output_tokens": int(row[2] or 0),
        "model_version": row[3],
        "generated_at": row[4],
    }


def suggestions_for_date(d: date) -> pd.DataFrame:
    """Suggestions for digest date `d`, read from the cache's output_json.

    We read from the cache (not the signals table) because cache.as_of is
    the canonical "what was the digest for this date" — signal persistence
    timing varies (backfill stamps midnight UTC of as_of; live runs use
    now()). The cache is the single source of truth.

    Returns columns:
      channel, symbol, direction, conviction, target_weight,
      time_horizon_days, stop_loss_pct, rationale, counter_argument,
      entry_price, current_price, unrealized_pct
    """
    with _conn() as con:
        row = con.execute(
            """
            SELECT output_json FROM llm_signal_cache
            WHERE as_of = ? ORDER BY generated_at DESC LIMIT 1
            """,
            [d],
        ).fetchone()
        if not row:
            return pd.DataFrame()
        output = json.loads(row[0])

        # Latest prices for MTM
        prices = {
            r[0]: float(r[1]) for r in con.execute(
                """
                SELECT mb.symbol, mb.adj_close FROM market_bars mb
                JOIN (SELECT symbol, MAX(bar_date) AS bd FROM market_bars GROUP BY symbol) lb
                  ON lb.symbol = mb.symbol AND lb.bd = mb.bar_date
                """
            ).fetchall()
        }

        # Paper-trade entries keyed by (channel, symbol) for any trades whose
        # source signal was generated near this digest date (within +/-2 days
        # of as_of+1 trading day, which is typical entry timing).
        entries = {
            (r[0], r[1]): float(r[2]) for r in con.execute(
                """
                SELECT t.channel, t.symbol, t.price
                FROM trades t
                WHERE t.placed_at::DATE BETWEEN ? AND ?
                """,
                [d, d.replace(day=min(d.day, 28))],  # bounded; we just need any nearby
            ).fetchall()
        }

    rows: list[dict[str, Any]] = []
    for channel_key, channel_name in (
        ("channel_a_suggestions", "steady_alpha"),
        ("channel_b_suggestions", "aggressive_growth"),
    ):
        for sug in output.get(channel_key, []):
            sym = (sug.get("symbol") or "").upper().strip()
            entry_px = entries.get((channel_name, sym))
            cur_px = prices.get(sym)
            unreal = None
            if entry_px and cur_px and entry_px > 0:
                unreal = (cur_px - entry_px) / entry_px
            rows.append({
                "channel": channel_name,
                "symbol": sym,
                "direction": (sug.get("direction") or "").lower(),
                "conviction": sug.get("conviction"),
                "target_weight": sug.get("target_weight"),
                "time_horizon_days": sug.get("time_horizon_days"),
                "stop_loss_pct": sug.get("stop_loss_pct"),
                "rationale": sug.get("rationale", ""),
                "counter_argument": sug.get("counter_argument", ""),
                "entry_price": entry_px,
                "current_price": cur_px,
                "unrealized_pct": unreal,
            })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(
            ["channel", "conviction"], ascending=[True, False]
        ).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Open paper trades (MTM)
# ---------------------------------------------------------------------------


def open_paper_trades_mtm() -> pd.DataFrame:
    """Currently open paper trades with live MTM and time remaining."""
    with _conn() as con:
        df = con.execute(
            """
            WITH latest_bar AS (
                SELECT symbol, MAX(bar_date) AS bd FROM market_bars GROUP BY symbol
            ),
            cur AS (
                SELECT mb.symbol, mb.bar_date AS as_of, mb.adj_close AS cur_px
                FROM market_bars mb
                JOIN latest_bar lb ON lb.symbol = mb.symbol AND lb.bd = mb.bar_date
            )
            SELECT
                t.id,
                t.placed_at::DATE AS entry_date,
                t.channel,
                t.symbol,
                t.direction,
                t.price AS entry_px,
                cur.cur_px AS current_px,
                cur.as_of AS mark_date,
                (cur.cur_px - t.price)/NULLIF(t.price,0) AS unrealized,
                s.time_horizon_days,
                s.conviction,
                s.stop_loss_pct,
                s.rationale,
                date_diff('day', t.placed_at::DATE, cur.as_of) AS days_held,
                s.time_horizon_days
                  - date_diff('day', t.placed_at::DATE, cur.as_of) AS days_left
            FROM trades t
            LEFT JOIN trade_outcomes o ON o.trade_id = t.id
            LEFT JOIN signals s ON s.id = t.source_signal_id
            LEFT JOIN cur ON cur.symbol = t.symbol
            WHERE t.status = 'paper_filled' AND o.trade_id IS NULL
            ORDER BY t.placed_at DESC, t.channel, t.symbol
            """
        ).df()
    return df


# ---------------------------------------------------------------------------
# Track record
# ---------------------------------------------------------------------------


def channel_stats(forward_only: bool = False, model_version: str = "llm-opus-4-7-v1") -> pd.DataFrame:
    """Per-channel summary stats. If forward_only, exclude trades whose
    source signal was persisted via cache backfill (we use evaluated_at
    being recent as a proxy — strict forward filter is hard without an
    explicit `source='backfill'` column)."""
    where = "WHERE t.status = 'paper_filled'"
    if forward_only:
        # Treat 'forward' as: trade.placed_at >= 2026-05-30 (today).
        # When the auto-run starts populating, this naturally fills.
        where += " AND t.placed_at::DATE >= CURRENT_DATE"
    with _conn() as con:
        df = con.execute(
            f"""
            SELECT
                t.channel,
                COUNT(o.trade_id) AS n_scored,
                AVG(o.return_pct) AS avg_ret,
                AVG(o.alpha) AS avg_alpha,
                AVG(CASE WHEN o.direction_correct THEN 1.0 ELSE 0.0 END) AS win_rate,
                SUM(CASE WHEN o.return_pct > 0 THEN o.return_pct ELSE 0 END) AS gross_win,
                SUM(CASE WHEN o.return_pct < 0 THEN -o.return_pct ELSE 0 END) AS gross_loss,
                AVG(o.days_held) AS avg_days
            FROM trades t
            LEFT JOIN trade_outcomes o ON o.trade_id = t.id
            {where}
            GROUP BY t.channel
            ORDER BY t.channel
            """
        ).df()
    if not df.empty:
        df["profit_factor"] = df.apply(
            lambda r: (r["gross_win"] / r["gross_loss"]) if (r["gross_loss"] or 0) > 0 else None,
            axis=1,
        )
    return df


def per_symbol_stats(channel: str, min_trades: int = 2) -> pd.DataFrame:
    with _conn() as con:
        df = con.execute(
            """
            SELECT
                t.symbol,
                COUNT(*) AS n,
                AVG(o.return_pct) AS avg_ret,
                AVG(o.alpha) AS avg_alpha,
                AVG(CASE WHEN o.direction_correct THEN 1.0 ELSE 0.0 END) AS win_rate
            FROM trades t
            JOIN trade_outcomes o ON o.trade_id = t.id
            WHERE t.channel = ?
            GROUP BY t.symbol
            HAVING COUNT(*) >= ?
            ORDER BY AVG(o.alpha) DESC
            """,
            [channel, min_trades],
        ).df()
    return df


def cumulative_alpha_curve(channel: str) -> pd.DataFrame:
    """For each scored trade in chronological order, running average alpha."""
    with _conn() as con:
        df = con.execute(
            """
            SELECT
                o.evaluated_at::DATE AS d,
                o.alpha,
                o.return_pct,
                o.benchmark_return_pct
            FROM trades t
            JOIN trade_outcomes o ON o.trade_id = t.id
            WHERE t.channel = ?
            ORDER BY o.evaluated_at
            """,
            [channel],
        ).df()
    if df.empty:
        return df
    df["cum_alpha"] = df["alpha"].cumsum()
    df["cum_ret"] = df["return_pct"].cumsum()
    df["cum_bench"] = df["benchmark_return_pct"].cumsum()
    return df


def simulate_virtual_portfolio(
    initial: float = 100_000.0,
    position_size_pct: float = 0.05,
    use_signal_weights: bool = False,
) -> dict[str, Any]:
    """Realistic multi-position portfolio simulation (D1/D2).

    For each channel, walk every paper trade chronologically by ENTRY date.
    At entry: deduct (position_size_pct × current NAV) from cash, open a
    position. At exit (closed trades use trade_outcomes; open trades use
    latest bar MTM): credit cash with position_size × (1 + return_pct).

    Returns:
      {
        "nav_curve":  DataFrame[date, series, nav]   — event-driven NAV per channel + SPY
        "final_nav":  {channel: float}
        "final_open": {channel: list of open position dicts at last sim date}
        "params":     {initial, position_size_pct, use_signal_weights}
      }

    Caveats:
      - Cash earns 0 (no T-bill interest on idle capital).
      - Stop-out exits use the same exit date the scorer set.
      - Currently-open trades MTM'd to latest available bar (treated as if
        "exited now" for the NAV curve's tail).
      - Doesn't model overnight gap / slippage / commissions.
      - `use_signal_weights=True` would scale by signal.target_weight; v1
        ignores it and uses uniform position_size_pct so the simulation
        is deterministic and channel-fair.
    """
    from collections import defaultdict
    from datetime import datetime, timedelta

    with _conn() as con:
        # All trades + their outcomes (or NULL for still-open)
        rows = con.execute(
            """
            WITH latest_bar AS (
                SELECT symbol, MAX(bar_date) AS bd FROM market_bars GROUP BY symbol
            ),
            cur AS (
                SELECT mb.symbol, mb.bar_date AS as_of, mb.adj_close AS cur_px
                FROM market_bars mb
                JOIN latest_bar lb ON lb.symbol = mb.symbol AND lb.bd = mb.bar_date
            )
            SELECT t.id, t.channel, t.symbol, t.direction,
                   t.placed_at::DATE AS entry_date, t.price AS entry_px,
                   -- Real exit date: entry + days_held from scorer (NOT
                   -- evaluated_at, which is just when the row was written).
                   CASE WHEN o.trade_id IS NOT NULL
                        THEN t.placed_at::DATE + (o.days_held || ' days')::INTERVAL
                        ELSE NULL
                   END AS exit_date_actual,
                   o.return_pct,
                   cur.cur_px, cur.as_of AS cur_date,
                   s.time_horizon_days
            FROM trades t
            LEFT JOIN trade_outcomes o ON o.trade_id = t.id
            LEFT JOIN signals s ON s.id = t.source_signal_id
            LEFT JOIN cur ON cur.symbol = t.symbol
            WHERE t.status = 'paper_filled'
            ORDER BY t.placed_at
            """
        ).fetchall()

        if not rows:
            return {"nav_curve": pd.DataFrame(columns=["date", "series", "nav"]),
                    "final_nav": {}, "final_open": {}, "params": {}}

        # SPY benchmark price series for the full span
        def _to_date(d):
            return d.date() if hasattr(d, "date") else d
        all_dates = [_to_date(r[4]) for r in rows if r[4]]
        span_start = min(all_dates)
        span_end = max(
            (_to_date(r[6]) for r in rows if r[6] is not None),
            default=max(all_dates),
        )
        # Extend span_end to the latest available bar if any open trades exist
        latest_cur = max(
            (_to_date(r[9]) for r in rows if r[9] is not None),
            default=None,
        )
        if latest_cur and latest_cur > span_end:
            span_end = latest_cur
        spy_rows = con.execute(
            "SELECT bar_date, adj_close FROM market_bars "
            "WHERE symbol = 'SPY' AND bar_date BETWEEN ? AND ? ORDER BY bar_date",
            [span_start, span_end],
        ).fetchall()

    # Build per-channel event timeline
    # Events: (type, date, trade_id, symbol, realized_return, status)
    # Closed trades get both open + close events; open trades get only
    # an open event and carry an mtm_return for end-of-sim mark.
    events_by_channel: dict[str, list[tuple]] = defaultdict(list)
    open_mtm: dict[int, float] = {}  # trade_id → current MTM return (open trades only)
    direction_sign = {"buy": 1, "add": 1, "hold": 1,
                      "sell": -1, "exit": -1, "reduce": -1}
    for (trade_id, channel, symbol, direction, entry_date, entry_px,
         exit_date_actual, return_pct, cur_px, cur_date, horizon) in rows:
        if entry_date is None:
            continue
        if return_pct is not None:
            # Scored — both open and close events
            # exit_date_actual = entry + days_held (real bar date, not "today")
            if hasattr(exit_date_actual, "date"):
                exit_date = exit_date_actual.date()
            else:
                exit_date = exit_date_actual or entry_date
            realized = float(return_pct)
            events_by_channel[channel].append(
                ("open", entry_date, trade_id, symbol, 0.0)
            )
            events_by_channel[channel].append(
                ("close", exit_date, trade_id, symbol, realized)
            )
        else:
            # Still open — only an open event; track MTM for end-of-sim mark
            if cur_px is None or entry_px is None or entry_px <= 0:
                continue
            sign = direction_sign.get((direction or "").lower(), 1)
            mtm = sign * (float(cur_px) - float(entry_px)) / float(entry_px)
            events_by_channel[channel].append(
                ("open", entry_date, trade_id, symbol, 0.0)
            )
            open_mtm[trade_id] = mtm

    # Simulate
    nav_rows: list[dict[str, Any]] = []
    final_nav: dict[str, float] = {}
    final_open: dict[str, list[dict]] = {ch: [] for ch in events_by_channel}

    for channel, events in events_by_channel.items():
        events.sort(key=lambda e: (e[1], 0 if e[0] == "close" else 1))
        # Process closes before opens on the same day so cash frees up first

        nav = initial
        cash = initial
        open_positions: dict[int, dict] = {}  # trade_id -> {entry_value, ...}
        nav_rows.append({"date": events[0][1] - timedelta(days=1),
                         "series": channel, "nav": nav})

        for ev_type, ev_date, trade_id, symbol, realized in events:
            if ev_type == "open":
                position_value = min(nav * position_size_pct, max(cash, 0.0))
                if position_value <= 0:
                    continue
                cash -= position_value
                open_positions[trade_id] = {
                    "symbol": symbol,
                    "entry_value": position_value,
                    "entry_date": ev_date,
                }
            else:  # close
                pos = open_positions.pop(trade_id, None)
                if pos is None:
                    continue
                exit_value = pos["entry_value"] * (1.0 + realized)
                cash += exit_value
            # NAV = cash + value of all open positions (at entry_value
            # — MTM gets realized on close events)
            nav = cash + sum(p["entry_value"] for p in open_positions.values())
            nav_rows.append({"date": ev_date, "series": channel, "nav": nav})

        # End-of-sim mark-to-market for positions still open
        # (their cur_px MTM tracked in open_mtm by trade_id)
        if open_positions:
            mtm_total = sum(
                p["entry_value"] * (1.0 + open_mtm.get(tid, 0.0))
                for tid, p in open_positions.items()
            )
            nav = cash + mtm_total
            # Push one final NAV point at "today" (latest cur_date)
            mtm_dates = [c[9] for c in rows if c[9] is not None]
            mark_date = max(mtm_dates) if mtm_dates else events[-1][1]
            nav_rows.append({"date": mark_date, "series": channel, "nav": nav})

            final_open[channel] = []
            for tid, p in open_positions.items():
                mtm = open_mtm.get(tid, 0.0)
                final_open[channel].append({
                    **p,
                    "trade_id": tid,
                    "current_value": p["entry_value"] * (1.0 + mtm),
                    "unrealized_pct": mtm,
                })
        final_nav[channel] = nav

    # SPY benchmark normalized to start at `initial`
    if spy_rows:
        spy_first = float(spy_rows[0][1])
        for bar_date, adj_close in spy_rows:
            nav_rows.append({
                "date": bar_date,
                "series": "SPY (benchmark)",
                "nav": initial * float(adj_close) / spy_first,
            })
        final_nav["SPY (benchmark)"] = initial * float(spy_rows[-1][1]) / spy_first

    return {
        "nav_curve": pd.DataFrame(nav_rows),
        "final_nav": final_nav,
        "final_open": final_open,
        "params": {
            "initial": initial,
            "position_size_pct": position_size_pct,
            "use_signal_weights": use_signal_weights,
        },
    }


def total_counts() -> dict[str, int]:
    with _conn() as con:
        open_n = con.execute(
            """
            SELECT COUNT(*) FROM trades t
            LEFT JOIN trade_outcomes o ON o.trade_id = t.id
            WHERE t.status = 'paper_filled' AND o.trade_id IS NULL
            """
        ).fetchone()[0]
        scored_n = con.execute("SELECT COUNT(*) FROM trade_outcomes").fetchone()[0]
        signal_n = con.execute("SELECT COUNT(*) FROM signals").fetchone()[0]
        digest_n = con.execute("SELECT COUNT(*) FROM llm_signal_cache").fetchone()[0]
    return {
        "open_trades": open_n,
        "scored_trades": scored_n,
        "signals": signal_n,
        "digests": digest_n,
    }


# ---------------------------------------------------------------------------
# Run log
# ---------------------------------------------------------------------------


def build_snapshot_markdown(d: date) -> tuple[str, list[str], str, float]:
    """Rebuild the LLM snapshot for a given date. Returns
    (markdown, notable_events, regime_label, regime_confidence).

    This re-runs `build_snapshot` with the current universe — it's a
    point-in-time view of what the LLM *would* see if asked about that
    date today. Macro/calendar features are deterministic from the DB
    so it matches what the live digest used (modulo GDELT, which is
    only ~30d). Free; no API call.
    """
    from alpha_engine.llm.context import build_snapshot

    with _conn() as con:
        universe = [
            r[0] for r in con.execute(
                "SELECT symbol FROM instruments WHERE active = TRUE ORDER BY symbol"
            ).fetchall()
        ]
        snap = build_snapshot(con, universe, as_of=d)
    return snap.markdown, snap.notable_events, snap.regime_label, float(snap.regime_confidence)


def price_history(symbol: str, since: date) -> pd.DataFrame:
    """Daily bars for a symbol since `since`. Columns: bar_date, adj_close."""
    with _conn() as con:
        df = con.execute(
            """
            SELECT bar_date, adj_close FROM market_bars
            WHERE symbol = ? AND bar_date >= ?
            ORDER BY bar_date
            """,
            [symbol, since],
        ).df()
    return df


def trade_detail(trade_id: int) -> dict[str, Any] | None:
    """Full context for one trade: trade row, source signal, current price."""
    with _conn() as con:
        row = con.execute(
            """
            SELECT t.id, t.placed_at, t.channel, t.symbol, t.direction,
                   t.price, t.status,
                   s.conviction, s.target_weight, s.time_horizon_days,
                   s.stop_loss_pct, s.rationale, s.counter_argument,
                   s.generated_at
            FROM trades t
            LEFT JOIN signals s ON s.id = t.source_signal_id
            WHERE t.id = ?
            """,
            [trade_id],
        ).fetchone()
        if not row:
            return None
        # Latest price
        cur = con.execute(
            """
            SELECT bar_date, adj_close FROM market_bars
            WHERE symbol = ? ORDER BY bar_date DESC LIMIT 1
            """,
            [row[3]],
        ).fetchone()
    placed_at = row[1]
    entry_date = placed_at.date() if hasattr(placed_at, "date") else placed_at
    cur_px = float(cur[1]) if cur else None
    entry_px = float(row[5]) if row[5] is not None else None
    unreal = None
    if entry_px and cur_px and entry_px > 0:
        unreal = (cur_px - entry_px) / entry_px
    return {
        "id": row[0],
        "entry_date": entry_date,
        "channel": row[2],
        "symbol": row[3],
        "direction": row[4],
        "entry_px": entry_px,
        "status": row[6],
        "conviction": row[7],
        "target_weight": row[8],
        "time_horizon_days": row[9],
        "stop_loss_pct": row[10],
        "rationale": row[11],
        "counter_argument": row[12],
        "signal_generated_at": row[13],
        "current_px": cur_px,
        "current_as_of": cur[0] if cur else None,
        "unrealized_pct": unreal,
    }


def scored_trade_symbols() -> list[str]:
    """Distinct symbols among trades that have been scored. Used to
    decide whether to show the survivorship-bias warning on the dashboard."""
    with _conn() as con:
        rows = con.execute(
            """
            SELECT DISTINCT t.symbol
            FROM trades t JOIN trade_outcomes o ON o.trade_id = t.id
            ORDER BY t.symbol
            """
        ).fetchall()
    return [r[0] for r in rows]


def survivorship_affected_in_scored() -> list[str]:
    """Subset of scored-trade symbols that are individual equities."""
    from alpha_engine.backtest.warnings import affected_symbols

    syms = scored_trade_symbols()
    if not syms:
        return []
    with _conn() as con:
        return affected_symbols(con, syms)


def all_known_symbols() -> list[str]:
    """All symbols with at least one bar in market_bars. Includes:
      - 115 active instruments (LLM-visible universe)
      - ~400 Phase C bars-only S&P 500 tickers not in instruments
    Used by the Lookup page to autocomplete any ticker we have data for.
    """
    with _conn() as con:
        rows = con.execute(
            "SELECT DISTINCT symbol FROM market_bars ORDER BY symbol"
        ).fetchall()
    return [r[0] for r in rows]


def symbol_info(symbol: str) -> dict[str, Any] | None:
    """Per-symbol metadata + bar coverage + recent prices. Returns None
    if no bars exist."""
    sym = symbol.upper().strip()
    with _conn() as con:
        # Coverage
        meta = con.execute(
            """
            SELECT COUNT(*), MIN(bar_date), MAX(bar_date)
            FROM market_bars WHERE symbol = ?
            """,
            [sym],
        ).fetchone()
        if not meta or not meta[0]:
            return None
        n_bars, first_bar, last_bar = meta
        # Instrument metadata (may not exist if Phase C bars-only)
        inst = con.execute(
            "SELECT name, instrument_type, sector, active FROM instruments WHERE symbol = ?",
            [sym],
        ).fetchone()
        # First and last adj_close
        bounds = con.execute(
            """
            SELECT (SELECT adj_close FROM market_bars WHERE symbol=? ORDER BY bar_date ASC LIMIT 1),
                   (SELECT adj_close FROM market_bars WHERE symbol=? ORDER BY bar_date DESC LIMIT 1)
            """,
            [sym, sym],
        ).fetchone()
        first_px, last_px = float(bounds[0]), float(bounds[1])
    name = inst[0] if inst else "(bars-only — not in active universe)"
    return {
        "symbol": sym,
        "name": name,
        "instrument_type": inst[1] if inst else None,
        "sector": inst[2] if inst else None,
        "in_universe": bool(inst and inst[3]),
        "n_bars": int(n_bars),
        "first_bar": first_bar,
        "last_bar": last_bar,
        "first_px": first_px,
        "last_px": last_px,
        "total_return_pct": (last_px - first_px) / first_px if first_px > 0 else 0.0,
    }


def price_history_with_benchmark(symbol: str, benchmark: str = "SPY") -> pd.DataFrame:
    """Daily bars for a symbol since the symbol's first bar, plus benchmark
    aligned. Returns columns: date, symbol_price, symbol_normalized,
    benchmark_price, benchmark_normalized (both normalized to 100 at start).
    """
    sym = symbol.upper().strip()
    with _conn() as con:
        # Find symbol's start date
        first = con.execute(
            "SELECT MIN(bar_date) FROM market_bars WHERE symbol = ?", [sym]
        ).fetchone()
        if not first or not first[0]:
            return pd.DataFrame()
        start = first[0]
        df = con.execute(
            """
            SELECT s.bar_date AS date,
                   s.adj_close AS symbol_price,
                   b.adj_close AS benchmark_price
            FROM market_bars s
            LEFT JOIN market_bars b
              ON b.symbol = ? AND b.bar_date = s.bar_date
            WHERE s.symbol = ? AND s.bar_date >= ?
            ORDER BY s.bar_date
            """,
            [benchmark, sym, start],
        ).df()
    if df.empty:
        return df
    # Normalize both to 100 at start (first non-NaN benchmark row)
    first_sym = float(df["symbol_price"].iloc[0])
    bench_first_idx = df["benchmark_price"].first_valid_index()
    if bench_first_idx is None:
        df["benchmark_price"] = float("nan")
        df["benchmark_normalized"] = float("nan")
    else:
        first_bench = float(df["benchmark_price"].loc[bench_first_idx])
        df["benchmark_normalized"] = df["benchmark_price"] / first_bench * 100.0
    df["symbol_normalized"] = df["symbol_price"] / first_sym * 100.0
    return df


def digest_narrative(digest_date: date) -> dict[str, Any]:
    """Return the LLM's top-line context for this digest:
      - market_summary: 1-3 sentence read of current conditions
      - key_themes: list of bullet-style themes driving the view
      - risk_notes: list of specific risks to monitor

    Returns empty fields if the digest isn't cached.
    """
    with _conn() as con:
        row = con.execute(
            "SELECT output_json FROM llm_signal_cache "
            "WHERE as_of = ? ORDER BY generated_at DESC LIMIT 1",
            [digest_date],
        ).fetchone()
    if not row:
        return {"market_summary": "", "key_themes": [], "risk_notes": []}
    try:
        output = json.loads(row[0])
    except json.JSONDecodeError:
        return {"market_summary": "", "key_themes": [], "risk_notes": []}
    return {
        "market_summary": output.get("market_summary") or "",
        "key_themes": list(output.get("key_themes") or []),
        "risk_notes": list(output.get("risk_notes") or []),
    }


def bar_freshness(stale_threshold_days: int = 3) -> dict[str, Any]:
    """Per-symbol latest-bar-date check vs the expected most-recent
    trading day. Surfaces silent yfinance failures.

    "Expected latest trading day" = the most recent past trading day
    (accounting for weekends + NYSE holidays). After 4 PM ET on a
    trading day, today's bar should also be in the DB — but yfinance
    can lag by an hour or two, so we treat 1-day-behind as normal
    during US business hours and flag 2+.

    Returns:
      {
        "expected_date": date     — the trading day we expect to have bars for
        "summary": {
          "fresh": int,           — symbols at expected_date
          "behind_1": int,        — 1 trading day behind (yesterday's bar)
          "behind_2plus": int,    — 2+ behind (worth checking)
          "stale": int,           — > stale_threshold_days behind
        }
        "stale_symbols": [{symbol, last_bar, days_behind}]
        "open_trade_symbols_stale": [...]  — subset of stale that we hold
        "key_status": str         — overall status: "fresh" | "ok" | "stale"
      }
    """
    from datetime import date as _date, timedelta
    from alpha_engine.calendars.scheduled import is_trading_day

    # Walk back from today to find the most recent past trading day
    today = _date.today()
    expected = today
    # If today is a trading day, expect today's data (after close)
    # If not, walk back
    while not is_trading_day(expected):
        expected = expected - timedelta(days=1)

    # Per-symbol latest bar date — only for symbols in our universe
    # (so Phase C bars-only symbols don't pollute the check)
    with _conn() as con:
        rows = con.execute(
            """
            SELECT i.symbol, MAX(mb.bar_date) AS last_bar
            FROM instruments i
            LEFT JOIN market_bars mb ON mb.symbol = i.symbol
            WHERE i.active = TRUE
            GROUP BY i.symbol
            ORDER BY i.symbol
            """
        ).fetchall()
        # Symbols currently held in open paper trades — extra sensitive
        open_syms = {
            r[0] for r in con.execute(
                """
                SELECT DISTINCT t.symbol FROM trades t
                LEFT JOIN trade_outcomes o ON o.trade_id = t.id
                WHERE t.status = 'paper_filled' AND o.trade_id IS NULL
                """
            ).fetchall()
        }

    summary = {"fresh": 0, "behind_1": 0, "behind_2plus": 0, "stale": 0, "no_data": 0}
    stale_symbols: list[dict] = []
    open_stale: list[dict] = []

    for sym, last_bar in rows:
        if last_bar is None:
            summary["no_data"] += 1
            stale_symbols.append({"symbol": sym, "last_bar": None, "days_behind": None})
            if sym in open_syms:
                open_stale.append({"symbol": sym, "last_bar": None, "days_behind": None})
            continue

        # Count trading days between last_bar and expected
        days_behind = 0
        cursor = last_bar
        while cursor < expected:
            cursor = cursor + timedelta(days=1)
            if is_trading_day(cursor):
                days_behind += 1

        if days_behind == 0:
            summary["fresh"] += 1
        elif days_behind == 1:
            summary["behind_1"] += 1
        else:
            summary["behind_2plus"] += 1
            if days_behind > stale_threshold_days:
                summary["stale"] += 1
            entry = {"symbol": sym, "last_bar": last_bar, "days_behind": days_behind}
            stale_symbols.append(entry)
            if sym in open_syms:
                open_stale.append(entry)

    stale_symbols.sort(key=lambda r: (r["days_behind"] or 9999), reverse=True)
    open_stale.sort(key=lambda r: (r["days_behind"] or 9999), reverse=True)

    # Overall status
    if summary["stale"] > 0 or summary["no_data"] > 0:
        key_status = "stale"
    elif summary["behind_2plus"] > 0:
        key_status = "ok"  # 2-3 days behind is OK during transient yfinance lag
    else:
        key_status = "fresh"

    return {
        "expected_date": expected,
        "summary": summary,
        "stale_symbols": stale_symbols,
        "open_trade_symbols_stale": open_stale,
        "key_status": key_status,
        "stale_threshold_days": stale_threshold_days,
    }


def channel_crosscheck(digest_date: date) -> dict[str, Any]:
    """Find symbols that appear in BOTH channels of a digest. Returns
    agreements (both channels same direction = stronger signal) and
    contradictions (channels disagree = warning).

    Direction-buckets for agreement-checking:
      LONG = {buy, add}
      SHORT = {sell, exit, reduce}
      HOLD = {hold}
    Hold is informational, never counts as agreement or contradiction.

    Returns:
      {
        "agreements":     [{symbol, direction_bucket, a_dir, a_conv, b_dir, b_conv, combined_conv}, ...]
        "contradictions": [{symbol, a_dir, a_conv, a_rationale, b_dir, b_conv, b_rationale}, ...]
        "lookup":         {symbol: "agreement_long" | "agreement_short" | "contradiction" | None}
      }
    """
    LONG = {"buy", "add"}
    SHORT = {"sell", "exit", "reduce"}

    def _bucket(d: str) -> str:
        d = (d or "").lower()
        if d in LONG: return "long"
        if d in SHORT: return "short"
        return "hold"

    with _conn() as con:
        row = con.execute(
            "SELECT output_json FROM llm_signal_cache "
            "WHERE as_of = ? ORDER BY generated_at DESC LIMIT 1",
            [digest_date],
        ).fetchone()
    if not row:
        return {"agreements": [], "contradictions": [], "lookup": {}}
    try:
        output = json.loads(row[0])
    except json.JSONDecodeError:
        return {"agreements": [], "contradictions": [], "lookup": {}}

    def _index(channel_list):
        return {
            (sug.get("symbol") or "").upper().strip(): sug
            for sug in channel_list or []
            if (sug.get("symbol") or "").strip()
        }

    a_map = _index(output.get("channel_a_suggestions", []))
    b_map = _index(output.get("channel_b_suggestions", []))
    common = sorted(set(a_map) & set(b_map))

    agreements: list[dict[str, Any]] = []
    contradictions: list[dict[str, Any]] = []
    lookup: dict[str, str] = {}

    for sym in common:
        a = a_map[sym]; b = b_map[sym]
        a_dir = (a.get("direction") or "").lower()
        b_dir = (b.get("direction") or "").lower()
        a_buck = _bucket(a_dir)
        b_buck = _bucket(b_dir)
        a_conv = float(a.get("conviction") or 0)
        b_conv = float(b.get("conviction") or 0)
        if a_buck == "hold" or b_buck == "hold":
            continue  # one side is no-op, nothing to flag
        if a_buck == b_buck:
            # AGREEMENT: amplifier signal
            agreements.append({
                "symbol": sym,
                "direction_bucket": a_buck,
                "a_dir": a_dir, "a_conv": a_conv,
                "b_dir": b_dir, "b_conv": b_conv,
                # Simple combined score: max + 0.5*(min) — rewards strong+strong
                "combined_conv": max(a_conv, b_conv) + 0.5 * min(a_conv, b_conv),
            })
            lookup[sym] = f"agreement_{a_buck}"
        else:
            # CONTRADICTION
            contradictions.append({
                "symbol": sym,
                "a_dir": a_dir, "a_conv": a_conv,
                "a_rationale": (a.get("rationale") or "")[:140],
                "b_dir": b_dir, "b_conv": b_conv,
                "b_rationale": (b.get("rationale") or "")[:140],
            })
            lookup[sym] = "contradiction"

    agreements.sort(key=lambda r: r["combined_conv"], reverse=True)
    contradictions.sort(key=lambda r: max(r["a_conv"], r["b_conv"]), reverse=True)
    return {"agreements": agreements, "contradictions": contradictions, "lookup": lookup}


def digest_diff(
    new_date: date, old_date: date, conviction_delta_threshold: float = 0.5
) -> dict[str, Any]:
    """Compute the per-channel delta between two cached digests.

    Buckets:
      - new:        symbols in `new_date` not present in `old_date`
      - dropped:    symbols in `old_date` not present in `new_date`
      - flipped:    same symbol, different direction
      - conv_up:    same symbol+direction, conviction up by ≥ threshold
      - conv_down:  same symbol+direction, conviction down by ≥ threshold

    Returns:
      {
        "new_date": date, "old_date": date,
        "steady_alpha": {new: [...], dropped: [...], flipped: [...], conv_up: [...], conv_down: [...]},
        "aggressive_growth": {...},
        "total_changes": int,
      }
    Each row in a bucket is a dict: {symbol, direction, conviction, prior_direction?, prior_conviction?, rationale}
    """
    def _load_outputs(d: date) -> dict[str, Any]:
        with _conn() as con:
            row = con.execute(
                "SELECT output_json FROM llm_signal_cache "
                "WHERE as_of = ? ORDER BY generated_at DESC LIMIT 1",
                [d],
            ).fetchone()
        if not row:
            return {}
        try:
            return json.loads(row[0])
        except json.JSONDecodeError:
            return {}

    def _by_symbol(channel_list: list[dict]) -> dict[str, dict]:
        """Index a channel's suggestions by uppercase symbol."""
        out: dict[str, dict] = {}
        for sug in channel_list or []:
            sym = (sug.get("symbol") or "").upper().strip()
            if not sym:
                continue
            out[sym] = sug
        return out

    new_output = _load_outputs(new_date)
    old_output = _load_outputs(old_date)

    result: dict[str, Any] = {
        "new_date": new_date,
        "old_date": old_date,
        "total_changes": 0,
    }

    for ch_key, ch_name in (
        ("channel_a_suggestions", "steady_alpha"),
        ("channel_b_suggestions", "aggressive_growth"),
    ):
        new_map = _by_symbol(new_output.get(ch_key, []))
        old_map = _by_symbol(old_output.get(ch_key, []))

        new_syms = set(new_map) - set(old_map)
        dropped_syms = set(old_map) - set(new_map)
        common = set(new_map) & set(old_map)

        buckets: dict[str, list[dict]] = {
            "new": [], "dropped": [], "flipped": [], "conv_up": [], "conv_down": [],
        }
        for s in sorted(new_syms):
            sug = new_map[s]
            buckets["new"].append({
                "symbol": s,
                "direction": (sug.get("direction") or "").lower(),
                "conviction": float(sug.get("conviction") or 0),
                "rationale": (sug.get("rationale") or "")[:140],
            })
        for s in sorted(dropped_syms):
            sug = old_map[s]
            buckets["dropped"].append({
                "symbol": s,
                "prior_direction": (sug.get("direction") or "").lower(),
                "prior_conviction": float(sug.get("conviction") or 0),
            })
        for s in sorted(common):
            new_sug = new_map[s]
            old_sug = old_map[s]
            new_dir = (new_sug.get("direction") or "").lower()
            old_dir = (old_sug.get("direction") or "").lower()
            new_conv = float(new_sug.get("conviction") or 0)
            old_conv = float(old_sug.get("conviction") or 0)
            if new_dir != old_dir:
                buckets["flipped"].append({
                    "symbol": s,
                    "direction": new_dir,
                    "conviction": new_conv,
                    "prior_direction": old_dir,
                    "prior_conviction": old_conv,
                    "rationale": (new_sug.get("rationale") or "")[:140],
                })
                continue
            delta = new_conv - old_conv
            if delta >= conviction_delta_threshold:
                buckets["conv_up"].append({
                    "symbol": s, "direction": new_dir, "conviction": new_conv,
                    "prior_conviction": old_conv, "delta": delta,
                })
            elif delta <= -conviction_delta_threshold:
                buckets["conv_down"].append({
                    "symbol": s, "direction": new_dir, "conviction": new_conv,
                    "prior_conviction": old_conv, "delta": delta,
                })

        # Sort conv_up/down by abs(delta) desc
        buckets["conv_up"].sort(key=lambda r: r["delta"], reverse=True)
        buckets["conv_down"].sort(key=lambda r: r["delta"])

        result[ch_name] = buckets
        result["total_changes"] += sum(len(v) for v in buckets.values())

    return result


def today_action_items(
    high_conv_threshold: float = 7.5,
    due_within_days: int = 5,
    drawdown_threshold: float = -0.05,
) -> dict[str, Any]:
    """Synthesize "what to look at today" for the Suggestions landing card.

    Returns a dict with:
      - new_high_conv: list of (channel, symbol, direction, conviction,
        rationale_short) from the latest cached digest with conv >= threshold
        and an actionable direction
      - stopped_out_today: list of (channel, symbol, direction, return_pct,
        days_held) for trades whose outcomes were written today with notes
        like 'stopped out%'
      - due_soon: list of (channel, symbol, entry_date, days_left,
        unrealized_pct) for open trades whose horizon ends within N days
      - drawdown_alerts: list of (channel, symbol, entry_date, unrealized_pct,
        days_held) for open trades currently down past threshold from entry
      - latest_digest_date: date or None
    """
    from datetime import datetime, timedelta

    with _conn() as con:
        # 1. Latest cached digest's high-conv actionable picks (from output_json)
        new_high_conv: list[dict[str, Any]] = []
        row = con.execute(
            "SELECT as_of, output_json FROM llm_signal_cache "
            "ORDER BY as_of DESC, generated_at DESC LIMIT 1"
        ).fetchone()
        latest_digest_date = row[0] if row else None
        if row:
            try:
                output = json.loads(row[1])
            except json.JSONDecodeError:
                output = {}
            actionable = {"buy", "add"}
            for ch_key, ch_name in (
                ("channel_a_suggestions", "steady_alpha"),
                ("channel_b_suggestions", "aggressive_growth"),
            ):
                for sug in output.get(ch_key, []):
                    direction = (sug.get("direction") or "").lower()
                    if direction not in actionable:
                        continue
                    try:
                        conv = float(sug.get("conviction") or 0)
                    except (TypeError, ValueError):
                        conv = 0.0
                    if conv < high_conv_threshold:
                        continue
                    new_high_conv.append({
                        "channel": ch_name,
                        "symbol": (sug.get("symbol") or "").upper(),
                        "direction": direction,
                        "conviction": conv,
                        "rationale_short": (sug.get("rationale") or "")[:140],
                    })
            new_high_conv.sort(key=lambda r: r["conviction"], reverse=True)

        # 2. Stopped-out today
        today = datetime.utcnow().date()
        stopped_today = con.execute(
            """
            SELECT t.channel, t.symbol, t.direction, o.return_pct, o.days_held
            FROM trade_outcomes o
            JOIN trades t ON t.id = o.trade_id
            WHERE o.notes LIKE 'stopped out%'
              AND o.evaluated_at::DATE = ?
            ORDER BY o.return_pct ASC
            """,
            [today],
        ).fetchall()
        stopped_out_today = [
            {
                "channel": r[0], "symbol": r[1], "direction": r[2],
                "return_pct": float(r[3] or 0), "days_held": int(r[4] or 0),
            }
            for r in stopped_today
        ]

        # 3. Open trades due to score within N days
        due = con.execute(
            f"""
            WITH latest_bar AS (
                SELECT symbol, MAX(bar_date) AS bd FROM market_bars GROUP BY symbol
            ),
            cur AS (
                SELECT mb.symbol, mb.adj_close AS cur_px
                FROM market_bars mb
                JOIN latest_bar lb ON lb.symbol = mb.symbol AND lb.bd = mb.bar_date
            )
            SELECT t.channel, t.symbol, t.placed_at::DATE AS entry_date,
                   s.time_horizon_days,
                   date_diff('day', t.placed_at::DATE, CURRENT_DATE) AS days_held,
                   s.time_horizon_days
                     - date_diff('day', t.placed_at::DATE, CURRENT_DATE) AS days_left,
                   t.price AS entry_px, cur.cur_px,
                   (cur.cur_px - t.price)/NULLIF(t.price, 0) AS unrealized
            FROM trades t
            LEFT JOIN trade_outcomes o ON o.trade_id = t.id
            LEFT JOIN signals s ON s.id = t.source_signal_id
            LEFT JOIN cur ON cur.symbol = t.symbol
            WHERE t.status = 'paper_filled' AND o.trade_id IS NULL
              AND s.time_horizon_days IS NOT NULL
              AND s.time_horizon_days
                  - date_diff('day', t.placed_at::DATE, CURRENT_DATE) BETWEEN 0 AND ?
            ORDER BY days_left ASC, t.symbol
            """,
            [due_within_days],
        ).fetchall()
        due_soon = [
            {
                "channel": r[0], "symbol": r[1], "entry_date": r[2],
                "days_left": int(r[5] or 0),
                "unrealized": float(r[8] or 0) if r[8] is not None else None,
            }
            for r in due
        ]

        # 4. Open trades currently down past threshold (computed against
        #    entry, direction-adjusted for shorts)
        dd_rows = con.execute(
            """
            WITH latest_bar AS (
                SELECT symbol, MAX(bar_date) AS bd FROM market_bars GROUP BY symbol
            ),
            cur AS (
                SELECT mb.symbol, mb.adj_close AS cur_px
                FROM market_bars mb
                JOIN latest_bar lb ON lb.symbol = mb.symbol AND lb.bd = mb.bar_date
            )
            SELECT t.channel, t.symbol, t.direction, t.placed_at::DATE AS entry_date,
                   t.price AS entry_px, cur.cur_px,
                   date_diff('day', t.placed_at::DATE, CURRENT_DATE) AS days_held,
                   CASE
                       WHEN t.direction IN ('buy','add','hold')
                            THEN (cur.cur_px - t.price)/NULLIF(t.price, 0)
                       ELSE -((cur.cur_px - t.price)/NULLIF(t.price, 0))
                   END AS unrealized
            FROM trades t
            LEFT JOIN trade_outcomes o ON o.trade_id = t.id
            LEFT JOIN cur ON cur.symbol = t.symbol
            WHERE t.status = 'paper_filled' AND o.trade_id IS NULL
              AND cur.cur_px IS NOT NULL
            """,
        ).fetchall()
        drawdown_alerts = []
        for r in dd_rows:
            unreal = float(r[7] or 0) if r[7] is not None else 0.0
            if unreal <= drawdown_threshold:
                drawdown_alerts.append({
                    "channel": r[0], "symbol": r[1], "direction": r[2],
                    "entry_date": r[3], "unrealized": unreal,
                    "days_held": int(r[6] or 0),
                })
        drawdown_alerts.sort(key=lambda r: r["unrealized"])

    return {
        "new_high_conv": new_high_conv,
        "stopped_out_today": stopped_out_today,
        "due_soon": due_soon,
        "drawdown_alerts": drawdown_alerts,
        "latest_digest_date": latest_digest_date,
        "high_conv_threshold": high_conv_threshold,
        "due_within_days": due_within_days,
        "drawdown_threshold": drawdown_threshold,
    }


def tail_run_log(n_lines: int = 200) -> str:
    """Return the last N lines of daily_paper_trade.log, or empty string."""
    from pathlib import Path

    path = Path("data") / "daily_paper_trade.log"
    if not path.exists():
        return ""
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except OSError:
        return ""
    return "".join(lines[-n_lines:])


def last_run_summary() -> dict[str, Any] | None:
    """Get the most recent auto-run status.

    Prefers `data/last_run_status.json` (written by post_run_check.ps1 —
    higher fidelity, includes error_reason). Falls back to parsing the
    log if the JSON is missing (legacy path or first-time setup).

    Returns a dict with:
      - started_at: datetime or None
      - status: 'ok' | 'skipped_weekend' | 'skipped_holiday' | 'error' | 'unknown'
      - cost_usd: float or None
      - opened: int or None
      - scored: int or None
      - gdelt_warning: bool
      - error_reason: str or None (from JSON status only)
      - last_lines: str (tail for popover)
      - age_hours: float (since started_at)
    """
    import json as _json
    import re
    from datetime import datetime
    from pathlib import Path

    # 1. Try the structured JSON first (preferred path)
    json_path = Path("data") / "last_run_status.json"
    if json_path.exists():
        try:
            # utf-8-sig handles the BOM that Windows PowerShell prepends.
            with json_path.open("r", encoding="utf-8-sig") as f:
                data = _json.load(f)
            started_at = None
            ts_raw = data.get("timestamp")
            if ts_raw:
                try:
                    # ISO 8601 with timezone (e.g. 2026-06-01T17:00:00.123-06:00)
                    started_at = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
                    # Strip tzinfo for comparison with datetime.now()
                    started_at = started_at.replace(tzinfo=None)
                except (TypeError, ValueError):
                    pass

            log_tail = data.get("log_tail", "") or ""

            # Re-derive a few fields from the tail (the JSON doesn't carry them)
            cost = None
            m = re.search(r"Cost:\s*\$([\d.]+)", log_tail)
            if m:
                cost = float(m.group(1))
            opened = scored = None
            m = re.search(r"opened (\d+) new paper trades", log_tail)
            if m:
                opened = int(m.group(1))
            m = re.search(r"scored (\d+) trades", log_tail)
            if m:
                scored = int(m.group(1))

            age_hours = None
            if started_at:
                age_hours = (datetime.now() - started_at).total_seconds() / 3600.0

            return {
                "started_at": started_at,
                "status": data.get("status", "unknown"),
                "cost_usd": cost,
                "opened": opened,
                "scored": scored,
                "gdelt_warning": "[warn] gdelt ingest exited non-zero" in log_tail,
                "error_reason": data.get("error_reason"),
                "last_lines": log_tail[-2000:],
                "age_hours": age_hours,
            }
        except (OSError, ValueError, _json.JSONDecodeError):
            pass  # fall through to log parsing

    # 2. Legacy: parse the log file
    path = Path("data") / "daily_paper_trade.log"
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            content = f.read()
    except OSError:
        return None

    # Find the last "===== <anything> =====" header.  Windows %DATE% can be
    # "Mon 06/01/2026" or just "06/01/2026", and %TIME% can use 12h/24h with
    # 1-2 decimal seconds — grab the whole interior and split flexibly.
    headers = list(re.finditer(r"=====\s+(?P<hdr>.+?)\s+=====", content))
    if not headers:
        return None

    last_header = headers[-1]
    last_block = content[last_header.start():]
    hdr_str = last_header.group("hdr").strip()

    # Strip optional weekday prefix (Mon/Tue/...)
    parts = hdr_str.split()
    if parts and parts[0].rstrip(",").lower()[:3] in (
        "mon", "tue", "wed", "thu", "fri", "sat", "sun"
    ):
        parts = parts[1:]
    core = " ".join(parts)

    # Try a range of plausible Windows date+time formats
    started_at: datetime | None = None
    candidate_fmts = [
        "%m/%d/%Y %H:%M:%S.%f",
        "%m/%d/%Y %H:%M:%S",
        "%m/%d/%Y %I:%M:%S.%f %p",
        "%m/%d/%Y %I:%M:%S %p",
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
    ]
    for fmt in candidate_fmts:
        try:
            started_at = datetime.strptime(core, fmt)
            break
        except ValueError:
            continue
    if started_at is None and len(parts) >= 1:
        # Last-resort: just parse the date, lose the time
        for d_fmt in ("%m/%d/%Y", "%Y-%m-%d"):
            try:
                started_at = datetime.strptime(parts[0], d_fmt)
                break
            except ValueError:
                continue

    # Classify outcome
    status = "ok"
    if "Skipping: " in last_block and "weekend" in last_block.lower():
        status = "skipped_weekend"
    elif "Skipping digest generation" in last_block:
        status = "skipped_holiday"
    elif "Traceback" in last_block or "[ERROR]" in last_block:
        status = "error"
    elif "ANTHROPIC_API_KEY not set" in last_block:
        status = "error"

    # Extract cost ($0.1482 etc.)
    cost = None
    m = re.search(r"Generated digest\.\s+Cost:\s+\$([\d.]+)", last_block)
    if m:
        cost = float(m.group(1))

    # Extract opened / scored counts
    opened = scored = None
    m = re.search(r"opened (\d+) new paper trades", last_block)
    if m:
        opened = int(m.group(1))
    m = re.search(r"scored (\d+) trades", last_block)
    if m:
        scored = int(m.group(1))

    # GDELT warning flag
    gdelt_warning = "[warn] gdelt ingest exited non-zero" in last_block

    age_hours = None
    if started_at:
        from datetime import datetime as _dt
        age_hours = (_dt.now() - started_at).total_seconds() / 3600.0

    return {
        "started_at": started_at,
        "status": status,
        "cost_usd": cost,
        "opened": opened,
        "scored": scored,
        "gdelt_warning": gdelt_warning,
        "error_reason": None,
        "last_lines": last_block[-2000:],  # tail for popover
        "age_hours": age_hours,
    }


# ---------------------------------------------------------------------------
# ML signal layer (ml_signals table + validation JSON)
# ---------------------------------------------------------------------------

ML_MODEL_LABELS = {
    "ml-momentum-v1": "Momentum composite",
    "ml-xgb-v1": "XGBoost (walk-forward)",
}


def latest_ml_date() -> Optional[date]:
    with _conn() as con:
        row = con.execute("SELECT MAX(signal_date) FROM ml_signals").fetchone()
    return row[0] if row and row[0] else None


def available_ml_dates() -> list[date]:
    with _conn() as con:
        rows = con.execute(
            "SELECT DISTINCT signal_date FROM ml_signals ORDER BY signal_date DESC"
        ).fetchall()
    return [r[0] for r in rows]


def ml_signals_for_date(d: date, model_version: str) -> pd.DataFrame:
    """Full ranked cross-section for one date + model, with instrument names."""
    with _conn() as con:
        return con.execute(
            """
            SELECT m.*, i.name AS instrument_name, i.instrument_type, i.sector
            FROM ml_signals m
            LEFT JOIN instruments i USING (symbol)
            WHERE m.signal_date = ? AND m.model_version = ?
            ORDER BY m.rank
            """,
            [d, model_version],
        ).fetch_df()


def ml_models_for_date(d: date) -> list[str]:
    with _conn() as con:
        rows = con.execute(
            "SELECT DISTINCT model_version FROM ml_signals WHERE signal_date = ? "
            "ORDER BY model_version",
            [d],
        ).fetchall()
    return [r[0] for r in rows]


def ml_model_consensus(d: date) -> pd.DataFrame:
    """Symbols where BOTH models agree on BUY or AVOID — the strongest ML
    signal we have. Disagreements are surfaced separately (informative:
    composite chases trends, XGB has learned some mean-reversion)."""
    with _conn() as con:
        return con.execute(
            """
            SELECT a.symbol, i.name AS instrument_name,
                   a.action AS momentum_action, b.action AS xgb_action,
                   a.rank AS momentum_rank, b.rank AS xgb_rank,
                   a.n_universe,
                   a.mom_12_1, a.dist_200ma, a.rsi_14, a.vol_30d
            FROM ml_signals a
            JOIN ml_signals b
              ON a.signal_date = b.signal_date AND a.symbol = b.symbol
             AND a.model_version = 'ml-momentum-v1'
             AND b.model_version = 'ml-xgb-v1'
            LEFT JOIN instruments i ON i.symbol = a.symbol
            WHERE a.signal_date = ?
            ORDER BY a.rank
            """,
            [d],
        ).fetch_df()


def ml_action_lookup(d: date) -> dict[str, dict]:
    """symbol -> {action, rank, n, xgb_action} for the ML run closest at or
    before `d` (so the Suggestions page can badge LLM picks even when the
    digest and ML dates differ by a day)."""
    with _conn() as con:
        row = con.execute(
            "SELECT MAX(signal_date) FROM ml_signals WHERE signal_date <= ?", [d]
        ).fetchone()
        if not row or not row[0]:
            return {}
        ml_d = row[0]
        rows = con.execute(
            """
            SELECT a.symbol, a.action, a.rank, a.n_universe, b.action,
                   a.dist_50ma, a.rsi_14
            FROM ml_signals a
            LEFT JOIN ml_signals b
              ON b.signal_date = a.signal_date AND b.symbol = a.symbol
             AND b.model_version = 'ml-xgb-v1'
            WHERE a.signal_date = ? AND a.model_version = 'ml-momentum-v1'
            """,
            [ml_d],
        ).fetchall()
    return {
        r[0]: {"action": r[1], "rank": r[2], "n": r[3], "xgb_action": r[4],
               "dist_50ma": r[5], "rsi_14": r[6], "ml_date": ml_d}
        for r in rows
    }


def ml_llm_agreement(digest_date: date) -> pd.DataFrame:
    """Cross-reference the LLM digest's picks against the ML ranks.

    LLM buy/add = LONG, sell/exit/reduce = SHORT. Agreement when LONG meets
    ML BUY (or SHORT meets AVOID); conflict when they point opposite ways.
    Independent signal sources agreeing is the strongest evidence either
    produces."""
    sugg = suggestions_for_date(digest_date)
    if sugg.empty:
        return pd.DataFrame()
    lookup = ml_action_lookup(digest_date)
    if not lookup:
        return pd.DataFrame()

    rows = []
    for _, r in sugg.iterrows():
        ml = lookup.get(r["symbol"])
        if ml is None:
            continue
        d = (r["direction"] or "").lower()
        side = "LONG" if d in ("buy", "add") else "SHORT" if d in ("sell", "exit", "reduce") else None
        if side is None:
            continue
        verdict = "neutral"
        if (side == "LONG" and ml["action"] == "BUY") or (
            side == "SHORT" and ml["action"] == "AVOID"
        ):
            verdict = "agree"
        elif (side == "LONG" and ml["action"] == "AVOID") or (
            side == "SHORT" and ml["action"] == "BUY"
        ):
            verdict = "conflict"
        rows.append(
            {
                "symbol": r["symbol"],
                "channel": r["channel"],
                "llm_direction": d,
                "llm_conviction": r["conviction"],
                "ml_action": ml["action"],
                "ml_rank": ml["rank"],
                "ml_n": ml["n"],
                "verdict": verdict,
            }
        )
    return pd.DataFrame(rows)


def ml_validation() -> Optional[dict]:
    """Parsed data/ml_validation.json, or None if validation hasn't run."""
    from pathlib import Path

    p = Path(__file__).resolve().parent.parent / "data" / "ml_validation.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def ml_forward_performance(horizon: int = 21) -> dict[str, Any]:
    """Forward BUY−AVOID spread per ML cohort, measured only on signal dates
    that have `horizon` trading days of bars after them. The live,
    contamination-free counterpart to the walk-forward validation panel.
    See alpha_engine.ml.forward_eval for the methodology."""
    from alpha_engine.ml.forward_eval import compute_forward_performance

    with _conn() as con:
        return compute_forward_performance(con, horizon=horizon)


def portfolio_action_center(
    horizon_days: int = 7,
) -> Optional[dict[str, Any]]:
    """Load the real-holdings snapshot (data/real_holdings.json) and run the
    concentration + earnings risk engine, returning the ranked Action Center
    payload. None when no snapshot exists yet."""
    from datetime import date as _date
    from pathlib import Path

    from alpha_engine.risk.earnings_guard import upcoming_earnings
    from alpha_engine.risk.portfolio import concentration_report, rank_actions

    p = Path(__file__).resolve().parent.parent / "data" / "real_holdings.json"
    if not p.exists():
        return None
    try:
        snap = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None

    holdings = snap.get("holdings", [])
    if not holdings:
        return None
    report = concentration_report(holdings)
    cash = float(snap.get("cash") or 0.0)
    cash_weight = cash / (report["total_value"] + cash) if (report["total_value"] + cash) else None

    values = {h["symbol"].upper(): float(h["value"]) for h in holdings}
    try:
        as_of = _date.fromisoformat(snap.get("as_of")) if snap.get("as_of") else _date.today()
    except (TypeError, ValueError):
        as_of = _date.today()
    with _conn() as con:
        earnings = upcoming_earnings(
            con, list(values), as_of, horizon_days=horizon_days, values=values
        )
    actions = rank_actions(report, upcoming_earnings=earnings, cash_weight=cash_weight)

    # Semis-cluster trend + vol-drag, computed from a liquid proxy (SMH,
    # falling back to SOXX). The trend says whether a de-risk is urgent
    # (broken trend) or just right-sizing (intact); the vol-drag is the
    # mean-independent compounding tax the cluster pays.
    from alpha_engine.risk.portfolio import annualized_vol_drag, trend_state

    semis_trend = None
    for proxy in ("SMH", "SOXX"):
        with _conn() as con:
            px = [r[0] for r in con.execute(
                "SELECT adj_close FROM market_bars WHERE symbol = ? ORDER BY bar_date",
                [proxy],
            ).fetchall()]
        ts = trend_state(px, window=200)
        if ts is None:
            continue
        rets = [px[i] / px[i - 1] - 1.0 for i in range(1, len(px))]
        vd = annualized_vol_drag(rets)
        semis_trend = {"proxy": proxy, **ts, **vd}
        break

    return {
        "account": snap.get("account", ""),
        "as_of": snap.get("as_of", ""),
        "cash": cash,
        "cash_weight": cash_weight,
        "total_equity": report["total_value"],
        "report": report,
        "actions": actions,
        "earnings": earnings,
        "semis_trend": semis_trend,
    }


def feedback_loop_behavior() -> dict[str, Any]:
    """Per-model_version behavior comparison (v1 vs v3-fb): conviction-
    calibration slope, duplicate-of-holding share, action mix, and repeated
    misses. See alpha_engine.llm.feedback_eval for the methodology."""
    from alpha_engine.llm.feedback_eval import compute_feedback_loop_behavior

    with _conn() as con:
        return compute_feedback_loop_behavior(con)


def price_history_with_technicals(symbol: str, since: date) -> pd.DataFrame:
    """Price history from `since` with 50/200-day SMA and 14-day RSI columns.

    Pulls ~320 extra calendar days before `since` so the long MA and the
    RSI EMA are fully warmed up at the first displayed row, then clips.
    Columns: date, price, sma50, sma200, rsi14.
    """
    from datetime import timedelta

    import numpy as np

    with _conn() as con:
        df = con.execute(
            "SELECT bar_date AS date, adj_close AS price FROM market_bars "
            "WHERE symbol = ? AND bar_date >= ? ORDER BY bar_date",
            [symbol, since - timedelta(days=320)],
        ).fetch_df()
    if df.empty:
        return df
    df["sma50"] = df["price"].rolling(50).mean()
    df["sma200"] = df["price"].rolling(200).mean()

    delta = df["price"].diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / 14, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1.0 / 14, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    df["rsi14"] = (100.0 - 100.0 / (1.0 + rs)).fillna(50.0)

    df["date"] = pd.to_datetime(df["date"])
    return df[df["date"] >= pd.Timestamp(since)].reset_index(drop=True)


def conviction_calibration() -> pd.DataFrame:
    """Win rate and avg alpha per (channel, conviction bucket) — the same
    calibration table the LLM now sees in its snapshot feedback section.
    Only trades whose horizon has logically completed count."""
    with _conn() as con:
        return con.execute(
            """
            SELECT t.channel,
                   CASE WHEN s.conviction >= 8 THEN '8.0+'
                        WHEN s.conviction >= 7 THEN '7.0-7.9'
                        ELSE '<7.0' END AS bucket,
                   COUNT(*) AS n_scored,
                   AVG(CASE WHEN o.return_pct > 0 THEN 1.0 ELSE 0.0 END) AS win_rate,
                   AVG(o.alpha) AS avg_alpha,
                   AVG(o.return_pct) AS avg_return
            FROM trades t
            JOIN trade_outcomes o ON o.trade_id = t.id
            JOIN signals s ON s.id = t.source_signal_id
            WHERE t.status = 'paper_filled'
              AND s.conviction IS NOT NULL
              AND (t.placed_at::DATE + o.days_held * INTERVAL 1 DAY) <= CURRENT_DATE
            GROUP BY 1, 2
            ORDER BY 1, 2 DESC
            """
        ).df()


def execution_timing_rows() -> pd.DataFrame:
    """Per-trade entry-timing comparison: for every scored trade that has a
    counterfactual price, the return under next-OPEN entry and under
    next-CLOSE entry over the same exit. Works for both cohorts — for a
    next_open trade the actual return IS the open-entry return and
    alt_entry_return_pct is the close-entry one; for a legacy next_close
    trade it's reversed. `gap` = open_return - close_return = the per-trade
    value of entering a session earlier."""
    with _conn() as con:
        df = con.execute(
            """
            SELECT
                t.channel,
                t.entry_style,
                t.symbol,
                CASE WHEN t.entry_style = 'next_open'
                     THEN o.return_pct ELSE o.alt_entry_return_pct END AS open_return,
                CASE WHEN t.entry_style = 'next_open'
                     THEN o.alt_entry_return_pct ELSE o.return_pct END AS close_return,
                o.days_held
            FROM trades t
            JOIN trade_outcomes o ON o.trade_id = t.id
            WHERE o.alt_entry_return_pct IS NOT NULL
              AND (o.notes IS NULL OR o.notes NOT LIKE '%no exit%')
            """
        ).df()
    if not df.empty:
        df["gap"] = df["open_return"] - df["close_return"]
    return df
