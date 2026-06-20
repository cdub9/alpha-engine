"""Page 2 — Open Paper Trades with live MTM."""

from __future__ import annotations

from datetime import timedelta

import altair as alt
import pandas as pd
import streamlit as st

from dashboard import glossary as gl
from dashboard import queries as q


def _color_pnl(val):
    if pd.isna(val):
        return ""
    color = "#1f7a3a" if val >= 0 else "#a02d2d"
    return f"color: {color}; font-weight: 600"


def render() -> None:
    st.title("📂 Open Paper Trades")

    df = q.open_paper_trades_mtm()
    if df.empty:
        st.info("No open paper trades. Run `paper_trader.py open` after generating a digest.")
        return

    st.caption(
        f"{len(df)} open trade(s). Mark date is each symbol's latest bar in our DB."
    )

    # Filters
    f1, f2 = st.columns([1, 1])
    with f1:
        channels = ["(all)"] + sorted(df["channel"].dropna().unique().tolist())
        ch = st.selectbox("Channel", channels, index=0)
    with f2:
        order = st.selectbox(
            "Sort by",
            ["unrealized desc", "unrealized asc", "days_left asc", "entry_date desc"],
            index=0,
        )

    view = df.copy()
    if ch != "(all)":
        view = view[view["channel"] == ch]
    sort_map = {
        "unrealized desc":   ("unrealized", False),
        "unrealized asc":    ("unrealized", True),
        "days_left asc":     ("days_left", True),
        "entry_date desc":   ("entry_date", False),
    }
    col, asc = sort_map[order]
    view = view.sort_values(col, ascending=asc, na_position="last")

    # Aggregate stats
    a, b, c, d = st.columns(4)
    a.metric("Open trades", len(view), help="Paper trades currently in their horizon window.")
    avg = view["unrealized"].mean()
    b.metric("Avg unrealized", f"{(avg or 0)*100:+.2f}%", help=gl.UNREALIZED)
    winners = (view["unrealized"] > 0).sum()
    c.metric("In the green", f"{winners}/{len(view)}",
             help="How many open positions are currently profitable on a mark-to-market basis.")
    due_soon = (view["days_left"] <= 7).sum()
    d.metric("Due within 7d", int(due_soon),
             help="Positions whose horizon ends within 7 days. Scorer will close these next run.")

    # Pretty table
    show = view[[
        "entry_date", "channel", "symbol", "direction",
        "entry_px", "current_px", "unrealized",
        "days_held", "days_left", "conviction", "rationale",
    ]].rename(columns={
        "entry_date": "Entry",
        "channel": "Channel",
        "symbol": "Sym",
        "direction": "Dir",
        "entry_px": "Entry $",
        "current_px": "Now $",
        "unrealized": "Unrealized",
        "days_held": "Held",
        "days_left": "Left",
        "conviction": "Conv",
        "rationale": "Why",
    })

    styled = (
        show.style
        .format({
            "Entry $": "${:,.2f}",
            "Now $": "${:,.2f}",
            "Unrealized": "{:+.2%}",
            "Conv": "{:.1f}",
        })
        .map(_color_pnl, subset=["Unrealized"])
    )
    st.dataframe(styled, use_container_width=True, hide_index=True, height=500)

    # Drill-down: pick a trade ID to see full context + price chart
    st.markdown("---")
    st.subheader("Trade detail")
    ids = view["id"].tolist()
    if not ids:
        return
    labels = {
        int(r.id): f"#{int(r.id)}  {r.symbol:<6}  {r.channel:<18}  entry {r.entry_date}"
        for r in view.itertuples()
    }
    picked_id = st.selectbox(
        "Drill into a trade",
        options=ids,
        format_func=lambda i: labels[int(i)],
        index=0,
        key="open_trades_drilldown",
    )
    _render_trade_detail(int(picked_id))


def _render_trade_detail(trade_id: int) -> None:
    d = q.trade_detail(trade_id)
    if not d:
        st.warning("Trade not found.")
        return

    # Header metrics
    a, b, c, d_col = st.columns(4)
    a.metric("Symbol", d["symbol"])
    b.metric("Channel", d["channel"], help="steady_alpha targets SPY +3-5%; aggressive_growth targets 2× SPY.")
    c.metric("Entry $", f"${d['entry_px']:.2f}" if d["entry_px"] else "—",
             help="Adj close on the first trading day after the digest.")
    unreal = d["unrealized_pct"]
    if unreal is not None:
        d_col.metric(
            "Unrealized",
            f"{unreal*100:+.2f}%",
            delta=f"${(d['current_px']-d['entry_px']):+.2f}",
            help=gl.UNREALIZED,
        )
    else:
        d_col.metric("Unrealized", "—")

    # Sub-row of signal context
    e, f, g, h = st.columns(4)
    e.metric("Conviction", f"{d['conviction']:.1f}" if d["conviction"] else "—",
             help=gl.CONVICTION)
    f.metric(
        "Target weight",
        f"{d['target_weight']*100:.1f}%" if d["target_weight"] else "—",
        help=gl.TARGET_WEIGHT,
    )
    g.metric("Horizon", f"{int(d['time_horizon_days'])}d" if d["time_horizon_days"] else "—",
             help=gl.TIME_HORIZON)
    stop = d["stop_loss_pct"]
    h.metric("Stop loss", f"{-abs(stop)*100:.1f}%" if stop else "—",
             help=gl.STOP_LOSS_PCT)

    # Rationale & counter
    st.markdown("**Rationale.** " + (d.get("rationale") or "_(none)_"))
    if d.get("counter_argument"):
        st.markdown("**Counter-argument.** " + d["counter_argument"])

    # Price chart since entry with entry/stop lines + 50/200-MA overlays
    # and an RSI subplot (TA shipped with the v2-ta prompt — same numbers
    # the LLM now sees in its snapshot).
    since = d["entry_date"] - timedelta(days=5)
    bars = q.price_history_with_technicals(d["symbol"], since)
    if bars.empty:
        st.info("No price history available.")
        return

    base = alt.Chart(bars).encode(x=alt.X("date:T", title=None))
    line = base.mark_line(color="#1f77b4").encode(
        y=alt.Y("price:Q", title="Adj close", scale=alt.Scale(zero=False)),
        tooltip=["date:T", alt.Tooltip("price:Q", format="$,.2f")],
    )
    sma50_line = base.mark_line(color="#c98410", strokeWidth=1.2).encode(
        y="sma50:Q",
        tooltip=["date:T", alt.Tooltip("sma50:Q", format="$,.2f", title="50-day MA")],
    )
    sma200_line = base.mark_line(color="#7a4fb0", strokeWidth=1.2).encode(
        y="sma200:Q",
        tooltip=["date:T", alt.Tooltip("sma200:Q", format="$,.2f", title="200-day MA")],
    )

    # Reference lines: entry price and stop level
    layers = [line, sma50_line, sma200_line]
    if d["entry_px"]:
        entry_rule = alt.Chart(pd.DataFrame({"y": [d["entry_px"]]})).mark_rule(
            color="#666", strokeDash=[4, 4]
        ).encode(y="y:Q")
        layers.append(entry_rule)
    if d["entry_px"] and stop:
        stop_price = d["entry_px"] * (1 - abs(stop))
        stop_rule = alt.Chart(pd.DataFrame({"y": [stop_price]})).mark_rule(
            color="#a02d2d", strokeDash=[6, 4]
        ).encode(y="y:Q")
        layers.append(stop_rule)

    chart = alt.layer(*layers).properties(height=320)
    st.altair_chart(chart, use_container_width=True)

    rsi_base = alt.Chart(bars).encode(x=alt.X("date:T", title=None))
    rsi_line = rsi_base.mark_line(color="#1f77b4").encode(
        y=alt.Y("rsi14:Q", title="RSI-14", scale=alt.Scale(domain=[0, 100])),
        tooltip=["date:T", alt.Tooltip("rsi14:Q", format=".0f", title="RSI")],
    )
    rsi_bands = alt.Chart(pd.DataFrame({"y": [30, 70]})).mark_rule(
        color="#888", strokeDash=[3, 3]
    ).encode(y="y:Q")
    st.altair_chart(alt.layer(rsi_line, rsi_bands).properties(height=110),
                    use_container_width=True)
    st.caption(
        "Blue = adj close · orange = 50-day MA · purple = 200-day MA. "
        "Dashed gray = entry price; dashed red = stop level. "
        "RSI subplot: above 70 = stretched, below 30 = washed out."
    )
