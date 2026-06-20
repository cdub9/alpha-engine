"""Page 3 — Track Record (per-channel summary + cumulative alpha curves)."""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from dashboard import glossary as gl
from dashboard import queries as q


def _channel_card(row: pd.Series) -> None:
    ch = row["channel"]
    n = int(row.get("n_scored") or 0)
    if n == 0:
        st.metric(ch, "—", help="No scored trades yet")
        return
    avg_ret = float(row.get("avg_ret") or 0)
    avg_alpha = float(row.get("avg_alpha") or 0)
    win_rate = float(row.get("win_rate") or 0)
    pf = row.get("profit_factor")
    pf_str = f"{pf:.2f}" if pf is not None and not pd.isna(pf) else "—"
    st.markdown(f"#### {ch}")
    a, b, c, d, e = st.columns(5)
    a.metric("Scored trades", f"{n}", help="Number of paper trades whose horizon has elapsed and have a realized outcome.")
    b.metric("Avg return", f"{avg_ret*100:+.2f}%", help=gl.RETURN_PCT)
    c.metric("Avg alpha", f"{avg_alpha*100:+.2f}%", help=gl.ALPHA)
    d.metric("Win rate", f"{win_rate*100:.0f}%", help=gl.WIN_RATE)
    e.metric("Profit factor", pf_str, help=gl.PROFIT_FACTOR)


def _pct(x, signed=True) -> str:
    if x is None:
        return "—"
    return f"{x:+.2%}" if signed else f"{x:.0%}"


def _render_feedback_loop() -> None:
    """Did the v3 self-learning loop change behavior? Compares signal
    cohorts (model_version) on conviction-calibration slope, wasteful
    re-buy share, action mix, and repeated misses. Some metrics need
    matured v3 trades and stay pending until ~3 weeks of forward data."""
    fb = q.feedback_loop_behavior()
    order = fb.get("order", [])
    if len(order) < 2:
        return  # nothing to compare against until a second cohort exists

    st.subheader(
        "🔁 Feedback loop: did v3 change behavior?",
        help=(
            "Since the v3-fb prompt, the model sees its own open book and "
            "track record every day. This compares signal cohorts by "
            "model_version. Calibration slope and repeated misses need "
            "matured trades, so a fresh v3 cohort shows them as pending; "
            "re-buy share and action mix are available immediately."
        ),
    )
    cohorts = fb["cohorts"]

    # Side-by-side comparison table: one column per cohort.
    metrics = [
        ("Signals", lambda c: f"{c['n_signals']:,}"),
        ("Matured trades", lambda c: f"{c['n_matured']:,}"),
        ("New-buy re-buy share",
         lambda c: (f"{c['dup_share']:.0%} ({c['n_dup_buys']}/{c['n_new_buys']})"
                    if c["dup_share"] is not None else "—")),
        ("Calibration slope (8.0+ − <7.0 alpha)",
         lambda c: (_pct(c["calib_slope"])
                    + ("" if c["calib_slope_reliable"] else " (thin)")
                    if c["calib_slope"] is not None else "pending")),
        ("Repeated misses", lambda c: str(len(c["repeated_misses"]))),
    ]
    table = {"Metric": [label for label, _ in metrics]}
    for mv in order:
        c = cohorts[mv]
        table[c["label"]] = [fn(c) for _, fn in metrics]
    st.dataframe(pd.DataFrame(table), hide_index=True, use_container_width=True)
    st.caption(
        "Re-buy share = `buy` (open-new) picks that duplicated a name already "
        "held when generated; lower is tighter book discipline. Calibration "
        "slope > 0 means high-conviction picks beat low-conviction ones "
        "(the scale is right-side up). “(thin)” = a bucket under ~10 trades."
    )

    # Action-mix breakdown per cohort — the clearest early tell of the loop
    # nudging the model toward managing its book (more exit/reduce/hold).
    mix_rows = []
    all_dirs = sorted({d for mv in order for d in cohorts[mv]["action_mix"]})
    for mv in order:
        c = cohorts[mv]
        total = max(c["n_signals"], 1)
        row = {"Cohort": c["label"]}
        for d in all_dirs:
            n = c["action_mix"].get(d, 0)
            row[d] = f"{n} ({n/total:.0%})"
        mix_rows.append(row)
    with st.expander("Action mix per cohort"):
        st.dataframe(pd.DataFrame(mix_rows), hide_index=True,
                     use_container_width=True)
        st.caption(
            "Share of each cohort's signals by direction. A shift toward "
            "exit/reduce/hold is the feedback loop prompting the model to "
            "tend its existing positions rather than only open new ones."
        )


def render() -> None:
    st.title("📊 Track Record")

    # Survivorship-bias warning if any scored trade is an individual equity
    affected = q.survivorship_affected_in_scored()
    if affected:
        from alpha_engine.backtest.warnings import survivorship_warning_text

        with st.expander(
            f"⚠️  Survivorship-bias warning ({len(affected)} individual equities present)",
            expanded=False,
        ):
            st.markdown(survivorship_warning_text(affected, include_header=False))

    counts = q.total_counts()
    h1, h2, h3, h4 = st.columns(4)
    h1.metric("Total signals", f"{counts['signals']:,}",
              help="All LLM-generated signals ever persisted (incl. historical backfill).")
    h2.metric("Open trades", f"{counts['open_trades']:,}",
              help="Paper trades currently in their horizon window (not yet scored).")
    h3.metric("Scored trades", f"{counts['scored_trades']:,}",
              help="Paper trades that have completed and have a realized outcome in trade_outcomes.")
    h4.metric("Digests cached", f"{counts['digests']:,}",
              help="LLM digests in llm_signal_cache. Each cached digest represents one paid API call.")

    forward_only = st.toggle(
        "Forward-only (exclude pre-today backfill)",
        value=False,
        help=(
            "When ON, only counts paper trades placed on or after today. "
            "Historical backfill stats are training-data contaminated — "
            "the Opus model may 'remember' past outcomes. Forward stats "
            "are the only clean signal-quality measurement."
        ),
    )

    stats = q.channel_stats(forward_only=forward_only)
    if stats.empty:
        st.info(
            "No scored trades match the current filter. "
            "If you just turned on Forward-only, that's expected — clean "
            "data starts accumulating after the next auto-run."
        )
    else:
        for _, row in stats.iterrows():
            _channel_card(row)
            st.markdown("---")

    # Conviction calibration — the same table the LLM sees in its snapshot
    # feedback section since the v3-fb prompt. If 8.0+ underperforms <7.0,
    # the model's conviction scale is inverted and (per its new operating
    # principle) it should be self-correcting in subsequent digests.
    st.subheader(
        "🎯 Conviction calibration",
        help=(
            "Does higher conviction actually mean better outcomes? Win rate "
            "and average alpha per conviction bucket, completed trades only. "
            "The LLM sees this exact table in its daily snapshot and is "
            "instructed to recalibrate when buckets invert. Buckets under "
            "~10 trades are noise."
        ),
    )
    calib = q.conviction_calibration()
    if calib.empty:
        st.info("No completed trades with conviction data yet.")
    else:
        view = pd.DataFrame({
            "Channel": calib["channel"],
            "Conviction": calib["bucket"],
            "Scored": calib["n_scored"],
            "Win rate": calib["win_rate"].map(lambda x: f"{x:.0%}"),
            "Avg alpha": calib["avg_alpha"].map(lambda x: f"{x:+.2%}"),
            "Avg return": calib["avg_return"].map(lambda x: f"{x:+.2%}"),
        })
        st.dataframe(view, hide_index=True, use_container_width=True)
        # Flag inversion loudly — it's the single most actionable read here
        for ch in calib["channel"].unique():
            sub = calib[calib["channel"] == ch].set_index("bucket")
            if "8.0+" in sub.index and "<7.0" in sub.index:
                hi, lo = sub.loc["8.0+"], sub.loc["<7.0"]
                if hi["n_scored"] >= 10 and lo["n_scored"] >= 10 and hi["avg_alpha"] < lo["avg_alpha"]:
                    st.warning(
                        f"⚠️ **{ch}**: high-conviction (8.0+) picks are "
                        f"underperforming low-conviction (<7.0) picks "
                        f"({hi['avg_alpha']:+.1%} vs {lo['avg_alpha']:+.1%} avg alpha) — "
                        "the conviction scale is inverted. The model sees this "
                        "in its snapshot; watch whether it self-corrects."
                    )

    _render_feedback_loop()

    # Execution timing — what the next-open entry switch is worth. Every
    # scored trade carries the counterfactual return under the other entry
    # style (same exit), so this measures the latency gap directly.
    st.subheader(
        "⏱️ Execution timing: next-open vs next-close entry",
        help=(
            "The digest is generated after the close, so the tightest honest "
            "fill is the next session's OPEN (a market-on-open order). Entering "
            "at the next CLOSE instead cedes a full trading session. Each scored "
            "trade stores both returns over the same exit; the gap is the "
            "per-trade value of entering a session earlier."
        ),
    )
    timing = q.execution_timing_rows()
    if timing.empty:
        st.info("No scored trades with entry-timing data yet.")
    else:
        gap_mean = timing["gap"].mean()
        share_pos = (timing["gap"] > 0).mean()
        n_fwd = int((timing["entry_style"] == "next_open").sum())
        c1, c2, c3 = st.columns(3)
        c1.metric(
            "Avg gap (open − close)", f"{gap_mean:+.2%}",
            help="Average per-trade return advantage of next-open over next-close, same exit.",
        )
        c2.metric(
            "Trades where open wins", f"{share_pos:.0%}",
            help="Share of trades where entering at the next open beat the next close.",
        )
        c3.metric(
            "Live next-open trades", f"{n_fwd}",
            help="Scored trades actually executed under the new next-open rule "
                 "(the rest are historical next-close trades measured counterfactually).",
        )
        if gap_mean > 0:
            st.success(
                f"Entering at the next open has been worth **{gap_mean:+.2%} per trade** "
                f"on average across {len(timing)} scored trades — the latency the "
                "next-open switch removes. New trades now fill this way automatically."
            )
        else:
            st.info(
                f"Across {len(timing)} scored trades the next-open entry averaged "
                f"{gap_mean:+.2%} vs next-close — no material edge in this sample, "
                "but it removes a real execution risk at zero cost."
            )
        by_ch = (
            timing.groupby("channel")
            .agg(n=("gap", "size"), avg_gap=("gap", "mean"),
                 open_wins=("gap", lambda s: (s > 0).mean()))
            .reset_index()
        )
        by_ch["avg_gap"] = by_ch["avg_gap"].map(lambda x: f"{x:+.2%}")
        by_ch["open_wins"] = by_ch["open_wins"].map(lambda x: f"{x:.0%}")
        by_ch.columns = ["Channel", "Scored", "Avg gap", "Open wins"]
        st.dataframe(by_ch, hide_index=True, use_container_width=True)

    # Per-symbol top alphas
    st.subheader("Top symbols by alpha (≥2 trades)")
    tab_a, tab_b = st.tabs(["steady_alpha", "aggressive_growth"])
    for tab, ch in [(tab_a, "steady_alpha"), (tab_b, "aggressive_growth")]:
        with tab:
            sym_df = q.per_symbol_stats(ch)
            if sym_df.empty:
                st.info("No symbol-level data yet.")
                continue
            sym_df = sym_df.sort_values("avg_alpha", ascending=False).head(15)
            chart = (
                alt.Chart(sym_df)
                .mark_bar()
                .encode(
                    x=alt.X("avg_alpha:Q", title="Avg alpha", axis=alt.Axis(format="%")),
                    y=alt.Y("symbol:N", sort="-x", title=None),
                    color=alt.condition(
                        alt.datum.avg_alpha > 0,
                        alt.value("#1f7a3a"),
                        alt.value("#a02d2d"),
                    ),
                    tooltip=[
                        alt.Tooltip("symbol:N"),
                        alt.Tooltip("n:Q", title="trades"),
                        alt.Tooltip("avg_ret:Q", format=".2%", title="avg return"),
                        alt.Tooltip("avg_alpha:Q", format=".2%", title="avg alpha"),
                        alt.Tooltip("win_rate:Q", format=".0%", title="win rate"),
                    ],
                )
                .properties(height=400)
            )
            st.altair_chart(chart, use_container_width=True)

    # Virtual portfolio NAV vs SPY (D1+D2 combined)
    st.subheader("Virtual portfolio: $100K following each channel vs SPY")
    pos_pct = st.slider(
        "Position size (% of NAV per trade)",
        min_value=0.01, max_value=0.20, value=0.05, step=0.01,
        format="%.2f",
        help=(
            "Each new BUY/SHORT opens a position sized at this % of current NAV. "
            "5% means up to ~20 concurrent positions before running out of cash. "
            "Lower = more diversified, smaller per-trade impact."
        ),
    )
    sim = q.simulate_virtual_portfolio(initial=100_000.0, position_size_pct=pos_pct)
    nav_df = sim["nav_curve"]
    if nav_df.empty:
        st.info("No trades to simulate yet — chart will populate as paper trades accumulate.")
    else:
        st.caption(
            f"Multi-position simulation: each trade sized at {pos_pct*100:.0f}% of NAV, "
            f"cash earns 0%, no commissions/slippage. Open trades MTM'd to latest bar."
        )
        domain = sorted(nav_df["series"].unique())
        range_ = []
        for s in domain:
            if "steady" in s:
                range_.append("#1f7a3a")
            elif "aggressive" in s:
                range_.append("#a02d2d")
            else:
                range_.append("#888888")
        nav_chart = (
            alt.Chart(nav_df)
            .mark_line()
            .encode(
                x=alt.X("date:T", title=None),
                y=alt.Y("nav:Q", title="NAV ($)", scale=alt.Scale(zero=False)),
                color=alt.Color(
                    "series:N",
                    scale=alt.Scale(domain=domain, range=range_),
                    legend=alt.Legend(orient="bottom", title=None),
                ),
                tooltip=[
                    alt.Tooltip("date:T", title="Date"),
                    alt.Tooltip("series:N", title="Series"),
                    alt.Tooltip("nav:Q", format="$,.0f", title="NAV"),
                ],
            )
            .properties(height=340)
        )
        st.altair_chart(nav_chart, use_container_width=True)

        # Final NAV summary row
        final_nav = sim["final_nav"]
        if final_nav:
            cols = st.columns(len(final_nav))
            for col, (series, nav) in zip(cols, sorted(final_nav.items())):
                pnl = nav - 100_000.0
                col.metric(
                    series,
                    f"${nav:,.0f}",
                    delta=f"{pnl/100_000*100:+.1f}%",
                    help=gl.NAV,
                )

        # D8: Current "what would I own?" snapshot
        st.markdown("---")
        st.subheader("Current paper-portfolio holdings (sim end-state)")
        st.caption(
            "Positions still open in the virtual portfolio as of the latest sim event. "
            "These are 'what would you own right now if you'd followed every signal at "
            f"{pos_pct*100:.0f}% sizing.'"
        )
        any_open = False
        for channel, positions in sim["final_open"].items():
            if not positions:
                continue
            any_open = True
            st.markdown(f"**{channel}** ({len(positions)} open)")
            pos_df = pd.DataFrame(positions).rename(columns={
                "symbol": "Sym",
                "entry_date": "Entry",
                "entry_value": "Cost $",
                "current_value": "MTM $",
                "unrealized_pct": "Unrealized",
                "trade_id": "ID",
            })
            pos_df = pos_df[["ID", "Entry", "Sym", "Cost $", "MTM $", "Unrealized"]].sort_values(
                "Unrealized", ascending=False, na_position="last"
            )
            total_cost = pos_df["Cost $"].sum()
            total_mtm = pos_df["MTM $"].sum()
            cash = sim["final_nav"][channel] - total_mtm
            styled = pos_df.style.format({
                "Cost $": "${:,.0f}", "MTM $": "${:,.0f}",
                "Unrealized": "{:+.2%}",
            }).map(
                lambda v: ("color: #1f7a3a" if v >= 0 else "color: #a02d2d")
                if isinstance(v, float) else "",
                subset=["Unrealized"],
            )
            st.dataframe(styled, hide_index=True, use_container_width=True)
            st.caption(
                f"  Cost basis: ${total_cost:,.0f} · MTM: ${total_mtm:,.0f} · "
                f"Cash: ${cash:,.0f} · NAV: ${sim['final_nav'][channel]:,.0f}"
            )
        if not any_open:
            st.caption("All positions closed at end of simulation.")

    # Cumulative alpha curve
    st.subheader("Cumulative alpha over time")
    curve_data = []
    for ch in ("steady_alpha", "aggressive_growth"):
        c = q.cumulative_alpha_curve(ch)
        if c.empty:
            continue
        c = c[["d", "cum_alpha"]].copy()
        c["channel"] = ch
        curve_data.append(c)
    if curve_data:
        combined = pd.concat(curve_data, ignore_index=True)
        chart = (
            alt.Chart(combined)
            .mark_line(point=False)
            .encode(
                x=alt.X("d:T", title="Evaluation date"),
                y=alt.Y("cum_alpha:Q", title="Cumulative alpha", axis=alt.Axis(format="%")),
                color=alt.Color("channel:N", legend=alt.Legend(orient="bottom")),
                tooltip=["d:T", "channel:N", alt.Tooltip("cum_alpha:Q", format=".2%")],
            )
            .properties(height=320)
        )
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("No scored trades to plot yet.")
