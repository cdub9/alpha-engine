"""Page 1 — Today's Suggestions (or any picked date).

Two side-by-side channel panels. Each suggestion is an expandable card
showing direction, conviction, weight, horizon, stop, and the LLM's
rationale + counter-argument.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import streamlit as st

from dashboard import queries as q


DIRECTION_COLORS = {
    "buy": "#1f7a3a",      # green
    "add": "#1f7a3a",
    "hold": "#666666",     # gray
    "sell": "#a02d2d",     # red
    "exit": "#a02d2d",
    "reduce": "#a02d2d",
}
DIRECTION_ICONS = {
    "buy": "▲ BUY",
    "add": "▲ ADD",
    "hold": "■ HOLD",
    "sell": "▼ SELL",
    "exit": "▼ EXIT",
    "reduce": "▼ REDUCE",
}


def _fmt_pct(x) -> str:
    if x is None or pd.isna(x):
        return "—"
    return f"{x*100:+.2f}%"


def _fmt_money(x) -> str:
    if x is None or pd.isna(x):
        return "—"
    return f"${x:,.2f}"


def _render_suggestion(
    row: pd.Series,
    crosscheck_lookup: dict[str, str] | None = None,
    ml_lookup: dict[str, dict] | None = None,
) -> None:
    direction = (row["direction"] or "").lower()
    color = DIRECTION_COLORS.get(direction, "#444")
    icon = DIRECTION_ICONS.get(direction, direction.upper())
    conv = float(row["conviction"] or 0)
    weight = row.get("target_weight")
    horizon = row.get("time_horizon_days")
    stop = row.get("stop_loss_pct")

    # Cross-channel marker (A1+A2)
    cc_status = (crosscheck_lookup or {}).get(row["symbol"])
    cc_badge = ""
    if cc_status == "agreement_long":
        cc_badge = "<span style='color:#1f7a3a' title='Both channels agree (LONG)'>🟢 agrees</span>"
    elif cc_status == "agreement_short":
        cc_badge = "<span style='color:#a02d2d' title='Both channels agree (SHORT)'>🔴 agrees</span>"
    elif cc_status == "contradiction":
        cc_badge = "<span style='color:#c98410' title='Other channel disagrees on this name'>⚠️ disagrees</span>"

    # ML rank badge — the quantitative layer's independent read on this name.
    # Agreement (LLM long + ML BUY) is the strongest combined signal we have;
    # conflict is worth a pause before acting.
    ml = (ml_lookup or {}).get(row["symbol"])
    ml_badge = ""
    if ml:
        is_long = direction in ("buy", "add")
        is_short = direction in ("sell", "exit", "reduce")
        if (is_long and ml["action"] == "BUY") or (is_short and ml["action"] == "AVOID"):
            ml_color, ml_icon = "#1f7a3a", "🤖✓"
        elif (is_long and ml["action"] == "AVOID") or (is_short and ml["action"] == "BUY"):
            ml_color, ml_icon = "#c98410", "🤖✗"
        else:
            ml_color, ml_icon = "#888888", "🤖"
        ml_badge = (
            f"<span style='color:{ml_color}' title='ML signal layer: {ml['action']}, "
            f"rank {ml['rank']} of {ml['n']} by momentum composite ({ml['ml_date']})'>"
            f"{ml_icon} ML {ml['action']} #{ml['rank']}</span>"
        )

    header_bits = [
        f"<span style='color:{color}; font-weight:600'>{icon}</span>",
        f"<span style='font-size:1.1em; font-weight:700'>{row['symbol']}</span>",
        f"conv <b>{conv:.1f}</b>",
    ]
    if cc_badge:
        header_bits.append(cc_badge)
    if ml_badge:
        header_bits.append(ml_badge)
    if weight is not None and not pd.isna(weight):
        header_bits.append(f"weight <b>{float(weight)*100:.0f}%</b>")
    if horizon is not None and not pd.isna(horizon):
        header_bits.append(f"<b>{int(horizon)}d</b> horizon")
    if stop is not None and not pd.isna(stop):
        header_bits.append(f"stop <b>{-abs(float(stop))*100:.0f}%</b>")

    header_html = " &nbsp;·&nbsp; ".join(header_bits)

    # Unrealized P&L badge if a paper trade was opened
    unreal = row.get("unrealized_pct")
    if unreal is not None and not pd.isna(unreal):
        pnl_color = "#1f7a3a" if unreal >= 0 else "#a02d2d"
        header_html += (
            f" &nbsp;·&nbsp; <span style='color:{pnl_color}; font-weight:700'>"
            f"MTM {_fmt_pct(unreal)}</span>"
        )

    st.markdown(
        f"<div style='padding:8px 0; border-bottom:1px solid #eee'>{header_html}</div>",
        unsafe_allow_html=True,
    )

    # Inline TA tag — trend location + RSI, same numbers the LLM sees in its
    # snapshot ("trend +18% above 50-MA, RSI 76" per the FOLLOWUPS spec).
    if ml and ml.get("dist_50ma") is not None and ml.get("rsi_14") is not None:
        d50 = float(ml["dist_50ma"])
        rsi = float(ml["rsi_14"])
        rsi_note = " · stretched" if rsi >= 75 else " · washed out" if rsi <= 25 else ""
        st.caption(
            f"📐 trend {d50:+.1%} vs 50-MA · RSI {rsi:.0f}{rsi_note}"
        )

    with st.expander("rationale & counter-argument", expanded=False):
        st.markdown(f"**Rationale.** {row['rationale'] or '_(none)_'}")
        if row.get("counter_argument"):
            st.markdown(f"**Counter-argument.** {row['counter_argument']}")
        if row.get("entry_price") is not None and not pd.isna(row.get("entry_price")):
            st.caption(
                f"Paper entry {_fmt_money(row['entry_price'])} → "
                f"current {_fmt_money(row['current_price'])}"
            )


def _channel_panel(df: pd.DataFrame, channel: str, label: str,
                   crosscheck_lookup: dict[str, str] | None = None,
                   ml_lookup: dict[str, dict] | None = None) -> None:
    sub = df[df["channel"] == channel].reset_index(drop=True)
    st.subheader(label)
    if sub.empty:
        st.info("No suggestions for this channel on the selected date.")
        return
    st.caption(f"{len(sub)} suggestion(s) · sorted by conviction")
    for _, row in sub.iterrows():
        _render_suggestion(row, crosscheck_lookup=crosscheck_lookup,
                           ml_lookup=ml_lookup)


def _render_data_freshness() -> None:
    """Compact strip showing bar-freshness vs the expected latest trading
    day. Catches silent yfinance failures that would otherwise pollute
    every MTM and chart in the dashboard."""
    fr = q.bar_freshness()
    status = fr["key_status"]
    summary = fr["summary"]
    expected = fr["expected_date"]

    status_map = {
        "fresh": ("🟢", "green", "Fresh"),
        "ok":    ("🟡", "orange", "Slight lag"),
        "stale": ("🔴", "red", "STALE — investigate"),
    }
    icon, color, label = status_map.get(status, ("⚪", "gray", "Unknown"))
    bg_color = {"green": "#163d20", "orange": "#523c0d",
                "red": "#491414", "gray": "#2a2a2a"}.get(color, "#2a2a2a")

    bits = [
        f"<b>Data status</b>",
        f"{icon} {label}",
        f"expected through <b>{expected}</b>",
        f"<b>{summary['fresh']}</b> fresh",
    ]
    if summary["behind_1"]:
        bits.append(f"<b>{summary['behind_1']}</b> 1d behind")
    if summary["behind_2plus"]:
        bits.append(f"<span style='color:#ff8'>⚠️ <b>{summary['behind_2plus']}</b> 2+d behind</span>")
    if summary["stale"]:
        bits.append(f"<span style='color:#f88'>🔴 <b>{summary['stale']}</b> stale</span>")
    if summary["no_data"]:
        bits.append(f"<b>{summary['no_data']}</b> no data")

    line = " &nbsp;·&nbsp; ".join(bits)
    st.markdown(
        f"<div style='padding:6px 14px; background:{bg_color}; border-radius:6px; "
        f"border-left:4px solid {color}; margin-bottom:10px; font-size:0.85em'>{line}</div>",
        unsafe_allow_html=True,
    )

    # Detail expander if anything is stale or we're holding stale positions
    if fr["open_trade_symbols_stale"] or fr["summary"]["stale"]:
        with st.expander(
            f"⚠️ Stale-data details "
            f"({len(fr['open_trade_symbols_stale'])} stale held by open trades)",
            expanded=False,
        ):
            if fr["open_trade_symbols_stale"]:
                st.markdown("**Stale symbols you currently hold (open paper trades):**")
                for r in fr["open_trade_symbols_stale"]:
                    last = r["last_bar"] or "no data"
                    behind = r["days_behind"] or "—"
                    st.markdown(
                        f"&nbsp;&nbsp;🔴 **{r['symbol']}** · last bar {last} · "
                        f"<b>{behind}</b> trading days behind",
                        unsafe_allow_html=True,
                    )
                st.markdown("")
            other_stale = [
                r for r in fr["stale_symbols"]
                if r not in fr["open_trade_symbols_stale"]
            ]
            if other_stale:
                st.markdown(f"**Other stale universe symbols ({len(other_stale)}):**")
                for r in other_stale[:20]:
                    last = r["last_bar"] or "no data"
                    behind = r["days_behind"] or "—"
                    st.markdown(
                        f"&nbsp;&nbsp;• {r['symbol']} · {last} · {behind}d behind"
                    )


def _render_last_run_card() -> None:
    """Small card showing the most recent auto-run result. Pinned to the
    top of Suggestions so you can glance and know if the scheduled task
    fired correctly."""
    info = q.last_run_summary()
    if info is None:
        st.info(
            "🕓 No daily-run log yet. The scheduled task will produce one "
            "the next time it fires (weekdays 5 PM). To trigger manually:\n\n"
            "`schtasks /Run /TN \"AlphaEngine Daily Paper Trade\"`"
        )
        return

    status = info["status"]
    started = info["started_at"]
    age_h = info["age_hours"]
    age_str = (
        f"{age_h*60:.0f} min ago" if age_h is not None and age_h < 1
        else f"{age_h:.1f} h ago" if age_h is not None and age_h < 48
        else f"{age_h/24:.1f} d ago" if age_h is not None
        else "—"
    )
    started_str = started.strftime("%a %Y-%m-%d %H:%M") if started else "?"

    badge_map = {
        "ok": ("🟢 OK", "green"),
        "skipped_weekend": ("⚪ skipped (weekend)", "gray"),
        "skipped_holiday": ("🟡 skipped (holiday)", "orange"),
        "error": ("🔴 ERROR", "red"),
        "unknown": ("⚪ unknown", "gray"),
    }
    badge, color = badge_map.get(status, ("⚪ unknown", "gray"))

    bg_color = {
        "green": "#163d20", "orange": "#523c0d",
        "red": "#491414", "gray": "#2a2a2a",
    }.get(color, "#2a2a2a")

    bits = [f"<b>Last auto-run</b>", f"<b>{started_str}</b> ({age_str})", badge]
    if info["cost_usd"] is not None:
        bits.append(f"cost <b>${info['cost_usd']:.4f}</b>")
    if info["opened"] is not None:
        bits.append(f"opened <b>{info['opened']}</b>")
    if info["scored"] is not None:
        bits.append(f"scored <b>{info['scored']}</b>")
    if info["gdelt_warning"]:
        bits.append("⚠️ GDELT ingest had issues")

    line = " &nbsp;·&nbsp; ".join(bits)

    st.markdown(
        f"<div style='padding:8px 14px; background:{bg_color}; border-radius:6px; "
        f"border-left:4px solid {color}; margin-bottom:12px; font-size:0.9em'>{line}</div>",
        unsafe_allow_html=True,
    )

    if age_h is not None and age_h > 30:
        st.warning(
            f"⚠️  Last run was {age_str} — the daily auto-run may not have fired. "
            "Check the Run Log page or trigger manually with "
            "`schtasks /Run /TN \"AlphaEngine Daily Paper Trade\"`."
        )

    if status == "error":
        reason = info.get("error_reason")
        if reason:
            st.error(f"❌ Last auto-run failed: **{reason}**")
        with st.expander("Tail of last run log", expanded=False):
            st.code(info["last_lines"], language="text")


def _render_action_items_card() -> None:
    """Synthesized "what to look at today" card pinned above the digest.

    Reads four things from the DB and surfaces only the sections that have
    rows — so on a quiet day the card is small or absent rather than full
    of zeros."""
    items = q.today_action_items()
    sections: list[tuple[str, str, list[dict]]] = [
        ("🟢", f"High-conviction picks today (≥ {items['high_conv_threshold']:.1f})", items["new_high_conv"]),
        ("🔴", "Stopped out today (review)", items["stopped_out_today"]),
        ("🟠", f"Due to score in next {items['due_within_days']} days", items["due_soon"]),
        ("⚠️", f"Open positions down past {items['drawdown_threshold']*100:.0f}% (vs entry)", items["drawdown_alerts"]),
    ]
    populated = [(icon, title, rows) for icon, title, rows in sections if rows]

    if not populated:
        st.success(
            "📋 No urgent action items today. "
            "No high-conviction new picks, no stop-outs, nothing due to score soon, "
            "no positions in drawdown. Quiet day."
        )
        return

    total_items = sum(len(rows) for _, _, rows in populated)
    badge = f"{total_items} item{'s' if total_items != 1 else ''} need attention"

    with st.expander(f"📋 **Today's action items** · {badge}", expanded=True):
        for icon, title, rows in populated:
            st.markdown(f"**{icon} {title}** ({len(rows)})")
            for r in rows[:10]:
                bits = [f"`{r.get('channel', '')[:15]:<15s}`", f"**{r['symbol']:<5s}**"]
                if "direction" in r:
                    bits.append(f"{r['direction']}")
                if "conviction" in r:
                    bits.append(f"conv {r['conviction']:.1f}")
                if "unrealized" in r and r["unrealized"] is not None:
                    sign = "+" if r["unrealized"] >= 0 else ""
                    bits.append(f"{sign}{r['unrealized']*100:.1f}%")
                if "return_pct" in r:
                    bits.append(f"realized {r['return_pct']*100:+.1f}%")
                if "days_left" in r:
                    bits.append(f"{r['days_left']}d left")
                if "days_held" in r:
                    bits.append(f"held {r['days_held']}d")
                if r.get("rationale_short"):
                    bits.append(f"_{r['rationale_short'][:90]}…_")
                st.markdown("&nbsp;&nbsp;" + " · ".join(bits), unsafe_allow_html=True)
            if len(rows) > 10:
                st.caption(f"  … and {len(rows) - 10} more")
            st.markdown("")  # spacer


def _render_crosscheck_card(picked_date) -> None:
    """Surface multi-channel agreement/contradiction patterns.

    Both channels independently picking the same name = stronger signal.
    Channels disagreeing on the same name = something to scrutinize.
    """
    cc = q.channel_crosscheck(picked_date)
    if not cc["agreements"] and not cc["contradictions"]:
        return  # nothing notable

    n_agree = len(cc["agreements"])
    n_contra = len(cc["contradictions"])
    title_bits = []
    if n_agree:
        title_bits.append(f"🟢 {n_agree} agreement{'s' if n_agree != 1 else ''}")
    if n_contra:
        title_bits.append(f"⚠️ {n_contra} contradiction{'s' if n_contra != 1 else ''}")
    title = "**Cross-channel signals** · " + " · ".join(title_bits)

    with st.expander(f"🔀 {title}", expanded=True):
        if cc["agreements"]:
            st.markdown(
                "**Both channels agree** (stronger signal — independent confirmation):"
            )
            for r in cc["agreements"]:
                arrow = "▲ LONG" if r["direction_bucket"] == "long" else "▼ SHORT"
                color = "#1f7a3a" if r["direction_bucket"] == "long" else "#a02d2d"
                st.markdown(
                    f"&nbsp;&nbsp;<span style='color:{color}; font-weight:600'>{arrow}</span> "
                    f"**{r['symbol']}** · A: {r['a_dir']} conv <b>{r['a_conv']:.1f}</b> "
                    f"&nbsp;+&nbsp; B: {r['b_dir']} conv <b>{r['b_conv']:.1f}</b> "
                    f"&nbsp;·&nbsp; combined score <b>{r['combined_conv']:.1f}</b>",
                    unsafe_allow_html=True,
                )
            st.markdown("")
        if cc["contradictions"]:
            st.markdown(
                "**Channels disagree** (one is right; scrutinize the higher-conviction side):"
            )
            for r in cc["contradictions"]:
                a_color = "#1f7a3a" if r["a_dir"] in ("buy", "add") else "#a02d2d"
                b_color = "#1f7a3a" if r["b_dir"] in ("buy", "add") else "#a02d2d"
                st.markdown(
                    f"&nbsp;&nbsp;⚠️ **{r['symbol']}** &nbsp;·&nbsp; "
                    f"A: <span style='color:{a_color}; font-weight:600'>{r['a_dir'].upper()}</span> "
                    f"conv <b>{r['a_conv']:.1f}</b> "
                    f"&nbsp;vs&nbsp; "
                    f"B: <span style='color:{b_color}; font-weight:600'>{r['b_dir'].upper()}</span> "
                    f"conv <b>{r['b_conv']:.1f}</b>",
                    unsafe_allow_html=True,
                )
                with st.expander(f"  rationales for {r['symbol']}", expanded=False):
                    st.markdown(f"**A says:** {r['a_rationale']}")
                    st.markdown(f"**B says:** {r['b_rationale']}")


def _render_change_diff_card(picked_date, all_dates) -> None:
    """Render a diff between picked_date and the next-most-recent digest."""
    # Find the cached digest immediately before the picked one
    prior_candidates = [d for d in all_dates if d < picked_date]
    if not prior_candidates:
        return  # No prior to compare against
    prior_date = max(prior_candidates)

    diff = q.digest_diff(picked_date, prior_date)
    if diff["total_changes"] == 0:
        return  # Identical to prior — no point showing the panel

    DIR_COLOR = {
        "buy": "#1f7a3a", "add": "#1f7a3a", "hold": "#666",
        "sell": "#a02d2d", "exit": "#a02d2d", "reduce": "#a02d2d",
    }
    def _dir_chip(d: str) -> str:
        c = DIR_COLOR.get(d, "#444")
        return f"<span style='color:{c}; font-weight:600'>{d.upper()}</span>"

    with st.expander(
        f"🔁 **What changed since {prior_date}** · {diff['total_changes']} item(s)",
        expanded=True,
    ):
        st.caption(
            f"Comparing **{picked_date}** vs **{prior_date}** (the previous "
            f"cached digest). Conviction-only changes shown when |Δ| ≥ 0.5."
        )

        for ch_label, ch_key in (
            ("🟢 steady_alpha", "steady_alpha"),
            ("🚀 aggressive_growth", "aggressive_growth"),
        ):
            buckets = diff[ch_key]
            ch_total = sum(len(v) for v in buckets.values())
            if ch_total == 0:
                continue
            st.markdown(f"**{ch_label}** ({ch_total} changes)")

            if buckets["new"]:
                for r in buckets["new"]:
                    bits = [
                        f"🆕 NEW",
                        f"**{r['symbol']}**",
                        _dir_chip(r['direction']),
                        f"conv <b>{r['conviction']:.1f}</b>",
                        f"<i>{r['rationale'][:80]}…</i>" if r['rationale'] else "",
                    ]
                    st.markdown("&nbsp;&nbsp;" + " · ".join(b for b in bits if b),
                                unsafe_allow_html=True)

            if buckets["flipped"]:
                for r in buckets["flipped"]:
                    st.markdown(
                        f"&nbsp;&nbsp;🔁 FLIPPED · **{r['symbol']}** · "
                        f"{_dir_chip(r['prior_direction'])} → {_dir_chip(r['direction'])} "
                        f"(conv {r['prior_conviction']:.1f} → <b>{r['conviction']:.1f}</b>)",
                        unsafe_allow_html=True,
                    )

            if buckets["conv_up"]:
                for r in buckets["conv_up"]:
                    st.markdown(
                        f"&nbsp;&nbsp;⬆️ CONV UP · **{r['symbol']}** {_dir_chip(r['direction'])} · "
                        f"{r['prior_conviction']:.1f} → <b>{r['conviction']:.1f}</b> "
                        f"(+{r['delta']:.1f})",
                        unsafe_allow_html=True,
                    )

            if buckets["conv_down"]:
                for r in buckets["conv_down"]:
                    st.markdown(
                        f"&nbsp;&nbsp;⬇️ CONV DOWN · **{r['symbol']}** {_dir_chip(r['direction'])} · "
                        f"{r['prior_conviction']:.1f} → <b>{r['conviction']:.1f}</b> "
                        f"({r['delta']:.1f})",
                        unsafe_allow_html=True,
                    )

            if buckets["dropped"]:
                for r in buckets["dropped"]:
                    st.markdown(
                        f"&nbsp;&nbsp;⛔ DROPPED · **{r['symbol']}** "
                        f"(was {_dir_chip(r['prior_direction'])} conv {r['prior_conviction']:.1f})",
                        unsafe_allow_html=True,
                    )
            st.markdown("")  # spacer


def render() -> None:
    st.title("📈 Today's Suggestions")

    _render_data_freshness()
    _render_last_run_card()
    _render_action_items_card()

    dates = q.available_digest_dates()
    if not dates:
        st.warning(
            "No cached digests yet. Run `paper_trader.py run-day --generate` "
            "to create one (~$0.15)."
        )
        return

    col_date, col_meta = st.columns([1, 3])
    with col_date:
        picked: date = st.selectbox(
            "Digest date",
            options=dates,
            index=0,
            format_func=lambda d: d.isoformat(),
        )
    meta = q.digest_meta(picked)
    with col_meta:
        if meta:
            st.markdown(
                f"**Model:** `{meta.get('model_version','?')}` &nbsp;·&nbsp; "
                f"**Cost:** ${meta.get('cost_usd',0):.4f} &nbsp;·&nbsp; "
                f"**Tokens:** {meta.get('input_tokens',0):,} in / "
                f"{meta.get('output_tokens',0):,} out"
            )

    # Market narrative (B6) — the LLM's top-line read + themes + risks
    narrative = q.digest_narrative(picked)
    if narrative["market_summary"] or narrative["key_themes"] or narrative["risk_notes"]:
        st.markdown(
            "<div style='padding:10px 14px; background:#1a2a3a; border-radius:6px; "
            "border-left:4px solid #4a90e2; margin:10px 0 14px 0;'>"
            f"<b>Market read</b> &nbsp;·&nbsp; <i>{narrative['market_summary'] or '(no summary)'}</i>"
            "</div>",
            unsafe_allow_html=True,
        )
        cols = st.columns(2)
        with cols[0]:
            if narrative["key_themes"]:
                st.markdown("**📌 Key themes**")
                for t in narrative["key_themes"]:
                    st.markdown(f"&nbsp;&nbsp;• {t}")
        with cols[1]:
            if narrative["risk_notes"]:
                st.markdown("**⚠️ Risk notes**")
                for r in narrative["risk_notes"]:
                    st.markdown(f"&nbsp;&nbsp;• {r}")

    # "What changed since prior digest" panel (only renders when changes exist)
    _render_change_diff_card(picked, dates)

    # Cross-channel agreement / contradiction panel (only renders when relevant)
    _render_crosscheck_card(picked)

    df = q.suggestions_for_date(picked)
    if df.empty:
        st.info("No persisted signals for this date.")
        return

    crosscheck_lookup = q.channel_crosscheck(picked)["lookup"]

    ml_lookup = q.ml_action_lookup(picked)

    left, right = st.columns(2)
    with left:
        _channel_panel(df, "steady_alpha", "🟢 steady_alpha (SPY +3-5%)",
                       crosscheck_lookup=crosscheck_lookup, ml_lookup=ml_lookup)
    with right:
        _channel_panel(df, "aggressive_growth", "🚀 aggressive_growth (2× SPY)",
                       crosscheck_lookup=crosscheck_lookup, ml_lookup=ml_lookup)
