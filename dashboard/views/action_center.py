"""Page — Action Center.

One question, answered as order tickets: what do I do today? The page leads
with concrete SELL orders (exact shares, dollars, reason, timing) synthesized
from every analysis the app runs — concentration caps, the earnings guard,
the semis-trend state, and the ML ranks. Everything else is tucked into
expanders so the trades are the first and biggest thing on the screen.

Holdings come from data/real_holdings.json, refreshed via the brokerage
connector. Read-only.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard import queries as q

_WHEN_COLOR = {"Now": "#E24B4A"}


def _order_card(o: dict) -> None:
    sh = f"{o['shares']} sh" if o.get("shares") else "—"
    dollars = f"~${o['est_dollars']:,.0f}" if o.get("est_dollars") else ""
    when = o["when"]
    when_color = "#E24B4A" if when == "Now" else ("#BA7517" if when.startswith("Before") else "#888780")
    tgt = f" → {o['target_weight']:.0%}" if o.get("target_weight") is not None else ""
    cur = f"{o['current_weight']:.1%}" if o.get("current_weight") is not None else ""
    st.markdown(
        f"<div style='border:0.5px solid var(--border,#ccc);border-radius:12px;"
        f"padding:0.7rem 1rem;margin-bottom:8px'>"
        f"<div style='display:flex;justify-content:space-between;align-items:baseline'>"
        f"<span style='font-size:16px'><b>{o['action']} {sh} {o['symbol']}</b> "
        f"<span style='font-family:monospace;color:#888'>{dollars}</span></span>"
        f"<span style='font-size:12px;color:#fff;background:{when_color};"
        f"padding:2px 10px;border-radius:999px'>{when}</span></div>"
        f"<div style='font-size:13px;color:var(--text-secondary,#555);margin-top:4px'>"
        f"{o['reason']} <span style='color:#888'>({cur}{tgt})</span></div>"
        + (f"<div style='font-size:12px;color:#888;margin-top:2px'>{o['ml_note']}</div>"
           if o.get("ml_note") else "")
        + "</div>",
        unsafe_allow_html=True,
    )


def render() -> None:
    st.title("🎯 Action Center")

    data = q.portfolio_action_center()
    if data is None:
        st.info(
            "No holdings snapshot yet. Ask the assistant to refresh your "
            "positions via the brokerage connector."
        )
        return

    age = data.get("holdings_age_days") or 0
    if age > 3:
        st.warning(
            f"⚠️ Holdings snapshot is {age} days old ({data['as_of']}). Position "
            "sizes may have drifted — ask the assistant to refresh your positions. "
            "Earnings dates below are current as of today."
        )

    report = data["report"]
    total = data["total_equity"]
    top = report.get("top_name") or {}
    tr = data.get("semis_trend")
    plan = data["plan"]

    # ---- Compact health strip (one line, not four big cards) ----
    semis_w = report["clusters"].get("semis_ai_hw", {}).get("weight", 0.0)
    trend_txt = (
        f"trend {'▲ intact' if tr['above'] else '▼ broken'} ({tr['distance']:+.0%})"
        if tr else ""
    )
    st.caption(
        f"**${total:,.0f}** · semis **{semis_w:.0%}** · top **{top.get('symbol','—')} "
        f"{top.get('weight',0):.0%}** · cash {(data['cash_weight'] or 0):.0%} · {trend_txt}  "
        f"— {data['account']}, {data['as_of']}"
    )

    # ---- TODAY'S TRADES — the hero ----
    st.subheader("Today's trades")
    orders = plan["orders"]
    if not orders:
        st.success("Nothing to do today — no cap breaches or imminent earnings.")
    else:
        n = plan["summary"]["n_orders"]
        raised = plan["summary"]["sell_dollars_now"]
        st.caption(f"{n} order{'s' if n != 1 else ''} · ~${raised:,.0f} to raise, "
                   "then redeploy into cash or diversified holdings (your call).")
        for o in orders:
            _order_card(o)

    # Cluster status (trend-gated de-risk)
    if plan.get("cluster_note"):
        (st.warning if (tr and tr["above"]) else st.error)(plan["cluster_note"])
        if plan["armed"]:
            with st.expander(f"Armed cluster-cut orders ({len(plan['armed'])}) — fire on a trend break"):
                for o in plan["armed"]:
                    _order_card(o)

    # ---- Opportunity ideas — softer, signal-driven, honestly labeled ----
    opp = data.get("opportunity") or {"trims": [], "adds": []}
    if opp["trims"] or opp["adds"]:
        st.subheader("Opportunity ideas")
        st.caption(
            "From the app's return-side signals (ML rank, LLM digest, technicals). "
            "These are IDEAS, not orders — the app's return skill is still "
            "unproven (see the forward-validation panel on Track Record). Weigh "
            "them; don't obey them. Add ideas already respect your risk caps "
            "(nothing suggested that worsens an over-cap cluster)."
        )
        for idea in opp["trims"]:
            w = f" ({idea['weight']:.1%})" if idea.get("weight") is not None else ""
            st.markdown(f"🔻 **Trim {idea['symbol']}**{w} — {idea['reason']}")
        for idea in opp["adds"]:
            w = f" ({idea['weight']:.1%})" if idea.get("weight") is not None else ""
            st.markdown(f"🔼 **Consider adding {idea['symbol']}**{w} — {idea['reason']}")

        # Phase 3 — do these ideas actually work? (forward track record)
        rtr = q.reco_track_record()
        if rtr["n_matured"] == 0:
            st.caption(
                f"📈 Learning loop: {rtr['n_total']} idea(s) logged so far, none "
                f"matured yet — first forward scores (vs {rtr['benchmark']}, "
                f"{rtr['horizon']}-day) in ~{rtr['horizon']} trading days. Until "
                "then these are unproven."
            )
        else:
            add, trim = rtr["by_kind"]["add"], rtr["by_kind"]["trim"]
            bits = []
            if add["n"]:
                bits.append(f"adds {add['hit_rate']:.0%} hit / {add['avg_alpha']:+.1%} avg alpha (n={add['n']})")
            if trim["n"]:
                bits.append(f"trims {trim['hit_rate']:.0%} hit / {trim['avg_alpha']:+.1%} (n={trim['n']})")
            st.caption(
                f"📈 Track record vs {rtr['benchmark']} ({rtr['horizon']}-day): "
                + " · ".join(bits)
                + f" · {rtr['n_pending']} pending. "
                + ("Small sample — directional only." if rtr["n_matured"] < 20 else "")
            )

    st.divider()

    # ---- Market context (the holistic read) ----
    mc = data.get("market_context") or {}
    with st.expander("Market context — regime, themes, geopolitical"):
        if mc.get("regime"):
            st.markdown(
                f"**Regime:** {mc['regime']} "
                f"(confidence {mc.get('regime_confidence', 0):.0%}, "
                f"as of {mc.get('regime_date')})"
            )
        if mc.get("market_summary"):
            st.markdown(f"**Digest read ({mc.get('digest_date')}):** {mc['market_summary']}")
        themes = mc.get("key_themes") or []
        if themes:
            st.markdown("**Key themes:** " + " · ".join(str(t) for t in themes[:5]))
        risks = mc.get("risk_notes") or []
        if risks:
            st.markdown("**Risk notes:**")
            for r in risks[:5]:
                st.markdown(f"  - {r}")
        geo = mc.get("geopolitical") or []
        if geo:
            st.markdown("**Elevated geopolitical signals:** " + ", ".join(
                f"{g['name']}"
                + (f" (tone {g['avg_tone']:+.1f})" if g.get("avg_tone") is not None else "")
                for g in geo
            ))

    # ---- Everything else: collapsed ----
    with st.expander("Why — the analysis behind these trades"):
        for a in data["actions"]:
            icon = {"critical": "🔴", "high": "🟠", "watch": "⚪"}.get(a["severity"], "•")
            st.markdown(f"{icon} **{a['title']}** — {a['rationale']}")
        if tr:
            st.markdown(
                f"📈 **Semis trend:** {tr['proxy']} {tr['distance']:+.0%} vs its "
                f"200-day average; cluster vol ~{tr['vol']:.0%} → ~{tr['drag']:.0%}/yr "
                "compounding drag."
            )

    with st.expander("Where your money is"):
        clusters = report["clusters"]
        rows = sorted(
            ({"Cluster": k, "Weight": v["weight"], "Value": v["value"]}
             for k, v in clusters.items()),
            key=lambda r: r["Weight"], reverse=True,
        )
        st.dataframe(
            pd.DataFrame({
                "Cluster": [r["Cluster"] for r in rows],
                "Weight": [f"{r['Weight']:.1%}" for r in rows],
                "Value": [f"${r['Value']:,.0f}" for r in rows],
            }),
            hide_index=True, use_container_width=True,
        )
        st.caption("Target: no single name > 5%, semis cluster < 20%, total tech < 35%.")

    with st.expander("All positions — with signals"):
        names = report["names"]
        sig = data.get("signals") or {}

        def _rsi(s):
            v = sig.get(s, {}).get("rsi_14")
            return f"{v:.0f}" if v is not None else ""

        def _trend(s):
            v = sig.get(s, {}).get("dist_200ma")
            return f"{v:+.0%}" if v is not None else ""

        def _llm(s):
            d = sig.get(s, {}).get("llm_direction")
            return d or ""

        st.dataframe(
            pd.DataFrame({
                "Symbol": [n["symbol"] for n in names],
                "Value": [f"${n['value']:,.0f}" for n in names],
                "Weight": [f"{n['weight']:.1%}" for n in names],
                "ML": [data["ml_actions"].get(n["symbol"], "") for n in names],
                "vs 200MA": [_trend(n["symbol"]) for n in names],
                "RSI": [_rsi(n["symbol"]) for n in names],
                "LLM": [_llm(n["symbol"]) for n in names],
            }),
            hide_index=True, use_container_width=True,
        )
        st.caption("ML = cross-sectional rank bucket · vs 200MA = trend location · "
                   "RSI = 14-day · LLM = latest digest view (blank if not covered).")
