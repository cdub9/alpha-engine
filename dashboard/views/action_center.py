"""Page — Action Center.

The highest-value risk actions for the real brokerage book, ranked so the
most impactful move (trim the oversized single name, cut the concentrated
cluster, trim into earnings) is one glance away instead of buried across 60+
rows. Backed by alpha_engine.risk.portfolio + earnings_guard.

Holdings come from data/real_holdings.json (refreshed by pulling positions
from the brokerage). Read-only.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard import queries as q

_SEV = {
    "critical": ("🔴", "Critical"),
    "high": ("🟠", "High"),
    "watch": ("⚪", "Watch"),
}

# Stacked concentration bar order + colors (dark-mode safe hexes).
_BAND_ORDER = [
    ("semis_ai_hw", "Semis/AI-HW", "#E24B4A"),
    ("tech_growth_etf", "Tech ETF", "#EF9F27"),
    ("broad_index", "Broad", "#378ADD"),
    ("international", "Intl", "#1D9E75"),
    ("defensive", "Defensive", "#7F77DD"),
    ("bonds_credit", "Bonds", "#888780"),
    ("leveraged", "Leveraged", "#D4537E"),
    ("other", "Other", "#D3D1C7"),
]


def _action_row(a: dict) -> None:
    icon, label = _SEV.get(a["severity"], ("•", a["severity"].title()))
    c1, c2 = st.columns([5, 1])
    with c1:
        st.markdown(f"{icon} **{a['title']}**")
        st.caption(a["rationale"])
    with c2:
        st.markdown(
            f"<div style='text-align:right;font-family:monospace'>"
            f"{a['metric_now']} → {a['metric_target']}</div>",
            unsafe_allow_html=True,
        )


def render() -> None:
    st.title("🎯 Action Center")
    st.caption(
        "Your real book, ranked by what to fix first. Concentration and "
        "earnings risk on the highest-impact positions — act from the top."
    )

    data = q.portfolio_action_center()
    if data is None:
        st.info(
            "No holdings snapshot yet. Pull your brokerage positions into "
            "`data/real_holdings.json` to populate this page."
        )
        return

    report = data["report"]
    total = data["total_equity"]
    top = report.get("top_name") or {}

    # Headline risk metrics
    semis_w = report["clusters"].get("semis_ai_hw", {}).get("weight", 0.0)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Account value", f"${total:,.0f}")
    c2.metric("Semis / AI-hardware", f"{semis_w:.1%}",
              help="Share of the book in one correlated semiconductor cluster. Target < 20%.")
    c3.metric("Biggest single name",
              f"{top.get('symbol','—')} {top.get('weight',0):.1%}" if top else "—",
              help="Largest single-stock position. Target < 5%.")
    c4.metric("Cash", f"{(data['cash_weight'] or 0):.1%}")

    st.caption(f"Account {data['account']} · snapshot {data['as_of']}")
    st.divider()

    # The ranked actions — the centerpiece
    st.subheader("Do these first")
    actions = data["actions"]
    if not actions:
        st.success("No cap breaches or imminent earnings. Book looks balanced.")
    else:
        for a in actions:
            _action_row(a)
            st.markdown("")

    st.divider()

    # Concentration bar
    st.subheader("Where your money is")
    clusters = report["clusters"]
    bands = [(label, clusters.get(key, {}).get("weight", 0.0), color)
             for key, label, color in _BAND_ORDER
             if clusters.get(key, {}).get("weight", 0.0) > 0]
    bar = "".join(
        f"<div style='width:{w*100:.1f}%;background:{color};color:#fff;"
        f"font-size:11px;display:flex;align-items:center;justify-content:center;"
        f"overflow:hidden;white-space:nowrap' title='{label} {w:.1%}'>"
        f"{label if w > 0.06 else ''}</div>"
        for label, w, color in bands
    )
    st.markdown(
        f"<div style='display:flex;height:28px;border-radius:6px;overflow:hidden'>{bar}</div>",
        unsafe_allow_html=True,
    )
    st.caption("Target: no single name > 5%, semis cluster < 20%, total tech < 35%.")

    # Top positions table (the detail, available but not in the way)
    with st.expander("All positions by weight"):
        names = report["names"]
        view = pd.DataFrame({
            "Symbol": [n["symbol"] for n in names],
            "Value": [f"${n['value']:,.0f}" for n in names],
            "Weight": [f"{n['weight']:.1%}" for n in names],
        })
        st.dataframe(view, hide_index=True, use_container_width=True)
