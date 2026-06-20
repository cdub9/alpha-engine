"""Page 5 — Snapshot Inspector.

Shows the markdown snapshot the LLM was (or would be) shown for a picked
date. Useful for "why did it pick this?" — you can see the regime, macro
features, calendar context, geopolitical state, and per-ticker notes that
fed into the suggestions.

Note: this rebuilds the snapshot live from current DB state. Macro and
calendar features are deterministic — they'll match what the live digest
saw. GDELT only has ~30 days of history, so older dates will show "no
data" in that section (a known limitation, documented in FOLLOWUPS).
"""

from __future__ import annotations

from datetime import date

import streamlit as st

from dashboard import glossary as gl
from dashboard import queries as q


def render() -> None:
    st.title("🔍 Snapshot Inspector")
    st.caption(
        "The market context the LLM saw before generating suggestions. "
        "Rebuilt live from the DB — macro/calendar match the live digest. "
        "GDELT history is ~30 days, so older dates show empty geopolitical."
    )

    dates = q.available_digest_dates()
    if dates:
        default_date = dates[0]
    else:
        default_date = date.today()

    picked: date = st.date_input(
        "Date",
        value=default_date,
        help="Any date. Picking one in your cached-digest list lets you cross-reference with the Suggestions page.",
    )

    try:
        markdown, notable, regime, conf = q.build_snapshot_markdown(picked)
    except Exception as e:  # noqa: BLE001 — show error to user
        st.error(f"Snapshot build failed: {e}")
        return

    # Header
    a, b, c = st.columns(3)
    a.metric("Regime", regime.upper(), help=gl.REGIME)
    b.metric("Confidence", f"{conf:.0%}", help=gl.REGIME_CONFIDENCE)
    c.metric("Notable events", str(len(notable)),
             help="Calendar items the snapshot flagged: FOMC week, OpEx, earnings within 5 days, etc.")

    if notable:
        st.markdown("**Notable events:**")
        for ev in notable:
            st.markdown(f"- {ev}")
        st.markdown("---")

    st.markdown(markdown)
