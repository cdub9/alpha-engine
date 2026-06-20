"""Page 4 — Daily Run Log tail."""

from __future__ import annotations

import streamlit as st

from dashboard import queries as q


def render() -> None:
    st.title("📜 Daily Run Log")
    st.caption(
        "Tails `data/daily_paper_trade.log`. The scheduled task appends a new "
        "block each weekday at 5:00 PM local time."
    )

    n = st.slider("Lines to show (tail)", 50, 1000, 200, step=50)
    log = q.tail_run_log(n_lines=n)
    if not log:
        st.info(
            "No log file yet (or empty). The first entry will appear after the "
            "scheduled task fires for the first time. To trigger manually:\n\n"
            "```\nschtasks /Run /TN \"AlphaEngine Daily Paper Trade\"\n```"
        )
        return
    st.code(log, language="text")
