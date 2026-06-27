"""AlphaEngine local dashboard — Streamlit entry point.

Launch with:
    .venv\\Scripts\\streamlit run dashboard/app.py
or via:
    scripts\\dashboard.bat

Read-only — no API calls, no writes. Safe to leave running.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is importable when streamlit launches us from anywhere
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st  # noqa: E402
from streamlit_autorefresh import st_autorefresh  # noqa: E402

from dashboard.views import (  # noqa: E402
    action_center,
    lookup,
    ml_signals,
    open_trades,
    run_log,
    snapshot,
    suggestions,
    track_record,
)


def main() -> None:
    st.set_page_config(
        page_title="AlphaEngine",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    pages = {
        "Trading": [
            st.Page(action_center.render, title="Action Center", icon="🎯", url_path="action-center", default=True),
            st.Page(suggestions.render, title="Suggestions", icon="📈", url_path="suggestions"),
            st.Page(ml_signals.render, title="ML Signals", icon="🤖", url_path="ml-signals"),
            st.Page(open_trades.render, title="Open Trades", icon="📂", url_path="open-trades"),
        ],
        "Analysis": [
            st.Page(track_record.render, title="Track Record", icon="📊", url_path="track-record"),
            st.Page(snapshot.render, title="Snapshot", icon="🔍", url_path="snapshot"),
            st.Page(lookup.render, title="Lookup", icon="🔎", url_path="lookup"),
        ],
        "Ops": [
            st.Page(run_log.render, title="Run Log", icon="📜", url_path="run-log"),
        ],
    }
    nav = st.navigation(pages)

    with st.sidebar:
        st.markdown("### AlphaEngine")
        st.caption("Local-first signal dashboard")
        st.caption("Read-only view · No API calls")
        st.divider()

        # Auto-refresh — off by default. When on, re-runs the current page
        # at the chosen interval. Each tick re-queries DuckDB; the queries
        # are read-only and cheap so this is safe to leave running during
        # market hours.
        auto_on = st.toggle("Auto-refresh", value=False)
        if auto_on:
            interval_s = st.slider(
                "Refresh every (seconds)", min_value=10, max_value=300, value=60, step=10
            )
            st_autorefresh(interval=interval_s * 1000, key="dashboard_autorefresh")
            st.caption(f"Refreshing every {interval_s}s")

        st.divider()
        st.caption(
            "Generate a new digest:\n\n"
            "```\npython scripts/paper_trader.py run-day --generate\n```"
        )

    nav.run()


if __name__ == "__main__":
    main()
