"""TA snapshot section (FOLLOWUPS "Add technical analysis features to the
snapshot"): section content, breadth math, insufficient-history exclusion,
prompt operating principle, and the v2-ta version tag.
"""

from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd

from alpha_engine.llm.context import _format_technicals_section


def seed_bars(con: duckdb.DuckDBPyConnection, n_days: int = 300) -> pd.DataFrame:
    con.execute(
        "CREATE TABLE market_bars (symbol VARCHAR, bar_date DATE, open DOUBLE,"
        " high DOUBLE, low DOUBLE, close DOUBLE, adj_close DOUBLE, volume BIGINT,"
        " source VARCHAR, ingested_at TIMESTAMP)"
    )
    idx = pd.bdate_range("2024-01-01", periods=n_days)
    rng = np.random.default_rng(5)
    panel = {
        "UPP": 100.0 * np.exp(np.cumsum(rng.normal(0.002, 0.005, n_days))),
        "DWN": 100.0 * np.exp(np.cumsum(rng.normal(-0.002, 0.005, n_days))),
        "FLT": np.full(n_days, 100.0),
    }
    rows = []
    for sym, prices in panel.items():
        for d, p in zip(idx, prices):
            rows.append([sym, d.date(), p, p, p, p, p, 1000, "test", None])
    # NEWBIE: only 60 days of history — must be excluded from the section
    for d, p in zip(idx[-60:], panel["UPP"][-60:]):
        rows.append(["NEWBIE", d.date(), p, p, p, p, p, 1000, "test", None])
    con.executemany("INSERT INTO market_bars VALUES (?,?,?,?,?,?,?,?,?,?)", rows)
    return pd.DataFrame(panel, index=idx)


class TestTechnicalsSection:
    def test_section_structure_and_content(self):
        con = duckdb.connect(":memory:")
        panel = seed_bars(con)
        as_of = panel.index[-1].date()
        section = _format_technicals_section(
            con, ["UPP", "DWN", "FLT", "NEWBIE"], as_of
        )
        assert section.startswith("## Per-symbol technicals")
        assert "Breadth:" in section
        # Strong uptrend above its MAs, downtrend below
        upp_line = next(l for l in section.splitlines() if "**UPP**" in l)
        dwn_line = next(l for l in section.splitlines() if "**DWN**" in l)
        assert "+" in upp_line.split("50MA")[1][:8]
        assert "-" in dwn_line.split("50MA")[1][:8]
        # RSI present as integer, vol present as percent
        assert "RSI" in upp_line and "vol" in upp_line

    def test_insufficient_history_symbol_excluded(self):
        con = duckdb.connect(":memory:")
        panel = seed_bars(con)
        as_of = panel.index[-1].date()
        section = _format_technicals_section(
            con, ["UPP", "DWN", "FLT", "NEWBIE"], as_of
        )
        assert "NEWBIE" not in section

    def test_breadth_math(self):
        con = duckdb.connect(":memory:")
        panel = seed_bars(con)
        as_of = panel.index[-1].date()
        section = _format_technicals_section(con, ["UPP", "DWN", "FLT"], as_of)
        # UPP above its 50MA, DWN below, FLT exactly at it (not above):
        # breadth = 1 of 3 = 33%
        assert "33% of 3 names" in section

    def test_empty_universe_returns_empty(self):
        con = duckdb.connect(":memory:")
        seed_bars(con)
        assert _format_technicals_section(con, [], pd.Timestamp("2025-01-01").date()) == ""

    def test_point_in_time(self):
        """Section for an early as_of must not change when later bars exist."""
        con = duckdb.connect(":memory:")
        panel = seed_bars(con)
        early = panel.index[280].date()
        with_future = _format_technicals_section(con, ["UPP", "DWN"], early)
        con.execute("DELETE FROM market_bars WHERE bar_date > ?", [early])
        without_future = _format_technicals_section(con, ["UPP", "DWN"], early)
        assert with_future == without_future


class TestPromptAndVersion:
    def test_prompt_contains_ta_principle(self):
        from alpha_engine.llm.prompts import SYSTEM_PROMPT

        assert "TA is confirmation, not a primary signal" in SYSTEM_PROMPT
        assert "Per-symbol technicals" in SYSTEM_PROMPT

    def test_model_version_bumped_past_v1(self):
        # The TA change must not write into the v1 cohort — exact current
        # version is asserted in test_llm_feedback.py; here we only guard
        # the cohort separation that the TA A/B comparison depends on.
        import inspect

        from alpha_engine.backtest.llm_advisor import DEFAULT_MODEL_VERSION
        from alpha_engine.llm.parser import persist_signals

        assert DEFAULT_MODEL_VERSION != "llm-opus-4-7-v1"
        sig = inspect.signature(persist_signals)
        assert sig.parameters["model_version"].default == DEFAULT_MODEL_VERSION

    def test_prompt_and_section_heading_match(self):
        # The prompt references the snapshot section by its exact heading;
        # if context.py's heading ever drifts, the LLM's instructions would
        # point at a section that no longer exists.
        import alpha_engine.llm.context as ctx

        source = open(ctx.__file__, encoding="utf-8").read()
        assert '"## Per-symbol technicals"' in source
