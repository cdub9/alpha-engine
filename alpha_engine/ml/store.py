"""Persistence for daily ML signals.

One row per (signal_date, symbol, model_version): the score, the rank
within that day's cross-section, the action bucket, and the raw feature
values (so the dashboard can show WHY a name ranks where it does without
recomputing anything).

Same-day re-runs replace prior rows for the same model_version — mirrors
the LLM digest's persist_signals behavior so downstream consumers never
double-count.
"""

from __future__ import annotations

from datetime import date
from typing import Optional

import duckdb
import pandas as pd

from alpha_engine.core.logging import get_logger
from alpha_engine.ml.features import FEATURE_COLUMNS

log = get_logger(__name__)


def persist_ml_signals(
    con: duckdb.DuckDBPyConnection,
    signal_date: date,
    scores: pd.Series,
    actions: pd.Series,
    features: pd.DataFrame,
    model_version: str,
) -> int:
    """Write one day's cross-section. Returns rows written.

    `scores` indexed by symbol (NaN rows are skipped), `actions` aligned,
    `features` the raw feature frame from compute_features.
    """
    valid = scores.dropna().sort_values(ascending=False)
    if valid.empty:
        log.warning("ml_persist_empty", date=str(signal_date))
        return 0

    n = len(valid)
    con.execute(
        "DELETE FROM ml_signals WHERE signal_date = ? AND model_version = ?",
        [signal_date, model_version],
    )

    rows = []
    for rank, (sym, score) in enumerate(valid.items(), start=1):
        feat_vals = [
            float(features.at[sym, c]) if pd.notna(features.at[sym, c]) else None
            for c in FEATURE_COLUMNS
        ]
        rows.append(
            [signal_date, sym, model_version, float(score), rank, n,
             str(actions.get(sym)), *feat_vals]
        )

    placeholders = ",".join(["?"] * (7 + len(FEATURE_COLUMNS)))
    con.executemany(
        f"INSERT INTO ml_signals (signal_date, symbol, model_version, score, "
        f"rank, n_universe, action, {', '.join(FEATURE_COLUMNS)}) "
        f"VALUES ({placeholders})",
        rows,
    )
    log.info("ml_signals_persisted", date=str(signal_date), rows=len(rows),
             model=model_version)
    return len(rows)


def load_ml_signals(
    con: duckdb.DuckDBPyConnection,
    signal_date: date,
    model_version: Optional[str] = None,
) -> pd.DataFrame:
    """All rows for a date (latest model_version if not pinned), ordered by rank."""
    if model_version is None:
        row = con.execute(
            "SELECT model_version FROM ml_signals WHERE signal_date = ? "
            "ORDER BY generated_at DESC LIMIT 1",
            [signal_date],
        ).fetchone()
        if not row:
            return pd.DataFrame()
        model_version = row[0]
    return con.execute(
        "SELECT * FROM ml_signals WHERE signal_date = ? AND model_version = ? "
        "ORDER BY rank",
        [signal_date, model_version],
    ).fetch_df()


def latest_ml_signal_date(
    con: duckdb.DuckDBPyConnection,
) -> Optional[date]:
    row = con.execute("SELECT MAX(signal_date) FROM ml_signals").fetchone()
    return row[0] if row and row[0] else None
