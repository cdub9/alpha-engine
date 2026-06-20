"""Storage layer: upserts typed domain objects into DuckDB.

Uses INSERT ... ON CONFLICT to upsert by natural key. Batches writes for
performance. All upserts are idempotent — safe to re-run.
"""

from __future__ import annotations

from typing import Iterable

import duckdb

from alpha_engine.core.logging import get_logger
from alpha_engine.core.types import (
    Instrument,
    MacroObservation,
    MarketBar,
)
from alpha_engine.data.gdelt import GdeltDailyPoint

log = get_logger(__name__)


def upsert_market_bars(
    con: duckdb.DuckDBPyConnection, bars: Iterable[MarketBar]
) -> int:
    """Upsert market bars by (symbol, bar_date). Returns count inserted."""
    rows = [
        (
            b.symbol,
            b.bar_date,
            b.open,
            b.high,
            b.low,
            b.close,
            b.adj_close,
            b.volume,
            b.source,
        )
        for b in bars
    ]
    if not rows:
        return 0

    con.executemany(
        """
        INSERT INTO market_bars
            (symbol, bar_date, open, high, low, close, adj_close, volume, source)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (symbol, bar_date) DO UPDATE SET
            open      = EXCLUDED.open,
            high      = EXCLUDED.high,
            low       = EXCLUDED.low,
            close     = EXCLUDED.close,
            adj_close = EXCLUDED.adj_close,
            volume    = EXCLUDED.volume,
            source    = EXCLUDED.source
        """,
        rows,
    )
    log.info("upsert_market_bars", count=len(rows))
    return len(rows)


def upsert_macro_observations(
    con: duckdb.DuckDBPyConnection, observations: Iterable[MacroObservation]
) -> int:
    """Upsert macro observations by (series_id, obs_date)."""
    rows = [(o.series_id, o.obs_date, o.value, o.source) for o in observations]
    if not rows:
        return 0

    con.executemany(
        """
        INSERT INTO macro_series (series_id, obs_date, value, source)
        VALUES (?, ?, ?, ?)
        ON CONFLICT (series_id, obs_date) DO UPDATE SET
            value  = EXCLUDED.value,
            source = EXCLUDED.source
        """,
        rows,
    )
    log.info("upsert_macro_observations", count=len(rows))
    return len(rows)


def upsert_geopolitical_points(
    con: duckdb.DuckDBPyConnection,
    signal_name: str,
    points: Iterable[GdeltDailyPoint],
    source: str = "gdelt_doc",
) -> int:
    """Upsert GDELT daily points by (signal_name, signal_date).

    `source` records provenance ('gdelt_doc' or 'gdelt_bq'). A later upsert
    overwrites an earlier one for the same (signal, day) regardless of
    source, so a full BigQuery backfill cleanly supersedes DOC rows and the
    series stays internally consistent (same normalization across dates)."""
    rows = [
        (
            signal_name,
            p.signal_date,
            p.volume_intensity,
            p.avg_tone,
            p.raw_query,
            source,
        )
        for p in points
    ]
    if not rows:
        return 0

    # DuckDB's ON CONFLICT DO UPDATE SET treats bare CURRENT_TIMESTAMP as
    # a column reference (it's not a SQL keyword in that position). Use
    # now() which is unambiguously a function call.
    con.executemany(
        """
        INSERT INTO geopolitical_signals
            (signal_name, signal_date, volume_intensity, avg_tone, raw_query, source)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT (signal_name, signal_date) DO UPDATE SET
            volume_intensity = EXCLUDED.volume_intensity,
            avg_tone         = EXCLUDED.avg_tone,
            raw_query        = EXCLUDED.raw_query,
            source           = EXCLUDED.source,
            fetched_at       = now()
        """,
        rows,
    )
    log.info("upsert_geopolitical_points", signal=signal_name, count=len(rows), source=source)
    return len(rows)


def upsert_instruments(
    con: duckdb.DuckDBPyConnection, instruments: Iterable[Instrument]
) -> int:
    """Upsert instruments by symbol."""
    rows = [
        (
            i.symbol,
            i.name,
            i.instrument_type.value,
            i.sector,
            i.industry,
            i.exchange,
            i.currency,
            i.active,
        )
        for i in instruments
    ]
    if not rows:
        return 0

    con.executemany(
        """
        INSERT INTO instruments
            (symbol, name, instrument_type, sector, industry, exchange, currency, active)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (symbol) DO UPDATE SET
            name            = EXCLUDED.name,
            instrument_type = EXCLUDED.instrument_type,
            sector          = EXCLUDED.sector,
            industry        = EXCLUDED.industry,
            exchange        = EXCLUDED.exchange,
            currency        = EXCLUDED.currency,
            active          = EXCLUDED.active
        """,
        rows,
    )
    log.info("upsert_instruments", count=len(rows))
    return len(rows)
