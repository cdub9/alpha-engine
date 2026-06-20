"""Database connection management for DuckDB.

A simple wrapper: opens the DB file (creating it if missing), applies the
schema, and returns a connection. Connections are not pooled — DuckDB is
embedded and a single process opens it.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import duckdb

from alpha_engine.core.config import get_settings
from alpha_engine.core.logging import get_logger

log = get_logger(__name__)

SCHEMA_PATH = Path(__file__).parent / "schema.sql"

# Additive column migrations for DBs created before a column existed.
# `CREATE TABLE IF NOT EXISTS` never alters an existing table, so new
# columns on existing tables are backfilled here. DuckDB's ALTER TABLE
# ADD COLUMN rejects inline constraints (NOT NULL / DEFAULT), so each
# entry is (table, column, bare_type, backfill_value): we add the column
# bare, then UPDATE existing rows to the backfill value when one is given.
# Fresh DBs get the full constraint from schema.sql's CREATE TABLE; the
# nominal nullability difference on migrated DBs is harmless because every
# row is populated. Safe to run on every startup.
_COLUMN_MIGRATIONS: list[tuple[str, str, str, object]] = [
    ("trades", "entry_style", "VARCHAR", "next_close"),
    ("trades", "alt_entry_price", "DOUBLE", None),
    ("trade_outcomes", "alt_entry_return_pct", "DOUBLE", None),
    ("geopolitical_signals", "source", "VARCHAR", "gdelt_doc"),
]


def _apply_column_migrations(con: duckdb.DuckDBPyConnection) -> None:
    for table, column, bare_type, backfill in _COLUMN_MIGRATIONS:
        table_exists = con.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?",
            [table],
        ).fetchone()[0]
        if not table_exists:
            continue  # nothing to migrate (e.g. a partial/test schema)
        present = con.execute(
            "SELECT COUNT(*) FROM information_schema.columns "
            "WHERE table_name = ? AND column_name = ?",
            [table, column],
        ).fetchone()[0]
        if present:
            continue
        con.execute(f"ALTER TABLE {table} ADD COLUMN {column} {bare_type}")
        if backfill is not None:
            con.execute(
                f"UPDATE {table} SET {column} = ? WHERE {column} IS NULL",
                [backfill],
            )
        log.info("schema_migration_applied", table=table, column=column)


def init_schema(con: duckdb.DuckDBPyConnection | None = None) -> None:
    """Create all tables, sequences, and indexes, then apply additive
    column migrations. Idempotent."""
    if con is None:
        con = duckdb.connect(str(get_settings().db_path))
        close_after = True
    else:
        close_after = False

    sql = SCHEMA_PATH.read_text(encoding="utf-8")
    con.execute(sql)
    _apply_column_migrations(con)
    log.info("schema_initialized", db_path=str(get_settings().db_path))

    if close_after:
        con.close()


def get_connection(read_only: bool = False) -> duckdb.DuckDBPyConnection:
    """Open a connection to the DuckDB file. Caller is responsible for
    closing it (or use the `connection()` context manager)."""
    settings = get_settings()
    return duckdb.connect(str(settings.db_path), read_only=read_only)


@contextmanager
def connection(read_only: bool = False) -> Iterator[duckdb.DuckDBPyConnection]:
    """Context manager for a DuckDB connection."""
    con = get_connection(read_only=read_only)
    try:
        yield con
    finally:
        con.close()
