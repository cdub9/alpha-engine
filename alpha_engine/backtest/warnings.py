"""Surface survivorship-bias warnings on backtests involving individual equities.

The universe in `instruments` is *today's* list. Companies delisted,
bankrupted, or removed from the S&P 500 since the backtest start date
don't exist in our market_bars. Backtests that long individual names
will systematically overstate returns — the failures are missing from
history.

ETFs (SPY, QQQ, VOO, IWM, sector SPDRs, etc.) hold the historical
composition by construction, so they are *not* affected.

This module gives backtest entrypoints (CLI scripts, dashboard pages) a
single, consistent way to detect the bias and surface a warning.
"""

from __future__ import annotations

from typing import Iterable

import duckdb


SURVIVORSHIP_WARNING_HEADER = "SURVIVORSHIP BIAS WARNING"

SURVIVORSHIP_WARNING_BODY = (
    "This backtest references individual equity tickers. The universe is "
    "today's list of survivors — companies delisted, bankrupted, or "
    "removed from their index since the backtest start are absent from "
    "the data. Results are therefore systematically biased upward.\n\n"
    "Clean baselines: SPY, VOO, QQQ, IWM, sector ETFs (these hold the "
    "historical composition by construction).\n\n"
    "To honestly evaluate individual-name strategies, point-in-time index "
    "membership data (Norgate, Sharadar, CRSP) is required. Until then, "
    "treat individual-name backtest numbers as upper bounds, not "
    "forecasts."
)


def affected_symbols(
    con: duckdb.DuckDBPyConnection, symbols: Iterable[str]
) -> list[str]:
    """Return the subset of `symbols` that are individual equities
    (instrument_type='equity'). These are the ones subject to the bias.

    ETFs, bond ETFs, leveraged ETFs are NOT affected — they hold the
    historical composition. The exclusion is purely on instrument_type.
    """
    syms = sorted({s.upper() for s in symbols if s})
    if not syms:
        return []
    placeholders = ",".join(["?"] * len(syms))
    rows = con.execute(
        f"""
        SELECT symbol FROM instruments
        WHERE instrument_type = 'equity'
          AND symbol IN ({placeholders})
        ORDER BY symbol
        """,
        syms,
    ).fetchall()
    return [r[0] for r in rows]


def survivorship_warning_text(
    affected: list[str], max_show: int = 12, include_header: bool = True
) -> str:
    """Format the warning. Returns empty string if `affected` is empty.

    The text is intended for both CLI (wrapped in a Rich Panel by the
    caller) and dashboard surfaces (rendered as markdown).
    """
    if not affected:
        return ""
    shown = ", ".join(affected[:max_show])
    overflow = f" (+{len(affected) - max_show} more)" if len(affected) > max_show else ""
    parts = []
    if include_header:
        parts.append(f"⚠️  {SURVIVORSHIP_WARNING_HEADER}")
        parts.append("")
    parts.append(f"Affected symbols ({len(affected)}): {shown}{overflow}")
    parts.append("")
    parts.append(SURVIVORSHIP_WARNING_BODY)
    return "\n".join(parts)


def has_individual_equities(
    con: duckdb.DuckDBPyConnection, symbols: Iterable[str]
) -> bool:
    """Cheap boolean check — does this symbol set include any equity?"""
    return len(affected_symbols(con, symbols)) > 0
