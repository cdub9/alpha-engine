"""yfinance market data provider.

Unofficial Yahoo Finance wrapper. Free, no API key required. Suitable for
daily bars; not real-time. For Phase 2 production, migrate to Polygon.io.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Iterable, Optional

import pandas as pd
import yfinance as yf

from alpha_engine.core.logging import get_logger
from alpha_engine.core.types import MarketBar
from alpha_engine.data.base import DataProvider

log = get_logger(__name__)


class YFinanceProvider(DataProvider[MarketBar]):
    name = "yfinance"

    def fetch(
        self,
        symbols: list[str] | str,
        start: Optional[date] = None,
        end: Optional[date] = None,
        period: Optional[str] = None,
    ) -> Iterable[MarketBar]:
        """Fetch daily OHLCV bars for one or more symbols.

        Either provide (start, end) or `period` (e.g. '5y', '1mo')."""
        if isinstance(symbols, str):
            symbols = [symbols]
        symbols = [s.upper() for s in symbols]

        log.info("yf_fetch", symbols=symbols, start=str(start), period=period)

        kwargs: dict = {
            "tickers": " ".join(symbols),
            "interval": "1d",
            "auto_adjust": False,
            "progress": False,
            "group_by": "ticker",
            # Single-threaded: yfinance's per-process sqlite tz-cache races
            # under threads=True, causing "database is locked" on Windows.
            "threads": False,
        }
        if period:
            kwargs["period"] = period
        else:
            kwargs["start"] = start
            kwargs["end"] = (end or date.today()) + timedelta(days=1)

        df = yf.download(**kwargs)
        if df is None or df.empty:
            log.warning("yf_empty_response", symbols=symbols)
            return

        yield from _frame_to_bars(df, symbols)


def _frame_to_bars(df: pd.DataFrame, symbols: list[str]) -> Iterable[MarketBar]:
    """Convert yfinance dataframe to MarketBar records. yfinance returns
    a multi-index column frame when multiple tickers are passed, and a flat
    frame for a single ticker."""
    if len(symbols) == 1:
        sym = symbols[0]
        for ts, row in df.iterrows():
            bar = _row_to_bar(sym, ts, row)
            if bar:
                yield bar
        return

    # Multi-symbol: columns are (ticker, field)
    for sym in symbols:
        if sym not in df.columns.levels[0]:
            log.warning("yf_symbol_missing", symbol=sym)
            continue
        sub = df[sym].dropna(how="all")
        for ts, row in sub.iterrows():
            bar = _row_to_bar(sym, ts, row)
            if bar:
                yield bar


def _row_to_bar(symbol: str, ts, row) -> Optional[MarketBar]:
    try:
        bar_date = ts.date() if hasattr(ts, "date") else ts
        if isinstance(bar_date, datetime):
            bar_date = bar_date.date()

        # Some rows may have NaN if the symbol wasn't trading that day
        open_ = float(row["Open"])
        high = float(row["High"])
        low = float(row["Low"])
        close = float(row["Close"])
        # yfinance with auto_adjust=False provides "Adj Close"
        adj_close = float(row.get("Adj Close", close))
        volume = int(row.get("Volume", 0) or 0)

        # Sanity: skip rows where prices are NaN
        if any(pd.isna(x) for x in (open_, high, low, close, adj_close)):
            return None

        return MarketBar(
            symbol=symbol,
            bar_date=bar_date,
            open=open_,
            high=high,
            low=low,
            close=close,
            adj_close=adj_close,
            volume=volume,
            source="yfinance",
        )
    except (KeyError, ValueError, TypeError) as e:
        log.debug("yf_bar_parse_error", symbol=symbol, error=str(e))
        return None
