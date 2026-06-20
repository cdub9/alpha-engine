"""End-to-end smoke test for the AlphaEngine foundation.

Run with: python scripts/verify_setup.py

Validates:
  1. Database can be created from schema
  2. yfinance can fetch and store price data
  3. FRED can fetch and store macro data (if FRED_API_KEY is set)
  4. VaR / CVaR computes correctly on real data
  5. Correlation matrix computes on a small basket
  6. Kelly Criterion works for continuous and discrete cases

Exits non-zero on any failure.
"""

from __future__ import annotations

import sys
import traceback
from datetime import date, timedelta

# Windows terminals default to cp1252 which cannot encode Rich's box-drawing
# characters. Force UTF-8 before anything imports Rich's renderer.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

import numpy as np
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from alpha_engine.core.config import get_settings
from alpha_engine.core.logging import configure_logging
from alpha_engine.core.types import Instrument, InstrumentType
from alpha_engine.data import (
    FredClient,
    YFinanceProvider,
    upsert_instruments,
    upsert_macro_observations,
    upsert_market_bars,
)
from alpha_engine.db import get_connection, init_schema
from alpha_engine.risk import (
    historical_var_cvar,
    kelly_continuous,
    kelly_discrete,
    parametric_var,
    returns_from_prices,
    rolling_correlation_matrix,
    summarize_correlation,
)

console = Console()
BASKET = ["SPY", "QQQ", "TLT", "GLD", "XLE"]


def step(title: str) -> None:
    console.rule(f"[bold cyan]{title}[/]")


def check(condition: bool, msg: str) -> None:
    if condition:
        console.print(f"  [green]PASS[/] {msg}")
    else:
        console.print(f"  [red]FAIL[/] {msg}")
        raise AssertionError(msg)


def main() -> int:
    configure_logging(level="INFO")
    settings = get_settings()

    console.print(
        Panel.fit(
            f"[bold]AlphaEngine setup verification[/]\n"
            f"DB: {settings.db_path}\n"
            f"Channels: {', '.join(c.value for c in settings.channels)}",
            title="AlphaEngine",
            border_style="cyan",
        )
    )

    # ------------------------------------------------------------------
    step("1. Initialize database")
    # ------------------------------------------------------------------
    init_schema()
    with get_connection(read_only=True) as con:
        tables = con.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'main' ORDER BY table_name"
        ).fetchall()
    table_names = [t[0] for t in tables]
    check(len(table_names) >= 10, f"created {len(table_names)} tables")
    console.print(f"  tables: {', '.join(table_names)}")

    # ------------------------------------------------------------------
    step("2. Upsert instrument universe")
    # ------------------------------------------------------------------
    instruments = []
    for universe_name, items in settings.universe.universes.items():
        for spec in items:
            instruments.append(
                Instrument(
                    symbol=spec.symbol,
                    name=spec.name,
                    instrument_type=spec.type,
                    sector=spec.sector,
                )
            )
    with get_connection() as con:
        n = upsert_instruments(con, instruments)
    check(n > 0, f"inserted {n} instruments")

    # ------------------------------------------------------------------
    step("3. Pull market data via yfinance")
    # ------------------------------------------------------------------
    yf = YFinanceProvider()
    start = date.today() - timedelta(days=365 * 2)  # 2 years for VaR/correlation
    bars = list(yf.fetch(BASKET, start=start))
    check(len(bars) > 0, f"fetched {len(bars)} bars across {len(BASKET)} symbols")
    with get_connection() as con:
        upsert_market_bars(con, bars)

    # ------------------------------------------------------------------
    step("4. Pull macro data from FRED")
    # ------------------------------------------------------------------
    if not settings.fred_api_key:
        console.print(
            "  [yellow]SKIP[/] FRED_API_KEY not set. Get one free at "
            "https://fred.stlouisfed.org/docs/api/api_key.html"
        )
    else:
        total = 0
        with FredClient() as fred, get_connection() as con:
            # Just pull 2 series for the smoke test (DGS10, CPIAUCSL)
            for spec in settings.fred_series[:2]:
                obs = list(fred.fetch(spec.id, observation_start=start))
                upsert_macro_observations(con, obs)
                total += len(obs)
                console.print(f"  fetched {len(obs)} obs for {spec.id} ({spec.name})")
        check(total > 0, f"loaded {total} macro observations")

    # ------------------------------------------------------------------
    step("5. Compute VaR / CVaR on SPY")
    # ------------------------------------------------------------------
    with get_connection(read_only=True) as con:
        spy_df = con.execute(
            "SELECT bar_date, adj_close FROM market_bars "
            "WHERE symbol = 'SPY' ORDER BY bar_date"
        ).fetch_df()

    spy_df["bar_date"] = pd.to_datetime(spy_df["bar_date"])
    spy_df = spy_df.set_index("bar_date")
    spy_returns = np.log(spy_df["adj_close"] / spy_df["adj_close"].shift(1)).dropna()

    hist_95 = historical_var_cvar(spy_returns.values, confidence=0.95)
    hist_99 = historical_var_cvar(spy_returns.values, confidence=0.99)
    param_95 = parametric_var(spy_returns.values, confidence=0.95)

    var_table = Table(title="SPY 1-day VaR / CVaR", show_header=True)
    var_table.add_column("Method")
    var_table.add_column("Confidence", justify="right")
    var_table.add_column("VaR", justify="right")
    var_table.add_column("CVaR", justify="right")
    var_table.add_column("Samples", justify="right")
    for r in (hist_95, hist_99, param_95):
        var_table.add_row(
            r.method,
            f"{r.confidence:.0%}",
            f"{r.var:.2%}",
            f"{r.cvar:.2%}",
            str(r.sample_size),
        )
    console.print(var_table)
    check(0 < hist_95.var < 0.1, "historical 95% VaR within plausible range")
    check(hist_99.var > hist_95.var, "99% VaR exceeds 95% VaR")
    check(hist_95.cvar >= hist_95.var, "CVaR >= VaR (always true by definition)")

    # ------------------------------------------------------------------
    step("6. Compute correlation matrix on basket")
    # ------------------------------------------------------------------
    with get_connection(read_only=True) as con:
        basket_df = con.execute(
            f"""
            SELECT bar_date, symbol, adj_close
            FROM market_bars
            WHERE symbol IN ({",".join("?" * len(BASKET))})
            ORDER BY bar_date
            """,
            BASKET,
        ).fetch_df()

    basket_df["bar_date"] = pd.to_datetime(basket_df["bar_date"])
    wide = basket_df.pivot(index="bar_date", columns="symbol", values="adj_close")
    returns = returns_from_prices(wide, method="log")

    corr = rolling_correlation_matrix(returns, window=60)
    report = summarize_correlation(
        corr, n_observations=60, threshold=settings.risk.correlation_alert_threshold
    )

    corr_table = Table(title="60-day Correlation Matrix")
    corr_table.add_column("", style="bold")
    for sym in corr.columns:
        corr_table.add_column(sym, justify="right")
    for sym, row in corr.iterrows():
        corr_table.add_row(sym, *(f"{v:.2f}" for v in row))
    console.print(corr_table)
    console.print(
        f"  avg pairwise: [bold]{report.avg_pairwise:.3f}[/] "
        f"(warn threshold {report.threshold})"
    )
    if report.regime_warning:
        console.print("  [yellow]regime warning: correlations elevated[/]")
    check(report.n_assets == len(BASKET), f"computed correlation for {len(BASKET)} assets")

    # ------------------------------------------------------------------
    step("7. Kelly Criterion sizing")
    # ------------------------------------------------------------------
    # Continuous: based on SPY's actual return distribution
    spy_kelly = kelly_continuous(
        spy_returns.values,
        fraction=settings.risk.kelly_fraction,
        max_size=0.30,
    )
    # Discrete: hypothetical bet with 55% win rate, 1.5x payoff
    discrete_kelly = kelly_discrete(
        win_probability=0.55,
        win_payoff=1.5,
        fraction=settings.risk.kelly_fraction,
    )

    kelly_table = Table(title="Kelly Criterion Sizing")
    kelly_table.add_column("Scenario")
    kelly_table.add_column("Full Kelly", justify="right")
    kelly_table.add_column("Fraction", justify="right")
    kelly_table.add_column("Recommended", justify="right")
    kelly_table.add_column("Capped?", justify="right")
    kelly_table.add_row(
        "SPY (continuous, daily)",
        f"{spy_kelly.full_kelly:.3f}",
        f"{spy_kelly.fraction_applied:.2f}",
        f"{spy_kelly.recommended:.3f}",
        "yes" if spy_kelly.capped else "no",
    )
    kelly_table.add_row(
        "Hypothetical (55% win, 1.5x)",
        f"{discrete_kelly.full_kelly:.3f}",
        f"{discrete_kelly.fraction_applied:.2f}",
        f"{discrete_kelly.recommended:.3f}",
        "yes" if discrete_kelly.capped else "no",
    )
    console.print(kelly_table)
    check(discrete_kelly.full_kelly > 0, "positive edge produces positive Kelly")
    check(discrete_kelly.recommended <= discrete_kelly.max_size_cap, "size respects cap")

    # ------------------------------------------------------------------
    console.rule()
    console.print(
        Panel.fit(
            "[bold green]All checks passed.[/]\n"
            f"Database: {settings.db_path}\n"
            f"Bars stored: {len(bars)}\n"
            f"Ready for Phase 2: signal generation.",
            border_style="green",
        )
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        console.print(f"\n[red bold]FAILED:[/] {exc}\n")
        traceback.print_exc()
        sys.exit(1)
