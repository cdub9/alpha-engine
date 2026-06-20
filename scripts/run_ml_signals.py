"""Generate and persist today's ML signal cross-section.

Ranks every active universe instrument by the momentum composite (and
optionally the walk-forward XGBoost model), buckets into BUY / HOLD /
AVOID, and persists to ml_signals for the dashboard.

Free to run — pure local compute on bars already in the DB. Intended to
run daily right after the bar refresh (wired into daily_paper_trade.bat),
but safe to run manually any time:

    python scripts/run_ml_signals.py                 # latest bar date, both models
    python scripts/run_ml_signals.py --date 2026-06-10
    python scripts/run_ml_signals.py --skip-xgb      # composite only
"""

from __future__ import annotations

import sys
from datetime import date as date_t
from datetime import datetime, timedelta

import typer
from rich.console import Console
from rich.table import Table

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

from alpha_engine.core.logging import configure_logging
from alpha_engine.db import get_connection, init_schema
from alpha_engine.ml.advisor import _XGB_LOOKBACK_DAYS, load_price_panel
from alpha_engine.ml.features import MIN_HISTORY, compute_features
from alpha_engine.ml.model import MomentumComposite, WalkForwardXGB, assign_actions
from alpha_engine.ml.store import persist_ml_signals

console = Console()
app = typer.Typer(add_completion=False, help=__doc__)


def _universe_symbols(con) -> list[str]:
    """Active universe instruments, excluding leveraged ETFs — ranking a
    3x fund on its own momentum double-counts the underlying move."""
    rows = con.execute(
        "SELECT symbol FROM instruments WHERE active AND instrument_type != 'leveraged_etf' "
        "ORDER BY symbol"
    ).fetchall()
    return [r[0] for r in rows]


@app.command()
def main(
    signal_date: str = typer.Option(None, "--date", help="YYYY-MM-DD; default latest bar date"),
    skip_xgb: bool = typer.Option(False, "--skip-xgb"),
    log_level: str = typer.Option("INFO", "--log-level"),
) -> None:
    configure_logging(level=log_level)
    init_schema()  # idempotent — ensures ml_signals exists

    with get_connection() as con:
        if signal_date:
            as_of = datetime.strptime(signal_date, "%Y-%m-%d").date()
        else:
            as_of = con.execute("SELECT MAX(bar_date) FROM market_bars").fetchone()[0]
        assert isinstance(as_of, date_t)

        universe = _universe_symbols(con)
        prices = load_price_panel(con, universe, as_of, _XGB_LOOKBACK_DAYS)
        if prices.empty or len(prices) < MIN_HISTORY:
            console.print("[red]Not enough price history to rank. Run backfill first.[/]")
            raise typer.Exit(1)

        feats = compute_features(prices)

        total = 0
        for label, model in (
            ("momentum", MomentumComposite()),
            *(() if skip_xgb else (("xgb", WalkForwardXGB()),)),
        ):
            if isinstance(model, WalkForwardXGB):
                model.maybe_retrain(prices)
            scores = model.score_cross_section(feats)
            actions = assign_actions(scores)
            n = persist_ml_signals(con, as_of, scores, actions, feats, model.version)
            total += n
            console.print(f"[green]{model.version}[/]: {n} symbols ranked for {as_of}")

            top = scores.dropna().nlargest(10)
            bottom = scores.dropna().nsmallest(5)
            table = Table(title=f"{label} — top 10 / bottom 5 of {scores.notna().sum()}")
            table.add_column("Symbol")
            table.add_column("Score", justify="right")
            table.add_column("Action")
            table.add_column("12-1 mom", justify="right")
            table.add_column("vs 200MA", justify="right")
            for sym, sc in top.items():
                table.add_row(sym, f"{sc:+.2f}", str(actions[sym]),
                              f"{feats.at[sym, 'mom_12_1']:+.1%}",
                              f"{feats.at[sym, 'dist_200ma']:+.1%}")
            table.add_row("…", "", "", "", "")
            for sym, sc in bottom.items():
                table.add_row(sym, f"{sc:+.2f}", str(actions[sym]),
                              f"{feats.at[sym, 'mom_12_1']:+.1%}",
                              f"{feats.at[sym, 'dist_200ma']:+.1%}")
            console.print(table)

        if total == 0:
            raise typer.Exit(1)


if __name__ == "__main__":
    app()
