"""Print the day's recommended trades — a standalone, Streamlit-free brief.

The scheduled morning routine runs this after refreshing the holdings
snapshot and earnings, so it can deliver "here are today's trades" as text
(message / notification) without needing the local dashboard.

Reads data/real_holdings.json and applies the same rules as the Action
Center: single-name + cluster concentration caps, the earnings-blackout
guard (window relative to today), and the semis-trend gate. Degrades
gracefully — concentration always works; earnings need a populated
calendar; the trend needs SMH/SOXX bars (skipped if absent).
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import typer
from rich.console import Console

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

from alpha_engine.db import get_connection
from alpha_engine.risk.earnings_guard import upcoming_earnings
from alpha_engine.risk.portfolio import (
    annualized_vol_drag,
    concentration_report,
    trend_state,
)
from alpha_engine.risk.trade_plan import build_trade_plan

console = Console()
app = typer.Typer(add_completion=False, help=__doc__)

_HOLDINGS = Path(__file__).resolve().parent.parent / "data" / "real_holdings.json"


def _semis_trend(con) -> dict | None:
    for proxy in ("SMH", "SOXX"):
        px = [r[0] for r in con.execute(
            "SELECT adj_close FROM market_bars WHERE symbol = ? ORDER BY bar_date",
            [proxy],
        ).fetchall()]
        ts = trend_state(px, window=200)
        if ts is None:
            continue
        rets = [px[i] / px[i - 1] - 1.0 for i in range(1, len(px))]
        return {"proxy": proxy, **ts, **annualized_vol_drag(rets)}
    return None


@app.command()
def main(horizon_days: int = typer.Option(10, "--earnings-window")) -> None:
    if not _HOLDINGS.exists():
        console.print("[red]No holdings snapshot — pull positions first.[/]")
        raise typer.Exit(1)
    snap = json.loads(_HOLDINGS.read_text(encoding="utf-8"))
    holdings = snap.get("holdings", [])
    if not holdings:
        console.print("[red]Holdings snapshot is empty.[/]")
        raise typer.Exit(1)

    report = concentration_report(holdings)
    total = report["total_value"]
    values = {h["symbol"].upper(): float(h["value"]) for h in holdings}
    today = date.today()

    trend = None
    earnings: list = []
    try:
        with get_connection(read_only=True) as con:
            trend = _semis_trend(con)
            earnings = upcoming_earnings(con, list(values), today,
                                         horizon_days=horizon_days, values=values)
    except Exception as exc:  # no DB / empty DB in a fresh env — degrade
        console.print(f"[dim](trend/earnings unavailable: {exc})[/]")

    plan = build_trade_plan(holdings, report, report["caps"],
                            trend=trend, earnings=earnings)

    semis_w = report["clusters"].get("semis_ai_hw", {}).get("weight", 0.0)
    top = report.get("top_name") or {}
    console.print(f"[bold]AlphaEngine — trades for {today}[/]")
    console.print(
        f"Account {snap.get('account','')} · ${total:,.0f} · semis {semis_w:.0%} · "
        f"top {top.get('symbol','—')} {top.get('weight',0):.0%}"
        + (f" · semis trend {'▲' if trend['above'] else '▼'} {trend['distance']:+.0%}"
           if trend else "")
    )
    age = (today - date.fromisoformat(snap["as_of"])).days if snap.get("as_of") else 0
    if age > 3:
        console.print(f"[yellow]⚠ holdings snapshot is {age} days old — sizes may have drifted.[/]")

    console.print("\n[bold]Today's trades[/]")
    if not plan["orders"]:
        console.print("  [green]Nothing to do — no cap breaches or imminent earnings.[/]")
    for o in plan["orders"]:
        sh = f"{o['shares']} sh" if o["shares"] else ""
        console.print(
            f"  • [bold]{o['action']} {sh} {o['symbol']}[/] ~${o['est_dollars']:,.0f}  "
            f"[{o['when']}] — {o['reason']}"
            + (f"  ({o['ml_note']})" if o["ml_note"] else "")
        )
    if plan.get("cluster_note"):
        console.print(f"\n[dim]{plan['cluster_note']}[/]")


if __name__ == "__main__":
    app()
