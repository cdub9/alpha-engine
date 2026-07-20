"""Print the day's brief — trades, opportunity ideas, and market context.

The scheduled morning routine runs this after refreshing holdings, earnings,
held-name bars, and the book digest, so it can deliver the full picture as
text (message / notification): the deterministic risk trades first, then the
digest-driven opportunity ideas (clearly labeled unproven), then a one-line
market read.

Reuses the Action Center's data layer (queries.portfolio_action_center) — no
Streamlit — so the brief and the dashboard always agree.
"""

from __future__ import annotations

import sys
from pathlib import Path

import typer
from rich.console import Console

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

# `dashboard` is a top-level package (not pip-installed like alpha_engine),
# so add the project root to the path when run as a script.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dashboard import queries as q

console = Console()
app = typer.Typer(add_completion=False, help=__doc__)


@app.command()
def main() -> None:
    d = q.portfolio_action_center()
    if d is None:
        console.print("[red]No holdings snapshot — pull positions first.[/]")
        raise typer.Exit(1)

    report = d["report"]
    total = d["total_equity"]
    top = report.get("top_name") or {}
    tr = d.get("semis_trend")
    semis_w = report["clusters"].get("semis_ai_hw", {}).get("weight", 0.0)

    from datetime import date
    console.print(f"[bold]AlphaEngine — brief for {date.today()}[/]")
    console.print(
        f"{d.get('account','')} · ${total:,.0f} · semis {semis_w:.0%} · "
        f"top {top.get('symbol','—')} {top.get('weight',0):.0%} · "
        f"cash {(d['cash_weight'] or 0):.0%}"
        + (f" · semis trend {'up' if tr['above'] else 'DOWN'} {tr['distance']:+.0%}" if tr else "")
    )
    if (d.get("holdings_age_days") or 0) > 3:
        console.print(f"[yellow]! holdings snapshot is {d['holdings_age_days']} days old.[/]")

    # 1) Risk trades — the high-confidence, deterministic layer.
    plan = d["plan"]
    console.print("\n[bold]Today's trades[/] [dim](risk layer — act on these)[/]")
    if not plan["orders"]:
        console.print("  [green]Nothing required — no cap breaches or imminent earnings.[/]")
    for o in plan["orders"]:
        sh = f"{o['shares']} sh" if o["shares"] else ""
        console.print(f"  - {o['action']} {sh} {o['symbol']} ~${o['est_dollars']:,.0f} "
                      f"[{o['when']}] — {o['reason']}")

    # 2) Opportunity ideas — softer, digest-driven, honestly labeled.
    opp = d.get("opportunity") or {"adds": [], "trims": []}
    if opp["adds"] or opp["trims"]:
        console.print("\n[bold]Opportunity ideas[/] [dim](from the digest's signals — "
                      "UNPROVEN skill; weigh, don't obey; already cap-gated)[/]")
        for a in opp["adds"]:
            console.print(f"  - Consider ADD {a['symbol']} — {a['reason']}")
        for t in opp["trims"]:
            console.print(f"  - Consider TRIM {t['symbol']} — {t['reason']}")

    # 3) One-line market context.
    mc = d.get("market_context") or {}
    if mc.get("market_summary"):
        console.print(f"\n[dim]Market read ({mc.get('digest_date')}): "
                      f"{mc['market_summary'][:280]}[/]")
    if plan.get("cluster_note"):
        console.print(f"[dim]{plan['cluster_note']}[/]")


if __name__ == "__main__":
    app()
