"""Write data/real_holdings.json from a normalized positions payload.

The scheduled morning routine pulls positions + quotes from the brokerage
connector, normalizes them to the shape below, and hands them here so the
snapshot the Action Center reads is always written the same way (value,
cluster tag, totals, today's date) regardless of who/what refreshed it.

Input JSON:
  {
    "account": "****5210",
    "cash": 5373.35,
    "holdings": [{"symbol": "MU", "quantity": 19.7, "price": 1122.99}, ...]
  }

Source of the JSON (first match wins):
  python scripts/write_holdings.py --input payload.json
  python scripts/write_holdings.py '{"account": "...", ...}'
  echo '{...}' | python scripts/write_holdings.py
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

from alpha_engine.risk.portfolio import cluster_of

console = Console()
app = typer.Typer(add_completion=False, help=__doc__)

_OUT = Path(__file__).resolve().parent.parent / "data" / "real_holdings.json"


@app.command()
def main(
    payload: str = typer.Argument("", help="Inline JSON payload (optional)"),
    input: str = typer.Option("", "--input", help="Path to a JSON payload file"),
) -> None:
    if input:
        raw = Path(input).read_text(encoding="utf-8")
    elif payload:
        raw = payload
    else:
        raw = sys.stdin.read()
    if not raw.strip():
        console.print("[red]No payload provided (arg, --input, or stdin).[/]")
        raise typer.Exit(1)

    data = json.loads(raw)
    holdings = []
    for h in data.get("holdings", []):
        sym = h["symbol"].upper().strip()
        qty = float(h["quantity"])
        price = float(h["price"])
        holdings.append({
            "symbol": sym,
            "quantity": qty,
            "price": price,
            "value": round(qty * price, 2),
            "cluster": cluster_of(sym),
        })
    if not holdings:
        console.print("[red]Payload had no holdings.[/]")
        raise typer.Exit(1)

    snap = {
        "account": data.get("account", ""),
        "as_of": date.today().isoformat(),
        "cash": round(float(data.get("cash", 0.0)), 2),
        "total_equity": round(sum(h["value"] for h in holdings), 2),
        "holdings": holdings,
    }
    _OUT.parent.mkdir(exist_ok=True)
    _OUT.write_text(json.dumps(snap, indent=2), encoding="utf-8")
    console.print(
        f"[green]Wrote {_OUT.name}[/]: {len(holdings)} holdings, "
        f"equity ${snap['total_equity']:,.0f}, as of {snap['as_of']}."
    )


if __name__ == "__main__":
    app()
