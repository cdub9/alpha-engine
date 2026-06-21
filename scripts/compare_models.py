"""A/B two primary models on the SAME daily snapshot (clean head-to-head).

Builds one snapshot for `--as-of`, sends the identical system prompt +
snapshot to two models, and reports how much they agree (symbol overlap,
direction agreement, conviction gap) plus the cost of each. This is the
immediate, decisive read on "does the cheaper model pick like Opus?"

It costs TWO primary calls (~$0.25 total for Opus + Sonnet) and writes
NOTHING to the DB — it's a pure dry comparison.

    python scripts/compare_models.py                         # today, opus vs sonnet
    python scripts/compare_models.py --as-of 2026-06-18
    python scripts/compare_models.py --model-b claude-haiku-4-5
    python scripts/compare_models.py --save out.json

The single-day agreement tells you whether a switch is LOW-RISK. The skill
verdict is the forward A/B: run both models nightly under their own cohort
tags and compare on the dashboard once trades mature, e.g.

    python scripts/paper_trader.py run-day --generate                          # Opus cohort
    python scripts/paper_trader.py run-day --generate --primary-model claude-sonnet-4-6
"""

from __future__ import annotations

import json
import sys
from datetime import date, datetime

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

from alpha_engine.core.config import get_settings
from alpha_engine.core.logging import configure_logging
from alpha_engine.db import get_connection
from alpha_engine.llm.client import LLMClient
from alpha_engine.llm.compare import compare_outputs
from alpha_engine.llm.context import build_snapshot
from alpha_engine.llm.prompts import OUTPUT_SCHEMA, SYSTEM_PROMPT, USER_MESSAGE_TEMPLATE

console = Console()
app = typer.Typer(add_completion=False, help=__doc__)


def _fmt_opt(x, kind: str) -> str:
    if x is None:
        return "—"
    if kind == "pct":
        return f"{x:.0%}"
    if kind == "f2":
        return f"{x:.2f}"
    return str(x)


@app.command()
def main(
    as_of: str = typer.Option("", help="YYYY-MM-DD; defaults to today"),
    model_a: str = typer.Option("claude-opus-4-7", "--model-a", help="Incumbent"),
    model_b: str = typer.Option("claude-sonnet-4-6", "--model-b", help="Challenger"),
    effort: str = typer.Option("high", help="Primary effort for BOTH calls"),
    save: str = typer.Option("", help="Path to write outputs + comparison JSON"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the cost confirmation"),
    log_level: str = typer.Option("WARNING", "--log-level"),
) -> None:
    configure_logging(level=log_level)
    settings = get_settings()
    if not settings.anthropic_api_key:
        console.print("[red bold]ANTHROPIC_API_KEY not set.[/] Add it to .env.")
        raise typer.Exit(1)

    target = datetime.strptime(as_of, "%Y-%m-%d").date() if as_of else date.today()

    console.print(
        Panel(
            f"[yellow]A/B for {target}: [bold]{model_a}[/] vs [bold]{model_b}[/][/]\n"
            f"Two paid primary calls (~$0.25 total). Writes nothing to the DB.",
            border_style="yellow",
        )
    )
    if not yes and not typer.confirm("Proceed?", default=True):
        raise typer.Exit(0)

    # ONE snapshot, fed identically to both models.
    with get_connection(read_only=True) as con:
        universe = [
            r[0] for r in con.execute(
                "SELECT symbol FROM instruments WHERE active = TRUE ORDER BY symbol"
            ).fetchall()
        ]
        snapshot = build_snapshot(con, universe=universe, as_of=target)

    user_message = USER_MESSAGE_TEMPLATE.format(snapshot_markdown=snapshot.markdown)
    client = LLMClient()

    responses = {}
    for label, model in (("a", model_a), ("b", model_b)):
        console.print(f"[cyan]Calling {model}…[/]")
        responses[label] = client.call_structured(
            system_prompt=SYSTEM_PROMPT,
            user_message=user_message,
            output_schema=OUTPUT_SCHEMA,
            model=model,
            effort=effort,
        )

    ra, rb = responses["a"], responses["b"]
    cmp = compare_outputs(ra.parsed, rb.parsed)

    # --- Agreement ---------------------------------------------------------
    ov = cmp["overall"]
    agree = Table(title="Agreement (same snapshot, both models)", header_style="bold")
    agree.add_column("Scope")
    agree.add_column("Symbol overlap", justify="right")
    agree.add_column("Shared", justify="right")
    agree.add_column("Direction agree", justify="right")
    agree.add_column("Conviction MAE", justify="right")
    for label, ch in cmp["by_channel"].items():
        agree.add_row(
            label,
            _fmt_opt(ch["symbol_jaccard"], "pct"),
            f"{ch['n_shared']} ({ch['n_a']}/{ch['n_b']})",
            _fmt_opt(ch["direction_agreement"], "pct"),
            _fmt_opt(ch["conviction_mae"], "f2"),
        )
    agree.add_row(
        "[bold]overall[/]",
        f"[bold]{_fmt_opt(ov['symbol_jaccard'], 'pct')}[/]",
        f"[bold]{ov['n_shared']}[/]",
        f"[bold]{_fmt_opt(ov['direction_agreement'], 'pct')}[/]",
        f"[bold]{_fmt_opt(ov['conviction_mae'], 'f2')}[/]",
    )
    console.print(agree)

    # Direction disagreements + picks unique to each model (the interesting bits)
    for label, ch in cmp["by_channel"].items():
        if ch["diff_direction"]:
            console.print(f"\n[bold]{label} — direction conflicts:[/]")
            for d in ch["diff_direction"]:
                console.print(
                    f"  • {d['symbol']}: {model_a}={d['a_dir']} "
                    f"({_fmt_opt(d['a_conv'], 'f2')}) vs {model_b}={d['b_dir']} "
                    f"({_fmt_opt(d['b_conv'], 'f2')})"
                )
        if ch["only_a"]:
            console.print(f"  [dim]{label} only in {model_a}: {', '.join(ch['only_a'])}[/]")
        if ch["only_b"]:
            console.print(f"  [dim]{label} only in {model_b}: {', '.join(ch['only_b'])}[/]")

    # --- Cost --------------------------------------------------------------
    cost = Table(title="Cost (this single-day comparison)", header_style="bold")
    cost.add_column("Model")
    cost.add_column("Input", justify="right")
    cost.add_column("Output", justify="right")
    cost.add_column("Cost (USD)", justify="right")
    for model, r in ((model_a, ra), (model_b, rb)):
        cost.add_row(model, f"{r.input_tokens:,}", f"{r.output_tokens:,}",
                     f"${r.cost_estimate_usd:.4f}")
    console.print(cost)

    a_cost, b_cost = ra.cost_estimate_usd, rb.cost_estimate_usd
    if a_cost > 0:
        saving = (a_cost - b_cost) / a_cost
        verb = "cheaper" if saving > 0 else "more expensive"
        console.print(
            f"\n[bold]{model_b}[/] is [bold]{abs(saving):.0%} {verb}[/] than "
            f"{model_a} on this run (${b_cost:.4f} vs ${a_cost:.4f}). "
            f"At ~21 trading days/mo that is ~${(a_cost - b_cost) * 21:+.2f}/mo."
        )

    console.print(
        "\n[dim]Single-day agreement gauges switch RISK, not skill. For the "
        "skill verdict, run both nightly under their own cohort tags and "
        "compare on the dashboard once trades mature.[/]"
    )

    if save:
        payload = {
            "as_of": target.isoformat(),
            "model_a": model_a, "model_b": model_b,
            "comparison": cmp,
            "cost": {model_a: a_cost, model_b: b_cost},
            "output_a": ra.parsed, "output_b": rb.parsed,
        }
        with open(save, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)
        console.print(f"[green]Saved {save}[/]")


if __name__ == "__main__":
    app()
