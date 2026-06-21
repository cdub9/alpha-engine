"""Run the daily LLM digest and print suggestions.

Usage:
    python scripts/run_digest.py
    python scripts/run_digest.py --no-dissent              # cheaper, faster
    python scripts/run_digest.py --no-persist              # dry run
    python scripts/run_digest.py --effort max              # max quality, ~2x cost
    python scripts/run_digest.py --dump-snapshot           # print the snapshot too
    python scripts/run_digest.py --as-of 2026-05-20        # historical
"""

from __future__ import annotations

import json
import sys
from datetime import date, datetime
from typing import Optional

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
from alpha_engine.llm import run_digest
from alpha_engine.llm.digest import DigestRun

console = Console()
app = typer.Typer(add_completion=False, help=__doc__)


REGIME_COLOR = {
    "expansion_low_vol": "green",
    "expansion_high_vol": "yellow",
    "late_cycle": "magenta",
    "recession": "red",
    "recovery": "cyan",
    "unknown": "white",
}


def _direction_color(direction: str) -> str:
    return {
        "buy": "green",
        "add": "green",
        "sell": "red",
        "exit": "red",
        "reduce": "yellow",
        "hold": "white",
    }.get(direction, "white")


def _print_suggestions(channel_label: str, color: str, suggestions: list[dict]) -> None:
    if not suggestions:
        console.print(
            f"\n[bold {color}]{channel_label}[/]: [dim]no suggestions[/]"
        )
        return
    console.print(f"\n[bold {color}]{channel_label}[/] ({len(suggestions)} suggestions)")
    table = Table(show_header=True, header_style="bold")
    table.add_column("Symbol", style="cyan")
    table.add_column("Direction", justify="center")
    table.add_column("Conv", justify="right")
    table.add_column("Weight", justify="right")
    table.add_column("Horizon", justify="right")
    table.add_column("Stop", justify="right")
    table.add_column("Rationale (excerpt)", max_width=80)
    for s in suggestions:
        d_color = _direction_color(s.get("direction", ""))
        table.add_row(
            s.get("symbol", "?"),
            f"[{d_color}]{s.get('direction', '?')}[/]",
            f"{s.get('conviction', 0):.1f}",
            f"{(s.get('target_weight') or 0):.1%}" if s.get("target_weight") else "—",
            f"{s.get('time_horizon_days', '—')}d",
            f"{(s.get('stop_loss_pct') or 0):.1%}" if s.get("stop_loss_pct") else "—",
            (s.get("rationale", "") or "")[:200] + (
                "..." if len(s.get("rationale", "")) > 200 else ""
            ),
        )
    console.print(table)


def _print_run(run: DigestRun, dump_snapshot: bool) -> None:
    snap = run.snapshot

    if dump_snapshot:
        console.print()
        console.rule("Snapshot (sent to LLM)")
        console.print(snap.markdown)
        console.rule("End snapshot")

    color = REGIME_COLOR.get(snap.regime_label, "white")
    console.print()
    console.print(
        Panel(
            f"[bold {color}]{snap.regime_label.upper()}[/]  "
            f"(confidence {snap.regime_confidence:.2f})\n"
            + (
                f"\n[yellow]Notable: {' • '.join(snap.notable_events)}[/]"
                if snap.notable_events
                else "[dim]No notable events flagged[/]"
            ),
            title=f"Market state — {snap.as_of}",
            border_style=color,
        )
    )

    output = run.final_output
    console.print(f"\n[bold]Market summary:[/] {output.get('market_summary', '')}")

    themes = output.get("key_themes", [])
    if themes:
        console.print("\n[bold]Key themes:[/]")
        for t in themes:
            console.print(f"  • {t}")

    _print_suggestions(
        "Channel A (steady_alpha)", "blue", output.get("channel_a_suggestions", [])
    )
    _print_suggestions(
        "Channel B (aggressive_growth)", "magenta", output.get("channel_b_suggestions", [])
    )

    risks = output.get("risk_notes", [])
    if risks:
        console.print("\n[bold red]Risk notes:[/]")
        for r in risks:
            console.print(f"  • {r}")

    if run.dissents:
        console.print(f"\n[bold]Dissent layer:[/] challenged {len(run.dissents)} suggestion(s)")
        for ch, sym, d in run.dissents:
            adj = f"{d.conviction_adjustment:+d}"
            strong = " [red bold](STRONG → demoted to hold)[/]" if d.is_strong_counter else ""
            console.print(f"  • {ch} / {sym}: adjustment {adj}{strong}")
            console.print(f"    [dim]{d.counter_argument[:200]}{'...' if len(d.counter_argument) > 200 else ''}[/]")

    # Cost summary
    primary = run.primary_response
    console.print()
    cost_table = Table(title="Cost summary", show_header=True, header_style="bold")
    cost_table.add_column("Call")
    cost_table.add_column("Input", justify="right")
    cost_table.add_column("Output", justify="right")
    cost_table.add_column("Cache write", justify="right")
    cost_table.add_column("Cache read", justify="right")
    cost_table.add_column("Cost (USD)", justify="right")
    cost_table.add_row(
        "primary",
        f"{primary.input_tokens:,}",
        f"{primary.output_tokens:,}",
        f"{primary.cache_creation_tokens:,}",
        f"{primary.cache_read_tokens:,}",
        f"${primary.cost_estimate_usd:.4f}",
    )
    # Dedupe dissent responses by id() — batch dissent shares one LLMResponse
    seen_response_ids: set[int] = set()
    for ch, sym, d in run.dissents:
        r = d.raw_response
        rid = id(r)
        if rid in seen_response_ids:
            continue
        seen_response_ids.add(rid)
        symbols_for_response = [
            s for _, s, dd in run.dissents if id(dd.raw_response) == rid
        ]
        label = (
            f"dissent (batch, {len(symbols_for_response)} suggestions)"
            if len(symbols_for_response) > 1
            else f"dissent ({sym})"
        )
        cost_table.add_row(
            label,
            f"{r.input_tokens:,}",
            f"{r.output_tokens:,}",
            f"{r.cache_creation_tokens:,}",
            f"{r.cache_read_tokens:,}",
            f"${r.cost_estimate_usd:.4f}",
        )
    cost_table.add_row(
        "[bold]TOTAL[/]", "", "", "", "", f"[bold]${run.total_cost_usd:.4f}[/]"
    )
    console.print(cost_table)

    if run.persisted:
        console.print(
            f"\n[green]Persisted:[/] "
            f"A={run.persisted.inserted_a} (skipped {run.persisted.skipped_a}), "
            f"B={run.persisted.inserted_b} (skipped {run.persisted.skipped_b})"
        )


@app.command()
def main(
    as_of: str = typer.Option("", help="YYYY-MM-DD; defaults to today"),
    no_dissent: bool = typer.Option(False, "--no-dissent"),
    dissent_threshold: float = typer.Option(
        7.5, help="Only challenge suggestions with conviction >= this"
    ),
    dissent_model: str = typer.Option(
        "claude-haiku-4-5",
        help="Model for batch dissent. Use 'claude-opus-4-7' for deeper counter-args.",
    ),
    primary_model: str = typer.Option(
        "claude-opus-4-7",
        "--primary-model",
        help="Model for the primary suggestion call. Pass 'claude-sonnet-4-6' "
             "to A/B a cheaper model; persisted signals are tagged by model.",
    ),
    no_persist: bool = typer.Option(False, "--no-persist"),
    effort: str = typer.Option("high", help="low|medium|high|xhigh|max (primary call)"),
    dump_snapshot: bool = typer.Option(False, "--dump-snapshot"),
    save_json: str = typer.Option(
        "", help="Path to write the full raw LLM output as JSON"
    ),
    repeat: int = typer.Option(
        1,
        help=(
            "Run the digest N times back-to-back. Used to VERIFY prompt caching: "
            "calls 2..N should show non-zero cache_read_tokens on the primary call. "
            "EACH CALL COSTS MONEY (~$0.15). Default 1 (single normal run)."
        ),
    ),
    log_level: str = typer.Option("WARNING", "--log-level"),
) -> None:
    configure_logging(level=log_level)
    settings = get_settings()

    if not settings.anthropic_api_key:
        console.print(
            "[red bold]ANTHROPIC_API_KEY not set.[/]\n"
            "Get a key at https://console.anthropic.com/settings/keys\n"
            "Then add it to .env as ANTHROPIC_API_KEY=sk-ant-..."
        )
        raise typer.Exit(1)

    target_date = (
        datetime.strptime(as_of, "%Y-%m-%d").date() if as_of else date.today()
    )

    repeat = max(1, int(repeat))
    if repeat > 1:
        est_cost = 0.15 * repeat
        console.print(
            Panel(
                f"[yellow]Cache-verification mode: {repeat} consecutive runs.[/]\n"
                f"Expected cost: ~${est_cost:.2f} (each call ~$0.15).\n"
                f"Calls 2..{repeat} should show non-zero cache_read_tokens.",
                border_style="yellow",
            )
        )

    cache_verification: list[tuple[int, int, int]] = []

    for i in range(1, repeat + 1):
        if repeat > 1:
            console.print(f"\n[bold magenta]===== Run {i}/{repeat} =====[/]")

        console.print(
            f"[bold cyan]Running digest for {target_date}[/] "
            f"(primary {primary_model} effort={effort}, dissent={not no_dissent} "
            f"@ {dissent_model} threshold {dissent_threshold}, "
            f"persist={not no_persist})"
        )

        run = run_digest(
            as_of=target_date,
            enable_dissent=not no_dissent,
            dissent_model=dissent_model,
            primary_model=primary_model,
            dissent_min_conviction=dissent_threshold,
            persist=not no_persist,
            effort=effort,
        )
        _print_run(run, dump_snapshot=dump_snapshot)

        if save_json and i == repeat:
            with open(save_json, "w", encoding="utf-8") as f:
                json.dump(run.final_output, f, indent=2)
            console.print(f"\n[green]Saved raw output to {save_json}[/]")

        # Capture cache stats for the primary call
        p = run.primary_response
        cache_verification.append(
            (i, p.cache_creation_tokens, p.cache_read_tokens)
        )

    if repeat > 1:
        table = Table(title="Cache hit verification — primary call across runs")
        table.add_column("Run", justify="right")
        table.add_column("cache_creation_tokens", justify="right")
        table.add_column("cache_read_tokens", justify="right")
        table.add_column("verdict")
        for run_n, creat, read in cache_verification:
            if run_n == 1:
                verdict = "[dim]baseline (creation)[/]"
            elif read > 0:
                verdict = "[green]CACHE HIT[/]"
            else:
                verdict = "[red]CACHE MISS — caching not working[/]"
            table.add_row(str(run_n), f"{creat:,}", f"{read:,}", verdict)
        console.print(table)

        all_hits = all(r[2] > 0 for r in cache_verification[1:])
        if all_hits and repeat > 1:
            console.print(
                "[green bold]✅ Prompt caching is working — confirms ~$2-3/mo savings vs uncached.[/]"
            )
        elif repeat > 1:
            console.print(
                "[red bold]❌ Cache misses detected.[/] Possible causes: "
                "5-min TTL expired between calls, system prompt changed, "
                "or cache_control header not applied. Investigate llm/client.py."
            )


if __name__ == "__main__":
    app()
