"""Paper-trading runner — track LLM signal outcomes over time without
spending API credits unless asked.

ALL SUBCOMMANDS ARE FREE EXCEPT `run-day --generate`, which makes one
Opus 4.7 + Haiku 4.5 digest call (~$0.15). Estimated cost is shown
before any API call.

Subcommands:
  backfill   — persist signals from llm_signal_cache into signals table
               (free; for retrofitting historical paper trades)
  open       — open paper trades for an already-persisted digest date
               (free)
  score      — score paper trades whose horizon has elapsed (free)
  status     — print the paper-trading track record (free)
  run-day    — open + score for one date; with --generate also runs a
               fresh digest first (only step that costs money)

Typical flows:
  # Catch up from cached history (free)
  python scripts/paper_trader.py backfill
  python scripts/paper_trader.py open --all
  python scripts/paper_trader.py score
  python scripts/paper_trader.py status

  # Daily forward use (paid; ~$0.15)
  python scripts/paper_trader.py run-day --generate
"""

from __future__ import annotations

import sys
from datetime import date, datetime, timedelta
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

from alpha_engine.calendars.scheduled import is_nyse_holiday, is_trading_day
from alpha_engine.backtest.llm_advisor import (
    DEFAULT_MODEL_VERSION,
    DEFAULT_PRIMARY_MODEL,
    config_hash,
    get_cached_output,
    model_version_for,
    store_cached_output,
)
from alpha_engine.core.config import get_settings
from alpha_engine.core.logging import configure_logging
from alpha_engine.db import get_connection, init_schema
from alpha_engine.llm.prompts import SYSTEM_PROMPT
from alpha_engine.paper import (
    backfill_signals_from_cache,
    open_paper_trades_for_date,
    score_due_paper_trades,
)

console = Console()
app = typer.Typer(add_completion=False, help=__doc__)


def _full_universe(con) -> list[str]:
    rows = con.execute(
        "SELECT symbol FROM instruments WHERE active = TRUE ORDER BY symbol"
    ).fetchall()
    return [r[0] for r in rows]


# ---------------------------------------------------------------------------
# Subcommand: backfill — persist signals from llm_signal_cache (FREE)
# ---------------------------------------------------------------------------


@app.command()
def backfill(
    model_version: str = typer.Option(DEFAULT_MODEL_VERSION),
    log_level: str = typer.Option("WARNING", "--log-level"),
) -> None:
    """[FREE] Persist signals from llm_signal_cache into the signals table.

    Useful when you generated digests with persist=False (e.g. the
    backtest) and want to use them for paper trading retrospectively.
    Idempotent."""
    configure_logging(level=log_level)
    init_schema()
    with get_connection() as con:
        n = backfill_signals_from_cache(con, model_version=model_version)
    console.print(f"[green]Backfilled {n} signals from cached digests.[/]")


# ---------------------------------------------------------------------------
# Subcommand: open — open paper trades for already-persisted signals (FREE)
# ---------------------------------------------------------------------------


@app.command()
def open(  # noqa: A001 — shadowing built-in is intentional and contextual
    digest_date: str = typer.Option(
        "", help="YYYY-MM-DD; defaults to today. Ignored if --all is set."
    ),
    all_dates: bool = typer.Option(
        False, "--all", help="Open paper trades for every persisted digest date."
    ),
    min_conviction: float = typer.Option(
        6.0, help="Only open trades for signals at/above this conviction."
    ),
    model_version: str = typer.Option(DEFAULT_MODEL_VERSION),
    log_level: str = typer.Option("WARNING", "--log-level"),
) -> None:
    """[FREE] Open paper trades for already-persisted signals."""
    configure_logging(level=log_level)
    init_schema()

    with get_connection() as con:
        if all_dates:
            dates = [
                r[0]
                for r in con.execute(
                    "SELECT DISTINCT DATE(generated_at) AS d FROM signals "
                    "WHERE model_version = ? ORDER BY d",
                    [model_version],
                ).fetchall()
            ]
        else:
            target = (
                datetime.strptime(digest_date, "%Y-%m-%d").date()
                if digest_date
                else date.today()
            )
            dates = [target]

        totals = {
            "opened": 0,
            "non_actionable": 0,
            "below_conv": 0,
            "no_price": 0,
            "seen": 0,
        }
        for d in dates:
            r = open_paper_trades_for_date(
                con, d, model_version=model_version, min_conviction=min_conviction
            )
            totals["opened"] += r.paper_trades_opened
            totals["non_actionable"] += r.skipped_non_actionable
            totals["below_conv"] += r.skipped_below_conviction
            totals["no_price"] += r.skipped_no_entry_price
            totals["seen"] += r.signals_seen

    console.print(
        f"Opened [green]{totals['opened']}[/] paper trades across "
        f"{len(dates)} digest date(s).\n"
        f"  signals seen:        {totals['seen']}\n"
        f"  skipped non-action:  {totals['non_actionable']} (hold/sell/exit/reduce)\n"
        f"  skipped low conv:    {totals['below_conv']} (< {min_conviction})\n"
        f"  skipped no price:    {totals['no_price']}"
    )


# ---------------------------------------------------------------------------
# Subcommand: score — grade trades whose horizon has elapsed (FREE)
# ---------------------------------------------------------------------------


@app.command()
def score(
    as_of: str = typer.Option(
        "", help="YYYY-MM-DD; defaults to today. Trades exiting <= this are scored."
    ),
    default_horizon: int = typer.Option(
        30, help="Default horizon (days) for signals without one."
    ),
    log_level: str = typer.Option("WARNING", "--log-level"),
) -> None:
    """[FREE] Score every paper trade whose planned exit has elapsed."""
    configure_logging(level=log_level)
    init_schema()
    target = datetime.strptime(as_of, "%Y-%m-%d").date() if as_of else date.today()

    with get_connection() as con:
        r = score_due_paper_trades(con, as_of=target, default_horizon_days=default_horizon)

    console.print(
        f"Inspected [bold]{r.inspected}[/] open paper trades.\n"
        f"  scored:        [green]{r.scored}[/]\n"
        f"  still open:    {r.still_open}\n"
        f"  no exit price: {r.no_exit_price}"
    )


# ---------------------------------------------------------------------------
# Subcommand: status — print the paper-trading track record (FREE)
# ---------------------------------------------------------------------------


def _channel_stats(con, channel: str) -> dict:
    row = con.execute(
        """
        SELECT
            COUNT(*) AS n,
            AVG(o.return_pct) AS avg_ret,
            AVG(o.alpha) AS avg_alpha,
            AVG(CASE WHEN o.direction_correct THEN 1.0 ELSE 0.0 END) AS win_rate,
            SUM(CASE WHEN o.return_pct > 0 THEN o.return_pct ELSE 0 END) AS gross_win,
            SUM(CASE WHEN o.return_pct < 0 THEN -o.return_pct ELSE 0 END) AS gross_loss,
            AVG(o.days_held) AS avg_days
        FROM trades t
        JOIN trade_outcomes o ON o.trade_id = t.id
        WHERE t.channel = ?
        """,
        [channel],
    ).fetchone()
    n, avg_ret, avg_alpha, win_rate, gross_win, gross_loss, avg_days = row
    return {
        "n": n or 0,
        "avg_ret": avg_ret or 0.0,
        "avg_alpha": avg_alpha or 0.0,
        "win_rate": win_rate or 0.0,
        "profit_factor": (gross_win / gross_loss) if (gross_loss or 0) > 0 else 0.0,
        "avg_days": avg_days or 0.0,
    }


def _per_symbol_stats(con, channel: str, limit: int = 10) -> list[tuple]:
    return con.execute(
        f"""
        SELECT t.symbol, COUNT(*), AVG(o.return_pct), AVG(o.alpha)
        FROM trades t
        JOIN trade_outcomes o ON o.trade_id = t.id
        WHERE t.channel = ?
        GROUP BY t.symbol
        HAVING COUNT(*) >= 2
        ORDER BY AVG(o.alpha) DESC
        LIMIT {int(limit)}
        """,
        [channel],
    ).fetchall()


@app.command()
def status(
    log_level: str = typer.Option("WARNING", "--log-level"),
) -> None:
    """[FREE] Print paper-trading track record by channel."""
    configure_logging(level=log_level)
    init_schema()

    with get_connection(read_only=True) as con:
        # Top-level counts
        open_count = con.execute(
            """
            SELECT COUNT(*) FROM trades t
            LEFT JOIN trade_outcomes o ON o.trade_id = t.id
            WHERE t.status = 'paper_filled' AND o.trade_id IS NULL
            """
        ).fetchone()[0]
        scored_count = con.execute(
            "SELECT COUNT(*) FROM trade_outcomes"
        ).fetchone()[0]

        console.print(
            Panel(
                f"[bold]Paper trading track record[/]\n"
                f"open: {open_count}    scored: {scored_count}",
                border_style="cyan",
            )
        )

        table = Table(title="Per-channel summary (scored trades only)")
        table.add_column("Channel", style="cyan")
        table.add_column("N", justify="right")
        table.add_column("Avg ret", justify="right")
        table.add_column("Avg alpha", justify="right")
        table.add_column("Win rate", justify="right")
        table.add_column("Profit factor", justify="right")
        table.add_column("Avg days held", justify="right")
        for ch in ("steady_alpha", "aggressive_growth"):
            s = _channel_stats(con, ch)
            if s["n"] == 0:
                table.add_row(ch, "0", "—", "—", "—", "—", "—")
                continue
            table.add_row(
                ch,
                str(s["n"]),
                f"{s['avg_ret']:+.2%}",
                f"{s['avg_alpha']:+.2%}",
                f"{s['win_rate']:.0%}",
                f"{s['profit_factor']:.2f}",
                f"{s['avg_days']:.0f}d",
            )
        console.print(table)

        for ch in ("steady_alpha", "aggressive_growth"):
            rows = _per_symbol_stats(con, ch, limit=10)
            if not rows:
                continue
            t = Table(title=f"{ch} — top symbols by alpha (≥2 trades)")
            t.add_column("Symbol", style="cyan")
            t.add_column("N", justify="right")
            t.add_column("Avg ret", justify="right")
            t.add_column("Avg alpha", justify="right")
            for sym, n, avg_ret, avg_alpha in rows:
                t.add_row(
                    sym,
                    str(n),
                    f"{(avg_ret or 0):+.2%}",
                    f"{(avg_alpha or 0):+.2%}",
                )
            console.print(t)


# ---------------------------------------------------------------------------
# Subcommand: run-day — open + score for one date (FREE unless --generate)
# ---------------------------------------------------------------------------


@app.command(name="run-day")
def run_day(
    as_of: str = typer.Option("", help="YYYY-MM-DD; defaults to today"),
    generate: bool = typer.Option(
        False,
        "--generate",
        help="Run a fresh digest if not cached. THIS COSTS MONEY (~$0.15).",
    ),
    min_conviction: float = typer.Option(6.0),
    default_horizon: int = typer.Option(30),
    primary_model: str = typer.Option(
        DEFAULT_PRIMARY_MODEL,
        "--primary-model",
        help="Model for the primary digest call. Pass 'claude-sonnet-4-6' to "
             "A/B a cheaper model — it lands in its own cohort automatically.",
    ),
    model_version: str = typer.Option(
        "",
        help="Cohort tag for signals/cache. Empty = auto-derive from "
             "--primary-model (recommended; keeps Opus/Sonnet cohorts apart).",
    ),
    effort: str = typer.Option("high"),
    force: bool = typer.Option(
        False,
        "--force",
        help="Override the non-trading-day skip (run --generate even on holidays/weekends).",
    ),
    log_level: str = typer.Option("WARNING", "--log-level"),
) -> None:
    """Open + score paper trades for `as_of`. With --generate, run a
    fresh digest first (only step that costs money). Skips the paid
    generate path on weekends and NYSE holidays unless --force is given;
    free scoring/opening still run so trades catch up."""
    configure_logging(level=log_level)
    init_schema()
    target = datetime.strptime(as_of, "%Y-%m-%d").date() if as_of else date.today()
    # Auto-derive the cohort tag from the model unless overridden, so a
    # Sonnet A/B can't accidentally write under the Opus tag.
    model_version = model_version or model_version_for(primary_model)

    settings = get_settings()
    with get_connection(read_only=True) as con:
        universe = _full_universe(con)
        cfg_hash = config_hash(SYSTEM_PROMPT, universe)
        cached = get_cached_output(con, target, model_version, cfg_hash)

    # Holiday / weekend skip — only affects the paid --generate path.
    # Scoring + opening below still run so we catch up on any pending work.
    if generate and not is_trading_day(target) and not force:
        if target.weekday() >= 5:
            reason = f"{target:%A}"
        elif is_nyse_holiday(target):
            reason = "NYSE holiday"
        else:
            reason = "non-trading day"
        console.print(
            Panel(
                f"[yellow]Skipping digest generation for {target} ({reason}).[/]\n"
                f"No API call made (saved ~$0.15). Pass [bold]--force[/] to override.",
                border_style="yellow",
            )
        )
        generate = False

    if cached is None and generate:
        if not settings.anthropic_api_key:
            console.print("[red]ANTHROPIC_API_KEY not set; cannot generate.[/]")
            raise typer.Exit(1)
        console.print(
            Panel(
                f"[yellow]About to spend ~$0.10-0.16 on a {primary_model} digest "
                f"for {target}[/] (cohort [bold]{model_version}[/]).\n"
                f"Re-run without --generate to skip.",
                border_style="yellow",
            )
        )
        from alpha_engine.llm.digest import run_digest

        run = run_digest(
            as_of=target,
            universe=universe,
            enable_dissent=False,  # cheaper; dissent for live trading is optional
            persist=True,           # also persist into signals
            primary_model=primary_model,
            model_version=model_version,
            effort=effort,
            dissent_model="claude-haiku-4-5",
        )
        with get_connection() as con:
            store_cached_output(
                con,
                as_of=target,
                model_version=model_version,
                cfg_hash=cfg_hash,
                output=run.final_output,
                universe=universe,
                input_tokens=run.primary_response.input_tokens,
                output_tokens=run.primary_response.output_tokens,
                cost_usd=run.total_cost_usd,
            )
        console.print(f"[green]Generated digest. Cost: ${run.total_cost_usd:.4f}[/]")
    elif cached is None:
        console.print(
            f"[yellow]No cached digest for {target}. "
            f"Run with --generate to spend ~$0.15, or pick a date you have cached.[/]"
        )

    # Backfill signals from cache (idempotent; covers the case where we
    # generated above but persist=False, or any past dates).
    with get_connection() as con:
        backfill_signals_from_cache(con, model_version=model_version)

        # Open paper trades for ALL unprocessed signals (not just today's).
        # This self-heals if yesterday's signals couldn't be opened yet
        # because the next trading day's bar wasn't available — they'll get
        # picked up automatically the next time `run-day` runs.
        pending_dates = [
            r[0]
            for r in con.execute(
                """
                SELECT DISTINCT DATE(s.generated_at) AS d
                FROM signals s
                LEFT JOIN trades t ON t.source_signal_id = s.id
                WHERE s.model_version = ? AND t.id IS NULL
                ORDER BY d
                """,
                [model_version],
            ).fetchall()
        ]
        total_opened = 0
        total_no_price = 0
        for d in pending_dates:
            r = open_paper_trades_for_date(
                con, d, model_version=model_version, min_conviction=min_conviction
            )
            total_opened += r.paper_trades_opened
            total_no_price += r.skipped_no_entry_price

        scored = score_due_paper_trades(
            con, as_of=target, default_horizon_days=default_horizon
        )

    console.print(
        f"\n[bold]{target}[/]: opened {total_opened} new paper trades "
        f"(across {len(pending_dates)} pending digest date(s)), "
        f"scored {scored.scored} trades whose horizon ended."
    )
    if total_no_price:
        console.print(
            f"  ({total_no_price} signals deferred until next trading day's bar lands)"
        )
    if scored.still_open:
        console.print(f"  ({scored.still_open} trades still in their horizon window)")


if __name__ == "__main__":
    app()
