"""Pull GDELT geopolitical signals into the database.

For each query in config/geopolitical.yaml, fetches daily timeline
volume + tone over the requested window and upserts into the
geopolitical_signals table.

Usage:
    python scripts/ingest_gdelt.py                  # default 30d
    python scripts/ingest_gdelt.py --timespan 90d   # longer history
    python scripts/ingest_gdelt.py --only iran_conflict,oil_disruption
"""

from __future__ import annotations

import sys

import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

from alpha_engine.core.config import get_settings
from alpha_engine.core.logging import configure_logging, get_logger
from alpha_engine.data import GDELTClient, upsert_geopolitical_points
from alpha_engine.db import get_connection, init_schema

console = Console()
log = get_logger(__name__)
app = typer.Typer(add_completion=False, help=__doc__)


@app.command()
def ingest(
    timespan: str = typer.Option(
        "30d",
        help="GDELT timespan: e.g. 7d, 30d, 90d, 1y. Default 30d.",
    ),
    only: str = typer.Option(
        "",
        help="Comma-separated list of signal names to ingest (default: all).",
    ),
    polite_sleep: float = typer.Option(
        0.5, help="Seconds between API calls (be courteous)"
    ),
    log_level: str = typer.Option("WARNING", "--log-level"),
) -> None:
    configure_logging(level=log_level)
    init_schema()

    settings = get_settings()
    all_signals = settings.geopolitical.signals
    if not all_signals:
        console.print("[red]No geopolitical signals configured.[/]")
        raise typer.Exit(1)

    if only:
        wanted = {s.strip() for s in only.split(",")}
        signals = [s for s in all_signals if s.name in wanted]
        missing = wanted - {s.name for s in signals}
        if missing:
            console.print(
                f"[yellow]Unknown signal names ignored: {', '.join(missing)}[/]"
            )
    else:
        signals = all_signals

    console.print(
        f"[bold cyan]Ingesting {len(signals)} GDELT signal(s)[/]  "
        f"timespan={timespan}"
    )

    total_points = 0
    with (
        GDELTClient() as client,
        Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.fields[signal]}"),
            TimeElapsedColumn(),
            console=console,
        ) as progress,
    ):
        task = progress.add_task(
            "fetching", total=len(signals), signal=""
        )
        for i, sig in enumerate(signals):
            progress.update(task, signal=sig.name)
            try:
                points = client.fetch_signal(
                    signal_name=sig.name,
                    query=sig.query,
                    timespan=timespan,
                    polite_sleep=polite_sleep,
                )
                with get_connection() as con:
                    n = upsert_geopolitical_points(con, sig.name, points)
                total_points += n
            except Exception as exc:
                log.error(
                    "ingest_signal_failed", signal=sig.name, error=str(exc)
                )
            progress.advance(task)

    console.print(
        f"\n[bold green]Done.[/]  Stored {total_points:,} signal-days "
        f"across {len(signals)} signals."
    )

    # Quick coverage summary
    with get_connection(read_only=True) as con:
        rows = con.execute(
            """
            SELECT signal_name, COUNT(*) AS n_days,
                   MIN(signal_date) AS first_date,
                   MAX(signal_date) AS last_date,
                   ROUND(AVG(volume_intensity)::DOUBLE, 5) AS avg_vol,
                   ROUND(AVG(avg_tone)::DOUBLE, 2) AS avg_tone
            FROM geopolitical_signals
            GROUP BY signal_name
            ORDER BY signal_name
            """
        ).fetchall()

    console.print("\n[bold]Coverage summary:[/]")
    for r in rows:
        name, n_days, first_d, last_d, vol, tone = r
        console.print(
            f"  {name:25s}  {n_days:3d} days  "
            f"{first_d} -> {last_d}  avg_vol={vol}  avg_tone={tone}"
        )


if __name__ == "__main__":
    app()
