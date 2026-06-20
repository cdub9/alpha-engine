"""Ingest GDELT geopolitical signals from BigQuery (historical-capable).

Unlike the DOC API path (scripts/ingest_gdelt.py, ~30-day rolling window),
this pulls YEARS of history so the geopolitical layer can finally be
backtested. One scan per time-chunk computes every signal at once.

COST SAFETY (read this):
  BigQuery bills by bytes SCANNED, not rows returned. On a non-partitioned
  table a date filter does NOT reduce the scan. So this script ALWAYS
  dry-runs first (free, hits no data) and refuses any chunk whose estimate
  exceeds --max-gb. The BigQuery free tier is 1 TB scanned/month; beyond it
  the rate is ~$6.25/TB. The default table is the *_partitioned GKG so date
  filters prune; --no-theme-match drops the largest column (V2Themes) for a
  cheaper entity-only match.

Auth: needs Application Default Credentials and a billing-enabled project.
    pip install -e ".[bigquery]"
    gcloud auth application-default login      # one-time
    python scripts/ingest_gdelt_bq.py --project YOUR_GCP_PROJECT --dry-run

Examples:
    # See the cost first, write nothing:
    python scripts/ingest_gdelt_bq.py --project my-proj --start 2015-01-01 --dry-run
    # Backfill a year once you're happy with the estimate:
    python scripts/ingest_gdelt_bq.py --project my-proj --start 2024-01-01 --end 2025-01-01
    # Full history, chunked by year, each chunk gated at 100 GB:
    python scripts/ingest_gdelt_bq.py --project my-proj --start 2015-01-01 --max-gb 100
"""

from __future__ import annotations

import sys
from datetime import date, datetime, timedelta

import typer
from rich.console import Console
from rich.table import Table

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

from alpha_engine.core.config import get_settings
from alpha_engine.core.logging import configure_logging
from alpha_engine.data import upsert_geopolitical_points
from alpha_engine.data.gdelt_bigquery import (
    DEFAULT_GKG_TABLE,
    DEFAULT_PARTITION_FIELD,
    build_gkg_query,
)
from alpha_engine.db import get_connection, init_schema

console = Console()
app = typer.Typer(add_completion=False, help=__doc__)

# BigQuery on-demand price as of 2026; only used to show a rough projection.
_USD_PER_TB = 6.25
_FREE_TB_PER_MONTH = 1.0


def _year_chunks(start: date, end: date) -> list[tuple[date, date]]:
    """[start, end) split on calendar-year boundaries (end exclusive)."""
    chunks = []
    cur = start
    while cur < end:
        nxt = min(date(cur.year + 1, 1, 1), end)
        chunks.append((cur, nxt))
        cur = nxt
    return chunks


@app.command()
def ingest(
    project: str = typer.Option("", help="GCP project id (or set GOOGLE_CLOUD_PROJECT)."),
    start: str = typer.Option("2015-01-01", help="Start date YYYY-MM-DD (GKG begins 2015)."),
    end: str = typer.Option("", help="End date YYYY-MM-DD exclusive (default: today)."),
    table: str = typer.Option(DEFAULT_GKG_TABLE, help="GKG table. Prefer the *_partitioned one."),
    partitioned: bool = typer.Option(
        True, help="Table is date-partitioned (prune on _PARTITIONTIME). "
                   "Pass --no-partitioned for the plain gdelt-bq.gdeltv2.gkg."
    ),
    theme_match: bool = typer.Option(
        True, help="Match on AllNames + V2Themes. --no-theme-match is cheaper "
                   "(entity-only, drops the large V2Themes column)."
    ),
    max_gb: float = typer.Option(100.0, help="Refuse any chunk whose dry-run estimate exceeds this."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Estimate + print only; write nothing."),
    yes: bool = typer.Option(False, "--yes", help="Skip the interactive spend confirmation."),
    log_level: str = typer.Option("WARNING", "--log-level"),
) -> None:
    import os

    configure_logging(level=log_level)
    project = project or os.environ.get("GOOGLE_CLOUD_PROJECT", "")
    if not project:
        console.print("[red]No --project and GOOGLE_CLOUD_PROJECT unset.[/]")
        raise typer.Exit(1)

    start_d = datetime.strptime(start, "%Y-%m-%d").date()
    end_d = datetime.strptime(end, "%Y-%m-%d").date() if end else date.today() + timedelta(days=1)
    if end_d <= start_d:
        console.print("[red]--end must be after --start.[/]")
        raise typer.Exit(1)

    signals = [s for s in get_settings().geopolitical.signals if s.bq_match]
    if not signals:
        console.print("[red]No signals have a bq_match in config/geopolitical.yaml.[/]")
        raise typer.Exit(1)

    partition_field = DEFAULT_PARTITION_FIELD if partitioned else None
    chunks = _year_chunks(start_d, end_d)
    console.print(
        f"[bold cyan]BigQuery GKG ingest[/]  project=[bold]{project}[/]  "
        f"{start_d} → {end_d}  ({len(signals)} signals, {len(chunks)} year-chunk(s))\n"
        f"table={table}  theme_match={theme_match}  max_gb={max_gb}"
    )

    init_schema()

    # Lazy client (import guarded so --help works without the SDK).
    from alpha_engine.data.gdelt_bigquery import GDELTBigQueryClient

    try:
        client = GDELTBigQueryClient(project=project)
    except ImportError as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(1)

    # --- Estimate every chunk first (free) so the user sees total cost. ----
    plans = []
    total_gb = 0.0
    est_table = Table(title="Dry-run cost estimate (bytes BigQuery would scan)")
    est_table.add_column("Chunk")
    est_table.add_column("Est. scan", justify="right")
    est_table.add_column("Status", justify="right")
    for lo, hi in chunks:
        sql = build_gkg_query(signals, lo, hi, table=table,
                              partition_field=partition_field, theme_match=theme_match)
        try:
            gb = client.estimate_gb(sql)
        except Exception as exc:
            console.print(f"[red]Dry-run failed for {lo}→{hi}: {exc}[/]")
            raise typer.Exit(1)
        over = gb > max_gb
        plans.append((lo, hi, sql, gb, over))
        total_gb += gb
        est_table.add_row(
            f"{lo} → {hi}", f"{gb:,.1f} GB",
            "[red]OVER max-gb[/]" if over else "[green]ok[/]",
        )
    console.print(est_table)
    proj_cost = max(0.0, (total_gb / 1024) - _FREE_TB_PER_MONTH) * _USD_PER_TB
    console.print(
        f"\nTotal estimated scan: [bold]{total_gb:,.1f} GB[/] "
        f"({total_gb/1024:.2f} TB).  Free tier: {_FREE_TB_PER_MONTH:.0f} TB/month. "
        f"Projected charge beyond free tier this run: [bold]${proj_cost:,.2f}[/] "
        f"(at ${_USD_PER_TB}/TB)."
    )

    runnable = [p for p in plans if not p[4]]
    blocked = [p for p in plans if p[4]]
    if blocked:
        console.print(
            f"[yellow]{len(blocked)} chunk(s) exceed --max-gb {max_gb} and will be "
            f"skipped. Raise --max-gb, narrow the dates, or use --no-theme-match.[/]"
        )

    if dry_run:
        console.print("[bold]--dry-run: nothing written.[/]")
        return
    if not runnable:
        console.print("[red]Every chunk is over the gate; nothing to run.[/]")
        raise typer.Exit(1)
    if not yes:
        ok = typer.confirm(
            f"Run {len(runnable)} chunk(s), scanning ~{sum(p[3] for p in runnable):,.0f} GB?"
        )
        if not ok:
            console.print("Aborted.")
            raise typer.Exit(0)

    # --- Execute the approved chunks. --------------------------------------
    from alpha_engine.data.gdelt_bigquery import rows_to_points

    total_rows = 0
    for lo, hi, sql, gb, _ in runnable:
        console.print(f"[bold]Running[/] {lo} → {hi}  (~{gb:,.1f} GB)…")
        rows = client.run(sql)
        per_signal = rows_to_points(rows, signals)
        with get_connection() as con:
            for name, points in per_signal.items():
                total_rows += upsert_geopolitical_points(con, name, points, source="gdelt_bq")
        console.print(f"  stored {sum(len(v) for v in per_signal.values()):,} signal-days")

    console.print(f"\n[bold green]Done.[/] Upserted {total_rows:,} signal-day rows from BigQuery.")

    with get_connection(read_only=True) as con:
        rows = con.execute(
            """
            SELECT signal_name, source, COUNT(*) n,
                   MIN(signal_date) f, MAX(signal_date) l
            FROM geopolitical_signals GROUP BY 1,2 ORDER BY 1,2
            """
        ).fetchall()
    summary = Table(title="geopolitical_signals coverage")
    for c in ("Signal", "Source", "Days", "First", "Last"):
        summary.add_column(c)
    for name, source, n, f, l in rows:
        summary.add_row(name, source, str(n), str(f), str(l))
    console.print(summary)


if __name__ == "__main__":
    app()
