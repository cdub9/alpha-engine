"""Inspect classified regimes.

Shows:
  1. Current regime classification with full reasoning
  2. Timeline of regime changes (compressed runs)
  3. Regime distribution stats

Usage:
    python scripts/inspect_regimes.py
"""

from __future__ import annotations

import json
import sys
from datetime import date

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

from alpha_engine.core.types import MarketRegime
from alpha_engine.db import get_connection
from alpha_engine.regime import REGIME_DESCRIPTIONS, classify, extract_features

console = Console()


REGIME_COLOR = {
    MarketRegime.EXPANSION_LOW_VOL.value: "green",
    MarketRegime.EXPANSION_HIGH_VOL.value: "yellow",
    MarketRegime.LATE_CYCLE.value: "magenta",
    MarketRegime.RECESSION.value: "red",
    MarketRegime.RECOVERY.value: "cyan",
    MarketRegime.UNKNOWN.value: "white",
}


def _color(regime: str) -> str:
    return REGIME_COLOR.get(regime, "white")


def main() -> None:
    today = date.today()

    # --- 1. Current classification (live, not from cache) ----------------
    with get_connection(read_only=True) as con:
        features = extract_features(con, today)
        assessment = classify(features)

    color = _color(assessment.regime.value)
    body = [
        f"[bold {color}]{assessment.regime.value.upper()}[/]   "
        f"confidence: {assessment.confidence:.2f}",
        "",
        REGIME_DESCRIPTIONS[assessment.regime],
        "",
        "[bold]Reasoning:[/]",
    ]
    for r in assessment.reasoning:
        body.append(f"  • {r}")
    console.print(
        Panel(
            "\n".join(body),
            title=f"Current regime ({today})",
            border_style=color,
        )
    )

    # --- 2. Timeline of changes -----------------------------------------
    with get_connection(read_only=True) as con:
        rows = con.execute(
            "SELECT classification_date, regime, confidence, features_json "
            "FROM regime_classifications "
            "WHERE model_version = 'rule_v1' "
            "ORDER BY classification_date"
        ).fetchall()

    if not rows:
        console.print(
            "\n[yellow]No regime history found. "
            "Run scripts/classify_regimes.py first.[/]"
        )
        return

    # Compress: keep only rows where regime changed from the previous one
    runs: list[tuple[date, date, str, float, list[str]]] = []
    cur_start = rows[0][0]
    cur_regime = rows[0][1]
    cur_conf = rows[0][2]
    cur_reasoning = json.loads(rows[0][3] or "{}").get("reasoning", [])

    for i in range(1, len(rows)):
        d, regime, conf, fjson = rows[i]
        if regime != cur_regime:
            runs.append(
                (cur_start, rows[i - 1][0], cur_regime, cur_conf, cur_reasoning)
            )
            cur_start = d
            cur_regime = regime
            cur_conf = conf
            cur_reasoning = json.loads(fjson or "{}").get("reasoning", [])
    runs.append((cur_start, rows[-1][0], cur_regime, cur_conf, cur_reasoning))

    table = Table(title="Regime timeline")
    table.add_column("Start")
    table.add_column("End")
    table.add_column("Regime")
    table.add_column("Weeks", justify="right")
    table.add_column("Conf", justify="right")
    table.add_column("Primary reason")
    for s, e, regime, conf, reasoning in runs:
        weeks = (e - s).days // 7 + 1
        color = _color(regime)
        primary = reasoning[0] if reasoning else ""
        table.add_row(
            str(s),
            str(e),
            f"[{color}]{regime}[/]",
            str(weeks),
            f"{conf:.2f}",
            primary[:60] + ("..." if len(primary) > 60 else ""),
        )
    console.print()
    console.print(table)

    # --- 3. Distribution stats ------------------------------------------
    dist: dict[str, int] = {}
    for _, regime, _, _ in [(r[0], r[1], r[2], r[3]) for r in rows]:
        dist[regime] = dist.get(regime, 0) + 1
    total = sum(dist.values())

    dist_table = Table(title="Regime distribution (weekly observations)")
    dist_table.add_column("Regime")
    dist_table.add_column("Count", justify="right")
    dist_table.add_column("% of history", justify="right")
    for regime, count in sorted(dist.items(), key=lambda x: -x[1]):
        color = _color(regime)
        dist_table.add_row(
            f"[{color}]{regime}[/]",
            str(count),
            f"{count / total * 100:.1f}%" if total else "—",
        )
    console.print()
    console.print(dist_table)


if __name__ == "__main__":
    main()
