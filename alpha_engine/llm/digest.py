"""End-to-end digest orchestrator.

  1. Build the daily snapshot (context.py)
  2. Call LLM for primary suggestions (client.py + prompts.py)
  3. For each high-conviction suggestion, run dissent (dissent.py)
  4. Persist final suggestions to the signals table (parser.py)
  5. Return a structured DigestRun for the CLI / dashboard

Dissent is gated by `dissent_min_conviction` — we don't burn an API call
on a low-conviction "pass" suggestion. Default 6.0 means we only
challenge suggestions in the "act on this" range.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Optional

import duckdb

from alpha_engine.core.logging import get_logger
from alpha_engine.db import get_connection
from alpha_engine.llm.client import LLMClient, LLMResponse
from alpha_engine.llm.context import DailySnapshot, build_snapshot
from alpha_engine.llm.dissent import (
    DissentResult,
    apply_dissent_to_suggestion,
    generate_batch_dissent,
    generate_dissent,
)
from alpha_engine.llm.parser import PersistResult, persist_signals
from alpha_engine.llm.prompts import (
    OUTPUT_SCHEMA,
    SYSTEM_PROMPT,
    USER_MESSAGE_TEMPLATE,
)

log = get_logger(__name__)


@dataclass
class DigestRun:
    """Output of one digest run."""

    as_of: date
    snapshot: DailySnapshot
    primary_response: LLMResponse
    primary_output: dict
    dissents: list[tuple[str, str, DissentResult]] = field(default_factory=list)
    # (channel_label, symbol, dissent_result)
    final_output: dict = field(default_factory=dict)
    persisted: Optional[PersistResult] = None
    total_cost_usd: float = 0.0


def _default_universe(con: duckdb.DuckDBPyConnection) -> list[str]:
    """Universe for the LLM = all active instruments in the DB."""
    rows = con.execute(
        "SELECT symbol FROM instruments WHERE active = TRUE ORDER BY symbol"
    ).fetchall()
    return [r[0] for r in rows]


def run_digest(
    *,
    as_of: Optional[date] = None,
    universe: Optional[list[str]] = None,
    dissent_min_conviction: float = 7.5,
    enable_dissent: bool = True,
    dissent_model: str = "claude-haiku-4-5",
    persist: bool = True,
    effort: str = "high",
    con: Optional[duckdb.DuckDBPyConnection] = None,
) -> DigestRun:
    """Run the digest end-to-end.

    Args:
        as_of: target date for the snapshot. Defaults to today.
        universe: tickers to include. Defaults to all active instruments.
        dissent_min_conviction: only challenge suggestions at/above this.
            Default 7.5 — only the highest-conviction picks are challenged,
            which is where dissent has the most value vs. cost.
        enable_dissent: set False to skip dissent calls entirely (cheaper).
        dissent_model: model used for the batch dissent call. Defaults to
            Haiku 4.5 (~5x cheaper than Opus for bounded reasoning).
        persist: write to signals table. Set False for dry runs.
        effort: LLM effort level for the PRIMARY call. "high" is the
            recommended minimum for intelligence-sensitive work; "max" is
            even higher but ~2x cost.
        con: existing DB connection. If None, opens one.
    """
    owned_con = con is None
    if owned_con:
        con = get_connection(read_only=False)

    try:
        universe = universe or _default_universe(con)
        snapshot = build_snapshot(con, universe=universe, as_of=as_of)

        client = LLMClient()

        # --- Primary call ----------------------------------------------
        user_message = USER_MESSAGE_TEMPLATE.format(
            snapshot_markdown=snapshot.markdown
        )
        primary = client.call_structured(
            system_prompt=SYSTEM_PROMPT,
            user_message=user_message,
            output_schema=OUTPUT_SCHEMA,
            effort=effort,
        )
        primary_output = primary.parsed

        # --- Dissent (batched) ----------------------------------------
        # Collect ALL high-conviction suggestions across both channels,
        # then challenge them in a single batched call. This is much
        # cheaper than one call per suggestion (most cost is per-call
        # overhead like resending the snapshot context).
        dissents: list[tuple[str, str, DissentResult]] = []
        if enable_dissent:
            to_challenge: list[tuple[str, dict]] = []
            for channel_label, key in (
                ("steady_alpha", "channel_a_suggestions"),
                ("aggressive_growth", "channel_b_suggestions"),
            ):
                for sug in primary_output.get(key, []):
                    if float(sug.get("conviction", 0)) >= dissent_min_conviction:
                        to_challenge.append((channel_label, sug))

            log.info(
                "dissent_batch_planned",
                count=len(to_challenge),
                model=dissent_model,
                threshold=dissent_min_conviction,
            )

            if to_challenge:
                try:
                    dissent_map = generate_batch_dissent(
                        client=client,
                        suggestions=to_challenge,
                        snapshot_markdown=snapshot.markdown,
                        model=dissent_model,
                    )
                except Exception as exc:
                    log.warning("batch_dissent_failed", error=str(exc))
                    dissent_map = {}

                # Apply each returned dissent to the matching suggestion
                for channel_label, key in (
                    ("steady_alpha", "channel_a_suggestions"),
                    ("aggressive_growth", "channel_b_suggestions"),
                ):
                    new_list = []
                    for sug in primary_output.get(key, []):
                        symbol = (sug.get("symbol") or "").upper().strip()
                        dissent = dissent_map.get((channel_label, symbol))
                        if dissent is not None:
                            dissents.append((channel_label, symbol, dissent))
                            sug = apply_dissent_to_suggestion(sug, dissent)
                            if dissent.is_strong_counter:
                                sug["direction"] = "hold"
                        new_list.append(sug)
                    primary_output[key] = new_list

        # --- Persist --------------------------------------------------
        persisted: Optional[PersistResult] = None
        if persist:
            persisted = persist_signals(
                con,
                primary_output=primary_output,
                snapshot_universe=universe,
                generated_at=datetime.now(timezone.utc),
            )

        # Batch dissent: all entries share one LLMResponse. Dedupe by id()
        # so we don't count the same call cost N times.
        seen_responses: set[int] = set()
        dissent_cost = 0.0
        for _, _, d in dissents:
            rid = id(d.raw_response)
            if rid in seen_responses:
                continue
            seen_responses.add(rid)
            dissent_cost += d.raw_response.cost_estimate_usd
        total_cost = primary.cost_estimate_usd + dissent_cost
        log.info(
            "digest_run_complete",
            as_of=str(snapshot.as_of),
            primary_cost=round(primary.cost_estimate_usd, 4),
            dissent_count=len(dissents),
            total_cost_usd=round(total_cost, 4),
            inserted=persisted.total_inserted if persisted else 0,
        )

        return DigestRun(
            as_of=snapshot.as_of,
            snapshot=snapshot,
            primary_response=primary,
            primary_output=primary_output,
            dissents=dissents,
            final_output=primary_output,
            persisted=persisted,
            total_cost_usd=total_cost,
        )
    finally:
        if owned_con:
            con.close()
