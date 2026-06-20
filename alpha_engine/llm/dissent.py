"""Dissent layer: asks Claude to argue against a primary suggestion.

Why bother: confirmation bias is one of the most predictable causes of bad
trades. If the primary signal generator produced a thesis it was already
predisposed toward, asking the *same model* to articulate the strongest
counterargument forces it to look for what it missed. Counter is added to
the signal record (`counter_argument` column), and conviction can be
adjusted downward when the counter is compelling.

Implemented as a separate API call (lower max_tokens, smaller schema)
rather than asking the primary call to produce both. This keeps the
primary's reasoning focused and means we can skip dissent entirely if
cost matters for a particular run.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from alpha_engine.core.logging import get_logger
from alpha_engine.llm.client import LLMClient, LLMResponse
from alpha_engine.llm.prompts import (
    BATCH_DISSENT_OUTPUT_SCHEMA,
    BATCH_DISSENT_SYSTEM_PROMPT,
    BATCH_DISSENT_USER_TEMPLATE,
    DISSENT_OUTPUT_SCHEMA,
    DISSENT_SYSTEM_PROMPT,
    DISSENT_USER_TEMPLATE,
)

log = get_logger(__name__)


@dataclass
class DissentResult:
    counter_argument: str
    conviction_adjustment: int        # signed: typically -3 to +1
    is_strong_counter: bool
    raw_response: LLMResponse


def generate_dissent(
    *,
    client: LLMClient,
    channel: str,
    suggestion: dict[str, Any],
    snapshot_markdown: str,
    effort: str = "medium",          # lower effort — dissent is bounded
) -> DissentResult:
    """Generate a counterargument for a primary suggestion.

    `suggestion` is one of the dict objects from the primary call's
    channel_a_suggestions / channel_b_suggestions arrays.
    """
    user_message = DISSENT_USER_TEMPLATE.format(
        channel=channel,
        symbol=suggestion.get("symbol", "—"),
        direction=suggestion.get("direction", "—"),
        conviction=suggestion.get("conviction", "—"),
        rationale=suggestion.get("rationale", "—"),
        snapshot_markdown=snapshot_markdown,
    )

    response = client.call_structured(
        system_prompt=DISSENT_SYSTEM_PROMPT,
        user_message=user_message,
        output_schema=DISSENT_OUTPUT_SCHEMA,
        max_tokens=2000,
        effort=effort,
    )

    parsed = response.parsed
    log.info(
        "dissent_generated",
        channel=channel,
        symbol=suggestion.get("symbol"),
        adjustment=parsed.get("conviction_adjustment"),
        is_strong=parsed.get("is_strong_counter"),
        cost_usd=round(response.cost_estimate_usd, 4),
    )
    # Clamp adjustment to expected range since the schema can't enforce it
    adjustment = max(-3, min(1, int(parsed["conviction_adjustment"])))

    return DissentResult(
        counter_argument=parsed["counter_argument"],
        conviction_adjustment=adjustment,
        is_strong_counter=bool(parsed["is_strong_counter"]),
        raw_response=response,
    )


def apply_dissent_to_suggestion(
    suggestion: dict[str, Any], dissent: DissentResult
) -> dict[str, Any]:
    """Returns a copy of `suggestion` with conviction adjusted and the
    counter argument attached. If the counter is strong, conviction is
    floored at 0 and direction may be changed to 'hold' by the caller."""
    new = dict(suggestion)
    original = float(suggestion.get("conviction", 0))
    adjusted = max(0.0, min(10.0, original + dissent.conviction_adjustment))
    new["conviction"] = adjusted
    new["counter_argument"] = dissent.counter_argument
    return new


# ---------------------------------------------------------------------------
# Batch dissent — one API call covers many suggestions
# ---------------------------------------------------------------------------


def _format_suggestion_block(items: list[tuple[str, dict[str, Any]]]) -> str:
    """Format a list of (channel, suggestion) tuples as a numbered block
    for the user message."""
    lines: list[str] = []
    for i, (channel, sug) in enumerate(items, start=1):
        lines.append(
            f"{i}. CHANNEL: {channel}\n"
            f"   SYMBOL: {sug.get('symbol', '?')}\n"
            f"   DIRECTION: {sug.get('direction', '?')}\n"
            f"   CONVICTION: {sug.get('conviction', '?')}/10\n"
            f"   RATIONALE: {sug.get('rationale', '?')}"
        )
    return "\n\n".join(lines)


def generate_batch_dissent(
    *,
    client: LLMClient,
    suggestions: list[tuple[str, dict[str, Any]]],
    snapshot_markdown: str,
    model: str = "claude-haiku-4-5",
    max_tokens: int = 6000,
    effort: str = "medium",
) -> dict[tuple[str, str], DissentResult]:
    """Generate counter-arguments for many suggestions in a single API call.

    Args:
        client: LLMClient
        suggestions: list of (channel_label, suggestion_dict) tuples
        snapshot_markdown: the same snapshot the primary call used
        model: which model to use. Defaults to Haiku 4.5 — dissent is a
            bounded reasoning task and Haiku gives ~5x cost savings.
            Pass "claude-opus-4-7" if you want primary-quality reasoning.
        max_tokens: total output budget across all dissents.
        effort: ignored on Haiku (the client handles the gating).

    Returns:
        dict mapping (channel, symbol) -> DissentResult. Missing keys mean
        the model didn't return a dissent for that suggestion.
    """
    if not suggestions:
        return {}

    user_message = BATCH_DISSENT_USER_TEMPLATE.format(
        snapshot_markdown=snapshot_markdown,
        n=len(suggestions),
        suggestion_block=_format_suggestion_block(suggestions),
    )

    response = client.call_structured(
        system_prompt=BATCH_DISSENT_SYSTEM_PROMPT,
        user_message=user_message,
        output_schema=BATCH_DISSENT_OUTPUT_SCHEMA,
        model=model,
        max_tokens=max_tokens,
        effort=effort,
    )

    parsed = response.parsed
    out: dict[tuple[str, str], DissentResult] = {}
    for entry in parsed.get("dissents", []):
        symbol = (entry.get("symbol") or "").upper().strip()
        channel = (entry.get("channel") or "").strip()
        if not symbol or not channel:
            log.warning("batch_dissent_skipped_no_key", entry=entry)
            continue
        adjustment = max(-3, min(1, int(entry.get("conviction_adjustment", 0))))
        out[(channel, symbol)] = DissentResult(
            counter_argument=entry.get("counter_argument", ""),
            conviction_adjustment=adjustment,
            is_strong_counter=bool(entry.get("is_strong_counter", False)),
            raw_response=response,  # all entries share the single response
        )

    log.info(
        "batch_dissent_complete",
        model=model,
        requested=len(suggestions),
        returned=len(out),
        cost_usd=round(response.cost_estimate_usd, 4),
    )
    return out
