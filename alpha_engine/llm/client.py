"""Thin wrapper around the Anthropic SDK with our defaults baked in.

Defaults (per current best practice for intelligence-sensitive work on
Opus 4.7):
  - Model: claude-opus-4-7
  - thinking: {"type": "adaptive"}  (REQUIRED on 4.7 — budget_tokens 400s)
  - output_config.effort: "high"
  - Prompt caching: cache_control on the system prompt (it's stable across
    runs; cache write is 1.25x, reads are 0.1x, so any 2+ calls within 5
    min net out cheaper)

Note: temperature / top_p / top_k are REMOVED on Opus 4.7 and will return
400 if sent. We don't set them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import anthropic

from alpha_engine.core.config import get_settings
from alpha_engine.core.logging import get_logger

log = get_logger(__name__)


DEFAULT_MODEL = "claude-opus-4-7"
DEFAULT_MAX_TOKENS = 8000


# Per-model pricing per 1M tokens (cached: 2026-04-29). Cache write = 1.25x
# input price; cache read = 0.1x input price.
_MODEL_PRICING: dict[str, tuple[float, float]] = {
    # model_id: (input_per_1M, output_per_1M)
    "claude-opus-4-7":   (5.00, 25.00),
    "claude-opus-4-6":   (5.00, 25.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-haiku-4-5":  (1.00,  5.00),
}


@dataclass
class LLMResponse:
    """What we hand back to callers — the parsed output plus metadata."""

    parsed: dict[str, Any]                # validated JSON output
    raw_text: str                         # full text content for inspection
    model: str
    stop_reason: str
    input_tokens: int
    output_tokens: int
    cache_creation_tokens: int            # cached this turn (1.25x cost)
    cache_read_tokens: int                # served from cache (0.1x cost)
    cost_estimate_usd: float

    @property
    def cache_hit_ratio(self) -> float:
        """Fraction of input that was served from cache."""
        total = self.input_tokens + self.cache_read_tokens + self.cache_creation_tokens
        return self.cache_read_tokens / total if total else 0.0


def _estimate_cost(
    model: str,
    input_tok: int,
    output_tok: int,
    cache_creation: int,
    cache_read: int,
) -> float:
    """Estimate USD cost using per-model pricing. Unknown models fall back
    to Opus 4.7 rates (conservative)."""
    in_price, out_price = _MODEL_PRICING.get(model, _MODEL_PRICING[DEFAULT_MODEL])
    return (
        input_tok * in_price
        + output_tok * out_price
        + cache_creation * in_price * 1.25
        + cache_read * in_price * 0.10
    ) / 1_000_000.0


class LLMClient:
    """Wrapper around anthropic.Anthropic with our defaults."""

    def __init__(self, api_key: Optional[str] = None) -> None:
        settings = get_settings()
        key = api_key or settings.anthropic_api_key
        if not key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY not set. Get a key at console.anthropic.com "
                "and add it to .env"
            )
        self._client = anthropic.Anthropic(api_key=key)

    def call_structured(
        self,
        *,
        system_prompt: str,
        user_message: str,
        output_schema: dict[str, Any],
        model: str = DEFAULT_MODEL,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        effort: str = "high",
    ) -> LLMResponse:
        """Make a call requesting structured JSON output validated against schema.

        The system prompt is marked cacheable. Within a 5-minute window, repeated
        calls with the same system prompt cost ~0.1x for the system tokens
        (subject to the 4096-token minimum prefix on Opus 4.7).

        Model-specific feature gating (per Anthropic docs):
          - Haiku 4.5 does NOT support `output_config.effort` or adaptive thinking
          - Opus 4.7 / 4.6 / Sonnet 4.6 support both
        """
        is_haiku = model.startswith("claude-haiku")
        log.info(
            "llm_call_start",
            model=model,
            effort=("n/a" if is_haiku else effort),
            system_len=len(system_prompt),
            user_len=len(user_message),
        )

        output_config: dict[str, Any] = {
            "format": {
                "type": "json_schema",
                "schema": output_schema,
            },
        }
        if not is_haiku:
            output_config["effort"] = effort

        request_kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "output_config": output_config,
            "system": [
                {
                    "type": "text",
                    "text": system_prompt,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            "messages": [{"role": "user", "content": user_message}],
        }
        if not is_haiku:
            request_kwargs["thinking"] = {"type": "adaptive"}

        try:
            response = self._client.messages.create(**request_kwargs)
        except anthropic.BadRequestError as e:
            log.error("llm_bad_request", error=str(e), error_type=e.type)
            raise
        except anthropic.RateLimitError as e:
            log.error("llm_rate_limited", error=str(e))
            raise
        except anthropic.APIError as e:
            log.error("llm_api_error", error=str(e), status=getattr(e, "status", None))
            raise

        # Extract text blocks (skip thinking blocks if any made it through)
        raw_text = "".join(
            b.text for b in response.content if getattr(b, "type", None) == "text"
        )

        # Parse the JSON from the model response
        # output_config.format=json_schema constrains the model but we still
        # need to parse the JSON ourselves from raw_text
        import json
        try:
            parsed = json.loads(raw_text)
        except json.JSONDecodeError as e:
            log.error("llm_invalid_json", error=str(e), raw=raw_text[:500])
            raise RuntimeError(
                f"LLM returned invalid JSON despite schema constraint: {e}"
            )

        usage = response.usage
        cache_creation = getattr(usage, "cache_creation_input_tokens", 0) or 0
        cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0

        cost = _estimate_cost(
            model, usage.input_tokens, usage.output_tokens, cache_creation, cache_read
        )

        log.info(
            "llm_call_complete",
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cache_creation=cache_creation,
            cache_read=cache_read,
            cost_usd=round(cost, 4),
            stop_reason=response.stop_reason,
        )

        return LLMResponse(
            parsed=parsed,
            raw_text=raw_text,
            model=response.model,
            stop_reason=response.stop_reason or "unknown",
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cache_creation_tokens=cache_creation,
            cache_read_tokens=cache_read,
            cost_estimate_usd=cost,
        )
