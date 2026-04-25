"""Per-model USD price map (USD per 1M tokens).

Update this map when OpenRouter pricing changes. The audit log
records cost in USD; the daily budget is enforced against this.

Prices are tuples ``(prompt_usd_per_1m, completion_usd_per_1m)``.
Unknown models fall back to ``DEFAULT``.
"""
from __future__ import annotations

PRICES: dict[str, tuple[float, float]] = {
    # xAI Grok 4.x family (OpenRouter retail). Verify before relying on these
    # for budget enforcement; pricing drifts and the values below are a
    # conservative best-guess based on prior generations.
    "x-ai/grok-4.20":              (3.0, 15.0),
    "x-ai/grok-4.20-multi-agent":  (5.0, 25.0),
    "x-ai/grok-4.1-fast":          (0.20, 0.50),
    "x-ai/grok-4.1-fast:online":   (0.20, 0.50),
    # Legacy aliases kept so old configs still load.
    "x-ai/grok-4":         (3.0, 15.0),
    "x-ai/grok-4:online":  (3.0, 15.0),
    "x-ai/grok-4-fast":    (0.20, 0.50),
    # Common fallbacks for offline tests.
    "openai/gpt-4o-mini": (0.15, 0.60),
    "anthropic/claude-3.5-haiku": (0.80, 4.0),
    # Mock model used in offline tests.
    "mock/mock-1": (0.0, 0.0),
}

DEFAULT: tuple[float, float] = (3.0, 15.0)


def cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Return USD cost for one call given token usage."""
    p, c = PRICES.get(model, DEFAULT)
    return prompt_tokens * p / 1_000_000.0 + completion_tokens * c / 1_000_000.0
