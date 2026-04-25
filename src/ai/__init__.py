"""AI layer: OpenRouter client, prompts, schemas, budget, audit, mock."""
from .audit import AuditWriter
from .budget import BudgetTracker
from .client import AIClient
from .mock import MockRouter
from .prices import cost_usd
from .prompts import (
    PROMPT_VERSION,
    render_deep_prompt,
    render_review_prompt,
    render_watchlist_prompt,
)
from .schemas import (
    Catalyst,
    DeepSignal,
    ReviewResponse,
    SocialPulse,
    WatchlistResponse,
    WatchlistSelection,
)

__all__ = [
    "AIClient",
    "AuditWriter",
    "BudgetTracker",
    "Catalyst",
    "DeepSignal",
    "MockRouter",
    "PROMPT_VERSION",
    "ReviewResponse",
    "SocialPulse",
    "WatchlistResponse",
    "WatchlistSelection",
    "cost_usd",
    "render_deep_prompt",
    "render_review_prompt",
    "render_watchlist_prompt",
]

