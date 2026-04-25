"""Pydantic schemas for the LLM responses (v3).

Schemas:
- ``WatchlistResponse``  (Prompt A)
- ``DeepSignal``         (Prompt B)
- ``ReviewResponse``     (Prompt C - position review, restored Phase 12)

Every schema echoes ``prompt_version`` so audit logs can join responses
to the prompt template that produced them.

The ``check_consistency`` helper on ``DeepSignal`` returns a list of
warnings. Two or more warnings should force ``action="flat"`` upstream
(see docs/03_AI_LAYER.md section 4).
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


_CATALYST_TAGS = (
    "listing", "delisting", "news", "macro", "onchain",
    "derivatives", "social", "technical", "hack", "partnership",
    "unlock", "airdrop", "governance",
)


class Catalyst(BaseModel):
    """A verifiable, time-stamped catalyst. ``source_url`` is required."""
    model_config = ConfigDict(extra="ignore")

    tag: Literal[
        "listing", "delisting", "news", "macro", "onchain",
        "derivatives", "social", "technical", "hack", "partnership",
        "unlock", "airdrop", "governance",
    ]
    summary: str = Field(..., max_length=240)
    source_url: str = Field(..., min_length=8, max_length=512)
    published_at: str | None = None      # ISO-8601 UTC
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class SocialPulse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    sentiment: Literal["bullish", "bearish", "mixed", "neutral"] = "neutral"
    sentiment_score: float = Field(default=0.0, ge=-1.0, le=1.0)
    attention_delta: Literal["spiking", "elevated", "normal", "fading"] = "normal"
    shill_risk: Literal["low", "medium", "high"] = "low"
    notable_handles: list[str] = Field(default_factory=list)
    notes: str = Field(default="", max_length=240)


# ---- Watchlist (Prompt A) -------------------------------------------
class WatchlistSelection(BaseModel):
    model_config = ConfigDict(extra="ignore")

    symbol: str
    side: Literal["long", "short"]
    expected_move_pct: float = Field(..., ge=-100.0, le=500.0)
    confidence: float = Field(..., ge=0.0, le=1.0)
    thesis: str = Field(..., max_length=280)
    key_confluences: list[str] = Field(default_factory=list)
    catalysts: list[Catalyst] = Field(default_factory=list)
    social_pulse: SocialPulse | None = None
    risks: list[str] = Field(default_factory=list)


class WatchlistResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    prompt_version: str
    as_of: str
    market_regime: Literal[
        "risk-on", "risk-off", "memecoin-rotation",
        "altcoin-season", "chop", "unknown",
    ]
    regime_evidence: list[str] = Field(default_factory=list)
    reasoning: list[str] = Field(default_factory=list)
    selections: list[WatchlistSelection] = Field(default_factory=list)
    discarded_pumps: list[str] = Field(default_factory=list)
    notes: str = ""

    @field_validator("selections")
    @classmethod
    def _max_five(cls, v: list[WatchlistSelection]) -> list[WatchlistSelection]:
        if len(v) > 5:
            raise ValueError("at most 5 selections allowed")
        return v


# ---- Deep signal (Prompt B) -----------------------------------------
class DeepSignal(BaseModel):
    model_config = ConfigDict(extra="ignore")

    prompt_version: str
    symbol: str
    action: Literal["long", "short", "flat"]
    entry: float | None = None
    entry_trigger: float | None = None
    activation_kind: Literal["touch", "close_above", "close_below", "breakout"] | None = None
    stop_loss: float | None = None
    take_profit_1: float | None = None
    take_profit_2: float | None = None
    time_horizon_bars: int | None = Field(default=None, ge=6, le=120)
    confidence: float = Field(..., ge=0.0, le=1.0)
    expected_move_pct: float | None = None
    reasoning: list[str] = Field(default_factory=list)
    key_confluences: list[str] = Field(default_factory=list)
    catalysts: list[Catalyst] = Field(default_factory=list)
    rationale: str = ""
    invalidation: str = ""

    def check_consistency(self, *, mark_price: float | None = None) -> list[str]:
        """Return a list of warning strings. Empty list = clean."""
        warnings: list[str] = []
        if self.action == "flat":
            return warnings

        if self.entry is None or self.stop_loss is None or self.take_profit_1 is None:
            warnings.append("entry/sl/tp1 must all be set for non-flat action")
            return warnings

        e, sl, tp1 = self.entry, self.stop_loss, self.take_profit_1
        if self.action == "long":
            if not (sl < e < tp1):
                warnings.append("long requires sl < entry < tp1")
            if self.take_profit_2 is not None and not (tp1 <= self.take_profit_2):
                warnings.append("tp2 must be >= tp1 on long")
        else:  # short
            if not (tp1 < e < sl):
                warnings.append("short requires tp1 < entry < sl")
            if self.take_profit_2 is not None and not (tp1 >= self.take_profit_2):
                warnings.append("tp2 must be <= tp1 on short")

        risk = abs(e - sl)
        reward = abs(tp1 - e)
        if risk <= 0:
            warnings.append("zero risk distance")
        elif reward / risk < 2.0:
            warnings.append(f"R:R on TP1 below 2.0 ({reward / risk:.2f})")

        if self.time_horizon_bars is not None and not (8 <= self.time_horizon_bars <= 96):
            warnings.append("time_horizon_bars outside [8,96]")

        if mark_price is not None and mark_price > 0:
            drift = abs(e - mark_price) / mark_price
            if drift > 0.01:
                warnings.append(f"entry drifts {drift:.1%} from mark price (>1%)")

        return warnings


# ---- Review (Prompt C - restored Phase 12) --------------------------
class ReviewResponse(BaseModel):
    """LLM verdict on an open position after a hook fires."""
    model_config = ConfigDict(extra="ignore")

    prompt_version: str
    symbol: str
    action: Literal["hold", "tighten_stop", "scale_out", "stop", "flip"]
    new_stop_loss: float | None = None
    confidence: float = Field(..., ge=0.0, le=1.0)
    rationale: str = Field(default="", max_length=400)
    reasoning: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _stop_required(self) -> "ReviewResponse":
        # ``tighten_stop`` requires the new stop. ``stop`` and ``flip`` are
        # actioned by the loop without needing a price. ``scale_out`` and
        # ``hold`` do not touch the stop.
        if self.action == "tighten_stop" and self.new_stop_loss is None:
            raise ValueError("tighten_stop requires new_stop_loss")
        return self

