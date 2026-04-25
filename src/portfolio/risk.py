"""Risk sizing, exposure caps, and the loser-streak circuit breaker.

Pure math: no I/O, no broker state mutation. The broker calls
:func:`size_intent` to convert an :class:`Intent` plus current
portfolio context into an absolute ``qty`` (or a rejection reason).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from portfolio.intents import Intent


@dataclass(frozen=True)
class InstrumentSpec:
    """Minimum exchange clamp for sizing."""

    qty_step: float = 0.001
    min_order_qty: float = 0.0
    min_notional_usd: float = 5.0
    tick_size: float = 0.0001


@dataclass(frozen=True)
class RiskCaps:
    """Hard exposure caps + per-trade risk."""

    per_trade_risk_pct: float = 0.01
    max_concurrent_positions: int = 3
    max_symbol_exposure_pct: float = 0.40
    max_aggregate_exposure_pct: float = 1.50

    @classmethod
    def from_config(cls, cfg) -> "RiskCaps":
        return cls(
            per_trade_risk_pct=float(getattr(cfg, "per_trade_risk_pct", 0.01)),
            max_concurrent_positions=int(getattr(cfg, "max_concurrent_positions", 3)),
            max_symbol_exposure_pct=float(getattr(cfg, "max_symbol_exposure_pct", 0.40)),
            max_aggregate_exposure_pct=float(getattr(cfg, "max_aggregate_exposure_pct", 1.50)),
        )


@dataclass(frozen=True)
class SizingDecision:
    """Outcome of :func:`size_intent`."""

    qty: float
    notional_usd: float
    risk_usd: float
    reason: str = ""  # rejection reason; empty means accepted

    @property
    def accepted(self) -> bool:
        return self.qty > 0 and self.reason == ""


def _floor_to_step(value: float, step: float) -> float:
    if step <= 0:
        return float(value)
    return math.floor(value / step) * step


def size_intent(
    *,
    intent: Intent,
    equity_usd: float,
    open_positions: int,
    symbol_exposure_usd: float,
    aggregate_exposure_usd: float,
    instrument: InstrumentSpec,
    caps: RiskCaps,
    risk_multiplier: float = 1.0,
) -> SizingDecision:
    """Return a sizing decision per ``docs/04_EXECUTION.md`` section 3.1.

    Rejection reasons (string code):

    - ``concurrent_cap``: already at ``max_concurrent_positions``.
    - ``zero_sl_distance``: entry == stop_loss.
    - ``notional_floor``: floored qty * entry below ``min_notional``.
    - ``min_qty``: floored qty below ``min_order_qty``.
    - ``symbol_exposure_cap``: would breach the per-symbol exposure cap.
    - ``aggregate_exposure_cap``: would breach the book-wide cap.
    """
    if open_positions >= caps.max_concurrent_positions:
        return SizingDecision(qty=0.0, notional_usd=0.0, risk_usd=0.0,
                              reason="concurrent_cap")

    sl_distance = abs(intent.entry - intent.stop_loss)
    if sl_distance <= 0:
        return SizingDecision(qty=0.0, notional_usd=0.0, risk_usd=0.0,
                              reason="zero_sl_distance")

    risk_pct = max(0.0, caps.per_trade_risk_pct * float(risk_multiplier))
    risk_usd = max(0.0, equity_usd) * risk_pct
    raw_qty = risk_usd / sl_distance
    qty = _floor_to_step(raw_qty, instrument.qty_step)

    if qty < instrument.min_order_qty:
        return SizingDecision(qty=0.0, notional_usd=0.0, risk_usd=risk_usd,
                              reason="min_qty")

    notional = qty * intent.entry
    if notional < instrument.min_notional_usd:
        return SizingDecision(qty=0.0, notional_usd=notional, risk_usd=risk_usd,
                              reason="notional_floor")

    sym_cap = caps.max_symbol_exposure_pct * equity_usd
    if symbol_exposure_usd + notional > sym_cap:
        return SizingDecision(qty=0.0, notional_usd=notional, risk_usd=risk_usd,
                              reason="symbol_exposure_cap")

    agg_cap = caps.max_aggregate_exposure_pct * equity_usd
    if aggregate_exposure_usd + notional > agg_cap:
        return SizingDecision(qty=0.0, notional_usd=notional, risk_usd=risk_usd,
                              reason="aggregate_exposure_cap")

    return SizingDecision(qty=qty, notional_usd=notional, risk_usd=risk_usd)


def circuit_breaker_multiplier(loser_streak: int, *, threshold: int = 2) -> float:
    """Halve risk after ``threshold`` consecutive losers on the same UTC day."""
    return 0.5 if loser_streak >= threshold else 1.0


__all__ = [
    "InstrumentSpec",
    "RiskCaps",
    "SizingDecision",
    "circuit_breaker_multiplier",
    "size_intent",
]
