"""Pure trigger-gate logic.

`detect_trigger` is a pure function: given a fresh 15m bar (as a row of
features) plus the symbol's prior cooldown state, it returns a
``TriggerDecision`` with one of seven branches:

- ``no_bar``         -- the row is empty / missing required fields.
- ``no_flag``        -- no configured trigger flag fired on this bar.
- ``dup_bar``        -- this exact (symbol, bar_ts) was already evaluated.
- ``cooldown_active`` -- a flag fired but cooldown is still in effect
                          and price has not bypassed it.
- ``bypass_move``    -- cooldown still active by bar count, but price
                          moved enough to override.
- ``post_cooldown``  -- bar count cooldown elapsed naturally.
- ``fresh``          -- first time we ever see this symbol fire.

The function does NOT mutate state. The scanner owns the
``CooldownStore`` and writes back after a positive decision.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import pandas as pd

from core.config import Config


# Decision codes (string for jsonl + log readability).
DEC_FRESH = "fresh"
DEC_POST_COOLDOWN = "post_cooldown"
DEC_BYPASS_MOVE = "bypass_move"
DEC_COOLDOWN_ACTIVE = "cooldown_active"
DEC_DUP_BAR = "dup_bar"
DEC_NO_FLAG = "no_flag"
DEC_NO_BAR = "no_bar"

POSITIVE_DECISIONS = frozenset({DEC_FRESH, DEC_POST_COOLDOWN, DEC_BYPASS_MOVE})


@dataclass(frozen=True)
class CooldownState:
    """Per-symbol cooldown record. ``None`` means never fired."""

    last_bar_ts: pd.Timestamp | None = None
    last_close: float | None = None
    bars_since: int = 10**9  # effectively "infinity" for fresh symbols


@dataclass(frozen=True)
class TriggerDecision:
    """Outcome of evaluating one bar against the gate."""

    symbol: str
    bar_ts: pd.Timestamp | None
    decision: str
    flag: str | None = None  # which trigger flag fired (if any)
    close: float | None = None
    atr_pct: float | None = None
    move_pct: float | None = None  # |close_now - last_close| / last_close
    threshold_pct: float | None = None  # cooldown bypass threshold
    bars_elapsed: int | None = None
    reason: str = ""

    @property
    def fired(self) -> bool:
        return self.decision in POSITIVE_DECISIONS


def _coerce_ts(value: Any) -> pd.Timestamp | None:
    if value is None:
        return None
    try:
        ts = pd.Timestamp(value)
    except (TypeError, ValueError):
        return None
    if ts is pd.NaT:
        return None
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts


def detect_trigger(
    *,
    symbol: str,
    bar: Mapping[str, Any] | pd.Series | None,
    state: CooldownState,
    cfg: Config,
) -> TriggerDecision:
    """Evaluate one freshly-closed 15m bar against the trigger gate.

    Required keys in ``bar``: ``timestamp`` (or ``bar_ts``), ``close``,
    ``atr_14_pct``, plus the trigger flags listed in
    ``cfg.trigger_flags``. Missing fields => ``no_bar``.
    """
    if bar is None:
        return TriggerDecision(symbol=symbol, bar_ts=None,
                               decision=DEC_NO_BAR, reason="bar is None")
    if isinstance(bar, pd.Series):
        bar = bar.to_dict()

    bar_ts = _coerce_ts(bar.get("timestamp") or bar.get("bar_ts"))
    if bar_ts is None:
        return TriggerDecision(symbol=symbol, bar_ts=None,
                               decision=DEC_NO_BAR, reason="missing/invalid timestamp")

    close = bar.get("close")
    try:
        close = float(close) if close is not None else None
    except (TypeError, ValueError):
        close = None
    if close is None or close <= 0:
        return TriggerDecision(symbol=symbol, bar_ts=bar_ts,
                               decision=DEC_NO_BAR, reason="missing/invalid close")

    # Dedup: same bar already evaluated.
    if state.last_bar_ts is not None and bar_ts == state.last_bar_ts:
        return TriggerDecision(symbol=symbol, bar_ts=bar_ts, close=close,
                               decision=DEC_DUP_BAR, reason="bar_ts already evaluated")

    # Which configured flag fired?
    fired_flag: str | None = None
    for fname in cfg.trigger_flags:
        v = bar.get(fname)
        try:
            if v is not None and float(v) >= 1.0:
                fired_flag = fname
                break
        except (TypeError, ValueError):
            continue
    if fired_flag is None:
        return TriggerDecision(symbol=symbol, bar_ts=bar_ts, close=close,
                               decision=DEC_NO_FLAG, reason="no trigger flag set")

    # ATR (in %). Required for bypass calculation; 0 if missing.
    atr_pct_raw = bar.get("atr_14_pct")
    try:
        atr_pct = float(atr_pct_raw) if atr_pct_raw is not None else 0.0
    except (TypeError, ValueError):
        atr_pct = 0.0

    # Fresh symbol: never fired before.
    if state.last_bar_ts is None or state.last_close is None:
        return TriggerDecision(
            symbol=symbol, bar_ts=bar_ts, close=close, atr_pct=atr_pct,
            decision=DEC_FRESH, flag=fired_flag,
            bars_elapsed=None, reason="no prior trigger for symbol",
        )

    bars_elapsed = max(0, int(state.bars_since))
    cooldown = max(0, int(cfg.prompt_cooldown_candles))

    if bars_elapsed >= cooldown:
        return TriggerDecision(
            symbol=symbol, bar_ts=bar_ts, close=close, atr_pct=atr_pct,
            decision=DEC_POST_COOLDOWN, flag=fired_flag,
            bars_elapsed=bars_elapsed,
            reason=f"bars_elapsed={bars_elapsed} >= cooldown={cooldown}",
        )

    # Cooldown still active by bar count -- check price-move bypass.
    move_pct = abs(close - state.last_close) / state.last_close
    threshold_pct = max(
        cfg.cooldown_bypass_atr_mult * (atr_pct / 100.0),
        cfg.cooldown_bypass_floor_pct,
    )
    if move_pct >= threshold_pct:
        return TriggerDecision(
            symbol=symbol, bar_ts=bar_ts, close=close, atr_pct=atr_pct,
            decision=DEC_BYPASS_MOVE, flag=fired_flag,
            move_pct=move_pct, threshold_pct=threshold_pct,
            bars_elapsed=bars_elapsed,
            reason=f"move {move_pct:.4f} >= threshold {threshold_pct:.4f}",
        )

    return TriggerDecision(
        symbol=symbol, bar_ts=bar_ts, close=close, atr_pct=atr_pct,
        decision=DEC_COOLDOWN_ACTIVE, flag=fired_flag,
        move_pct=move_pct, threshold_pct=threshold_pct,
        bars_elapsed=bars_elapsed,
        reason=f"bars_elapsed={bars_elapsed} < cooldown={cooldown} "
               f"and move {move_pct:.4f} < threshold {threshold_pct:.4f}",
    )
