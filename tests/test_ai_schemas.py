"""Tests for ai/schemas.py.

Schema-only validation. No network. No filesystem.
"""
from __future__ import annotations

import pytest

from ai.schemas import (
    DeepSignal,
    WatchlistResponse,
    WatchlistSelection,
)
from ai.prompts import PROMPT_VERSION


def _make_signal(**over) -> DeepSignal:
    base = dict(
        prompt_version=PROMPT_VERSION,
        symbol="BTCUSDT",
        action="long",
        entry=100.0,
        stop_loss=98.0,
        take_profit_1=104.5,
        take_profit_2=108.0,
        time_horizon_bars=24,
        confidence=0.7,
        reasoning=["a", "b", "c"],
        rationale="r",
        invalidation="below sl",
    )
    base.update(over)
    return DeepSignal(**base)


class TestWatchlist:
    def test_max_five_selections(self):
        sels = [
            WatchlistSelection(symbol=f"X{i}USDT", side="long",
                               expected_move_pct=5.0, confidence=0.5,
                               thesis="t")
            for i in range(6)
        ]
        with pytest.raises(Exception):
            WatchlistResponse(prompt_version=PROMPT_VERSION, as_of="t",
                              market_regime="chop", selections=sels)

    def test_signed_expected_move(self):
        s = WatchlistSelection(symbol="X", side="short",
                               expected_move_pct=-12.5, confidence=0.6,
                               thesis="t")
        assert s.expected_move_pct == -12.5


class TestDeepConsistency:
    def test_clean_long_passes(self):
        sig = _make_signal()
        assert sig.check_consistency() == []

    def test_long_with_inverted_stop(self):
        sig = _make_signal(stop_loss=101.0)  # above entry
        warns = sig.check_consistency()
        assert any("sl < entry" in w for w in warns)

    def test_low_rr_warns(self):
        # entry=100, sl=99 (risk=1), tp1=101.5 (reward=1.5) -> R:R=1.5 < 2
        sig = _make_signal(stop_loss=99.0, take_profit_1=101.5, take_profit_2=103.0)
        warns = sig.check_consistency()
        assert any("R:R" in w for w in warns)

    def test_short_orientation(self):
        sig = _make_signal(action="short", entry=100.0, stop_loss=102.0,
                           take_profit_1=95.0, take_profit_2=92.0)
        assert sig.check_consistency() == []

    def test_flat_skips(self):
        sig = DeepSignal(prompt_version=PROMPT_VERSION, symbol="X",
                         action="flat", confidence=0.0)
        assert sig.check_consistency() == []

    def test_mark_drift_warns(self):
        sig = _make_signal()
        warns = sig.check_consistency(mark_price=110.0)  # ~10% drift
        assert any("drift" in w for w in warns)
