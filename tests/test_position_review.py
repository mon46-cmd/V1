"""Phase 12 - position-review hooks (offline).

Covers:
- TP1 hook fires exactly once per position.
- Drawdown hook fires on adverse move past the configured threshold.
- Regime-flip hook fires once on transition.
- Funding-approach hook fires inside the configured window.
- Per-position 1/hour cap throttles a second hook in the same hour.
- Budget exhaustion forces the AIClient synthetic-hold fallback and
  keeps audited "review" calls observable while NOT charging budget.
- Apply layer: ``tighten_stop`` updates the position's SL when valid;
  ``stop`` closes the remaining qty; invalid SLs are rejected.
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import pytest

from ai import AuditWriter, BudgetTracker
from ai.client import AIClient
from ai.schemas import DeepSignal, ReviewResponse
from core.config import load_config
from loops.exec import ExecConfig, ExecLoop, ReviewConfig
from loops.triggers import TriggerDecision
from portfolio.broker import Bar
from portfolio.intents import Tick


T0 = pd.Timestamp("2026-04-25T12:00:00Z")
BAR = pd.Timedelta(minutes=15)


# ---------------------------------------------------------------------
@dataclass
class StubAI:
    """Records every chat_deep / chat_review call for assertions."""

    deep_signal: DeepSignal
    review_action: str = "tighten_stop"
    review_calls: list[dict] = field(default_factory=list)
    deep_calls: list[dict] = field(default_factory=list)

    async def chat_deep(self, symbol: str, payload: dict) -> DeepSignal:
        self.deep_calls.append({"symbol": symbol, "payload": payload})
        return self.deep_signal.model_copy(update={"symbol": symbol})

    async def chat_review(self, symbol: str, payload: dict) -> ReviewResponse:
        self.review_calls.append({"symbol": symbol, "payload": payload})
        pos = payload["position"]
        new_sl: float | None = None
        if self.review_action == "tighten_stop":
            entry = float(pos["entry"])
            sl = float(pos["stop_loss"])
            if pos["side"] == "long":
                # Halve the SL distance.
                new_sl = sl + 0.5 * (entry - sl)
            else:
                new_sl = sl - 0.5 * (sl - entry)
        return ReviewResponse(
            prompt_version="v3.1",
            symbol=symbol,
            action=self.review_action,  # type: ignore[arg-type]
            new_stop_loss=new_sl,
            confidence=0.6,
            rationale="stub",
        )


def _signal_long(*, entry: float = 100.0, sl: float = 80.0,
                 tp1: float = 110.0) -> DeepSignal:
    return DeepSignal(
        prompt_version="v3.1",
        symbol="BTCUSDT",
        action="long",
        entry=entry,
        entry_trigger=entry,
        activation_kind="touch",
        stop_loss=sl,
        take_profit_1=tp1,
        take_profit_2=tp1 + (tp1 - entry),
        time_horizon_bars=8,
        confidence=0.7,
    )


def _trigger(close: float = 100.0) -> tuple[dict, TriggerDecision]:
    bar = {"symbol": "BTCUSDT", "timestamp": T0, "close": close, "atr_pct": 0.01}
    decision = TriggerDecision(
        symbol="BTCUSDT", bar_ts=T0, decision="fresh",
        flag="flag_volume_climax", close=close, atr_pct=0.01,
        move_pct=0.0, threshold_pct=0.0, bars_elapsed=999, reason="",
    )
    return bar, decision


def _build_loop(*, ai, review_cfg: ReviewConfig | None = None) -> ExecLoop:
    cfg = load_config()
    return ExecLoop.build(
        cfg=cfg,
        feature_cfg=None,
        ai=ai,
        run_id="reviewrun01",
        exec_cfg=ExecConfig(starting_equity_usd=10_000.0,
                            save_state_every_fill=True),
        review_cfg=review_cfg or ReviewConfig(),
    )


def _open_position(loop: ExecLoop, ai: StubAI, *, sl_dist: float = 20.0) -> str:
    """Activate one long position via the trigger->intent->fill path."""
    entry = 100.0
    ai.deep_signal = _signal_long(entry=entry, sl=entry - sl_dist,
                                   tp1=entry + 10.0)
    bar, decision = _trigger(close=entry)
    intent = asyncio.run(loop.on_trigger("BTCUSDT", bar, decision))
    assert intent is not None
    # Activate via a touch tick at entry.
    tick = Tick(ts=T0 + pd.Timedelta(seconds=10), price=entry, size=1.0,
                symbol="BTCUSDT")
    events = loop.watcher.process_tick(tick)
    assert events
    asyncio.run(loop.emit_event(events[0]))
    pos_ids = list(loop.broker.positions.keys())
    assert len(pos_ids) == 1
    return pos_ids[0]


# ---------------------------------------------------------------------
# TP1 hook
# ---------------------------------------------------------------------
def test_tp1_hook_fires_once(tmp_data_root):
    ai = StubAI(deep_signal=_signal_long(), review_action="hold")
    loop = _build_loop(ai=ai)
    pid = _open_position(loop, ai, sl_dist=20.0)

    # Bar that touches TP1 (110) -> tp1 fill.
    bar = Bar(symbol="BTCUSDT", ts=T0 + BAR,
              open=100.0, high=112.0, low=99.0, close=109.0)
    asyncio.run(loop.on_bar_async(bar))

    assert len(ai.review_calls) == 1
    assert ai.review_calls[0]["payload"]["trigger_reason"] == "tp1"
    # Same TP1 bar replayed: position no longer hits TP1 again.
    bar2 = Bar(symbol="BTCUSDT", ts=T0 + 2 * BAR,
               open=109.0, high=109.5, low=108.0, close=108.5)
    asyncio.run(loop.on_bar_async(bar2))
    # No new review call.
    assert len(ai.review_calls) == 1


# ---------------------------------------------------------------------
# Drawdown hook
# ---------------------------------------------------------------------
def test_drawdown_hook_fires_at_threshold(tmp_data_root):
    ai = StubAI(deep_signal=_signal_long(), review_action="hold")
    loop = _build_loop(ai=ai, review_cfg=ReviewConfig(
        drawdown_pct_of_sl_distance=0.6,
        min_review_interval_sec=3600.0,
    ))
    _open_position(loop, ai, sl_dist=20.0)

    # Mark drifts to 87 (entry=100, sl=80 -> sl_dist=20, adverse=13 = 65%).
    bar = Bar(symbol="BTCUSDT", ts=T0 + BAR,
              open=99.0, high=99.0, low=86.0, close=87.0)
    asyncio.run(loop.on_bar_async(bar))

    assert len(ai.review_calls) == 1
    assert ai.review_calls[0]["payload"]["trigger_reason"] == "drawdown"


def test_drawdown_below_threshold_does_not_fire(tmp_data_root):
    ai = StubAI(deep_signal=_signal_long(), review_action="hold")
    loop = _build_loop(ai=ai, review_cfg=ReviewConfig(
        drawdown_pct_of_sl_distance=0.6))
    _open_position(loop, ai, sl_dist=20.0)

    # Adverse 5 of 20 = 25% -> below threshold.
    bar = Bar(symbol="BTCUSDT", ts=T0 + BAR,
              open=99.0, high=99.0, low=94.0, close=95.0)
    asyncio.run(loop.on_bar_async(bar))
    assert ai.review_calls == []


# ---------------------------------------------------------------------
# 1/hour throttle
# ---------------------------------------------------------------------
def test_one_review_per_hour_per_position(tmp_data_root):
    ai = StubAI(deep_signal=_signal_long(), review_action="hold")
    loop = _build_loop(ai=ai, review_cfg=ReviewConfig(
        drawdown_pct_of_sl_distance=0.6,
        min_review_interval_sec=3600.0,
    ))
    _open_position(loop, ai, sl_dist=20.0)

    # Bar 1: drawdown fires.
    b1 = Bar(symbol="BTCUSDT", ts=T0 + BAR,
             open=99.0, high=99.0, low=86.0, close=87.0)
    asyncio.run(loop.on_bar_async(b1))
    assert len(ai.review_calls) == 1

    # Bar 2 (15m later): still in drawdown -> THROTTLED.
    b2 = Bar(symbol="BTCUSDT", ts=T0 + 2 * BAR,
             open=87.0, high=88.0, low=86.0, close=86.5)
    asyncio.run(loop.on_bar_async(b2))
    assert len(ai.review_calls) == 1

    # Bar far in the future (>= 1h after bar 1 review): allowed.
    b3 = Bar(symbol="BTCUSDT", ts=T0 + BAR + pd.Timedelta(hours=1, minutes=1),
             open=86.5, high=87.0, low=86.0, close=86.5)
    asyncio.run(loop.on_bar_async(b3))
    assert len(ai.review_calls) == 2


# ---------------------------------------------------------------------
# Regime flip
# ---------------------------------------------------------------------
def test_regime_flip_hook(tmp_data_root):
    ai = StubAI(deep_signal=_signal_long(), review_action="hold")
    loop = _build_loop(ai=ai)
    _open_position(loop, ai, sl_dist=20.0)

    # First call: just records the regime, no review.
    out = asyncio.run(loop.notify_regime_flip(
        "risk-on", mark_map={"BTCUSDT": 100.0},
        now=T0 + pd.Timedelta(seconds=30)))
    assert out == []
    assert ai.review_calls == []

    # Same regime: still no review.
    out = asyncio.run(loop.notify_regime_flip(
        "risk-on", mark_map={"BTCUSDT": 100.0},
        now=T0 + pd.Timedelta(seconds=60)))
    assert out == []

    # Flip: fires for every open position.
    out = asyncio.run(loop.notify_regime_flip(
        "risk-off", mark_map={"BTCUSDT": 99.0},
        now=T0 + pd.Timedelta(seconds=120)))
    assert len(out) == 1
    assert ai.review_calls[0]["payload"]["trigger_reason"] == "regime_flip"


# ---------------------------------------------------------------------
# Funding approach
# ---------------------------------------------------------------------
def test_funding_approach_inside_window(tmp_data_root):
    ai = StubAI(deep_signal=_signal_long(), review_action="hold")
    loop = _build_loop(ai=ai, review_cfg=ReviewConfig(
        funding_window_sec=30 * 60.0))
    _open_position(loop, ai, sl_dist=20.0)

    now = T0 + pd.Timedelta(seconds=30)
    # 20 minutes to funding -> inside window.
    next_funding = now + pd.Timedelta(minutes=20)
    out = asyncio.run(loop.on_funding_window(
        symbol="BTCUSDT", next_funding_at=next_funding,
        mark=100.0, now=now))
    assert len(out) == 1
    assert ai.review_calls[0]["payload"]["trigger_reason"] == "funding_approach"


def test_funding_outside_window_skipped(tmp_data_root):
    ai = StubAI(deep_signal=_signal_long(), review_action="hold")
    loop = _build_loop(ai=ai, review_cfg=ReviewConfig(
        funding_window_sec=30 * 60.0))
    _open_position(loop, ai, sl_dist=20.0)

    now = T0 + pd.Timedelta(seconds=30)
    next_funding = now + pd.Timedelta(hours=2)
    out = asyncio.run(loop.on_funding_window(
        symbol="BTCUSDT", next_funding_at=next_funding,
        mark=100.0, now=now))
    assert out == []


def test_funding_auto_stop_when_imminent_adverse_and_losing(tmp_data_root):
    """Inside the imminent window with an adverse funding rate AND
    unrealized PnL < 0, ``on_funding_window`` short-circuits the
    review LLM and closes the position via a ``funding`` fill.
    """
    ai = StubAI(deep_signal=_signal_long(), review_action="hold")
    loop = _build_loop(ai=ai, review_cfg=ReviewConfig(
        funding_window_sec=30 * 60.0))
    pid = _open_position(loop, ai, sl_dist=20.0)
    assert pid in loop.broker.positions

    now = T0 + pd.Timedelta(seconds=30)
    # 4 minutes to funding -> imminent.
    next_funding = now + pd.Timedelta(minutes=4)
    # Long + positive funding rate = adverse; mark below entry = losing.
    asyncio.run(loop.on_funding_window(
        symbol="BTCUSDT", next_funding_at=next_funding,
        mark=95.0, now=now, funding_rate=0.0008))
    assert pid not in loop.broker.positions, "auto-stop should close pos"


def test_funding_auto_stop_skips_when_winning(tmp_data_root):
    """Imminent + adverse but unrealized PnL > 0 -> no auto-stop,
    falls through to the normal LLM review path.
    """
    ai = StubAI(deep_signal=_signal_long(), review_action="hold")
    loop = _build_loop(ai=ai, review_cfg=ReviewConfig(
        funding_window_sec=30 * 60.0))
    pid = _open_position(loop, ai, sl_dist=20.0)

    now = T0 + pd.Timedelta(seconds=30)
    next_funding = now + pd.Timedelta(minutes=4)
    out = asyncio.run(loop.on_funding_window(
        symbol="BTCUSDT", next_funding_at=next_funding,
        mark=110.0, now=now, funding_rate=0.0008))
    assert pid in loop.broker.positions, "winning position must not auto-stop"
    assert len(out) == 1
    assert out[0].action == "hold"


# ---------------------------------------------------------------------
# Apply layer
# ---------------------------------------------------------------------
def test_apply_tighten_stop_updates_sl(tmp_data_root):
    ai = StubAI(deep_signal=_signal_long(), review_action="tighten_stop")
    loop = _build_loop(ai=ai, review_cfg=ReviewConfig(
        drawdown_pct_of_sl_distance=0.6))
    pid = _open_position(loop, ai, sl_dist=20.0)
    pos = loop.broker.positions[pid]
    assert pos.stop_loss == pytest.approx(80.0)

    # Adverse mark to fire drawdown.
    bar = Bar(symbol="BTCUSDT", ts=T0 + BAR,
              open=99.0, high=99.0, low=86.0, close=87.0)
    asyncio.run(loop.on_bar_async(bar))

    pos = loop.broker.positions[pid]
    # Stub raises SL by 50% of (entry - sl). Entry is ~100.02 after
    # 2bps slippage on the touch fill, so new_sl ~= 90.01.
    assert pos.stop_loss == pytest.approx(90.01, abs=0.05)


def test_apply_stop_closes_position(tmp_data_root):
    ai = StubAI(deep_signal=_signal_long(), review_action="stop")
    loop = _build_loop(ai=ai, review_cfg=ReviewConfig(
        drawdown_pct_of_sl_distance=0.6))
    pid = _open_position(loop, ai, sl_dist=20.0)

    bar = Bar(symbol="BTCUSDT", ts=T0 + BAR,
              open=99.0, high=99.0, low=86.0, close=87.0)
    asyncio.run(loop.on_bar_async(bar))

    assert pid not in loop.broker.positions


def test_apply_scale_out_does_not_requeue_review(tmp_data_root):
    """``scale_out`` synthesizes a tp1 fill. The TP1 hook must NOT
    re-queue another review for the same position (it would only get
    throttled, but that pollutes reviews.jsonl)."""
    ai = StubAI(deep_signal=_signal_long(), review_action="scale_out")
    loop = _build_loop(ai=ai, review_cfg=ReviewConfig(
        drawdown_pct_of_sl_distance=0.6))
    pid = _open_position(loop, ai, sl_dist=20.0)

    bar = Bar(symbol="BTCUSDT", ts=T0 + BAR,
              open=99.0, high=99.0, low=86.0, close=87.0)
    asyncio.run(loop.on_bar_async(bar))

    # Exactly one review call total.
    assert len(ai.review_calls) == 1
    pos = loop.broker.positions[pid]
    assert pos.tp1_filled is True
    # No throttled rows in the review audit.
    review_rows = [json.loads(l) for l in loop._reviews_path.read_text(
        encoding="utf-8").splitlines() if l.strip()]
    assert all(r["action"] != "throttled" for r in review_rows)


# ---------------------------------------------------------------------
# Budget exhaustion
# ---------------------------------------------------------------------
def test_budget_exhaustion_short_circuits_review(tmp_data_root):
    """When the daily budget is gone, ``AIClient`` returns the
    synthetic ``hold`` and writes a ``budget_exhausted`` warning. The
    review still appears in ``reviews.jsonl`` (action=hold)."""
    cfg = load_config()
    audit_dir = cfg.repo_root / "data" / "logs" / "ai_audit_test"
    audit_dir.mkdir(parents=True, exist_ok=True)
    audit = AuditWriter(audit_dir)
    # Tiny cap; first review estimate exceeds it.
    budget = BudgetTracker(daily_cap_usd=1e-6)
    client = AIClient(cfg, budget=budget, audit=audit)

    # Wrap the client with a deep stub so we don't hit chat_deep budget.
    deep_signal = _signal_long()

    class WrappedAI:
        chat_review = client.chat_review

        async def chat_deep(self, symbol: str, payload: dict) -> DeepSignal:
            return deep_signal.model_copy(update={"symbol": symbol})

    ai = WrappedAI()
    loop = _build_loop(ai=ai, review_cfg=ReviewConfig(
        drawdown_pct_of_sl_distance=0.6))
    _open_position(loop, ai, sl_dist=20.0)

    bar = Bar(symbol="BTCUSDT", ts=T0 + BAR,
              open=99.0, high=99.0, low=86.0, close=87.0)
    asyncio.run(loop.on_bar_async(bar))

    # Budget did NOT charge (synthetic short-circuit).
    assert budget.spent_usd == pytest.approx(0.0)

    # ai_warnings.jsonl should record budget_exhausted.
    warnings_path = audit_dir / "ai_warnings.jsonl"
    assert warnings_path.exists()
    rows = [json.loads(l) for l in warnings_path.read_text(
        encoding="utf-8").splitlines() if l.strip()]
    assert any(r["kind"] == "budget_exhausted" for r in rows)

    # Reviews jsonl should still capture the audited "hold" outcome.
    reviews_path = loop._reviews_path
    assert reviews_path is not None and reviews_path.exists()
    review_rows = [json.loads(l) for l in reviews_path.read_text(
        encoding="utf-8").splitlines() if l.strip()]
    assert any(r["action"] == "hold" for r in review_rows)
