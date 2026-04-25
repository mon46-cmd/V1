"""Phase 9 tests: intent + atomic confirmation."""
from __future__ import annotations

import pandas as pd
import pytest

from portfolio.intents import (
    ActivationWatcher,
    BookTop,
    Intent,
    IntentQueue,
    IntentStatus,
    Tick,
    WatcherConfig,
)

T0 = pd.Timestamp("2026-04-25T12:00:00Z")


def _intent(
    *,
    activation_kind: str = "touch",
    side: str = "long",
    entry: float = 100.0,
    entry_trigger: float | None = None,
    sl: float = 99.0,
    tp1: float = 102.0,
    expires_in: float = 180.0,
    intent_id: str = "TEST_ID",
    symbol: str = "BTCUSDT",
) -> Intent:
    return Intent(
        intent_id=intent_id,
        created_at=T0,
        expires_at=T0 + pd.Timedelta(seconds=expires_in),
        symbol=symbol,
        side=side,  # type: ignore[arg-type]
        entry=entry,
        entry_trigger=entry_trigger if entry_trigger is not None else entry,
        activation_kind=activation_kind,  # type: ignore[arg-type]
        stop_loss=sl,
        take_profit_1=tp1,
        take_profit_2=None,
        time_horizon_bars=16,
        qty=1.0,
        trigger_flag="flag_volume_climax",
        prompt_version="test",
    )


def _tick(price: float, ts_off_sec: float = 0.0, symbol: str = "BTCUSDT") -> Tick:
    return Tick(ts=T0 + pd.Timedelta(seconds=ts_off_sec), price=price,
                size=1.0, side="Buy", symbol=symbol)


# ----- queue + audit ----------------------------------------------------

def test_queue_audit_writes_jsonl(tmp_path):
    audit = tmp_path / "intents.jsonl"
    q = IntentQueue(audit_path=audit)
    intent = _intent()
    q.submit(intent)
    assert audit.exists()
    lines = audit.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1


# ----- touch activation -------------------------------------------------

def test_touch_long_fires_once_at_correct_price(tmp_path):
    q = IntentQueue(audit_path=tmp_path / "intents.jsonl")
    w = ActivationWatcher(queue=q)
    intent = _intent(activation_kind="touch", entry=100.0,
                     entry_trigger=100.0, sl=99.0, tp1=102.0)
    q.submit(intent)

    # First tick above trigger -> nothing.
    assert w.process_tick(_tick(100.5)) == []

    # Tick at trigger -> activated exactly once at trigger price.
    evs = w.process_tick(_tick(100.0, ts_off_sec=1.0))
    assert len(evs) == 1
    ev = evs[0]
    assert ev.kind == "activated"
    assert ev.intent.activated_price == 100.0
    assert ev.intent.status == IntentStatus.ACTIVATED

    # Subsequent ticks must not produce a second activation.
    assert w.process_tick(_tick(99.5, ts_off_sec=2.0)) == []
    assert w.process_tick(_tick(100.0, ts_off_sec=3.0)) == []


def test_touch_short_fires_when_price_rises_to_trigger(tmp_path):
    q = IntentQueue(audit_path=tmp_path / "intents.jsonl")
    w = ActivationWatcher(queue=q)
    intent = _intent(side="short", activation_kind="touch",
                     entry=100.0, entry_trigger=100.0,
                     sl=101.0, tp1=98.0)
    q.submit(intent)

    assert w.process_tick(_tick(99.5)) == []
    evs = w.process_tick(_tick(100.0, ts_off_sec=1.0))
    assert len(evs) == 1 and evs[0].kind == "activated"
    assert evs[0].intent.activated_price == 100.0


# ----- expiry -----------------------------------------------------------

def test_expiry_kills_non_activating_intent(tmp_path):
    q = IntentQueue(audit_path=tmp_path / "intents.jsonl")
    w = ActivationWatcher(queue=q)
    intent = _intent(expires_in=60.0)  # one-minute window
    q.submit(intent)

    # Before expiry: no event.
    evs = w.process_clock(T0 + pd.Timedelta(seconds=30))
    assert evs == []
    assert q.get(intent.intent_id).status == IntentStatus.ARMED

    # After expiry: expired event.
    evs = w.process_clock(T0 + pd.Timedelta(seconds=61))
    assert len(evs) == 1 and evs[0].kind == "expired"
    assert q.get(intent.intent_id).status == IntentStatus.EXPIRED

    # Idempotent: a later clock pass does nothing.
    assert w.process_clock(T0 + pd.Timedelta(seconds=120)) == []


# ----- SL invalidation --------------------------------------------------

def test_sl_before_entry_kills_intent(tmp_path):
    q = IntentQueue(audit_path=tmp_path / "intents.jsonl")
    w = ActivationWatcher(queue=q)
    intent = _intent(side="long", entry=100.0, entry_trigger=100.0,
                     sl=99.0, tp1=102.0)
    q.submit(intent)

    # Price spikes through SL before ever touching the entry trigger.
    # The trigger and SL straddle 100; a tick at 98.5 hits SL first
    # (price <= sl AND price <= trigger). We require SL to win.
    evs = w.process_tick(_tick(98.5, ts_off_sec=1.0))
    assert len(evs) == 1 and evs[0].kind == "killed"
    assert evs[0].reason == "sl_before_entry"
    assert q.get(intent.intent_id).status == IntentStatus.KILLED


def test_sl_short_invalidation(tmp_path):
    q = IntentQueue(audit_path=tmp_path / "intents.jsonl")
    w = ActivationWatcher(queue=q)
    intent = _intent(side="short", entry=100.0, entry_trigger=100.0,
                     sl=101.0, tp1=98.0)
    q.submit(intent)
    evs = w.process_tick(_tick(101.5, ts_off_sec=1.0))
    assert len(evs) == 1 and evs[0].kind == "killed"
    assert evs[0].reason == "sl_before_entry"


# ----- breakout requires depth ------------------------------------------

def test_breakout_requires_book_size_threshold(tmp_path):
    q = IntentQueue(audit_path=tmp_path / "intents.jsonl")
    w = ActivationWatcher(queue=q, cfg=WatcherConfig(
        breakout_min_book_usd=5_000.0,
    ))
    intent = _intent(activation_kind="breakout", side="long",
                     entry=100.0, entry_trigger=100.0,
                     sl=99.0, tp1=102.0)
    q.submit(intent)

    # Above trigger but thin book -> no activation.
    thin = BookTop(ts=T0 + pd.Timedelta(seconds=1),
                   bid=99.99, bid_size=10.0,
                   ask=100.05, ask_size=10.0,  # 100.05 * 10 = 1000.5 USD < 5000
                   symbol="BTCUSDT")
    assert w.process_book(thin) == []
    assert q.get(intent.intent_id).status == IntentStatus.ARMED

    # Above trigger AND deep enough -> activation.
    deep = BookTop(ts=T0 + pd.Timedelta(seconds=2),
                   bid=100.04, bid_size=100.0,
                   ask=100.06, ask_size=100.0,  # 10006 USD
                   symbol="BTCUSDT")
    evs = w.process_book(deep)
    assert len(evs) == 1 and evs[0].kind == "activated"
    assert evs[0].intent.activated_price == 100.06


def test_breakout_short_requires_bid_below_trigger_with_depth(tmp_path):
    q = IntentQueue(audit_path=tmp_path / "intents.jsonl")
    w = ActivationWatcher(queue=q,
                          cfg=WatcherConfig(breakout_min_book_usd=5_000.0))
    intent = _intent(activation_kind="breakout", side="short",
                     entry=100.0, entry_trigger=100.0,
                     sl=101.0, tp1=98.0)
    q.submit(intent)
    # Bid still above trigger -> no activation.
    above = BookTop(ts=T0 + pd.Timedelta(seconds=1),
                    bid=100.05, bid_size=200.0,
                    ask=100.10, ask_size=200.0, symbol="BTCUSDT")
    assert w.process_book(above) == []
    # Bid below trigger with depth -> activation at bid.
    below = BookTop(ts=T0 + pd.Timedelta(seconds=2),
                    bid=99.95, bid_size=200.0,
                    ask=99.97, ask_size=200.0, symbol="BTCUSDT")
    evs = w.process_book(below)
    assert len(evs) == 1
    assert evs[0].intent.activated_price == 99.95


# ----- close_above ------------------------------------------------------

def test_close_above_waits_for_candle_close(tmp_path):
    q = IntentQueue(audit_path=tmp_path / "intents.jsonl")
    w = ActivationWatcher(queue=q,
                          cfg=WatcherConfig(close_candle_seconds=1.0))
    intent = _intent(activation_kind="close_above", side="long",
                     entry=100.0, entry_trigger=100.0,
                     sl=99.0, tp1=102.0)
    q.submit(intent)

    # First tick opens the candle at 100.5; same second -> no close yet.
    assert w.process_tick(_tick(100.5, ts_off_sec=0.0)) == []
    assert w.process_tick(_tick(100.6, ts_off_sec=0.5)) == []

    # 1.1 s later: candle closes at 100.6 (the prior tick), but the new
    # close after consuming this tick is the new tick's price. Either
    # way it's > 100 -> activation.
    evs = w.process_tick(_tick(100.7, ts_off_sec=1.1))
    assert len(evs) == 1 and evs[0].kind == "activated"
    assert evs[0].intent.activated_price > 100.0


def test_close_above_does_not_fire_when_close_below_trigger(tmp_path):
    q = IntentQueue(audit_path=tmp_path / "intents.jsonl")
    w = ActivationWatcher(queue=q,
                          cfg=WatcherConfig(close_candle_seconds=1.0))
    intent = _intent(activation_kind="close_above", side="long",
                     entry=100.0, entry_trigger=100.0,
                     sl=98.0, tp1=102.0)
    q.submit(intent)
    # Open candle just above, but next 1s candle closes back below.
    w.process_tick(_tick(100.05, ts_off_sec=0.0))
    w.process_tick(_tick(100.10, ts_off_sec=0.5))
    # 1s later, candle closes at 99.50 -> NOT > 100 -> no activation.
    evs = w.process_tick(_tick(99.50, ts_off_sec=1.1))
    assert evs == []
    assert q.get(intent.intent_id).status == IntentStatus.ARMED


# ----- multi-intent isolation ------------------------------------------

def test_per_symbol_isolation(tmp_path):
    q = IntentQueue(audit_path=tmp_path / "intents.jsonl")
    w = ActivationWatcher(queue=q)
    a = _intent(intent_id="A", symbol="BTCUSDT", entry=100.0,
                entry_trigger=100.0, sl=99.0, tp1=102.0)
    b = _intent(intent_id="B", symbol="ETHUSDT", entry=200.0,
                entry_trigger=200.0, sl=198.0, tp1=204.0)
    q.submit(a)
    q.submit(b)

    # A tick on BTC must not touch the ETH intent.
    evs = w.process_tick(_tick(100.0, ts_off_sec=1.0, symbol="BTCUSDT"))
    assert len(evs) == 1 and evs[0].intent.symbol == "BTCUSDT"
    assert q.get("B").status == IntentStatus.ARMED

    # ETH tick activates B alone.
    evs = w.process_tick(_tick(200.0, ts_off_sec=2.0, symbol="ETHUSDT"))
    assert len(evs) == 1 and evs[0].intent.symbol == "ETHUSDT"


# ----- intent_from_signal ----------------------------------------------

def test_intent_from_signal_basic():
    from ai.schemas import DeepSignal

    from portfolio.intents import intent_from_signal

    sig = DeepSignal(
        prompt_version="v3.1",
        symbol="BTCUSDT",
        action="long",
        entry=100.0,
        entry_trigger=100.0,
        activation_kind="touch",
        stop_loss=99.0,
        take_profit_1=102.0,
        time_horizon_bars=16,
        confidence=0.7,
    )
    intent = intent_from_signal(
        signal=sig, qty=0.5, trigger_flag="flag_sweep_up", now=T0,
    )
    assert intent.symbol == "BTCUSDT"
    assert intent.side == "long"
    assert intent.qty == 0.5
    assert intent.trigger_flag == "flag_sweep_up"
    assert intent.expires_at > T0


def test_intent_from_signal_rejects_flat():
    from ai.schemas import DeepSignal

    from portfolio.intents import intent_from_signal

    sig = DeepSignal(
        prompt_version="v3.1", symbol="BTCUSDT", action="flat",
        confidence=0.2,
    )
    with pytest.raises(ValueError):
        intent_from_signal(signal=sig, qty=1.0, trigger_flag="x", now=T0)
