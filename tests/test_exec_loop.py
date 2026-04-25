"""Phase 11 - exec loop tests (offline)."""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass

import pandas as pd
import pytest

from ai.schemas import DeepSignal
from core.config import load_config
from loops.exec import ExecConfig, ExecLoop
from loops.triggers import TriggerDecision
from portfolio.broker import Bar
from portfolio.intents import (
    ActivationWatcher,
    BookTop,
    IntentEvent,
    IntentStatus,
    Tick,
)

T0 = pd.Timestamp("2026-04-25T12:00:00Z")
BAR = pd.Timedelta(minutes=15)


# ---------------------------------------------------------------------
@dataclass
class FakeAI:
    """Minimal AIClient stand-in: returns a canned ``DeepSignal``."""

    signal: DeepSignal

    async def chat_deep(self, symbol: str, payload: dict) -> DeepSignal:
        return self.signal.model_copy(update={"symbol": symbol})


def _signal_long(symbol: str = "BTCUSDT", *, entry: float = 100.0,
                 sl: float = 80.0, tp1: float = 110.0,
                 activation: str = "touch") -> DeepSignal:
    return DeepSignal(
        prompt_version="v3.1",
        symbol=symbol,
        action="long",
        entry=entry,
        entry_trigger=entry,
        activation_kind=activation,
        stop_loss=sl,
        take_profit_1=tp1,
        take_profit_2=tp1 + (tp1 - entry),
        time_horizon_bars=8,
        confidence=0.7,
    )


def _flat_signal(symbol: str = "BTCUSDT") -> DeepSignal:
    return DeepSignal(prompt_version="v3.1", symbol=symbol,
                      action="flat", confidence=0.4)


def _trigger(symbol: str = "BTCUSDT", *, close: float = 100.0,
             flag: str = "flag_volume_climax") -> tuple[dict, TriggerDecision]:
    bar = {
        "symbol": symbol,
        "timestamp": T0,
        "close": close,
        "atr_pct": 0.01,
    }
    decision = TriggerDecision(
        symbol=symbol, bar_ts=T0, decision="fresh",
        flag=flag, close=close, atr_pct=0.01, move_pct=0.0,
        threshold_pct=0.0, bars_elapsed=999, reason="",
    )
    return bar, decision


def _build_loop(tmp_data_root, ai: FakeAI, *,
                starting_equity: float = 10_000.0,
                save_every_fill: bool = True) -> ExecLoop:
    cfg = load_config()
    feature_cfg = None  # exec loop never calls features when scanner not used
    return ExecLoop.build(
        cfg=cfg,
        feature_cfg=feature_cfg,
        ai=ai,
        run_id="testrun01",
        exec_cfg=ExecConfig(starting_equity_usd=starting_equity,
                            save_state_every_fill=save_every_fill),
    )


# ---------------------------------------------------------------------
# on_trigger
# ---------------------------------------------------------------------

def test_on_trigger_queues_sized_intent(tmp_data_root):
    ai = FakeAI(signal=_signal_long(entry=100.0, sl=80.0, tp1=110.0))
    loop = _build_loop(tmp_data_root, ai)
    bar, decision = _trigger()

    intent = asyncio.run(loop.on_trigger("BTCUSDT", bar, decision))

    assert intent is not None
    # risk = 1% * 10k = 100 USD; sl_dist = 20 -> qty = 5.0.
    assert intent.qty == pytest.approx(5.0)
    assert intent.symbol == "BTCUSDT"
    # Default lifecycle entry: armed (no SL-before-entry seen yet).
    assert intent.status in {IntentStatus.PENDING, IntentStatus.ARMED}
    # Audit trail.
    audit = (loop._intents_path).read_text(encoding="utf-8").splitlines()
    assert len(audit) == 1
    rec = json.loads(audit[0])
    assert rec["event"] == "submit"
    assert rec["intent"]["qty"] == pytest.approx(5.0)


def test_on_trigger_returns_none_for_flat(tmp_data_root):
    ai = FakeAI(signal=_flat_signal())
    loop = _build_loop(tmp_data_root, ai)
    bar, decision = _trigger()

    intent = asyncio.run(loop.on_trigger("BTCUSDT", bar, decision))
    assert intent is None
    assert loop.queue.all() == []


def test_on_trigger_rejects_when_sizing_fails(tmp_data_root):
    # zero SL distance -> sizing rejection.
    bad_signal = DeepSignal(
        prompt_version="v3.1", symbol="BTCUSDT", action="long",
        entry=100.0, entry_trigger=100.0, activation_kind="touch",
        stop_loss=99.999999, take_profit_1=110.0, time_horizon_bars=8,
        confidence=0.6,
    )
    # consistency may emit a warning but action is still long.
    ai = FakeAI(signal=bad_signal)
    loop = _build_loop(tmp_data_root, ai, starting_equity=10.0)
    bar, decision = _trigger()
    intent = asyncio.run(loop.on_trigger("BTCUSDT", bar, decision))
    # Either flat-on-2-warnings or rejected by risk: result is None.
    assert intent is None


# ---------------------------------------------------------------------
# emit_event - activation -> broker
# ---------------------------------------------------------------------

def test_activated_event_opens_broker_position(tmp_data_root):
    ai = FakeAI(signal=_signal_long(entry=100.0, sl=80.0, tp1=110.0,
                                     activation="breakout"))  # no slippage
    loop = _build_loop(tmp_data_root, ai)
    bar, decision = _trigger()
    intent = asyncio.run(loop.on_trigger("BTCUSDT", bar, decision))
    assert intent is not None

    # Drive an activation through the watcher: breakout requires an
    # `ask > trigger` book deeper than 5_000 USD.
    book = BookTop(ts=T0 + pd.Timedelta(seconds=10), bid=99.95, bid_size=10,
                   ask=100.10, ask_size=200, symbol="BTCUSDT")
    events = loop.watcher.process_book(book)
    assert len(events) == 1
    ev = events[0]
    assert ev.kind == "activated"

    asyncio.run(loop.emit_event(ev))

    assert len(loop.broker.positions) == 1
    pos = next(iter(loop.broker.positions.values()))
    assert pos.symbol == "BTCUSDT"
    assert pos.side == "long"
    # Entry fill recorded.
    fills = (loop._fills_path).read_text(encoding="utf-8").splitlines()
    assert len(fills) == 1
    rec = json.loads(fills[0])
    assert rec["kind"] == "entry"
    assert rec["symbol"] == "BTCUSDT"
    assert rec["qty"] == pytest.approx(intent.qty)
    # Cash decremented by fee.
    assert loop._cash_usd < 10_000.0
    assert loop._fees_paid_usd > 0


def test_killed_or_expired_event_does_not_open_position(tmp_data_root):
    ai = FakeAI(signal=_signal_long())
    loop = _build_loop(tmp_data_root, ai)
    bar, decision = _trigger()
    intent = asyncio.run(loop.on_trigger("BTCUSDT", bar, decision))
    assert intent is not None

    # Manually emit an expired event.
    expired = intent.__class__(
        **{**intent.__dict__, "status": IntentStatus.EXPIRED,
           "kill_reason": "expired"}
    )
    ev = IntentEvent(kind="expired", intent=expired, reason="expired")
    asyncio.run(loop.emit_event(ev))
    assert loop.broker.positions == {}


# ---------------------------------------------------------------------
# on_bar -> SL fill -> state file
# ---------------------------------------------------------------------

def test_full_cycle_trigger_to_sl_writes_state(tmp_data_root):
    # entry=100, sl=95: sl_dist=5, qty=20, notional=2000 (under 40% cap).
    ai = FakeAI(signal=_signal_long(entry=100.0, sl=95.0, tp1=110.0,
                                     activation="touch"))
    loop = _build_loop(tmp_data_root, ai)
    bar_in, decision = _trigger(close=100.0)
    intent = asyncio.run(loop.on_trigger("BTCUSDT", bar_in, decision))
    assert intent is not None

    # Touch activation: a tick at the trigger.
    tick = Tick(ts=T0 + pd.Timedelta(seconds=5), price=100.0, size=1.0,
                side="Buy", symbol="BTCUSDT")
    events = loop.watcher.process_tick(tick)
    assert events and events[0].kind == "activated"
    asyncio.run(loop.emit_event(events[0]))
    pos_qty = next(iter(loop.broker.positions.values())).remaining_qty

    # Bar 1: SL hit (low pierces 95).
    sl_bar = Bar(ts=T0 + BAR, open=99.5, high=100.0, low=94.5, close=95.2,
                 symbol="BTCUSDT")
    fills = loop.on_bar(sl_bar)
    assert len(fills) == 1
    assert fills[0].kind == "stop"
    # Pos closed.
    assert loop.broker.positions == {}

    # Loser streak incremented.
    assert loop._loser_streak == 1

    # State file written and consistent.
    state_raw = json.loads((loop._portfolio_path).read_text(encoding="utf-8"))
    assert state_raw["loser_streak"] == 1
    assert state_raw["closed_positions_24h"] == 1
    # Cash = starting - fees + pnl. PnL is negative.
    assert state_raw["realized_pnl_usd"] < 0
    assert state_raw["fees_paid_usd"] > 0
    # Equity = cash since no open positions.
    assert state_raw["equity_usd"] == pytest.approx(state_raw["cash_usd"])
    # Fills file has entry + stop.
    fill_lines = (loop._fills_path).read_text(encoding="utf-8").splitlines()
    assert len(fill_lines) == 2
    kinds = [json.loads(l)["kind"] for l in fill_lines]
    assert kinds == ["entry", "stop"]

    # Sanity: sl_dist=5, loss ~ -qty * 5 (modulo entry slippage).
    expected_loss = -(pos_qty * 5.0)
    assert state_raw["realized_pnl_usd"] == pytest.approx(expected_loss, rel=0.05)


def test_close_all_flattens_positions(tmp_data_root):
    ai = FakeAI(signal=_signal_long(entry=100.0, sl=80.0, tp1=110.0,
                                     activation="breakout"))
    loop = _build_loop(tmp_data_root, ai)
    bar, decision = _trigger()
    asyncio.run(loop.on_trigger("BTCUSDT", bar, decision))
    book = BookTop(ts=T0 + pd.Timedelta(seconds=10), bid=99.95, bid_size=10,
                   ask=100.10, ask_size=200, symbol="BTCUSDT")
    asyncio.run(loop.emit_event(loop.watcher.process_book(book)[0]))

    fills = loop.close_all(ts=T0 + BAR, price_map={"BTCUSDT": 105.0})
    assert len(fills) == 1
    assert fills[0].kind == "manual"
    assert fills[0].pnl_usd > 0
    assert loop.broker.positions == {}


def test_replay_matches_running_state(tmp_data_root):
    """Phase 10 invariant: replay_from_fills reproduces equity to 1e-6."""
    from portfolio.state import read_fills, replay_from_fills

    ai = FakeAI(signal=_signal_long(entry=100.0, sl=95.0, tp1=110.0,
                                     activation="touch"))
    loop = _build_loop(tmp_data_root, ai)
    bar, decision = _trigger()
    asyncio.run(loop.on_trigger("BTCUSDT", bar, decision))
    tick = Tick(ts=T0 + pd.Timedelta(seconds=5), price=100.0, size=1.0,
                side="Buy", symbol="BTCUSDT")
    asyncio.run(loop.emit_event(loop.watcher.process_tick(tick)[0]))
    loop.on_bar(Bar(ts=T0 + BAR, open=99.5, high=100.0, low=94.5, close=95.2,
                    symbol="BTCUSDT"))

    fills = read_fills(loop._fills_path)
    replayed = replay_from_fills(fills, starting_equity_usd=10_000.0,
                                 now=T0 + pd.Timedelta(hours=1))
    assert replayed.cash_usd == pytest.approx(loop._cash_usd, abs=1e-6)
    assert replayed.realized_pnl_usd == pytest.approx(
        loop._realized_pnl_usd, abs=1e-6,
    )
    assert replayed.fees_paid_usd == pytest.approx(
        loop._fees_paid_usd, abs=1e-6,
    )


def test_set_watchlist_propagates_to_state(tmp_data_root):
    ai = FakeAI(signal=_flat_signal())
    loop = _build_loop(tmp_data_root, ai)
    loop.set_watchlist(["BTCUSDT", "ETHUSDT"])
    state = loop.snapshot_state(now=T0)
    assert state.watchlist == ["BTCUSDT", "ETHUSDT"]


# ---------------------------------------------------------------------
# Phase 14.5 - safety hooks
# ---------------------------------------------------------------------

def test_set_watchlist_kills_dropped_pending_intents(tmp_data_root):
    """A pending intent on a symbol that is no longer watchlisted
    must be killed with reason ``watchlist_dropped``.
    """
    ai = FakeAI(signal=_signal_long(entry=100.0, sl=80.0, tp1=110.0))
    loop = _build_loop(tmp_data_root, ai)
    bar, decision = _trigger()
    intent = asyncio.run(loop.on_trigger("BTCUSDT", bar, decision))
    assert intent is not None
    loop.set_watchlist(["BTCUSDT"])
    assert all(not i.is_terminal() for i in loop.queue.active())

    killed = loop.set_watchlist(["ETHUSDT"])
    assert killed == 1
    survivor = loop.queue.get(intent.intent_id)
    assert survivor is not None
    assert survivor.status == IntentStatus.KILLED
    assert survivor.kill_reason == "watchlist_dropped"


def test_stop_file_blocks_on_trigger(tmp_data_root):
    """When the STOP sentinel exists, ``on_trigger`` returns ``None``
    without consulting the AI client.
    """
    ai = FakeAI(signal=_signal_long())
    loop = _build_loop(tmp_data_root, ai)
    sp = loop.stop_file_path()
    assert sp is not None
    sp.parent.mkdir(parents=True, exist_ok=True)
    sp.write_text("halt\n", encoding="utf-8")
    try:
        bar, decision = _trigger()
        result = asyncio.run(loop.on_trigger("BTCUSDT", bar, decision))
        assert result is None
        assert loop.stop_requested() is True
        assert loop.queue.all() == []
    finally:
        sp.unlink(missing_ok=True)


def test_replay_on_build_restores_running_counters(tmp_data_root):
    """Building a second ExecLoop on the same run_id must hydrate
    ``cash_usd`` / ``realized_pnl_usd`` / ``loser_streak`` from the
    pre-existing fills.jsonl.
    """
    from portfolio.state import read_fills, replay_from_fills
    ai = FakeAI(signal=_signal_long(entry=100.0, sl=80.0, tp1=110.0,
                                     activation="breakout"))
    loop_a = _build_loop(tmp_data_root, ai)
    bar, decision = _trigger()
    asyncio.run(loop_a.on_trigger("BTCUSDT", bar, decision))
    intent = loop_a.queue.active()[0]
    book = BookTop(ts=T0 + pd.Timedelta(seconds=10), bid=99.95, bid_size=10,
                   ask=100.10, ask_size=200, symbol="BTCUSDT")
    events = loop_a.watcher.process_book(book)
    for ev in events:
        asyncio.run(loop_a.emit_event(ev))
    # Drive a closing fill so realized PnL and fees are non-zero.
    closing_bar = Bar(symbol="BTCUSDT", ts=T0 + BAR,
                      open=110.0, high=110.5, low=109.0, close=109.5)
    loop_a.on_bar(closing_bar)

    cash_a = loop_a._cash_usd
    realized_a = loop_a._realized_pnl_usd
    fees_a = loop_a._fees_paid_usd

    # Now build a fresh loop on the same run_id; replay should
    # reconstruct the projection exactly.
    cfg = load_config()
    loop_b = ExecLoop.build(
        cfg=cfg, feature_cfg=None, ai=ai, run_id="testrun01",
        exec_cfg=ExecConfig(starting_equity_usd=10_000.0),
    )
    assert loop_b._cash_usd == pytest.approx(cash_a, abs=1e-6)
    assert loop_b._realized_pnl_usd == pytest.approx(realized_a, abs=1e-6)
    assert loop_b._fees_paid_usd == pytest.approx(fees_a, abs=1e-6)

    # Independent cross-check: a pure replay of fills.jsonl agrees too.
    pure = replay_from_fills(read_fills(loop_a._fills_path),
                             starting_equity_usd=10_000.0,
                             now=T0 + pd.Timedelta(hours=1))
    assert pure.cash_usd == pytest.approx(cash_a, abs=1e-6)


def test_build_deep_payload_merges_context_whitelist():
    """``_build_deep_payload`` merges the whitelisted context blocks
    and silently ignores anything else.
    """
    from loops.exec import _build_deep_payload
    bar, decision = _trigger()
    ctx = {
        "mtf": {"1h": [{"close": 99.0}], "4h": [{"close": 95.0}]},
        "deriv": {"funding_rate": 0.0001, "oi_usd": 1.5e9},
        "peer": {"cluster_id": 3, "rank": 1},
        "garbage": {"do": "not include"},
    }
    payload = _build_deep_payload("BTCUSDT", bar, decision, context=ctx)
    assert payload["symbol"] == "BTCUSDT"
    assert payload["mtf"]["1h"][0]["close"] == 99.0
    assert payload["deriv"]["funding_rate"] == 0.0001
    assert payload["peer"]["rank"] == 1
    assert "garbage" not in payload
