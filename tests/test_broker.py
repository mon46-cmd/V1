"""Phase 10 tests: paper broker + risk + state."""
from __future__ import annotations

import pandas as pd
import pytest

from portfolio.broker import Bar, Broker, BrokerConfig
from portfolio.intents import Intent, IntentStatus
from portfolio.risk import (
    InstrumentSpec,
    RiskCaps,
    circuit_breaker_multiplier,
    size_intent,
)
from portfolio.state import (
    PortfolioState,
    append_fill,
    load_state,
    read_fills,
    replay_from_fills,
    save_state,
)

T0 = pd.Timestamp("2026-04-25T12:00:00Z")
BAR_15M = pd.Timedelta(minutes=15)


def _activated(
    *,
    side: str = "long",
    entry: float = 100.0,
    sl: float = 99.0,
    tp1: float = 102.0,
    tp2: float | None = None,
    qty: float = 1.0,
    activation_kind: str = "touch",
    horizon: int = 16,
    intent_id: str = "I1",
    symbol: str = "BTCUSDT",
) -> Intent:
    return Intent(
        intent_id=intent_id,
        created_at=T0,
        expires_at=T0 + pd.Timedelta(seconds=180),
        symbol=symbol,
        side=side,  # type: ignore[arg-type]
        entry=entry,
        entry_trigger=entry,
        activation_kind=activation_kind,  # type: ignore[arg-type]
        stop_loss=sl,
        take_profit_1=tp1,
        take_profit_2=tp2,
        time_horizon_bars=horizon,
        qty=qty,
        trigger_flag="flag_volume_climax",
        prompt_version="test",
        status=IntentStatus.ACTIVATED,
        activated_price=entry,
        activated_at=T0,
    )


def _bar(*, ts_offset_min: float = 0.0, o: float, h: float, l: float, c: float,
         symbol: str = "BTCUSDT") -> Bar:
    return Bar(ts=T0 + pd.Timedelta(minutes=ts_offset_min),
               open=o, high=h, low=l, close=c, symbol=symbol)


# =====================================================================
# risk.py
# =====================================================================

def test_size_intent_basic():
    intent = _activated(entry=100.0, sl=80.0)  # 20.0 SL distance
    d = size_intent(
        intent=intent,
        equity_usd=10_000.0,
        open_positions=0,
        symbol_exposure_usd=0.0,
        aggregate_exposure_usd=0.0,
        instrument=InstrumentSpec(qty_step=0.001, min_order_qty=0.001,
                                  min_notional_usd=5.0),
        caps=RiskCaps(per_trade_risk_pct=0.01),
    )
    # risk = 100 USD, sl_dist = 20 -> raw qty = 5.0.
    assert d.accepted
    assert d.qty == pytest.approx(5.0)
    assert d.notional_usd == pytest.approx(500.0)
    assert d.risk_usd == pytest.approx(100.0)


def test_size_intent_concurrent_cap():
    intent = _activated()
    d = size_intent(
        intent=intent, equity_usd=10_000.0, open_positions=3,
        symbol_exposure_usd=0.0, aggregate_exposure_usd=0.0,
        instrument=InstrumentSpec(), caps=RiskCaps(max_concurrent_positions=3),
    )
    assert not d.accepted and d.reason == "concurrent_cap"


def test_size_intent_zero_sl_distance():
    intent = _activated(entry=100.0, sl=100.0)
    d = size_intent(
        intent=intent, equity_usd=10_000.0, open_positions=0,
        symbol_exposure_usd=0.0, aggregate_exposure_usd=0.0,
        instrument=InstrumentSpec(), caps=RiskCaps(),
    )
    assert d.reason == "zero_sl_distance"


def test_size_intent_notional_floor():
    # Risk so small the notional is below the floor.
    intent = _activated(entry=100.0, sl=99.0)
    d = size_intent(
        intent=intent, equity_usd=1.0, open_positions=0,
        symbol_exposure_usd=0.0, aggregate_exposure_usd=0.0,
        instrument=InstrumentSpec(qty_step=0.001, min_order_qty=0.001,
                                  min_notional_usd=50.0),
        caps=RiskCaps(per_trade_risk_pct=0.01),
    )
    # risk = 0.01 USD, sl_dist = 1 -> raw qty = 0.01, floored to 0.01,
    # notional = 1.00 USD < 50 -> reject.
    assert d.reason == "notional_floor"


def test_size_intent_symbol_exposure_cap():
    intent = _activated()
    d = size_intent(
        intent=intent, equity_usd=10_000.0, open_positions=0,
        symbol_exposure_usd=5_000.0,  # already past 0.4 * 10k = 4000? we sit at 5000
        aggregate_exposure_usd=5_000.0,
        instrument=InstrumentSpec(qty_step=1.0, min_order_qty=1.0,
                                  min_notional_usd=5.0),
        caps=RiskCaps(per_trade_risk_pct=0.01,
                      max_symbol_exposure_pct=0.40),
    )
    assert d.reason == "symbol_exposure_cap"


def test_size_intent_aggregate_cap():
    intent = _activated()
    d = size_intent(
        intent=intent, equity_usd=10_000.0, open_positions=0,
        symbol_exposure_usd=0.0,
        aggregate_exposure_usd=14_000.0,  # cap = 1.5 * 10k = 15k
        instrument=InstrumentSpec(qty_step=1.0, min_order_qty=1.0,
                                  min_notional_usd=5.0),
        caps=RiskCaps(per_trade_risk_pct=0.01,
                      max_aggregate_exposure_pct=1.50,
                      max_symbol_exposure_pct=10.0),
    )
    # qty=100, notional=10k -> 14k+10k > 15k cap.
    assert d.reason == "aggregate_exposure_cap"


def test_circuit_breaker_halves_after_two():
    assert circuit_breaker_multiplier(0) == 1.0
    assert circuit_breaker_multiplier(1) == 1.0
    assert circuit_breaker_multiplier(2) == 0.5
    assert circuit_breaker_multiplier(5) == 0.5


# =====================================================================
# broker.py - entry slippage + basic open
# =====================================================================

def test_open_from_intent_applies_entry_slippage_long_touch():
    b = Broker()
    intent = _activated(side="long", entry=100.0, sl=99.0, tp1=102.0,
                        activation_kind="touch", qty=1.0)
    pos, fill = b.open_from_intent(intent, fill_price=100.0, fill_ts=T0)
    # 2 bps slippage on a long touch -> fill > 100.
    assert pos.entry_price == pytest.approx(100.0 * 1.0002)
    assert fill.kind == "entry"
    assert fill.qty == 1.0
    # Fee = 6 bps * 100.02 * 1 = 0.060012.
    assert fill.fee_usd == pytest.approx(0.060012, rel=1e-6)


def test_open_from_intent_no_slippage_for_breakout():
    b = Broker()
    intent = _activated(activation_kind="breakout", entry=100.0,
                        sl=99.0, tp1=102.0)
    pos, fill = b.open_from_intent(intent, fill_price=100.0, fill_ts=T0)
    assert pos.entry_price == 100.0
    assert fill.price == 100.0


def test_open_rejects_non_activated_intent():
    import dataclasses

    b = Broker()
    intent = _activated()
    intent2 = dataclasses.replace(intent, status=IntentStatus.ARMED)
    with pytest.raises(ValueError):
        b.open_from_intent(intent2, fill_price=100.0, fill_ts=T0)


def test_implied_tp2_when_signal_omits_it():
    b = Broker()
    intent = _activated(entry=100.0, sl=99.0, tp1=102.0, tp2=None,
                        activation_kind="breakout")
    pos, _ = b.open_from_intent(intent, fill_price=100.0, fill_ts=T0)
    # tp2 = tp1 + (tp1 - entry) = 102 + 2 = 104.
    assert pos.take_profit_2 == pytest.approx(104.0)


# =====================================================================
# broker.py - intrabar SL/TP rules
# =====================================================================

def test_long_sl_hit_within_bar():
    b = Broker(cfg=BrokerConfig(entry_slippage_bps=0.0))
    intent = _activated(activation_kind="breakout",  # no slippage
                        entry=100.0, sl=99.0, tp1=102.0, tp2=104.0, qty=1.0)
    b.open_from_intent(intent, fill_price=100.0, fill_ts=T0)

    bar = _bar(ts_offset_min=15, o=100.5, h=100.8, l=98.5, c=99.2)
    fills = b.on_bar(bar)
    assert len(fills) == 1
    f = fills[0]
    assert f.kind == "stop"
    assert f.price == 99.0   # exact SL
    # PnL: (99 - 100) * 1 = -1
    assert f.pnl_usd == pytest.approx(-1.0)
    assert b.positions == {}


def test_long_sl_before_tp_when_both_touched():
    b = Broker(cfg=BrokerConfig(entry_slippage_bps=0.0))
    intent = _activated(activation_kind="breakout",
                        entry=100.0, sl=99.0, tp1=102.0, tp2=104.0)
    b.open_from_intent(intent, fill_price=100.0, fill_ts=T0)
    # Bar reaches both levels: SL must win.
    bar = _bar(ts_offset_min=15, o=100.0, h=102.5, l=98.5, c=101.0)
    fills = b.on_bar(bar)
    assert len(fills) == 1
    assert fills[0].kind == "stop"


def test_long_tp1_only_then_be_for_runner():
    b = Broker(cfg=BrokerConfig(entry_slippage_bps=0.0))
    intent = _activated(activation_kind="breakout",
                        entry=100.0, sl=99.0, tp1=102.0, tp2=104.0, qty=2.0)
    pos, _ = b.open_from_intent(intent, fill_price=100.0, fill_ts=T0)

    # Bar 1: TP1 only (high reaches 102.5, low stays above SL).
    fills = b.on_bar(_bar(ts_offset_min=15, o=100.5, h=102.5, l=99.5, c=102.2))
    kinds = [f.kind for f in fills]
    assert kinds == ["tp1"]
    assert fills[0].qty == pytest.approx(1.0)  # 50% scale-out
    # remaining position is now BE-armed at entry.
    assert pos.tp1_filled and pos.be_armed
    assert pos.stop_loss == 100.0
    assert pos.remaining_qty == pytest.approx(1.0)

    # Bar 2: pulls back to BE -> stop fills at entry, pnl=0.
    fills = b.on_bar(_bar(ts_offset_min=30, o=101.0, h=101.5, l=99.8, c=99.9))
    assert len(fills) == 1
    f = fills[0]
    assert f.kind == "stop"
    assert f.price == 100.0
    assert f.pnl_usd == pytest.approx(0.0)


def test_long_tp1_then_tp2_in_same_bar_be_arms_first():
    """After TP1 in a bar, BE arms; if low retests entry while high also
    reaches TP2 in the same bar, BE wins by the worst-case rule."""
    b = Broker(cfg=BrokerConfig(entry_slippage_bps=0.0))
    intent = _activated(activation_kind="breakout",
                        entry=100.0, sl=99.0, tp1=102.0, tp2=104.0, qty=2.0)
    b.open_from_intent(intent, fill_price=100.0, fill_ts=T0)
    # Bar reaches TP1 (102), TP2 (104), and pulls back to entry (100).
    fills = b.on_bar(_bar(ts_offset_min=15, o=101.0, h=104.5, l=99.95, c=103.5))
    kinds = [f.kind for f in fills]
    assert kinds == ["tp1", "stop"]  # BE wins on the runner
    # PnL = TP1 (+1*2.0?). Let's check: TP1 qty=1, pnl=(102-100)*1=2.
    # Stop at entry on remaining 1: pnl=0.
    assert fills[0].pnl_usd == pytest.approx(2.0)
    assert fills[1].pnl_usd == pytest.approx(0.0)


def test_long_tp1_then_tp2_clean_run():
    """TP1 + TP2 same bar but no retest of BE -> both fill positively."""
    b = Broker(cfg=BrokerConfig(entry_slippage_bps=0.0))
    intent = _activated(activation_kind="breakout",
                        entry=100.0, sl=99.0, tp1=102.0, tp2=104.0, qty=2.0)
    b.open_from_intent(intent, fill_price=100.0, fill_ts=T0)
    fills = b.on_bar(_bar(ts_offset_min=15, o=101.5, h=104.5, l=101.0, c=104.2))
    kinds = [f.kind for f in fills]
    assert kinds == ["tp1", "tp2"]
    # PnL: tp1 (102-100)*1 = 2, tp2 (104-100)*1 = 4.
    assert fills[0].pnl_usd == pytest.approx(2.0)
    assert fills[1].pnl_usd == pytest.approx(4.0)


def test_long_gap_through_sl_fills_at_open():
    b = Broker(cfg=BrokerConfig(entry_slippage_bps=0.0))
    intent = _activated(activation_kind="breakout",
                        entry=100.0, sl=99.0, tp1=102.0, tp2=104.0, qty=1.0)
    b.open_from_intent(intent, fill_price=100.0, fill_ts=T0)
    # Gap-down to 98: open <= sl, exit at open.
    fills = b.on_bar(_bar(ts_offset_min=15, o=98.0, h=98.5, l=97.5, c=97.8))
    assert len(fills) == 1
    assert fills[0].kind == "stop"
    assert fills[0].price == 98.0
    # PnL = (98 - 100) * 1 = -2
    assert fills[0].pnl_usd == pytest.approx(-2.0)


def test_long_gap_through_tp2():
    b = Broker(cfg=BrokerConfig(entry_slippage_bps=0.0))
    intent = _activated(activation_kind="breakout",
                        entry=100.0, sl=99.0, tp1=102.0, tp2=104.0, qty=2.0)
    b.open_from_intent(intent, fill_price=100.0, fill_ts=T0)
    # Gap up to 105 -> both TP1 and TP2 fill at open.
    fills = b.on_bar(_bar(ts_offset_min=15, o=105.0, h=105.5, l=104.5, c=105.2))
    kinds = [f.kind for f in fills]
    assert kinds == ["tp1", "tp2"]
    assert fills[0].price == 105.0 and fills[1].price == 105.0


def test_short_sl_within_bar():
    b = Broker(cfg=BrokerConfig(entry_slippage_bps=0.0))
    intent = _activated(activation_kind="breakout",
                        side="short", entry=100.0, sl=101.0, tp1=98.0,
                        tp2=96.0, qty=1.0)
    b.open_from_intent(intent, fill_price=100.0, fill_ts=T0)
    fills = b.on_bar(_bar(ts_offset_min=15, o=100.2, h=101.5, l=99.5, c=100.8))
    assert len(fills) == 1
    f = fills[0]
    assert f.kind == "stop"
    assert f.price == 101.0
    # Short PnL = (entry - exit) * qty = -1.
    assert f.pnl_usd == pytest.approx(-1.0)


def test_short_clean_tp1_tp2_run():
    b = Broker(cfg=BrokerConfig(entry_slippage_bps=0.0))
    intent = _activated(activation_kind="breakout",
                        side="short", entry=100.0, sl=101.0, tp1=98.0,
                        tp2=96.0, qty=2.0)
    b.open_from_intent(intent, fill_price=100.0, fill_ts=T0)
    fills = b.on_bar(_bar(ts_offset_min=15, o=99.5, h=99.8, l=95.5, c=96.0))
    kinds = [f.kind for f in fills]
    assert kinds == ["tp1", "tp2"]
    assert fills[0].pnl_usd == pytest.approx(2.0)   # (100-98)*1
    assert fills[1].pnl_usd == pytest.approx(4.0)   # (100-96)*1


def test_time_stop_closes_at_bar_close():
    b = Broker(cfg=BrokerConfig(entry_slippage_bps=0.0))
    intent = _activated(activation_kind="breakout",
                        entry=100.0, sl=98.0, tp1=105.0, tp2=110.0,
                        qty=1.0, horizon=2)
    b.open_from_intent(intent, fill_price=100.0, fill_ts=T0)
    # Bar 1: nothing happens (range 99.5..101).
    fills1 = b.on_bar(_bar(ts_offset_min=15, o=100.2, h=101.0, l=99.5, c=100.8))
    assert fills1 == []
    # Bar 2: still nothing intra, but bars_held becomes 2 -> time stop at close.
    fills2 = b.on_bar(_bar(ts_offset_min=30, o=100.5, h=101.2, l=99.7, c=100.6))
    assert len(fills2) == 1
    f = fills2[0]
    assert f.kind == "time"
    assert f.price == 100.6
    assert f.pnl_usd == pytest.approx(0.6)


def test_close_position_manual():
    b = Broker(cfg=BrokerConfig(entry_slippage_bps=0.0))
    intent = _activated(activation_kind="breakout",
                        entry=100.0, sl=99.0, tp1=102.0, tp2=104.0, qty=1.0)
    pos, _ = b.open_from_intent(intent, fill_price=100.0, fill_ts=T0)
    f = b.close_position(pos.position_id, price=101.0, ts=T0 + BAR_15M,
                         kind="manual")
    assert f is not None
    assert f.kind == "manual"
    assert f.pnl_usd == pytest.approx(1.0)
    # Idempotent.
    assert b.close_position(pos.position_id, price=101.0,
                            ts=T0 + BAR_15M) is None


# =====================================================================
# state.py - persistence + replay determinism
# =====================================================================

def _record_fills(broker_fills, fills_path):
    for f in broker_fills:
        append_fill(fills_path, f)


def test_state_save_load_roundtrip(tmp_path):
    s = PortfolioState(
        as_of=T0, equity_usd=10_000.0, cash_usd=10_000.0,
        realized_pnl_usd=0.0, fees_paid_usd=0.0,
        watchlist=["BTCUSDT", "ETHUSDT"],
    )
    p = tmp_path / "portfolio.json"
    save_state(s, p)
    assert p.exists()
    loaded = load_state(p)
    assert loaded.equity_usd == 10_000.0
    assert loaded.watchlist == ["BTCUSDT", "ETHUSDT"]
    # Second save creates a `.bak`.
    s2 = PortfolioState(**{**s.__dict__, "equity_usd": 11_000.0})
    save_state(s2, p)
    assert p.with_suffix(".json.bak").exists()


def test_replay_determinism(tmp_path):
    """Run a scenario, dump fills, replay, assert equity matches."""
    fills_path = tmp_path / "fills.jsonl"
    b = Broker(cfg=BrokerConfig(entry_slippage_bps=0.0))
    intent = _activated(activation_kind="breakout",
                        entry=100.0, sl=99.0, tp1=102.0, tp2=104.0, qty=2.0)
    pos, entry_fill = b.open_from_intent(intent, fill_price=100.0, fill_ts=T0)
    append_fill(fills_path, entry_fill)
    # Clean TP1 + TP2 in one bar.
    bar = _bar(ts_offset_min=15, o=101.5, h=104.5, l=101.0, c=104.2)
    out_fills = b.on_bar(bar)
    _record_fills(out_fills, fills_path)

    fills = read_fills(fills_path)
    assert len(fills) == 3  # entry + tp1 + tp2

    state = replay_from_fills(fills, starting_equity_usd=10_000.0,
                              now=T0 + pd.Timedelta(hours=1))
    # Realized PnL = 2 + 4 = 6. Fees = 6 bps * (100*2 + 102*1 + 104*1) per fill.
    expected_pnl = 6.0
    assert state.realized_pnl_usd == pytest.approx(expected_pnl)
    assert state.equity_usd == pytest.approx(state.cash_usd)
    # Cash = starting + pnl - fees.
    expected_fees = sum(f["fee_usd"] for f in fills)
    assert state.cash_usd == pytest.approx(
        10_000.0 + expected_pnl - expected_fees,
    )
    assert state.open_positions == []
    assert state.closed_positions_24h == 1
    assert state.loser_streak == 0


def test_replay_loser_streak_triggers_circuit_breaker(tmp_path):
    fills_path = tmp_path / "fills.jsonl"
    b = Broker(cfg=BrokerConfig(entry_slippage_bps=0.0))
    # Two losers back-to-back.
    for i in range(2):
        intent = _activated(activation_kind="breakout",
                            entry=100.0, sl=99.0, tp1=102.0, tp2=104.0,
                            qty=1.0, intent_id=f"I{i}")
        _, ef = b.open_from_intent(intent, fill_price=100.0,
                                   fill_ts=T0 + pd.Timedelta(minutes=30 * i))
        append_fill(fills_path, ef)
        out = b.on_bar(_bar(ts_offset_min=15 + 30 * i,
                            o=100.0, h=100.2, l=98.5, c=98.8))
        _record_fills(out, fills_path)

    fills = read_fills(fills_path)
    state = replay_from_fills(fills, starting_equity_usd=10_000.0,
                              now=T0 + pd.Timedelta(hours=2))
    assert state.loser_streak == 2
    assert state.risk_multiplier == 0.5


def test_replay_handles_tz_naive_fill_ts(tmp_path):
    """Regression: a fills.jsonl with a tz-naive `ts` must be coerced to UTC.

    Older runs (or hand-edited snapshots) can produce ts strings without
    a `+00:00` suffix. Replay must still produce tz-aware Timestamps so
    24h-cutoff comparisons don't crash with TypeError.
    """
    import json

    fills_path = tmp_path / "fills.jsonl"
    # Build a normal fills stream first so the schema is realistic.
    b = Broker(cfg=BrokerConfig(entry_slippage_bps=0.0))
    intent = _activated(activation_kind="breakout",
                        entry=100.0, sl=99.0, tp1=102.0, tp2=104.0, qty=2.0)
    _, entry_fill = b.open_from_intent(intent, fill_price=100.0, fill_ts=T0)
    append_fill(fills_path, entry_fill)
    out = b.on_bar(_bar(ts_offset_min=15, o=101.5, h=104.5, l=101.0, c=104.2))
    _record_fills(out, fills_path)

    # Now rewrite the file stripping any "+00:00" / "Z" tz markers.
    raw_lines = fills_path.read_text(encoding="utf-8").splitlines()
    rewritten = []
    for line in raw_lines:
        rec = json.loads(line)
        rec["ts"] = str(rec["ts"]).replace("+00:00", "").replace("Z", "")
        rewritten.append(json.dumps(rec, sort_keys=True))
    fills_path.write_text("\n".join(rewritten) + "\n", encoding="utf-8")

    fills = read_fills(fills_path)
    state = replay_from_fills(fills, starting_equity_usd=10_000.0,
                              now=T0 + pd.Timedelta(hours=1))
    # Counters survive (would be 0 if cutoff comparison silently failed).
    assert state.closed_positions_24h == 1
    assert state.realized_pnl_usd == pytest.approx(6.0)
