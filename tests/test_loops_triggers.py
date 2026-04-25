"""Offline tests for the trigger gate (Phase 8)."""
from __future__ import annotations

import pandas as pd
import pytest

from core.config import load_config
from loops.cooldowns import CooldownEntry, CooldownStore
from loops.triggers import (
    DEC_BYPASS_MOVE,
    DEC_COOLDOWN_ACTIVE,
    DEC_DUP_BAR,
    DEC_FRESH,
    DEC_NO_BAR,
    DEC_NO_FLAG,
    DEC_POST_COOLDOWN,
    CooldownState,
    detect_trigger,
)


def _bar(ts: str, close: float = 100.0, atr_pct: float = 1.0,
         flag: str | None = "flag_volume_climax", **extra) -> dict:
    row = {
        "timestamp": pd.Timestamp(ts, tz="UTC"),
        "close": close,
        "atr_14_pct": atr_pct,
        "flag_volume_climax": 0,
        "flag_sweep_up": 0,
        "flag_sweep_dn": 0,
    }
    if flag is not None:
        row[flag] = 1
    row.update(extra)
    return row


# ---- branch tests ----------------------------------------------------

def test_no_bar_when_bar_is_none(tmp_data_root):
    cfg = load_config()
    d = detect_trigger(symbol="BTCUSDT", bar=None, state=CooldownState(), cfg=cfg)
    assert d.decision == DEC_NO_BAR
    assert not d.fired


def test_no_bar_when_missing_timestamp(tmp_data_root):
    cfg = load_config()
    bar = _bar("2026-04-25T12:30:00Z")
    bar.pop("timestamp")
    d = detect_trigger(symbol="BTCUSDT", bar=bar, state=CooldownState(), cfg=cfg)
    assert d.decision == DEC_NO_BAR


def test_no_flag(tmp_data_root):
    cfg = load_config()
    bar = _bar("2026-04-25T12:30:00Z", flag=None)
    d = detect_trigger(symbol="BTCUSDT", bar=bar, state=CooldownState(), cfg=cfg)
    assert d.decision == DEC_NO_FLAG
    assert not d.fired


def test_fresh(tmp_data_root):
    cfg = load_config()
    bar = _bar("2026-04-25T12:30:00Z", close=100.0, flag="flag_sweep_up")
    d = detect_trigger(symbol="BTCUSDT", bar=bar, state=CooldownState(), cfg=cfg)
    assert d.decision == DEC_FRESH
    assert d.fired
    assert d.flag == "flag_sweep_up"
    assert d.close == 100.0


def test_dup_bar(tmp_data_root):
    cfg = load_config()
    ts = pd.Timestamp("2026-04-25T12:30:00Z")
    bar = _bar(ts.isoformat())
    state = CooldownState(last_bar_ts=ts, last_close=100.0, bars_since=0)
    d = detect_trigger(symbol="BTCUSDT", bar=bar, state=state, cfg=cfg)
    assert d.decision == DEC_DUP_BAR
    assert not d.fired


def test_post_cooldown(tmp_data_root):
    cfg = load_config()
    bar = _bar("2026-04-25T13:30:00Z", close=100.5)
    state = CooldownState(
        last_bar_ts=pd.Timestamp("2026-04-25T12:30:00Z"),
        last_close=100.0,
        bars_since=cfg.prompt_cooldown_candles,  # exactly elapsed
    )
    d = detect_trigger(symbol="BTCUSDT", bar=bar, state=state, cfg=cfg)
    assert d.decision == DEC_POST_COOLDOWN
    assert d.fired


def test_cooldown_active(tmp_data_root):
    cfg = load_config()
    # Tiny price move (<1bp), atr_pct small => below threshold, bars_since=1.
    bar = _bar("2026-04-25T12:45:00Z", close=100.001, atr_pct=0.5)
    state = CooldownState(
        last_bar_ts=pd.Timestamp("2026-04-25T12:30:00Z"),
        last_close=100.0,
        bars_since=1,
    )
    d = detect_trigger(symbol="BTCUSDT", bar=bar, state=state, cfg=cfg)
    assert d.decision == DEC_COOLDOWN_ACTIVE
    assert not d.fired
    assert d.threshold_pct is not None
    assert d.move_pct is not None and d.move_pct < d.threshold_pct


def test_bypass_move(tmp_data_root):
    cfg = load_config()
    # Price moved 5% with bars_since=1; threshold is max(0.8 * atr%, 0.01)
    # = max(0.008, 0.01) = 0.01 < 0.05 => bypass.
    bar = _bar("2026-04-25T12:45:00Z", close=105.0, atr_pct=1.0)
    state = CooldownState(
        last_bar_ts=pd.Timestamp("2026-04-25T12:30:00Z"),
        last_close=100.0,
        bars_since=1,
    )
    d = detect_trigger(symbol="BTCUSDT", bar=bar, state=state, cfg=cfg)
    assert d.decision == DEC_BYPASS_MOVE
    assert d.fired
    assert d.move_pct is not None and d.move_pct >= d.threshold_pct


# ---- CooldownStore ---------------------------------------------------

def test_cooldown_store_roundtrip(tmp_data_root):
    path = tmp_data_root / "cooldowns.json"
    store = CooldownStore.load(path)
    assert len(store) == 0

    ts = pd.Timestamp("2026-04-25T12:30:00Z")
    store._data["BTCUSDT"] = CooldownEntry(  # noqa: SLF001
        last_bar_ts=ts, last_close=65000.0, last_flag="flag_volume_climax",
    )
    store.save()
    assert path.exists()

    reloaded = CooldownStore.load(path)
    assert "BTCUSDT" in reloaded
    e = reloaded._data["BTCUSDT"]  # noqa: SLF001
    assert e.last_close == 65000.0
    assert e.last_flag == "flag_volume_climax"
    assert e.last_bar_ts == ts


def test_cooldown_store_state_for_computes_bars_since(tmp_data_root):
    path = tmp_data_root / "cooldowns.json"
    store = CooldownStore.load(path)
    ts0 = pd.Timestamp("2026-04-25T12:30:00Z")
    store._data["BTCUSDT"] = CooldownEntry(  # noqa: SLF001
        last_bar_ts=ts0, last_close=100.0, last_flag="flag_sweep_up",
    )
    state = store.state_for("BTCUSDT",
                            now_bar_ts=pd.Timestamp("2026-04-25T13:30:00Z"))
    assert state.last_close == 100.0
    assert state.bars_since == 4  # 60min / 15min


def test_cooldown_store_unknown_symbol_returns_empty_state(tmp_data_root):
    store = CooldownStore.load(tmp_data_root / "cooldowns.json")
    state = store.state_for("XYZUSDT")
    assert state.last_bar_ts is None
    assert state.last_close is None


# ---- integration on hand-crafted frame ------------------------------

def test_sequence_of_decisions(tmp_data_root):
    """Simulate 6 consecutive 15m bars and verify the full ledger."""
    cfg = load_config()
    store = CooldownStore.load(tmp_data_root / "cooldowns.json")
    ts0 = pd.Timestamp("2026-04-25T12:00:00Z")

    # Bar 0: flag fires from fresh -> FRESH.
    bar0 = _bar((ts0 + pd.Timedelta(minutes=0)).isoformat(), close=100.0)
    s0 = store.state_for("X", now_bar_ts=bar0["timestamp"])
    d0 = detect_trigger(symbol="X", bar=bar0, state=s0, cfg=cfg)
    assert d0.decision == DEC_FRESH
    store.record(d0)

    # Bar 1: same ts replayed -> DUP_BAR.
    s1 = store.state_for("X", now_bar_ts=bar0["timestamp"])
    d1 = detect_trigger(symbol="X", bar=bar0, state=s1, cfg=cfg)
    assert d1.decision == DEC_DUP_BAR

    # Bar 2: +15min, no flag -> NO_FLAG.
    bar2 = _bar((ts0 + pd.Timedelta(minutes=15)).isoformat(),
                close=100.05, flag=None)
    s2 = store.state_for("X", now_bar_ts=bar2["timestamp"])
    d2 = detect_trigger(symbol="X", bar=bar2, state=s2, cfg=cfg)
    assert d2.decision == DEC_NO_FLAG

    # Bar 3: +15min, tiny move flag fires -> COOLDOWN_ACTIVE
    # (bars_since=1 < cooldown=3, move below threshold).
    bar3 = _bar((ts0 + pd.Timedelta(minutes=15)).isoformat(),
                close=100.05, atr_pct=0.5)
    s3 = store.state_for("X", now_bar_ts=bar3["timestamp"])
    d3 = detect_trigger(symbol="X", bar=bar3, state=s3, cfg=cfg)
    assert d3.decision == DEC_COOLDOWN_ACTIVE

    # Bar 4: +30min, big move within cooldown -> BYPASS_MOVE
    # (bars_since=2 < cooldown=3, but +10% move exceeds threshold).
    bar4 = _bar((ts0 + pd.Timedelta(minutes=30)).isoformat(),
                close=110.0, atr_pct=1.0)
    s4 = store.state_for("X", now_bar_ts=bar4["timestamp"])
    d4 = detect_trigger(symbol="X", bar=bar4, state=s4, cfg=cfg)
    assert d4.decision == DEC_BYPASS_MOVE
    store.record(d4)

    # Bar 5: cooldown after bar4 fully elapsed -> POST_COOLDOWN.
    bar5 = _bar((ts0 + pd.Timedelta(minutes=30 + 15 * cfg.prompt_cooldown_candles)).isoformat(),
                close=110.5, atr_pct=1.0)
    s5 = store.state_for("X", now_bar_ts=bar5["timestamp"])
    d5 = detect_trigger(symbol="X", bar=bar5, state=s5, cfg=cfg)
    assert d5.decision == DEC_POST_COOLDOWN
