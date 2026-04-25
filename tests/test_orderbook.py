"""Offline tests for `downloader.orderbook.OrderBookL2`.

Feeds a synthesized snapshot + deltas and checks:
- top-of-book updates correctly after each delta;
- size "0" removes a level;
- gap in ``u`` raises ``BookGap`` and leaves state untouched;
- reset + fresh snapshot recovers cleanly.
"""
from __future__ import annotations

import pytest

from downloader.orderbook import BookGap, OrderBookL2


def _frame(kind: str, u: int, ts: int, bids: list[tuple[str, str]], asks: list[tuple[str, str]]):
    return {
        "topic": "orderbook.50.BTCUSDT",
        "type": kind,
        "ts": ts,
        "data": {
            "s": "BTCUSDT",
            "u": u,
            "b": [list(x) for x in bids],
            "a": [list(x) for x in asks],
        },
    }


def test_snapshot_sets_top_of_book():
    ob = OrderBookL2(symbol="BTCUSDT")
    ob.apply_frame(_frame(
        "snapshot", u=100, ts=1_700_000_000_000,
        bids=[("100.0", "1.0"), ("99.5", "2.0")],
        asks=[("100.5", "1.5"), ("101.0", "3.0")],
    ))
    assert ob.best_bid() == (100.0, 1.0)
    assert ob.best_ask() == (100.5, 1.5)
    assert ob.mid() == 100.25
    assert ob.stats.snapshots == 1
    assert ob.stats.last_u == 100


def test_delta_updates_and_removes():
    ob = OrderBookL2(symbol="BTCUSDT")
    ob.apply_frame(_frame(
        "snapshot", u=100, ts=1,
        bids=[("100.0", "1.0"), ("99.5", "2.0")],
        asks=[("100.5", "1.5")],
    ))
    # Bid level resize, ask level removed.
    ob.apply_frame(_frame("delta", u=101, ts=2,
        bids=[("100.0", "2.5")], asks=[("100.5", "0")]))
    assert ob.best_bid() == (100.0, 2.5)
    assert ob.best_ask() is None
    # New better ask added.
    ob.apply_frame(_frame("delta", u=102, ts=3,
        bids=[], asks=[("100.75", "0.5")]))
    assert ob.best_ask() == (100.75, 0.5)
    assert ob.stats.updates == 2
    assert ob.stats.last_u == 102


def test_many_deltas_keep_top_correct():
    ob = OrderBookL2(symbol="BTCUSDT")
    ob.apply_frame(_frame(
        "snapshot", u=0, ts=1,
        bids=[(f"{100.0 - 0.1*i:.2f}", "1.0") for i in range(10)],
        asks=[(f"{100.1 + 0.1*i:.2f}", "1.0") for i in range(10)],
    ))
    # Apply 50 sequential deltas with alternating bid/ask bumps.
    for i in range(1, 51):
        if i % 2 == 0:
            # Lift the best ask by pulling its size to zero and adding one above.
            bb = ob.best_ask()
            assert bb is not None
            p = bb[0]
            new_p = round(p + 0.01, 2)
            frame = _frame("delta", u=i, ts=i,
                bids=[], asks=[(f"{p:.2f}", "0"), (f"{new_p:.2f}", "1.0")])
        else:
            # Push best bid up.
            bb = ob.best_bid()
            assert bb is not None
            new_p = round(bb[0] + 0.01, 2)
            frame = _frame("delta", u=i, ts=i,
                bids=[(f"{new_p:.2f}", "1.0")], asks=[])
        ob.apply_frame(frame)
    assert ob.stats.updates == 50
    assert ob.stats.last_u == 50
    bb = ob.best_bid()
    ba = ob.best_ask()
    assert bb is not None and ba is not None
    assert ba[0] > bb[0]
    spread = ob.spread_bps()
    assert spread is not None and spread > 0


def test_gap_raises_book_gap_and_preserves_state():
    ob = OrderBookL2(symbol="BTCUSDT")
    ob.apply_frame(_frame(
        "snapshot", u=10, ts=1,
        bids=[("100.0", "1.0")], asks=[("100.5", "1.0")],
    ))
    ob.apply_frame(_frame("delta", u=11, ts=2,
        bids=[("100.0", "2.0")], asks=[]))
    assert ob.best_bid() == (100.0, 2.0)
    # Induce a gap: expected 12, send 15.
    with pytest.raises(BookGap) as info:
        ob.apply_frame(_frame("delta", u=15, ts=3,
            bids=[("101.0", "5.0")], asks=[]))
    assert info.value.expected == 12
    assert info.value.got == 15
    # State untouched by the bad delta.
    assert ob.best_bid() == (100.0, 2.0)
    assert ob.stats.last_u == 11


def test_reset_then_fresh_snapshot_recovers():
    ob = OrderBookL2(symbol="BTCUSDT")
    ob.apply_frame(_frame(
        "snapshot", u=10, ts=1,
        bids=[("100.0", "1.0")], asks=[("100.5", "1.0")],
    ))
    with pytest.raises(BookGap):
        ob.apply_frame(_frame("delta", u=99, ts=2, bids=[], asks=[]))
    ob.reset()
    assert ob.best_bid() is None and ob.best_ask() is None
    ob.apply_frame(_frame(
        "snapshot", u=500, ts=3,
        bids=[("200.0", "1.0")], asks=[("200.5", "1.0")],
    ))
    assert ob.best_bid() == (200.0, 1.0)
    assert ob.stats.last_u == 500


def test_unknown_frame_type_is_ignored():
    ob = OrderBookL2(symbol="BTCUSDT")
    ob.apply_frame(_frame(
        "snapshot", u=10, ts=1,
        bids=[("100.0", "1.0")], asks=[("100.5", "1.0")],
    ))
    before = (ob.best_bid(), ob.best_ask(), ob.stats.last_u)
    ob.apply_frame({"topic": "orderbook.50.BTCUSDT", "type": "weird", "ts": 4, "data": {}})
    assert (ob.best_bid(), ob.best_ask(), ob.stats.last_u) == before
