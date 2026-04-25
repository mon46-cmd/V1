"""L2 orderbook state machine for Bybit v5 ``orderbook.{depth}.{symbol}``.

Bybit emits:
- a ``snapshot`` frame carrying full depth and an update-id ``u``;
- subsequent ``delta`` frames whose ``u`` must equal ``prev_u + 1``.

Price levels are [priceStr, sizeStr]; a size of "0" removes the level.

This module is pure-Python, synchronous, and has no I/O. It raises
``BookGap`` on an update-id discontinuity; the caller must then
resubscribe the topic (which forces Bybit to re-emit ``snapshot``) and
hand the new frames back in.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from downloader.errors import DownloaderError

log = logging.getLogger(__name__)


class BookGap(DownloaderError):
    """Raised when a delta's ``u`` does not equal ``prev_u + 1``.

    The book state is left untouched when this is raised. Call
    ``reset()`` and feed a fresh snapshot to recover.
    """

    def __init__(self, symbol: str, expected: int, got: int) -> None:
        super().__init__(f"orderbook gap {symbol}: expected u={expected} got u={got}")
        self.symbol = symbol
        self.expected = expected
        self.got = got


@dataclass(slots=True)
class BookStats:
    snapshots: int = 0
    updates: int = 0
    last_u: int = -1
    last_ts_ms: int = 0


@dataclass
class OrderBookL2:
    symbol: str
    bids: dict[float, float] = field(default_factory=dict)
    asks: dict[float, float] = field(default_factory=dict)
    stats: BookStats = field(default_factory=BookStats)

    # ---- lifecycle ---------------------------------------------------
    def reset(self) -> None:
        self.bids.clear()
        self.asks.clear()
        self.stats = BookStats()

    # ---- frame dispatch ----------------------------------------------
    def apply_frame(self, frame: dict[str, Any]) -> None:
        """Apply a Bybit v5 orderbook frame.

        Raises ``BookGap`` if a delta update-id does not match the
        expected ``prev_u + 1``. On ``BookGap`` the book state is NOT
        mutated; the caller is expected to call ``reset()`` and feed a
        fresh snapshot.
        """
        kind = frame.get("type", "")
        data = frame.get("data") or {}
        u = int(data.get("u", -1))
        ts_ms = int(frame.get("ts", 0))
        if kind == "snapshot":
            self.bids.clear()
            self.asks.clear()
            self._apply_levels(data.get("b") or [], self.bids)
            self._apply_levels(data.get("a") or [], self.asks)
            self.stats.snapshots += 1
            self.stats.last_u = u
            self.stats.last_ts_ms = ts_ms
            return
        if kind != "delta":
            log.debug("ignoring non-snapshot/delta frame type=%r", kind)
            return
        if self.stats.last_u >= 0 and u != self.stats.last_u + 1:
            raise BookGap(self.symbol, self.stats.last_u + 1, u)
        self._apply_levels(data.get("b") or [], self.bids)
        self._apply_levels(data.get("a") or [], self.asks)
        self.stats.updates += 1
        self.stats.last_u = u
        self.stats.last_ts_ms = ts_ms

    @staticmethod
    def _apply_levels(levels: list[list[str]], side: dict[float, float]) -> None:
        for p_str, q_str in levels:
            try:
                p = float(p_str)
                q = float(q_str)
            except (TypeError, ValueError):
                continue
            if q <= 0:
                side.pop(p, None)
            else:
                side[p] = q

    # ---- queries -----------------------------------------------------
    def top(self, n: int = 10) -> dict[str, list[tuple[float, float]]]:
        bids = sorted(self.bids.items(), key=lambda kv: kv[0], reverse=True)[:n]
        asks = sorted(self.asks.items(), key=lambda kv: kv[0])[:n]
        return {"bids": bids, "asks": asks}

    def best_bid(self) -> tuple[float, float] | None:
        if not self.bids:
            return None
        p = max(self.bids)
        return (p, self.bids[p])

    def best_ask(self) -> tuple[float, float] | None:
        if not self.asks:
            return None
        p = min(self.asks)
        return (p, self.asks[p])

    def mid(self) -> float | None:
        bb = self.best_bid()
        ba = self.best_ask()
        if bb is None or ba is None:
            return None
        return (bb[0] + ba[0]) / 2.0

    def spread_bps(self) -> float | None:
        bb = self.best_bid()
        ba = self.best_ask()
        if bb is None or ba is None or bb[0] <= 0 or ba[0] <= 0:
            return None
        mid = (bb[0] + ba[0]) / 2.0
        return (ba[0] - bb[0]) / mid * 10_000.0
