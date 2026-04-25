"""Stitch daily tick archive and live WS trades into a single stream.

The archive covers ``[listing, today-1d]``; the live websocket covers
``[now_today, +inf)``. The pipeline:

1. Subscribes ``publicTrade.<SYM>`` (and optionally
   ``orderbook.<depth>.<SYM>``) via :class:`downloader.ws.WsClient`.
2. Buffers incoming trades in per-symbol, per-UTC-day buckets.
3. Flushes buckets to ``<cache_root>/ticks_live/<SYM>/<YYYY-MM-DD>.parquet``
   every ``flush_sec`` seconds (or on UTC-day rollover), dedup'd on
   ``trade_id`` and merged with any existing file atomically.
4. Optional order book state via :class:`downloader.orderbook.OrderBookL2`.
   On ``BookGap`` the topic is resubscribed and the book is reset; Bybit
   re-emits a snapshot which is applied cleanly.
5. ``read_continuous(sym, start, end)`` returns the union of archive
   + live ticks for the requested UTC-date range as a single
   dedup'd DataFrame sorted by timestamp.
"""
from __future__ import annotations

import asyncio
import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any

import pandas as pd

from core.config import Config
from downloader.archive import ArchiveClient
from downloader.cache import ParquetCache
from downloader.constants import (
    BYBIT_WS_LINEAR,
    CACHE_KIND_BOOK_TOP,
    CACHE_KIND_TICKS_LIVE,
    COLS_TICK,
)
from downloader.orderbook import BookGap, OrderBookL2
from downloader.ws import WsClient

log = logging.getLogger(__name__)


def _utc_today() -> date:
    return datetime.now(tz=timezone.utc).date()


@dataclass(slots=True)
class _LiveBucket:
    day: date
    rows: list[dict[str, Any]] = field(default_factory=list)
    seen_ids: set[str] = field(default_factory=set)


@dataclass(slots=True)
class TickPipelineStats:
    ticks_received: int = 0
    ticks_written: int = 0
    duplicates_dropped: int = 0
    book_updates: int = 0
    book_gaps: int = 0
    flushes: int = 0


class TickPipeline:
    """Stitch archive + live ticks into a gap-free per-day parquet record."""

    def __init__(
        self,
        cfg: Config,
        symbols: list[str],
        *,
        cache: ParquetCache,
        archive: ArchiveClient | None = None,
        ws_url: str = BYBIT_WS_LINEAR,
        flush_sec: float = 10.0,
        book_depth: int | None = 50,
        persist_book_top: bool = False,
    ) -> None:
        self._cfg = cfg
        self._symbols = [s.upper() for s in symbols]
        self._cache = cache
        self._archive = archive
        self._flush_sec = float(flush_sec)
        self._book_depth = book_depth
        self._persist_book_top = persist_book_top
        self._buckets: dict[str, _LiveBucket] = {
            s: _LiveBucket(day=_utc_today()) for s in self._symbols
        }
        self._books: dict[str, OrderBookL2] = (
            {s: OrderBookL2(symbol=s) for s in self._symbols} if book_depth else {}
        )
        self._book_snapshots: dict[str, deque[dict[str, Any]]] = defaultdict(
            lambda: deque(maxlen=10_000),
        )
        self._ws = WsClient(cfg, ws_url)
        self._stop = asyncio.Event()
        self._lock = asyncio.Lock()
        self._ws_url = ws_url
        self.stats = TickPipelineStats()

    # ---- lifecycle ---------------------------------------------------
    async def run(self, duration_sec: float | None = None) -> None:
        topics = self._build_topics()
        await self._ws.__aenter__()
        try:
            if not await self._ws.wait_connected(timeout=15.0):
                raise RuntimeError("ws did not connect within 15s")
            await self._ws.subscribe(topics)
            log.info("tick pipeline live: %d symbols, %d topics", len(self._symbols), len(topics))

            flusher = asyncio.create_task(self._flush_loop(), name="tick-flusher")
            consumer = asyncio.create_task(self._consumer_loop(), name="tick-consumer")
            end_at = None if duration_sec is None else asyncio.get_event_loop().time() + duration_sec
            try:
                while not self._stop.is_set():
                    if end_at is not None and asyncio.get_event_loop().time() >= end_at:
                        break
                    await asyncio.sleep(0.25)
            finally:
                for t in (flusher, consumer):
                    t.cancel()
                for t in (flusher, consumer):
                    try:
                        await t
                    except (asyncio.CancelledError, Exception):  # noqa: BLE001
                        pass
                await self._flush_all()
        finally:
            await self._ws.__aexit__(None, None, None)

    async def stop(self) -> None:
        self._stop.set()

    def book(self, symbol: str) -> OrderBookL2 | None:
        return self._books.get(symbol.upper())

    # ---- archive backfill -------------------------------------------
    async def backfill_archive(
        self,
        days: int = 7,
        *,
        overwrite: bool = False,
    ) -> dict[str, dict[str, int]]:
        if self._archive is None:
            raise RuntimeError("no ArchiveClient supplied")
        end = _utc_today() - timedelta(days=1)
        start = end - timedelta(days=days - 1)
        summary: dict[str, dict[str, int]] = {}
        for s in self._symbols:
            summary[s] = await self._archive.fetch_range(s, start, end, overwrite=overwrite)
        return summary

    # ---- internals ---------------------------------------------------
    def _build_topics(self) -> list[str]:
        topics: list[str] = []
        for s in self._symbols:
            topics.append(f"publicTrade.{s}")
            if self._book_depth:
                topics.append(f"orderbook.{self._book_depth}.{s}")
        return topics

    async def _consumer_loop(self) -> None:
        async for topic, payload, _srv_ms, _recv_ms in self._ws.messages():
            try:
                if topic.startswith("publicTrade."):
                    self._apply_trade(topic, payload)
                elif topic.startswith("orderbook."):
                    await self._apply_book(topic, payload)
            except Exception as exc:  # noqa: BLE001
                log.exception("ws consumer error on %s: %s", topic, exc)

    def _apply_trade(self, topic: str, frame: dict[str, Any]) -> None:
        symbol = topic.split(".", 1)[1].upper()
        bucket = self._buckets.get(symbol)
        if bucket is None:
            return
        data = frame.get("data") or []
        now_day = _utc_today()
        if now_day != bucket.day:
            self._rollover_sync(symbol, bucket, now_day)
            bucket = self._buckets[symbol]
        for t in data:
            try:
                tid = str(t.get("i", ""))
                if tid and tid in bucket.seen_ids:
                    self.stats.duplicates_dropped += 1
                    continue
                row = {
                    "timestamp": pd.Timestamp(int(t["T"]), unit="ms", tz="UTC"),
                    "symbol": symbol,
                    "side": str(t["S"]),
                    "size": float(t["v"]),
                    "price": float(t["p"]),
                    "trade_id": tid,
                }
            except (KeyError, ValueError, TypeError) as exc:
                log.debug("bad trade on %s: %s", topic, exc)
                continue
            bucket.rows.append(row)
            if tid:
                bucket.seen_ids.add(tid)
            self.stats.ticks_received += 1

    async def _apply_book(self, topic: str, frame: dict[str, Any]) -> None:
        parts = topic.split(".")
        if len(parts) != 3:
            return
        symbol = parts[2].upper()
        book = self._books.get(symbol)
        if book is None:
            return
        try:
            book.apply_frame(frame)
            self.stats.book_updates += 1
        except BookGap:
            self.stats.book_gaps += 1
            log.warning("book gap on %s; resetting and resubscribing", symbol)
            book.reset()
            await self._ws.resubscribe([topic])
            return
        if self._persist_book_top:
            bb = book.best_bid()
            ba = book.best_ask()
            if bb and ba:
                mid = (bb[0] + ba[0]) / 2.0
                self._book_snapshots[symbol].append({
                    "timestamp": pd.Timestamp(int(frame.get("ts", 0)), unit="ms", tz="UTC"),
                    "symbol": symbol,
                    "bid": bb[0], "bid_size": bb[1],
                    "ask": ba[0], "ask_size": ba[1],
                    "mid": mid,
                    "spread_bps": (ba[0] - bb[0]) / mid * 10_000.0,
                    "u": book.stats.last_u,
                })

    async def _flush_loop(self) -> None:
        while not self._stop.is_set():
            try:
                await asyncio.sleep(self._flush_sec)
            except asyncio.CancelledError:
                break
            await self._flush_all()

    async def _flush_all(self) -> None:
        async with self._lock:
            for symbol, bucket in self._buckets.items():
                if not bucket.rows:
                    continue
                self._persist_bucket(symbol, bucket)
                bucket.rows.clear()
            if self._persist_book_top:
                self._persist_book_snapshots()
            self.stats.flushes += 1

    def _persist_bucket(self, symbol: str, bucket: _LiveBucket) -> None:
        df_new = pd.DataFrame(bucket.rows, columns=list(COLS_TICK))
        if df_new.empty:
            return
        date_str = bucket.day.isoformat()
        existing = self._cache.read_daily(CACHE_KIND_TICKS_LIVE, symbol, date_str)
        if existing is None or existing.empty:
            merged = df_new
        else:
            merged = _dedup_ticks(pd.concat([existing, df_new], ignore_index=True))
        self._cache.write_daily(merged, CACHE_KIND_TICKS_LIVE, symbol, date_str)
        self.stats.ticks_written += len(df_new)

    def _persist_book_snapshots(self) -> None:
        for symbol, dq in self._book_snapshots.items():
            if not dq:
                continue
            rows = list(dq)
            dq.clear()
            df_new = pd.DataFrame(rows)
            if df_new.empty:
                continue
            day = _utc_today().isoformat()
            existing = self._cache.read_daily(CACHE_KIND_BOOK_TOP, symbol, day)
            if existing is None or existing.empty:
                merged = df_new
            else:
                merged = (
                    pd.concat([existing, df_new], ignore_index=True)
                    .sort_values("timestamp")
                    .reset_index(drop=True)
                )
            self._cache.write_daily(merged, CACHE_KIND_BOOK_TOP, symbol, day)

    def _rollover_sync(self, symbol: str, bucket: _LiveBucket, new_day: date) -> None:
        if bucket.rows:
            self._persist_bucket(symbol, bucket)
            bucket.rows.clear()
        self._buckets[symbol] = _LiveBucket(day=new_day)

    # ---- read helpers -----------------------------------------------
    def read_live_range(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        parts: list[pd.DataFrame] = []
        cur = start
        while cur <= end:
            df = self._cache.read_daily(CACHE_KIND_TICKS_LIVE, symbol, cur.isoformat())
            if df is not None:
                parts.append(df)
            cur += timedelta(days=1)
        if not parts:
            return pd.DataFrame(columns=list(COLS_TICK))
        return _dedup_ticks(pd.concat(parts, ignore_index=True))

    def read_continuous(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        """Union of archive + live ticks for ``[start, end]`` (UTC dates)."""
        parts: list[pd.DataFrame] = []
        if self._archive is not None:
            parts.append(self._archive.read_range(symbol, start, end))
        parts.append(self.read_live_range(symbol, start, end))
        non_empty = [p for p in parts if p is not None and not p.empty]
        if not non_empty:
            return pd.DataFrame(columns=list(COLS_TICK))
        return _dedup_ticks(pd.concat(non_empty, ignore_index=True))


def _dedup_ticks(df: pd.DataFrame) -> pd.DataFrame:
    """Dedup ticks on ``trade_id`` when available, else on the composite
    (timestamp_ms, side, size, price) key so that blank-id rows from
    unrelated prints are not collapsed together."""
    if df.empty:
        return df
    df = df.copy()
    has_id = df["trade_id"].astype(str) != ""
    with_id = df[has_id].drop_duplicates(subset=["trade_id"], keep="last")
    no_id = df[~has_id].copy()
    if not no_id.empty:
        ts_ms = (no_id["timestamp"].astype("int64") // 1_000_000).astype("int64")
        no_id["_k"] = (
            ts_ms.astype(str) + "|" + no_id["side"].astype(str)
            + "|" + no_id["size"].astype(str) + "|" + no_id["price"].astype(str)
        )
        no_id = no_id.drop_duplicates(subset=["_k"], keep="last").drop(columns=["_k"])
    out = pd.concat([with_id, no_id], ignore_index=True)
    return out.sort_values("timestamp").reset_index(drop=True)
