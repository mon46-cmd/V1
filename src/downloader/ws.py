"""Bybit v5 public WebSocket client.

Single async connection to one public endpoint (e.g.
``wss://stream.bybit.com/v5/public/linear``) with:

- auto-reconnect and jittered exponential backoff
- heartbeat ping frames every ``WS_PING_INTERVAL_SEC``
- subscription bookkeeping so reconnects replay the topic set

Two consumption modes are supported:

1. ``async for (topic, payload, server_ts_ms, recv_ts_ms) in ws.messages():``
   single linear stream of data frames (op/pong frames filtered out).
2. ``queue = ws.queue_for(topic)`` per-topic fan-out into an
   ``asyncio.Queue``; useful when different consumers want different
   topics concurrently.

Under the hood both modes are fed from one background reader task so
the WS is read exactly once.
"""
from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from types import TracebackType
from typing import Any, AsyncIterator

import websockets
from websockets.asyncio.client import ClientConnection

from core.config import Config
from downloader.constants import (
    WS_PING_INTERVAL_SEC,
    WS_QUEUE_MAX,
    WS_RECONNECT_BASE_SEC,
    WS_RECONNECT_MAX_SEC,
    WS_SUBSCRIBE_BATCH,
)

log = logging.getLogger(__name__)


def _now_ms() -> int:
    return int(time.time() * 1000)


class WsClient:
    """Async Bybit public WS client.

    Usage::

        async with WsClient(cfg, url) as ws:
            await ws.subscribe(["publicTrade.BTCUSDT"])
            async for topic, payload, srv_ms, recv_ms in ws.messages():
                ...
    """

    def __init__(
        self,
        cfg: Config,  # noqa: ARG002  - reserved for future knobs
        url: str,
        *,
        queue_maxsize: int = WS_QUEUE_MAX,
    ) -> None:
        self._url = url
        self._topics: set[str] = set()
        self._main_q: asyncio.Queue[tuple[str, dict[str, Any], int, int]] = asyncio.Queue(
            maxsize=queue_maxsize,
        )
        self._topic_qs: dict[str, asyncio.Queue[tuple[str, dict[str, Any], int, int]]] = {}
        self._stop = asyncio.Event()
        self._reader_task: asyncio.Task[None] | None = None
        self._ws: ClientConnection | None = None
        self._connected = asyncio.Event()
        # public counters
        self.msg_count = 0
        self.reconnect_count = 0
        self.dropped = 0
        self.last_recv_ms = 0

    # ---- context ----------------------------------------------------
    async def __aenter__(self) -> "WsClient":
        self._reader_task = asyncio.create_task(self._reader_loop(), name="ws-reader")
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.stop()

    # ---- public api -------------------------------------------------
    async def subscribe(self, topics: list[str]) -> None:
        new = [t for t in topics if t not in self._topics]
        self._topics.update(new)
        if self._ws is not None and self._connected.is_set() and new:
            for i in range(0, len(new), WS_SUBSCRIBE_BATCH):
                batch = new[i : i + WS_SUBSCRIBE_BATCH]
                await self._ws.send(json.dumps({"op": "subscribe", "args": batch}))

    async def unsubscribe(self, topics: list[str]) -> None:
        drop = [t for t in topics if t in self._topics]
        for t in drop:
            self._topics.discard(t)
        if self._ws is not None and self._connected.is_set() and drop:
            for i in range(0, len(drop), WS_SUBSCRIBE_BATCH):
                batch = drop[i : i + WS_SUBSCRIBE_BATCH]
                await self._ws.send(json.dumps({"op": "unsubscribe", "args": batch}))

    async def resubscribe(self, topics: list[str]) -> None:
        """Drop then re-add topics so Bybit re-emits a snapshot.

        Used by the orderbook path after a ``BookGap``.
        """
        if not topics or self._ws is None or not self._connected.is_set():
            return
        try:
            for i in range(0, len(topics), WS_SUBSCRIBE_BATCH):
                batch = topics[i : i + WS_SUBSCRIBE_BATCH]
                await self._ws.send(json.dumps({"op": "unsubscribe", "args": batch}))
            for i in range(0, len(topics), WS_SUBSCRIBE_BATCH):
                batch = topics[i : i + WS_SUBSCRIBE_BATCH]
                await self._ws.send(json.dumps({"op": "subscribe", "args": batch}))
        except Exception as exc:  # noqa: BLE001
            log.warning("resubscribe failed for %d topics: %s", len(topics), exc)

    def queue_for(self, topic: str) -> asyncio.Queue[tuple[str, dict[str, Any], int, int]]:
        """Return (creating if needed) a per-topic fan-out queue.

        A topic queue only receives frames whose ``topic`` field matches
        ``topic`` exactly. Messages are pushed to both the main stream
        and the per-topic queue.
        """
        q = self._topic_qs.get(topic)
        if q is None:
            q = asyncio.Queue(maxsize=WS_QUEUE_MAX)
            self._topic_qs[topic] = q
        return q

    async def messages(self) -> AsyncIterator[tuple[str, dict[str, Any], int, int]]:
        """Yield ``(topic, payload, server_ts_ms, recv_ts_ms)`` forever.

        Stops when ``stop()`` is called.
        """
        while not self._stop.is_set():
            try:
                item = await asyncio.wait_for(self._main_q.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            yield item

    async def wait_connected(self, timeout: float = 15.0) -> bool:
        try:
            await asyncio.wait_for(self._connected.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False

    async def stop(self) -> None:
        self._stop.set()
        if self._reader_task is not None:
            self._reader_task.cancel()
            try:
                await self._reader_task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
            self._reader_task = None
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:  # noqa: BLE001
                pass

    # ---- internals --------------------------------------------------
    async def _reader_loop(self) -> None:
        attempt = 0
        while not self._stop.is_set():
            try:
                async with websockets.connect(
                    self._url,
                    ping_interval=None,
                    close_timeout=5,
                    max_size=8 * 1024 * 1024,
                ) as ws:
                    self._ws = ws
                    self._connected.set()
                    attempt = 0
                    log.info("ws connected url=%s topics=%d", self._url, len(self._topics))
                    await self._subscribe_all()
                    ping_task = asyncio.create_task(self._ping_loop(ws), name="ws-ping")
                    try:
                        async for raw in ws:
                            self.msg_count += 1
                            recv_ms = _now_ms()
                            self.last_recv_ms = recv_ms
                            try:
                                frame = json.loads(raw)
                            except json.JSONDecodeError as exc:
                                log.warning("ws malformed json: %s", exc)
                                continue
                            self._dispatch(frame, recv_ms)
                    finally:
                        ping_task.cancel()
                        try:
                            await ping_task
                        except (asyncio.CancelledError, Exception):  # noqa: BLE001
                            pass
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                log.warning("ws dropped: %s", exc)
            finally:
                self._connected.clear()
                self._ws = None
            if self._stop.is_set():
                break
            self.reconnect_count += 1
            delay = min(WS_RECONNECT_MAX_SEC, WS_RECONNECT_BASE_SEC * (2 ** attempt))
            delay = random.uniform(delay / 2, delay)
            attempt = min(attempt + 1, 6)
            log.info("ws reconnect in %.1fs (attempt %d)", delay, self.reconnect_count)
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=delay)
                break
            except asyncio.TimeoutError:
                pass

    async def _ping_loop(self, ws: ClientConnection) -> None:
        try:
            while True:
                await asyncio.sleep(WS_PING_INTERVAL_SEC)
                await ws.send(json.dumps({"op": "ping"}))
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            log.debug("ping loop ended: %s", exc)

    async def _subscribe_all(self) -> None:
        if not self._topics or self._ws is None:
            return
        topics = sorted(self._topics)
        for i in range(0, len(topics), WS_SUBSCRIBE_BATCH):
            batch = topics[i : i + WS_SUBSCRIBE_BATCH]
            await self._ws.send(json.dumps({"op": "subscribe", "args": batch}))

    def _dispatch(self, frame: dict[str, Any], recv_ms: int) -> None:
        # Control frames (ack for subscribe/pong) have ``op`` but no ``topic``.
        topic = frame.get("topic")
        if not topic:
            return
        server_ms = int(frame.get("ts", 0) or 0)
        item = (topic, frame, server_ms, recv_ms)
        self._safe_put(self._main_q, item)
        q = self._topic_qs.get(topic)
        if q is not None:
            self._safe_put(q, item)

    def _safe_put(
        self,
        q: asyncio.Queue[tuple[str, dict[str, Any], int, int]],
        item: tuple[str, dict[str, Any], int, int],
    ) -> None:
        try:
            q.put_nowait(item)
        except asyncio.QueueFull:
            self.dropped += 1
            try:
                q.get_nowait()
            except asyncio.QueueEmpty:
                pass
            try:
                q.put_nowait(item)
            except asyncio.QueueFull:
                pass
