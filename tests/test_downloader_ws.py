"""LIVE Bybit WebSocket smoke test.

Subscribes to ``publicTrade.BTCUSDT`` for up to 30 s, asserts at least
10 frames arrive with monotonic ``recv_ts`` and well-formed topics.
Skipped when ``BYBIT_OFFLINE=1`` or ``stream.bybit.com:443`` is
unreachable.
"""
from __future__ import annotations

import asyncio
import os
import socket

import pytest


pytestmark = [
    pytest.mark.skipif(
        os.getenv("BYBIT_OFFLINE", "").lower() in ("1", "true", "yes"),
        reason="BYBIT_OFFLINE is set",
    ),
]


def _has_internet(host: str = "stream.bybit.com", port: int = 443, timeout: float = 3.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


if not _has_internet():  # pragma: no cover
    pytestmark.append(pytest.mark.skip(reason="stream.bybit.com unreachable"))


@pytest.mark.timeout(60)
async def test_ws_public_trade_stream(tmp_data_root) -> None:  # noqa: ARG001
    from core.config import load_config
    from downloader.constants import BYBIT_WS_LINEAR
    from downloader.ws import WsClient

    cfg = load_config()
    frames: list[tuple[str, int, int]] = []  # (topic, srv_ms, recv_ms)

    async with WsClient(cfg, BYBIT_WS_LINEAR) as ws:
        assert await ws.wait_connected(timeout=15.0)
        await ws.subscribe(["publicTrade.BTCUSDT"])

        async def collect() -> None:
            async for topic, _payload, srv, recv in ws.messages():
                frames.append((topic, srv, recv))
                if len(frames) >= 20:
                    return

        try:
            await asyncio.wait_for(collect(), timeout=30.0)
        except asyncio.TimeoutError:
            pass

    assert len(frames) >= 10, f"only {len(frames)} frames in 30s"
    assert all(t == "publicTrade.BTCUSDT" for t, _, _ in frames)
    recvs = [r for _, _, r in frames]
    # Monotonic non-decreasing recv timestamps.
    assert all(b >= a for a, b in zip(recvs, recvs[1:]))


@pytest.mark.timeout(60)
async def test_ws_subscribe_unsubscribe_resubscribe(tmp_data_root) -> None:  # noqa: ARG001
    """Sanity check control-plane ops against the real server."""
    from core.config import load_config
    from downloader.constants import BYBIT_WS_LINEAR
    from downloader.ws import WsClient

    cfg = load_config()
    async with WsClient(cfg, BYBIT_WS_LINEAR) as ws:
        assert await ws.wait_connected(timeout=15.0)
        await ws.subscribe(["publicTrade.BTCUSDT"])
        await asyncio.sleep(2.0)
        count_after_sub = ws.msg_count
        assert count_after_sub > 0
        await ws.unsubscribe(["publicTrade.BTCUSDT"])
        await ws.resubscribe(["publicTrade.BTCUSDT"])
        await asyncio.sleep(2.0)
        assert ws.msg_count >= count_after_sub
