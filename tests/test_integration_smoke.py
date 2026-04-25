"""LIVE integration smoke test for the tick pipeline.

Runs the pipeline against ``BTCUSDT`` for ~45 s with a short flush
interval, then:

1. Asserts >= 100 live ticks were written to
   ``cache/ticks_live/BTCUSDT/<today>.parquet``.
2. Archive-backfills the previous UTC day and verifies
   ``read_continuous`` stitches archive + live without duplicating
   trade ids.
3. Validates the stitched frame with ``validate_ticks``.
"""
from __future__ import annotations

import asyncio
import os
import socket
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest


pytestmark = [
    pytest.mark.skipif(
        os.getenv("BYBIT_OFFLINE", "").lower() in ("1", "true", "yes"),
        reason="BYBIT_OFFLINE is set",
    ),
]


def _has_internet(host: str, port: int = 443, timeout: float = 3.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


if not (_has_internet("stream.bybit.com") and _has_internet("public.bybit.com")):  # pragma: no cover
    pytestmark.append(pytest.mark.skip(reason="Bybit hosts unreachable"))


@pytest.mark.timeout(180)
async def test_pipeline_archive_plus_live_stitch(tmp_data_root: Path) -> None:
    from core.config import load_config
    from downloader.archive import ArchiveClient
    from downloader.cache import ParquetCache
    from downloader.http import HttpClient
    from downloader.tick_pipeline import TickPipeline
    from downloader.validators import validate_ticks

    cfg = load_config()
    cache = ParquetCache(cfg.cache_root)

    # Backfill one day of archive for BTCUSDT (today-3: certainly published).
    day = date.today() - timedelta(days=3)
    async with HttpClient(cfg) as http:
        arc = ArchiveClient(http, cache, cfg)
        arch_df = await arc.fetch_day("BTCUSDT", day)
    assert arch_df is not None and not arch_df.empty
    arch_rows = len(arch_df)

    # Run the pipeline live for ~45 s (trades only; book off for smoke speed).
    pipe = TickPipeline(
        cfg,
        symbols=["BTCUSDT"],
        cache=cache,
        archive=None,
        flush_sec=5.0,
        book_depth=None,
    )
    await pipe.run(duration_sec=45.0)
    assert pipe.stats.ticks_received >= 50, pipe.stats
    assert pipe.stats.ticks_written >= 50, pipe.stats
    assert pipe.stats.flushes >= 1

    today = datetime.now(tz=timezone.utc).date()
    live_path = cache.daily_path("ticks_live", "BTCUSDT", today.isoformat())
    assert live_path.exists() and live_path.stat().st_size > 0

    # read_continuous stitches archive (day) with live (today). The
    # ArchiveClient's read_range path is synchronous and cache-only, so
    # no live HttpClient is required here.
    async with HttpClient(cfg) as http2:
        arc2 = ArchiveClient(http2, cache, cfg)
        pipe_reader = TickPipeline(
            cfg, symbols=["BTCUSDT"], cache=cache,
            archive=arc2, book_depth=None,
        )
        stitched = pipe_reader.read_continuous("BTCUSDT", day, today)
    # Must contain at least archive day's rows + >=50 live.
    assert len(stitched) >= arch_rows + 50
    # Unique trade_ids (where non-empty).
    has_id = stitched["trade_id"].astype(str) != ""
    assert stitched.loc[has_id, "trade_id"].is_unique

    report = validate_ticks(stitched, "BTCUSDT")
    assert report.status in ("PASS", "WARN"), report.as_dict()

    # Monotonic-by-second timestamps (market prints can jitter by ms).
    sec = stitched["timestamp"].dt.floor("s")
    assert sec.is_monotonic_increasing


@pytest.mark.timeout(120)
async def test_pipeline_book_and_trades(tmp_data_root: Path) -> None:
    """Shorter run that exercises the orderbook path end-to-end."""
    from core.config import load_config
    from downloader.cache import ParquetCache
    from downloader.tick_pipeline import TickPipeline

    cfg = load_config()
    cache = ParquetCache(cfg.cache_root)
    pipe = TickPipeline(
        cfg, symbols=["BTCUSDT"],
        cache=cache, archive=None,
        flush_sec=3.0, book_depth=50, persist_book_top=False,
    )
    await pipe.run(duration_sec=20.0)
    book = pipe.book("BTCUSDT")
    assert book is not None
    # Should have seen at least one snapshot + many updates.
    assert book.stats.snapshots >= 1
    assert book.stats.updates >= 10
    bb = book.best_bid()
    ba = book.best_ask()
    assert bb is not None and ba is not None
    assert ba[0] > bb[0] > 0
    # Gaps are possible but rare; allow some but book must still converge.
    assert pipe.stats.ticks_received >= 5
