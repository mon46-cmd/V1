"""LIVE tests for the public daily tick archive.

Downloads exactly one day of BTCUSDT trade prints and verifies the
normalization pipeline. Skipped when ``BYBIT_OFFLINE=1`` or when
``public.bybit.com`` is unreachable.
"""
from __future__ import annotations

import os
import socket
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest


pytestmark = [
    pytest.mark.skipif(
        os.getenv("BYBIT_OFFLINE", "").lower() in ("1", "true", "yes"),
        reason="BYBIT_OFFLINE is set",
    ),
]


def _has_internet(host: str = "public.bybit.com", port: int = 443, timeout: float = 3.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


if not _has_internet():  # pragma: no cover
    pytestmark.append(pytest.mark.skip(reason="public.bybit.com unreachable"))


@pytest.mark.timeout(180)
async def test_archive_fetch_one_day(tmp_data_root: Path) -> None:
    from core.config import load_config
    from downloader.archive import ArchiveClient
    from downloader.cache import ParquetCache
    from downloader.http import HttpClient
    from downloader.validators import validate_ticks

    cfg = load_config()
    # Pick a day that is almost certainly published: 3 days ago.
    day = date.today() - timedelta(days=3)

    cache = ParquetCache(cfg.cache_root)
    async with HttpClient(cfg) as http:
        arc = ArchiveClient(http, cache, cfg)
        df = await arc.fetch_day("BTCUSDT", day)

    assert df is not None, f"archive missing for BTCUSDT {day}"
    assert not df.empty
    # Schema.
    for c in ("timestamp", "symbol", "side", "size", "price", "trade_id"):
        assert c in df.columns, c
    assert str(df["timestamp"].dt.tz) == "UTC"
    assert df["price"].dtype == "float64"
    assert df["size"].dtype == "float64"
    assert set(df["side"].unique()).issubset({"Buy", "Sell"})
    # A busy USDT perp has many thousands of prints a day.
    assert len(df) > 10_000
    # Monotone within ~1s (market data can have tiny backsteps).
    report = validate_ticks(df, "BTCUSDT")
    assert report.status in ("PASS", "WARN"), report.as_dict()

    # Cache file actually written.
    assert cache.daily_path("ticks_archive", "BTCUSDT", day.isoformat()).exists()

    # Second call is idempotent: no re-download, returns same row count.
    async with HttpClient(cfg) as http:
        arc2 = ArchiveClient(http, cache, cfg)
        df2 = await arc2.fetch_day("BTCUSDT", day)
    assert df2 is not None
    assert len(df2) == len(df)


@pytest.mark.timeout(30)
async def test_archive_404_returns_none(tmp_data_root: Path) -> None:
    """A symbol that does not exist must return None, not raise."""
    from core.config import load_config
    from downloader.archive import ArchiveClient
    from downloader.cache import ParquetCache
    from downloader.http import HttpClient

    cfg = load_config()
    cache = ParquetCache(cfg.cache_root)
    async with HttpClient(cfg) as http:
        arc = ArchiveClient(http, cache, cfg)
        df = await arc.fetch_day("DOESNOTEXIST123USDT", date.today() - timedelta(days=2))
    assert df is None
