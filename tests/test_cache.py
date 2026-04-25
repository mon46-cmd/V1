"""Offline tests for `downloader.cache.ParquetCache`.

These do no network I/O; they verify the atomic-write contract, the
append/merge-on-key behaviour, and the corrupt-file recovery path.
"""
from __future__ import annotations

import asyncio
import random
from pathlib import Path

import pandas as pd
import pytest

from downloader.cache import ParquetCache
from downloader.constants import COLS_KLINE


def _kline_df(start_ms: int, n: int) -> pd.DataFrame:
    ts = pd.to_datetime([start_ms + i * 60_000 for i in range(n)], unit="ms", utc=True)
    base = 100.0
    return pd.DataFrame({
        "timestamp": ts,
        "open": [base + i for i in range(n)],
        "high": [base + i + 1 for i in range(n)],
        "low": [base + i - 1 for i in range(n)],
        "close": [base + i + 0.5 for i in range(n)],
        "volume": [10.0 + i for i in range(n)],
        "turnover": [1000.0 + i * 10 for i in range(n)],
    })[list(COLS_KLINE)]


def test_atomic_write_and_read(tmp_path: Path) -> None:
    cache = ParquetCache(tmp_path)
    df = _kline_df(1_700_000_000_000, 10)
    p = cache.write(df, "klines", "BTCUSDT", "15")
    assert p.exists() and p.stat().st_size > 0
    # No stray .tmp.
    assert not list(p.parent.glob("*.tmp"))

    got = cache.read("klines", "BTCUSDT", "15")
    assert got is not None
    assert len(got) == 10
    assert got["timestamp"].is_monotonic_increasing
    assert got["close"].dtype == "float64"


def test_append_merges_on_timestamp(tmp_path: Path) -> None:
    cache = ParquetCache(tmp_path)
    a = _kline_df(1_700_000_000_000, 5)
    b = _kline_df(1_700_000_000_000 + 3 * 60_000, 5)  # overlaps last 2 of a

    cache.append(a, "klines", "BTCUSDT", "15")
    merged = cache.append(b, "klines", "BTCUSDT", "15")
    # 5 + 5 - 2 overlap = 8 unique timestamps.
    assert len(merged) == 8
    assert merged["timestamp"].is_monotonic_increasing
    assert merged["timestamp"].is_unique


def test_read_removes_corrupt_file(tmp_path: Path) -> None:
    cache = ParquetCache(tmp_path)
    p = cache.path("klines", "BTCUSDT", "15")
    p.write_bytes(b"not a parquet")  # definitely corrupt
    assert cache.read("klines", "BTCUSDT", "15") is None
    assert not p.exists()  # cleaned up


def test_read_removes_undersized_file(tmp_path: Path) -> None:
    cache = ParquetCache(tmp_path)
    p = cache.path("klines", "BTCUSDT", "15")
    p.write_bytes(b"x" * 10)  # below MIN_PARQUET_BYTES
    assert cache.read("klines", "BTCUSDT", "15") is None
    assert not p.exists()


def test_last_and_first_timestamp(tmp_path: Path) -> None:
    cache = ParquetCache(tmp_path)
    df = _kline_df(1_700_000_000_000, 10)
    cache.write(df, "klines", "BTCUSDT", "15")
    assert cache.first_timestamp("klines", "BTCUSDT", "15") == df["timestamp"].min()
    assert cache.last_timestamp("klines", "BTCUSDT", "15") == df["timestamp"].max()


def test_daily_path_and_roundtrip(tmp_path: Path) -> None:
    cache = ParquetCache(tmp_path)
    df = _kline_df(1_700_000_000_000, 3)
    cache.write_daily(df, "ticks_archive", "BTCUSDT", "2025-01-01")
    got = cache.read_daily("ticks_archive", "BTCUSDT", "2025-01-01")
    assert got is not None
    assert len(got) == 3


def test_inventory_lists_files(tmp_path: Path) -> None:
    cache = ParquetCache(tmp_path)
    cache.write(_kline_df(1_700_000_000_000, 2), "klines", "BTCUSDT", "15")
    cache.write(_kline_df(1_700_000_000_000, 2), "klines", "ETHUSDT", "60")
    inv = cache.inventory()
    assert set(inv["kind"]) == {"klines"}
    assert set(inv["symbol"]) == {"BTCUSDT", "ETHUSDT"}
    assert (inv["bytes"] > 0).all()


def test_concurrent_writers_do_not_corrupt(tmp_path: Path) -> None:
    """Many parallel writes to the same path must always leave a valid
    parquet on disk (os.replace is atomic on Windows and POSIX)."""
    cache = ParquetCache(tmp_path)

    async def writer(i: int) -> None:
        df = _kline_df(1_700_000_000_000 + i, 5 + (i % 3))
        await asyncio.sleep(random.uniform(0, 0.01))
        cache.write(df, "klines", "BTCUSDT", "15")

    async def main() -> None:
        await asyncio.gather(*(writer(i) for i in range(25)))

    asyncio.run(main())

    got = cache.read("klines", "BTCUSDT", "15")
    assert got is not None
    assert len(got) >= 5
    # No leftover tmp files.
    parent = cache.path("klines", "BTCUSDT", "15").parent
    assert not list(parent.glob("*.tmp"))


def test_empty_df_append_is_noop(tmp_path: Path) -> None:
    cache = ParquetCache(tmp_path)
    empty = _kline_df(0, 0)
    merged = cache.append(empty, "klines", "BTCUSDT", "15")
    assert merged.empty
    # Still nothing on disk.
    assert not cache.path("klines", "BTCUSDT", "15").exists() or \
        cache.read("klines", "BTCUSDT", "15") is None


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-q"])
