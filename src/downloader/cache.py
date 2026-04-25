"""Parquet cache with atomic writes.

Layout::

    <cache_root>/<kind>/<SYMBOL>/<interval>/<SYMBOL>_<kind>_<interval>.parquet
    <cache_root>/<kind>/<SYMBOL>/<SYMBOL>_<kind>.parquet            (no interval)
    <cache_root>/<kind>/<SYMBOL>/<YYYY-MM-DD>.parquet               (daily: ticks/book)

Guarantees:

- Writes are atomic: a ``.tmp`` sibling is flushed then ``os.replace`` d
  onto the final path. A crash mid-write leaves the previous good file
  intact; at worst a stale ``.tmp`` is left over and is harmless.
- Reads tolerate missing or corrupt files (they are removed and
  ``None`` is returned). Empty/undersized parquets are treated as
  corrupt.
- ``append(df, kind, symbol, interval, key=...)`` merges new rows into
  any existing file, dropping duplicates on ``key`` and keeping the
  latest version.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import pandas as pd

from downloader.constants import MIN_PARQUET_BYTES
from downloader.errors import CacheError

log = logging.getLogger(__name__)


class ParquetCache:
    def __init__(self, root: Path) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    # ---- paths --------------------------------------------------------
    def path(self, kind: str, symbol: str, interval: str = "") -> Path:
        if interval:
            d = self.root / kind / symbol / interval
            d.mkdir(parents=True, exist_ok=True)
            return d / f"{symbol}_{kind}_{interval}.parquet"
        d = self.root / kind / symbol
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{symbol}_{kind}.parquet"

    def daily_path(self, kind: str, symbol: str, date: str) -> Path:
        d = self.root / kind / symbol
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{date}.parquet"

    # ---- io -----------------------------------------------------------
    def read(self, kind: str, symbol: str, interval: str = "") -> pd.DataFrame | None:
        return self._read_path(self.path(kind, symbol, interval))

    def read_daily(self, kind: str, symbol: str, date: str) -> pd.DataFrame | None:
        return self._read_path(self.daily_path(kind, symbol, date))

    def _read_path(self, p: Path) -> pd.DataFrame | None:
        if not p.exists():
            return None
        if p.stat().st_size < MIN_PARQUET_BYTES:
            log.warning("undersized parquet %s (%d bytes), removing", p, p.stat().st_size)
            p.unlink(missing_ok=True)
            return None
        try:
            df = pd.read_parquet(p)
        except Exception as exc:  # noqa: BLE001
            log.warning("corrupt parquet %s, removing: %s", p, exc)
            p.unlink(missing_ok=True)
            return None
        return df if not df.empty else None

    def write(self, df: pd.DataFrame, kind: str, symbol: str, interval: str = "") -> Path:
        p = self.path(kind, symbol, interval)
        _atomic_write(p, df)
        return p

    def write_daily(self, df: pd.DataFrame, kind: str, symbol: str, date: str) -> Path:
        p = self.daily_path(kind, symbol, date)
        _atomic_write(p, df)
        return p

    def append(
        self,
        df: pd.DataFrame,
        kind: str,
        symbol: str,
        interval: str = "",
        *,
        key: str = "timestamp",
    ) -> pd.DataFrame:
        existing = self.read(kind, symbol, interval)
        if df is None or df.empty:
            return existing if existing is not None else pd.DataFrame()
        if existing is None or existing.empty:
            merged = df.sort_values(key).reset_index(drop=True)
        else:
            merged = (
                pd.concat([existing, df], ignore_index=True)
                .drop_duplicates(subset=[key], keep="last")
                .sort_values(key)
                .reset_index(drop=True)
            )
        self.write(merged, kind, symbol, interval)
        return merged

    def last_timestamp(self, kind: str, symbol: str, interval: str = "") -> pd.Timestamp | None:
        df = self.read(kind, symbol, interval)
        if df is None or df.empty or "timestamp" not in df.columns:
            return None
        return pd.Timestamp(df["timestamp"].max())

    def first_timestamp(self, kind: str, symbol: str, interval: str = "") -> pd.Timestamp | None:
        df = self.read(kind, symbol, interval)
        if df is None or df.empty or "timestamp" not in df.columns:
            return None
        return pd.Timestamp(df["timestamp"].min())

    def inventory(self) -> pd.DataFrame:
        """List every parquet under the cache root (debug/report helper)."""
        rows: list[dict[str, object]] = []
        if not self.root.exists():
            return pd.DataFrame()
        for p in self.root.rglob("*.parquet"):
            rel = p.relative_to(self.root)
            parts = rel.parts
            rows.append({
                "kind": parts[0] if len(parts) > 0 else "",
                "symbol": parts[1] if len(parts) > 1 else "",
                "subkey": "/".join(parts[2:-1]) if len(parts) > 3 else "",
                "file": parts[-1],
                "bytes": p.stat().st_size,
            })
        return pd.DataFrame(rows)


def _atomic_write(path: Path, df: pd.DataFrame) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        df.to_parquet(tmp, compression="zstd", index=False)
        os.replace(tmp, path)
    except Exception as exc:  # noqa: BLE001
        tmp.unlink(missing_ok=True)
        raise CacheError(f"write failed for {path}: {exc}") from exc
