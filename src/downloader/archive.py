"""Bybit public daily trade archive client.

Daily files live at::

    https://public.bybit.com/trading/<SYMBOL>/<SYMBOL><YYYY-MM-DD>.csv.gz

One file per symbol per UTC day since the contract listing. The CSV has
columns ``timestamp, symbol, side, size, price, tickDirection,
trdMatchID, grossValue, homeNotional, foreignNotional`` and is
gzip-encoded.

We normalize to the canonical tick schema (``COLS_TICK``) and persist
one parquet per day under::

    <cache_root>/ticks_archive/<SYMBOL>/<YYYY-MM-DD>.parquet

Idempotent: if the daily parquet already exists and is non-empty, the
fetch is skipped unless ``overwrite=True``.
"""
from __future__ import annotations

import gzip
import io
import logging
from datetime import date, timedelta
from typing import Callable

import pandas as pd

from core.config import Config
from downloader.cache import ParquetCache
from downloader.constants import COLS_TICK
from downloader.errors import HttpError
from downloader.http import HttpClient

log = logging.getLogger(__name__)

CACHE_KIND = "ticks_archive"


class ArchiveClient:
    def __init__(self, http: HttpClient, cache: ParquetCache, cfg: Config) -> None:
        self._http = http
        self._cache = cache
        self._base = cfg.bybit_archive_base.rstrip("/")

    def url_for(self, symbol: str, day: date) -> str:
        return f"{self._base}/{symbol}/{symbol}{day.isoformat()}.csv.gz"

    async def fetch_day(
        self,
        symbol: str,
        day: date,
        *,
        overwrite: bool = False,
    ) -> pd.DataFrame | None:
        """Download and cache one daily archive file.

        Returns the normalized DataFrame, or ``None`` if Bybit has no
        file for that (symbol, day) pair (404 - symbol not yet listed
        or long since delisted).
        """
        existing = self._cache.read_daily(CACHE_KIND, symbol, day.isoformat())
        if existing is not None and not overwrite:
            return existing
        url = self.url_for(symbol, day)
        try:
            raw = await self._http.get_bytes(url)
        except HttpError as exc:
            if exc.status == 404:
                log.info("archive miss %s %s (404)", symbol, day)
                return None
            raise
        df = _parse_archive_csv(raw, symbol)
        if df.empty:
            log.warning("archive %s %s parsed empty", symbol, day)
            return df
        self._cache.write_daily(df, CACHE_KIND, symbol, day.isoformat())
        return df

    async def fetch_range(
        self,
        symbol: str,
        start: date,
        end: date,
        *,
        overwrite: bool = False,
        on_progress: Callable[[str, date, pd.DataFrame | None], None] | None = None,
    ) -> dict[str, int]:
        """Fetch every daily file in ``[start, end]`` inclusive.

        After at least one success, three consecutive 404s within the
        last three days abort the forward walk (symbol likely delisted
        or data not yet published).
        """
        days_ok = 0
        days_missing = 0
        total_rows = 0
        seen_ok = False
        missing_streak = 0
        cur = start
        while cur <= end:
            df = await self.fetch_day(symbol, cur, overwrite=overwrite)
            if df is None:
                days_missing += 1
                missing_streak += 1
                if seen_ok and missing_streak >= 3 and cur > date.today() - timedelta(days=3):
                    log.info("archive: likely end of history for %s at %s", symbol, cur)
                    break
            else:
                seen_ok = True
                missing_streak = 0
                days_ok += 1
                total_rows += len(df)
            if on_progress is not None:
                on_progress(symbol, cur, df)
            cur += timedelta(days=1)
        return {"days_ok": days_ok, "days_missing": days_missing, "total_rows": total_rows}

    def read_range(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        """Read the union of cached daily files for ``[start, end]``. No fetch."""
        parts: list[pd.DataFrame] = []
        cur = start
        while cur <= end:
            df = self._cache.read_daily(CACHE_KIND, symbol, cur.isoformat())
            if df is not None:
                parts.append(df)
            cur += timedelta(days=1)
        if not parts:
            return pd.DataFrame(columns=list(COLS_TICK))
        return (
            pd.concat(parts, ignore_index=True)
            .drop_duplicates(subset=["trade_id"])
            .sort_values("timestamp")
            .reset_index(drop=True)
        )


def _parse_archive_csv(raw: bytes, symbol: str) -> pd.DataFrame:
    with gzip.GzipFile(fileobj=io.BytesIO(raw)) as gz:
        df = pd.read_csv(gz)
    if df.empty:
        return pd.DataFrame(columns=list(COLS_TICK))
    # ``timestamp`` column is seconds since epoch (float, sub-second).
    ts_sec = pd.to_numeric(df["timestamp"], errors="coerce").astype("float64")
    n = len(df)
    sym_series = (
        df["symbol"].astype(str) if "symbol" in df.columns
        else pd.Series([symbol] * n, dtype="object")
    )
    tid_series = (
        df["trdMatchID"].astype(str) if "trdMatchID" in df.columns
        else pd.Series([""] * n, dtype="object")
    )
    out = pd.DataFrame({
        "timestamp": pd.to_datetime((ts_sec * 1000).astype("int64"), unit="ms", utc=True),
        "symbol": sym_series,
        "side": df["side"].astype(str),
        "size": pd.to_numeric(df["size"], errors="coerce").astype("float64"),
        "price": pd.to_numeric(df["price"], errors="coerce").astype("float64"),
        "trade_id": tid_series,
    })
    return (
        out.dropna(subset=["price", "size"])
        .drop_duplicates(subset=["trade_id"])
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
