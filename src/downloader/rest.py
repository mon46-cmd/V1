"""Bybit v5 public REST client.

Every method returns a pandas DataFrame with UTC-tz timestamps and
float64 numeric columns, or a plain dict for non-tabular endpoints
(orderbook, tickers). Pagination is handled inside each method: give it
a window, receive every row.

Design rules followed here:

- Async only (shares the global rate limit via `HttpClient`).
- Missing/malformed numbers become NaN rather than raising.
- Duplicates on the natural key are dropped before returning.
- Empty responses return an empty DataFrame with the canonical schema
  so downstream `validators.py` never sees a shape surprise.
"""
from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from core.config import Config
from downloader.constants import (
    COLS_FUNDING,
    COLS_KLINE,
    COLS_LS,
    COLS_OI,
    COLS_PRICE_KLINE,
    COLS_TICK,
    LIMIT_FUNDING,
    LIMIT_KLINE,
    LIMIT_LS,
    LIMIT_OI,
    LIMIT_ORDERBOOK,
    LIMIT_RECENT_TRADES,
    PATH_FUNDING,
    PATH_INDEX_KLINE,
    PATH_INSTRUMENTS,
    PATH_KLINE,
    PATH_LS_RATIO,
    PATH_MARK_KLINE,
    PATH_OI,
    PATH_ORDERBOOK,
    PATH_PREMIUM_KLINE,
    PATH_RECENT_TRADES,
    PATH_TICKERS,
)
from downloader.http import HttpClient

log = logging.getLogger(__name__)


def _ms(ts: str | pd.Timestamp | int | float) -> int:
    if isinstance(ts, (int,)):
        return int(ts)
    if isinstance(ts, float):
        return int(ts)
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    else:
        t = t.tz_convert("UTC")
    return int(t.timestamp() * 1000)


def _to_float(v: Any) -> float:
    if v is None or v == "":
        return float("nan")
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


class RestClient:
    """Thin normalizing wrapper over Bybit v5 public endpoints."""

    def __init__(self, http: HttpClient, cfg: Config) -> None:
        self._http = http
        self._category = cfg.category

    # ---------------- klines family -----------------------------------
    async def klines(
        self,
        symbol: str,
        interval: str,
        start: str | pd.Timestamp | int,
        end: str | pd.Timestamp | int,
    ) -> pd.DataFrame:
        return await self._paginate_klines(PATH_KLINE, symbol, interval, start, end, price_only=False)

    async def mark_klines(
        self,
        symbol: str,
        interval: str,
        start: str | pd.Timestamp | int,
        end: str | pd.Timestamp | int,
    ) -> pd.DataFrame:
        return await self._paginate_klines(PATH_MARK_KLINE, symbol, interval, start, end, price_only=True)

    async def index_klines(
        self,
        symbol: str,
        interval: str,
        start: str | pd.Timestamp | int,
        end: str | pd.Timestamp | int,
    ) -> pd.DataFrame:
        return await self._paginate_klines(PATH_INDEX_KLINE, symbol, interval, start, end, price_only=True)

    async def premium_klines(
        self,
        symbol: str,
        interval: str,
        start: str | pd.Timestamp | int,
        end: str | pd.Timestamp | int,
    ) -> pd.DataFrame:
        return await self._paginate_klines(PATH_PREMIUM_KLINE, symbol, interval, start, end, price_only=True)

    async def _paginate_klines(
        self,
        path: str,
        symbol: str,
        interval: str,
        start: str | pd.Timestamp | int,
        end: str | pd.Timestamp | int,
        *,
        price_only: bool,
    ) -> pd.DataFrame:
        start_ms = _ms(start)
        end_ms = _ms(end)
        if end_ms <= start_ms:
            return _klines_empty(price_only)
        rows: list[list[Any]] = []
        cur_end = end_ms
        while cur_end > start_ms:
            params = {
                "category": self._category,
                "symbol": symbol,
                "interval": interval,
                "start": start_ms,
                "end": cur_end,
                "limit": LIMIT_KLINE,
            }
            result = await self._http.get_json(path, params)
            got = result.get("list") or []
            if not got:
                break
            rows.extend(got)
            oldest = int(got[-1][0])
            if len(got) < LIMIT_KLINE or oldest <= start_ms:
                break
            cur_end = oldest - 1
        return _klines_to_df(rows, price_only=price_only)

    # ---------------- funding -----------------------------------------
    async def funding(
        self,
        symbol: str,
        start: str | pd.Timestamp | int,
        end: str | pd.Timestamp | int,
    ) -> pd.DataFrame:
        start_ms, end_ms = _ms(start), _ms(end)
        out: list[dict[str, Any]] = []
        cur_end = end_ms
        while True:
            result = await self._http.get_json(
                PATH_FUNDING,
                {
                    "category": self._category,
                    "symbol": symbol,
                    "startTime": start_ms,
                    "endTime": cur_end,
                    "limit": LIMIT_FUNDING,
                },
            )
            got = result.get("list") or []
            if not got:
                break
            out.extend(got)
            oldest = min(int(r["fundingRateTimestamp"]) for r in got)
            if len(got) < LIMIT_FUNDING or oldest <= start_ms:
                break
            cur_end = oldest - 1

        if not out:
            return _empty_df(COLS_FUNDING, float_cols=("funding_rate",))
        df = pd.DataFrame(out)
        df["timestamp"] = pd.to_datetime(df["fundingRateTimestamp"].astype("int64"), unit="ms", utc=True)
        df["funding_rate"] = pd.to_numeric(df["fundingRate"], errors="coerce").astype("float64")
        df["symbol"] = df["symbol"].astype(str)
        return (
            df[list(COLS_FUNDING)]
            .drop_duplicates("timestamp")
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

    # ---------------- open interest -----------------------------------
    async def open_interest(
        self,
        symbol: str,
        interval: str,
        start: str | pd.Timestamp | int,
        end: str | pd.Timestamp | int,
    ) -> pd.DataFrame:
        start_ms, end_ms = _ms(start), _ms(end)
        out: list[dict[str, Any]] = []
        cursor = ""
        while True:
            params: dict[str, Any] = {
                "category": self._category,
                "symbol": symbol,
                "intervalTime": interval,
                "startTime": start_ms,
                "endTime": end_ms,
                "limit": LIMIT_OI,
            }
            if cursor:
                params["cursor"] = cursor
            result = await self._http.get_json(PATH_OI, params)
            got = result.get("list") or []
            if not got:
                break
            out.extend(got)
            cursor = result.get("nextPageCursor") or ""
            if not cursor or len(got) < LIMIT_OI:
                break
        if not out:
            return _empty_df(COLS_OI, float_cols=("open_interest",))
        df = pd.DataFrame(out)
        df["timestamp"] = pd.to_datetime(df["timestamp"].astype("int64"), unit="ms", utc=True)
        df["open_interest"] = pd.to_numeric(df["openInterest"], errors="coerce").astype("float64")
        return (
            df[list(COLS_OI)]
            .drop_duplicates("timestamp")
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

    # ---------------- long/short ratio --------------------------------
    async def long_short_ratio(
        self,
        symbol: str,
        interval: str,
        *,
        limit: int = LIMIT_LS,
    ) -> pd.DataFrame:
        """Aggregate account long/short ratio.

        Bybit exposes only the most recent ``limit`` points for this
        endpoint; start/end params are ignored by the exchange.
        """
        result = await self._http.get_json(
            PATH_LS_RATIO,
            {
                "category": self._category,
                "symbol": symbol,
                "period": interval,
                "limit": min(limit, LIMIT_LS),
            },
        )
        got = result.get("list") or []
        if not got:
            return _empty_df(COLS_LS, float_cols=("buy_ratio", "sell_ratio"))
        df = pd.DataFrame(got)
        df["timestamp"] = pd.to_datetime(df["timestamp"].astype("int64"), unit="ms", utc=True)
        df["buy_ratio"] = pd.to_numeric(df["buyRatio"], errors="coerce").astype("float64")
        df["sell_ratio"] = pd.to_numeric(df["sellRatio"], errors="coerce").astype("float64")
        return (
            df[list(COLS_LS)]
            .drop_duplicates("timestamp")
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

    # ---------------- tickers / instruments ---------------------------
    async def tickers(self) -> list[dict[str, Any]]:
        result = await self._http.get_json(PATH_TICKERS, {"category": self._category})
        return [_norm_ticker(t) for t in (result.get("list") or [])]

    async def ticker(self, symbol: str) -> dict[str, Any]:
        result = await self._http.get_json(
            PATH_TICKERS, {"category": self._category, "symbol": symbol},
        )
        lst = result.get("list") or []
        return _norm_ticker(lst[0]) if lst else {}

    async def instruments(self) -> pd.DataFrame:
        """Contract specs: tick size, qty step, status, launch time."""
        out: list[dict[str, Any]] = []
        cursor = ""
        while True:
            params: dict[str, Any] = {"category": self._category, "limit": 1000}
            if cursor:
                params["cursor"] = cursor
            result = await self._http.get_json(PATH_INSTRUMENTS, params)
            got = result.get("list") or []
            if not got:
                break
            out.extend(got)
            cursor = result.get("nextPageCursor") or ""
            if not cursor:
                break
        if not out:
            return pd.DataFrame()
        df = pd.json_normalize(out)
        if "launchTime" in df.columns:
            df["launchTime"] = pd.to_datetime(
                pd.to_numeric(df["launchTime"], errors="coerce"), unit="ms", utc=True,
            )
        return df

    # ---------------- orderbook snapshot ------------------------------
    async def orderbook(self, symbol: str, *, depth: int = LIMIT_ORDERBOOK) -> dict[str, Any]:
        depth = min(depth, LIMIT_ORDERBOOK)
        result = await self._http.get_json(
            PATH_ORDERBOOK, {"category": self._category, "symbol": symbol, "limit": depth},
        )
        bids = [(float(p), float(q)) for p, q in (result.get("b") or [])]
        asks = [(float(p), float(q)) for p, q in (result.get("a") or [])]
        return {
            "symbol": str(result.get("s", symbol)),
            "ts_ms": int(result.get("ts", 0)),
            "update_id": int(result.get("u", 0)),
            "bids": bids,
            "asks": asks,
        }

    # ---------------- recent trades (~last 1k) ------------------------
    async def recent_trades(self, symbol: str, *, limit: int = LIMIT_RECENT_TRADES) -> pd.DataFrame:
        result = await self._http.get_json(
            PATH_RECENT_TRADES,
            {"category": self._category, "symbol": symbol, "limit": min(limit, LIMIT_RECENT_TRADES)},
        )
        got = result.get("list") or []
        if not got:
            return _empty_df(COLS_TICK, float_cols=("size", "price"))
        df = pd.DataFrame(got)
        df["timestamp"] = pd.to_datetime(df["time"].astype("int64"), unit="ms", utc=True)
        df["symbol"] = df["symbol"].astype(str) if "symbol" in df.columns else symbol
        df["side"] = df["side"].astype(str)
        df["size"] = pd.to_numeric(df["size"], errors="coerce").astype("float64")
        df["price"] = pd.to_numeric(df["price"], errors="coerce").astype("float64")
        df["trade_id"] = (
            df["execId"].astype(str) if "execId" in df.columns
            else pd.Series([""] * len(df), dtype="object")
        )
        return (
            df[list(COLS_TICK)]
            .drop_duplicates("trade_id")
            .sort_values("timestamp")
            .reset_index(drop=True)
        )


# ----------------------- helpers ---------------------------------------
def _klines_to_df(rows: list[list[Any]], *, price_only: bool) -> pd.DataFrame:
    if not rows:
        return _klines_empty(price_only)
    if price_only:
        trimmed = [r[:5] for r in rows]
        df = pd.DataFrame(trimmed, columns=list(COLS_PRICE_KLINE))
    else:
        df = pd.DataFrame(rows, columns=list(COLS_KLINE))
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype("int64"), unit="ms", utc=True)
    for c in df.columns:
        if c == "timestamp":
            continue
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
    return df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)


def _klines_empty(price_only: bool) -> pd.DataFrame:
    cols = COLS_PRICE_KLINE if price_only else COLS_KLINE
    return _empty_df(cols, float_cols=tuple(c for c in cols if c != "timestamp"))


def _empty_df(cols: tuple[str, ...], *, float_cols: tuple[str, ...] = ()) -> pd.DataFrame:
    data: dict[str, pd.Series] = {}
    for c in cols:
        if c == "timestamp":
            data[c] = pd.Series(dtype="datetime64[ns, UTC]")
        elif c in float_cols:
            data[c] = pd.Series(dtype="float64")
        else:
            data[c] = pd.Series(dtype="object")
    return pd.DataFrame(data)


def _norm_ticker(t: dict[str, Any]) -> dict[str, Any]:
    return {
        "symbol": str(t.get("symbol", "")),
        "price": _to_float(t.get("lastPrice")),
        "bid": _to_float(t.get("bid1Price")),
        "ask": _to_float(t.get("ask1Price")),
        "mark_price": _to_float(t.get("markPrice")),
        "index_price": _to_float(t.get("indexPrice")),
        "volume_24h": _to_float(t.get("volume24h")),
        "turnover_24h": _to_float(t.get("turnover24h")),
        "open_interest": _to_float(t.get("openInterest")),
        "open_interest_value": _to_float(t.get("openInterestValue")),
        "funding_rate": _to_float(t.get("fundingRate")),
        "next_funding_ms": int(t.get("nextFundingTime", 0) or 0),
        "price_change_24h_pct": _to_float(t.get("price24hPcnt")),
        "high_24h": _to_float(t.get("highPrice24h")),
        "low_24h": _to_float(t.get("lowPrice24h")),
    }
