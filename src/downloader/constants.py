"""Bybit v5 endpoints, limits, interval maps, and canonical column schemas.

All magic strings and numbers used by the downloader live here. No other
module in `src/downloader/` should hard-code an endpoint, a limit, or a
column name.
"""
from __future__ import annotations

from typing import Final

# --- Public REST paths -------------------------------------------------
PATH_KLINE: Final[str] = "/v5/market/kline"
PATH_MARK_KLINE: Final[str] = "/v5/market/mark-price-kline"
PATH_INDEX_KLINE: Final[str] = "/v5/market/index-price-kline"
PATH_PREMIUM_KLINE: Final[str] = "/v5/market/premium-index-price-kline"
PATH_FUNDING: Final[str] = "/v5/market/funding/history"
PATH_OI: Final[str] = "/v5/market/open-interest"
PATH_TICKERS: Final[str] = "/v5/market/tickers"
PATH_INSTRUMENTS: Final[str] = "/v5/market/instruments-info"
PATH_ORDERBOOK: Final[str] = "/v5/market/orderbook"
PATH_RECENT_TRADES: Final[str] = "/v5/market/recent-trade"
PATH_LS_RATIO: Final[str] = "/v5/market/account-ratio"

# --- Pagination limits (Bybit v5 public caps) --------------------------
LIMIT_KLINE: Final[int] = 1000
LIMIT_FUNDING: Final[int] = 200
LIMIT_OI: Final[int] = 200
LIMIT_LS: Final[int] = 500
LIMIT_RECENT_TRADES: Final[int] = 1000
LIMIT_ORDERBOOK: Final[int] = 500

# --- Interval alphabets ------------------------------------------------
KLINE_INTERVALS: Final[tuple[str, ...]] = (
    "1", "3", "5", "15", "30", "60", "120", "240", "360", "720", "D", "W", "M",
)
OI_INTERVALS: Final[tuple[str, ...]] = ("5min", "15min", "30min", "1h", "4h", "1d")
LS_INTERVALS: Final[tuple[str, ...]] = ("5min", "15min", "30min", "1h", "4h", "4d")

KLINE_FREQ: Final[dict[str, str]] = {
    "1": "1min", "3": "3min", "5": "5min", "15": "15min", "30": "30min",
    "60": "60min", "120": "120min", "240": "240min", "360": "360min",
    "720": "720min", "D": "1D", "W": "1W", "M": "1MS",
}
OI_FREQ: Final[dict[str, str]] = {
    "5min": "5min", "15min": "15min", "30min": "30min",
    "1h": "60min", "4h": "240min", "1d": "1D",
}

# --- Canonical column schemas ------------------------------------------
COLS_KLINE: Final[tuple[str, ...]] = (
    "timestamp", "open", "high", "low", "close", "volume", "turnover",
)
COLS_PRICE_KLINE: Final[tuple[str, ...]] = (
    "timestamp", "open", "high", "low", "close",
)
COLS_FUNDING: Final[tuple[str, ...]] = ("timestamp", "symbol", "funding_rate")
COLS_OI: Final[tuple[str, ...]] = ("timestamp", "open_interest")
COLS_LS: Final[tuple[str, ...]] = ("timestamp", "buy_ratio", "sell_ratio")
COLS_TICK: Final[tuple[str, ...]] = (
    "timestamp", "symbol", "side", "size", "price", "trade_id",
)

# --- Cache gate --------------------------------------------------------
MIN_PARQUET_BYTES: Final[int] = 256

# --- Public WebSocket --------------------------------------------------
BYBIT_WS_LINEAR: Final[str] = "wss://stream.bybit.com/v5/public/linear"
BYBIT_WS_INVERSE: Final[str] = "wss://stream.bybit.com/v5/public/inverse"
BYBIT_WS_SPOT: Final[str] = "wss://stream.bybit.com/v5/public/spot"

WS_PING_INTERVAL_SEC: Final[float] = 20.0
WS_RECONNECT_BASE_SEC: Final[float] = 1.0
WS_RECONNECT_MAX_SEC: Final[float] = 30.0
WS_SUBSCRIBE_BATCH: Final[int] = 10
WS_QUEUE_MAX: Final[int] = 20_000

# Live tick cache kind; parquet files live at
# <cache_root>/ticks_live/<SYMBOL>/<YYYY-MM-DD>.parquet
CACHE_KIND_TICKS_LIVE: Final[str] = "ticks_live"
CACHE_KIND_TICKS_ARCHIVE: Final[str] = "ticks_archive"
CACHE_KIND_BOOK_TOP: Final[str] = "book_top"
