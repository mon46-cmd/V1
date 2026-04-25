"""Chart data helpers for the dashboard - candles + overlays + flags.

Pure-pandas, no dependency on the feature pipeline so the API can serve
chart data even on a fresh install without precomputed snapshots.

Layout assumed by :func:`load_candles`::

    <cache_root>/klines/<SYMBOL>/<INTERVAL>/<SYMBOL>_klines_<INTERVAL>.parquet

Timestamps in the parquet are UTC tz-aware (ms precision from Bybit).
The functions here normalise everything to int unix-seconds so that the
TradingView ``lightweight-charts`` front-end can consume them directly.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Bybit kline intervals (REST: numeric minutes or D/W/M).
_INTERVAL_RE = re.compile(r"^(?:[1-9]\d*|D|W|M)$")
_SYMBOL_RE = re.compile(r"^[A-Z0-9]{2,30}$")

INDICATORS_AVAILABLE = (
    "ema_8", "ema_21", "ema_50",
    "bb_mid", "bb_upper", "bb_lower",
    "vwap_20", "rsi_14", "atr_pct_14",
)
FLAGS_AVAILABLE = (
    "volume_climax", "sweep_up", "sweep_down",
    "macd_cross_up", "macd_cross_down",
    "rsi_overbought", "rsi_oversold",
)


def _safe_symbol(s: str) -> str:
    s = (s or "").upper().strip()
    if not _SYMBOL_RE.match(s):
        raise ValueError(f"invalid symbol: {s!r}")
    return s


def _safe_interval(tf: str) -> str:
    tf = (tf or "").strip()
    if not _INTERVAL_RE.match(tf):
        raise ValueError(f"invalid interval: {tf!r}")
    return tf


def candle_path(cache_root: Path, symbol: str, interval: str) -> Path:
    sym = _safe_symbol(symbol)
    iv = _safe_interval(interval)
    return cache_root / "klines" / sym / iv / f"{sym}_klines_{iv}.parquet"


def load_candles(
    cache_root: Path, symbol: str, interval: str, *, limit: int,
) -> pd.DataFrame:
    """Read the most recent ``limit`` candles from the parquet cache."""
    p = candle_path(cache_root, symbol, interval)
    cols = ["timestamp", "open", "high", "low", "close", "volume"]
    if not p.exists():
        return pd.DataFrame(columns=cols)
    df = pd.read_parquet(p)
    if "timestamp" not in df.columns:
        return pd.DataFrame(columns=cols)
    ts = df["timestamp"]
    if not pd.api.types.is_datetime64_any_dtype(ts):
        df["timestamp"] = pd.to_datetime(ts, utc=True)
    elif ts.dt.tz is None:
        df["timestamp"] = ts.dt.tz_localize("UTC")
    df = df.sort_values("timestamp").drop_duplicates("timestamp")
    if limit and len(df) > limit:
        df = df.tail(limit)
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------
# Indicator series (returned as same-length lists, ``None`` for NaN)
# ---------------------------------------------------------------------
def _series_to_list(s: pd.Series, *, ndigits: int = 8) -> list[float | None]:
    out: list[float | None] = []
    for v in s.tolist():
        if v is None or (isinstance(v, float) and np.isnan(v)):
            out.append(None)
        else:
            out.append(round(float(v), ndigits))
    return out


def compute_indicators(df: pd.DataFrame) -> dict[str, list[float | None]]:
    if df.empty:
        return {}
    c = df["close"].astype(float)
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    v = df["volume"].astype(float)

    out: dict[str, list[float | None]] = {}
    out["ema_8"] = _series_to_list(c.ewm(span=8, adjust=False).mean())
    out["ema_21"] = _series_to_list(c.ewm(span=21, adjust=False).mean())
    out["ema_50"] = _series_to_list(c.ewm(span=50, adjust=False).mean())

    sma20 = c.rolling(20).mean()
    std20 = c.rolling(20).std(ddof=0)
    out["bb_mid"] = _series_to_list(sma20)
    out["bb_upper"] = _series_to_list(sma20 + 2.0 * std20)
    out["bb_lower"] = _series_to_list(sma20 - 2.0 * std20)

    pv = (c * v).rolling(20).sum()
    vv = v.rolling(20).sum().replace(0, np.nan)
    out["vwap_20"] = _series_to_list(pv / vv)

    delta = c.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    dn = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    out["rsi_14"] = _series_to_list(rsi, ndigits=4)

    tr = pd.concat([
        (h - l),
        (h - c.shift()).abs(),
        (l - c.shift()).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / 14, adjust=False).mean()
    out["atr_pct_14"] = _series_to_list(
        (atr / c.replace(0, np.nan) * 100.0), ndigits=4
    )
    return out


# ---------------------------------------------------------------------
# Flag markers (point-in-time) - returned per flag name
# ---------------------------------------------------------------------
def compute_flag_markers(df: pd.DataFrame) -> dict[str, list[dict[str, Any]]]:
    if df.empty:
        return {k: [] for k in FLAGS_AVAILABLE}
    c = df["close"].astype(float)
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    v = df["volume"].astype(float)
    ts = (df["timestamp"].astype("int64") // 1_000_000_000).astype(int).tolist()
    n = len(df)
    out: dict[str, list[dict[str, Any]]] = {k: [] for k in FLAGS_AVAILABLE}

    sma_v = v.rolling(20).mean()
    for i in range(n):
        sv = sma_v.iloc[i]
        if pd.notna(sv) and sv > 0 and v.iloc[i] > 2.0 * sv:
            out["volume_climax"].append({
                "time": ts[i], "price": float(h.iloc[i]),
                "ratio": round(float(v.iloc[i] / sv), 2),
            })

    prior_high = h.shift().rolling(20).max()
    prior_low = l.shift().rolling(20).min()
    for i in range(n):
        ph = prior_high.iloc[i]
        pl = prior_low.iloc[i]
        if pd.notna(ph) and h.iloc[i] > ph and c.iloc[i] < ph:
            out["sweep_up"].append({"time": ts[i], "price": float(h.iloc[i])})
        if pd.notna(pl) and l.iloc[i] < pl and c.iloc[i] > pl:
            out["sweep_down"].append({"time": ts[i], "price": float(l.iloc[i])})

    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    sig = macd.ewm(span=9, adjust=False).mean()
    hist = macd - sig
    prev_hist = hist.shift()
    for i in range(n):
        ph = prev_hist.iloc[i]
        cur = hist.iloc[i]
        if pd.notna(ph) and pd.notna(cur):
            if cur > 0 and ph <= 0:
                out["macd_cross_up"].append({"time": ts[i], "price": float(c.iloc[i])})
            elif cur < 0 and ph >= 0:
                out["macd_cross_down"].append({"time": ts[i], "price": float(c.iloc[i])})

    delta = c.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    dn = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    for i in range(n):
        r = rsi.iloc[i]
        if pd.notna(r):
            if r >= 70:
                out["rsi_overbought"].append({"time": ts[i], "price": float(h.iloc[i])})
            elif r <= 30:
                out["rsi_oversold"].append({"time": ts[i], "price": float(l.iloc[i])})

    return out


def candles_payload(
    cache_root: Path, symbol: str, interval: str, *, limit: int,
) -> dict[str, Any]:
    df = load_candles(cache_root, symbol, interval, limit=limit)
    sym = _safe_symbol(symbol)
    iv = _safe_interval(interval)
    if df.empty:
        return {
            "symbol": sym, "tf": iv, "rows": [],
            "indicators": {}, "flags": {k: [] for k in FLAGS_AVAILABLE},
            "indicators_available": list(INDICATORS_AVAILABLE),
            "flags_available": list(FLAGS_AVAILABLE),
        }
    ts = (df["timestamp"].astype("int64") // 1_000_000_000).astype(int).tolist()
    o = df["open"].astype(float).tolist()
    h = df["high"].astype(float).tolist()
    lo = df["low"].astype(float).tolist()
    c = df["close"].astype(float).tolist()
    v = df["volume"].astype(float).tolist()
    rows = [
        {"time": ts[i], "open": o[i], "high": h[i],
         "low": lo[i], "close": c[i], "volume": v[i]}
        for i in range(len(df))
    ]
    return {
        "symbol": sym,
        "tf": iv,
        "rows": rows,
        "indicators": compute_indicators(df),
        "flags": compute_flag_markers(df),
        "indicators_available": list(INDICATORS_AVAILABLE),
        "flags_available": list(FLAGS_AVAILABLE),
    }


__all__ = [
    "INDICATORS_AVAILABLE", "FLAGS_AVAILABLE",
    "candle_path", "load_candles",
    "compute_indicators", "compute_flag_markers",
    "candles_payload",
]
