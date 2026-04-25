"""Schema, gap, and continuity validators for downloader outputs.

Each validator returns a dataclass report with a status in
``{"PASS", "WARN", "FAIL"}`` and a small ``as_dict`` for JSON logging.
Callers decide whether to act on a ``WARN``; ``FAIL`` means the data is
not fit for downstream use.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import pandas as pd

from downloader.constants import COLS_KLINE, COLS_TICK, KLINE_FREQ, OI_FREQ

Status = Literal["PASS", "WARN", "FAIL"]


@dataclass(slots=True)
class GridReport:
    kind: str
    symbol: str
    interval: str
    status: Status
    rows: int
    expected_rows: int
    missing_bars: int
    duplicates: int
    nulls: int
    first_ts: pd.Timestamp | None
    last_ts: pd.Timestamp | None
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind, "symbol": self.symbol, "interval": self.interval,
            "status": self.status, "rows": self.rows, "expected_rows": self.expected_rows,
            "missing_bars": self.missing_bars, "duplicates": self.duplicates,
            "nulls": self.nulls,
            "first_ts": str(self.first_ts) if self.first_ts is not None else "",
            "last_ts": str(self.last_ts) if self.last_ts is not None else "",
            "notes": "; ".join(self.notes),
        }


def validate_ohlcv(
    df: pd.DataFrame,
    symbol: str,
    interval: str,
    *,
    max_missing_frac: float = 0.02,
) -> GridReport:
    if df is None or df.empty:
        return GridReport(
            kind="klines", symbol=symbol, interval=interval, status="FAIL",
            rows=0, expected_rows=0, missing_bars=0, duplicates=0, nulls=0,
            first_ts=None, last_ts=None, notes=["empty"],
        )
    notes: list[str] = []
    missing_cols = [c for c in COLS_KLINE if c not in df.columns]
    if missing_cols:
        notes.append(f"missing cols {missing_cols}")
    ts = df["timestamp"]
    first_ts, last_ts = pd.Timestamp(ts.min()), pd.Timestamp(ts.max())
    dup = int(ts.duplicated().sum())
    freq = KLINE_FREQ.get(interval)
    expected, missing = 0, 0
    if freq is not None:
        grid = pd.date_range(start=first_ts, end=last_ts, freq=freq, tz="UTC")
        expected = len(grid)
        missing = max(0, expected - len(ts.unique()))
    nulls = int(df[list(COLS_KLINE)].isna().sum().sum()) if not missing_cols else 0
    neg = 0
    ohlc_bad = 0
    if not missing_cols:
        price_cols = ["open", "high", "low", "close"]
        neg = int((df[price_cols] <= 0).any(axis=1).sum())
        row_hi = df[price_cols].max(axis=1)
        row_lo = df[price_cols].min(axis=1)
        ohlc_bad = int(((df["high"] < row_hi) | (df["low"] > row_lo)).sum())
    status: Status = "PASS"
    if missing_cols or neg > 0 or ohlc_bad > 0:
        status = "FAIL"
        if neg:
            notes.append(f"{neg} non-positive prices")
        if ohlc_bad:
            notes.append(f"{ohlc_bad} OHLC violations")
    elif dup or nulls:
        status = "WARN"
    elif expected > 0 and missing / expected > max_missing_frac:
        status = "WARN"
        notes.append(f"{missing} missing bars ({missing / expected:.2%})")
    return GridReport(
        kind="klines", symbol=symbol, interval=interval, status=status,
        rows=len(df), expected_rows=expected, missing_bars=missing,
        duplicates=dup, nulls=nulls, first_ts=first_ts, last_ts=last_ts, notes=notes,
    )


def validate_grid(
    df: pd.DataFrame,
    *,
    kind: str,
    symbol: str,
    interval: str,
    freq: str,
    max_missing_frac: float = 0.05,
) -> GridReport:
    """Generic time-grid check for OI / mark / index / premium / funding."""
    if df is None or df.empty:
        return GridReport(
            kind=kind, symbol=symbol, interval=interval, status="FAIL",
            rows=0, expected_rows=0, missing_bars=0, duplicates=0, nulls=0,
            first_ts=None, last_ts=None, notes=["empty"],
        )
    notes: list[str] = []
    ts = df["timestamp"]
    first_ts, last_ts = pd.Timestamp(ts.min()), pd.Timestamp(ts.max())
    dup = int(ts.duplicated().sum())
    grid = pd.date_range(start=first_ts, end=last_ts, freq=freq, tz="UTC")
    expected = len(grid)
    missing = max(0, expected - len(ts.unique()))
    nulls = int(df.isna().sum().sum())
    status: Status = "PASS"
    if dup or nulls:
        status = "WARN"
    if expected > 0 and missing / expected > max_missing_frac:
        status = "WARN"
        notes.append(f"{missing} missing ({missing / expected:.2%})")
    return GridReport(
        kind=kind, symbol=symbol, interval=interval, status=status,
        rows=len(df), expected_rows=expected, missing_bars=missing,
        duplicates=dup, nulls=nulls, first_ts=first_ts, last_ts=last_ts, notes=notes,
    )


def oi_freq_for(interval: str) -> str:
    f = OI_FREQ.get(interval)
    if f is None:
        raise ValueError(f"unknown OI interval {interval!r}")
    return f


@dataclass(slots=True)
class TickReport:
    symbol: str
    status: Status
    rows: int
    first_ts: pd.Timestamp | None
    last_ts: pd.Timestamp | None
    duplicate_ids: int
    non_monotonic_ts: int
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {
            "symbol": self.symbol, "status": self.status, "rows": self.rows,
            "first_ts": str(self.first_ts) if self.first_ts is not None else "",
            "last_ts": str(self.last_ts) if self.last_ts is not None else "",
            "duplicate_ids": self.duplicate_ids,
            "non_monotonic_ts": self.non_monotonic_ts,
            "notes": "; ".join(self.notes),
        }


def validate_ticks(df: pd.DataFrame, symbol: str) -> TickReport:
    if df is None or df.empty:
        return TickReport(symbol, "FAIL", 0, None, None, 0, 0, ["empty"])
    missing_cols = [c for c in COLS_TICK if c not in df.columns]
    notes: list[str] = []
    if missing_cols:
        notes.append(f"missing cols {missing_cols}")
    dup = 0
    if "trade_id" in df.columns:
        dup = int(df["trade_id"].duplicated().sum())
    non_mono = int((df["timestamp"].diff().dt.total_seconds() < -1.0).sum())
    status: Status = "PASS"
    if missing_cols:
        status = "FAIL"
    elif dup:
        status = "WARN"
        notes.append(f"{dup} duplicate trade_ids")
    if non_mono:
        status = "WARN"
        notes.append(f"{non_mono} out-of-order timestamps (>1s backstep)")
    return TickReport(
        symbol=symbol, status=status, rows=len(df),
        first_ts=pd.Timestamp(df["timestamp"].min()),
        last_ts=pd.Timestamp(df["timestamp"].max()),
        duplicate_ids=dup, non_monotonic_ts=non_mono, notes=notes,
    )
