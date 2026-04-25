"""Universe scanner: filter Bybit linear perps down to the top N tradable.

The scanner runs in three conceptual steps:

1. **Collect.** Pull the full ``tickers`` snapshot and the
   ``instruments`` spec frame via :class:`downloader.rest.RestClient`.
2. **Filter.** Drop rows that fail any rule (wrong quote, not
   Trading, stablecoin / exclusion match, too-young listing, too-thin
   turnover, too-wide spread, too-low price). Rejection reasons are
   recorded so we can surface them in the CLI and the dashboard.
3. **Rank.** Sort survivors by 24h turnover descending and take the
   top ``cfg.universe_size`` rows.

The result is a DataFrame with one row per surviving symbol and enough
metadata for the snapshot builder to act on (tick size, qty step,
launch time, mark price, funding rate, etc.). A second DataFrame lists
every rejected symbol with a reason code so we can diagnose empty
universes quickly.

This module is pure I/O plus a pure filter; it does not touch the
websocket or the tick pipeline.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from core.config import Config
from core.paths import run_dir
from downloader.cache import ParquetCache
from downloader.http import HttpClient
from downloader.rest import RestClient

log = logging.getLogger(__name__)


# ----------------- rejection reason codes --------------------------------
REJ_NOT_LINEAR = "not_linear"
REJ_NOT_TRADING = "not_trading"
REJ_WRONG_QUOTE = "wrong_quote"
REJ_EXCLUDED_SYMBOL = "excluded_symbol"
REJ_EXCLUDED_SUBSTRING = "excluded_substring"
REJ_AGE_TOO_YOUNG = "age_too_young"
REJ_TURNOVER_TOO_LOW = "turnover_too_low"
REJ_PRICE_TOO_LOW = "price_too_low"
REJ_SPREAD_TOO_WIDE = "spread_too_wide"
REJ_NO_QUOTE = "no_quote"  # bid/ask missing -> cannot compute spread

UNIVERSE_COLUMNS: tuple[str, ...] = (
    "symbol",
    "price",
    "bid",
    "ask",
    "spread_bps",
    "mark_price",
    "index_price",
    "turnover_24h",
    "volume_24h",
    "open_interest",
    "open_interest_value",
    "funding_rate",
    "price_change_24h_pct",
    "tick_size",
    "qty_step",
    "min_order_qty",
    "launch_time",
    "age_days",
)


@dataclass(frozen=True)
class UniverseReport:
    """Summary of one universe-build run."""
    run_id: str
    size_requested: int
    size_returned: int
    total_candidates: int
    rejections: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "size_requested": self.size_requested,
            "size_returned": self.size_returned,
            "total_candidates": self.total_candidates,
            "rejections": dict(self.rejections),
        }


# ----------------- pure filter ------------------------------------------
def filter_universe(
    tickers: list[dict[str, Any]],
    instruments: pd.DataFrame,
    cfg: Config,
    *,
    now_ms: int | None = None,
    size: int | None = None,
    min_turnover_usd_24h: float | None = None,
    max_spread_bps: float | None = None,
    min_listing_age_days: int | None = None,
    min_price_usd: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pure filter: in -> (survivors, rejections).

    Parameters override ``cfg`` per call so the CLI can loosen filters
    without mutating the frozen config.

    Returns
    -------
    survivors : DataFrame with ``UNIVERSE_COLUMNS`` sorted by
        ``turnover_24h`` desc, capped at ``size`` rows.
    rejections : DataFrame with ``[symbol, reason, detail]``.
    """
    size = int(size if size is not None else cfg.universe_size)
    min_turnover = float(
        min_turnover_usd_24h if min_turnover_usd_24h is not None
        else cfg.min_turnover_usd_24h
    )
    max_spread = float(max_spread_bps if max_spread_bps is not None else cfg.max_spread_bps)
    min_age = int(min_listing_age_days if min_listing_age_days is not None else cfg.min_listing_age_days)
    min_price = float(min_price_usd if min_price_usd is not None else cfg.min_price_usd)
    now_ts = pd.Timestamp(now_ms, unit="ms", tz="UTC") if now_ms is not None else pd.Timestamp.now(tz="UTC")

    inst_by_sym = _index_instruments(instruments)
    exclude_set = {s.upper() for s in cfg.exclude_symbols}
    exclude_subs = tuple(s for s in cfg.exclude_substrings if s)
    quote = cfg.quote_currency.upper()

    rejections: list[dict[str, Any]] = []
    survivors: list[dict[str, Any]] = []

    for t in tickers:
        sym = str(t.get("symbol", "")).upper()
        if not sym:
            continue

        # Instrument spec gate (linear + Trading + quote).
        spec = inst_by_sym.get(sym)
        if spec is None:
            rejections.append({"symbol": sym, "reason": REJ_NOT_LINEAR, "detail": "no instrument row"})
            continue
        if (spec.get("contractType") or "").lower() not in ("linearperpetual", "linearfutures", ""):
            # Empty is permissible: some historical rows lack the field.
            pass
        if (spec.get("status") or "") != "Trading":
            rejections.append({"symbol": sym, "reason": REJ_NOT_TRADING, "detail": spec.get("status", "")})
            continue
        if (spec.get("quoteCoin") or "").upper() != quote:
            rejections.append({"symbol": sym, "reason": REJ_WRONG_QUOTE, "detail": spec.get("quoteCoin", "")})
            continue
        if sym in exclude_set:
            rejections.append({"symbol": sym, "reason": REJ_EXCLUDED_SYMBOL, "detail": ""})
            continue
        if any(s in sym for s in exclude_subs):
            rejections.append({"symbol": sym, "reason": REJ_EXCLUDED_SUBSTRING, "detail": ""})
            continue

        # Listing age.
        launch_ts = spec.get("launchTime")
        age_days = float("nan")
        if isinstance(launch_ts, pd.Timestamp) and not pd.isna(launch_ts):
            age_days = (now_ts - launch_ts).total_seconds() / 86_400.0
            if age_days < min_age:
                rejections.append({
                    "symbol": sym, "reason": REJ_AGE_TOO_YOUNG,
                    "detail": f"{age_days:.1f}d < {min_age}d",
                })
                continue

        price = _safe_float(t.get("price"))
        if not (price > min_price):
            rejections.append({
                "symbol": sym, "reason": REJ_PRICE_TOO_LOW,
                "detail": f"price={price}",
            })
            continue

        turnover = _safe_float(t.get("turnover_24h"))
        if not (turnover >= min_turnover):
            rejections.append({
                "symbol": sym, "reason": REJ_TURNOVER_TOO_LOW,
                "detail": f"{turnover:.0f} < {min_turnover:.0f}",
            })
            continue

        bid = _safe_float(t.get("bid"))
        ask = _safe_float(t.get("ask"))
        if not (bid > 0 and ask > 0 and ask >= bid):
            rejections.append({"symbol": sym, "reason": REJ_NO_QUOTE, "detail": f"bid={bid} ask={ask}"})
            continue
        mid = 0.5 * (bid + ask)
        spread_bps = (ask - bid) / mid * 10_000.0 if mid > 0 else float("inf")
        if spread_bps > max_spread:
            rejections.append({
                "symbol": sym, "reason": REJ_SPREAD_TOO_WIDE,
                "detail": f"{spread_bps:.2f}bps > {max_spread:.2f}bps",
            })
            continue

        survivors.append({
            "symbol": sym,
            "price": price,
            "bid": bid,
            "ask": ask,
            "spread_bps": spread_bps,
            "mark_price": _safe_float(t.get("mark_price")),
            "index_price": _safe_float(t.get("index_price")),
            "turnover_24h": turnover,
            "volume_24h": _safe_float(t.get("volume_24h")),
            "open_interest": _safe_float(t.get("open_interest")),
            "open_interest_value": _safe_float(t.get("open_interest_value")),
            "funding_rate": _safe_float(t.get("funding_rate")),
            "price_change_24h_pct": _safe_float(t.get("price_change_24h_pct")),
            "tick_size": _spec_float(spec, "priceFilter.tickSize"),
            "qty_step": _spec_float(spec, "lotSizeFilter.qtyStep"),
            "min_order_qty": _spec_float(spec, "lotSizeFilter.minOrderQty"),
            "launch_time": launch_ts if isinstance(launch_ts, pd.Timestamp) else pd.NaT,
            "age_days": age_days,
        })

    df = pd.DataFrame(survivors, columns=list(UNIVERSE_COLUMNS))
    if not df.empty:
        df = df.sort_values("turnover_24h", ascending=False, kind="mergesort").head(size).reset_index(drop=True)
    rej = pd.DataFrame(rejections, columns=["symbol", "reason", "detail"])
    return df, rej


# ----------------- orchestration ----------------------------------------
async def build_universe(
    cfg: Config,
    *,
    size: int | None = None,
    min_turnover_usd_24h: float | None = None,
    max_spread_bps: float | None = None,
    min_listing_age_days: int | None = None,
    min_price_usd: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Async end-to-end: fetch tickers + instruments then filter."""
    async with HttpClient(cfg) as http:
        rest = RestClient(http, cfg)
        tickers = await rest.tickers()
        instruments = await rest.instruments()
    return filter_universe(
        tickers, instruments, cfg,
        size=size,
        min_turnover_usd_24h=min_turnover_usd_24h,
        max_spread_bps=max_spread_bps,
        min_listing_age_days=min_listing_age_days,
        min_price_usd=min_price_usd,
    )


def save_universe(
    df: pd.DataFrame,
    run_id: str,
    cfg: Config,
    *,
    rejections: pd.DataFrame | None = None,
    cache: ParquetCache | None = None,
) -> Path:
    """Write the universe DataFrame atomically under the run directory.

    The file is also mirrored into the cache as ``universe/<run_id>.parquet``
    so downstream tools that only know the cache root can read it.
    """
    d = run_dir(cfg, run_id)
    path = d / "universe.parquet"
    _atomic_parquet(df, path)
    if rejections is not None and not rejections.empty:
        _atomic_parquet(rejections, d / "universe_rejections.parquet")
    if cache is not None:
        cache.write_daily(df, "universe", "run", run_id)
    return path


def load_universe(run_id: str, cfg: Config) -> pd.DataFrame | None:
    """Read back a previously saved universe for a given run id."""
    path = cfg.run_root / run_id / "universe.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


# ----------------- internals --------------------------------------------
def _index_instruments(instruments: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Index the normalized instruments frame by symbol."""
    if instruments is None or instruments.empty or "symbol" not in instruments.columns:
        return {}
    out: dict[str, dict[str, Any]] = {}
    recs = instruments.to_dict(orient="records")
    for r in recs:
        sym = str(r.get("symbol", "")).upper()
        if sym:
            out[sym] = r
    return out


def _safe_float(v: Any) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return f


def _spec_float(spec: dict[str, Any], key: str) -> float:
    v = spec.get(key)
    if v is None:
        return float("nan")
    return _safe_float(v)


def _atomic_parquet(df: pd.DataFrame, path: Path) -> None:
    import os
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(tmp, compression="zstd", index=False)
    os.replace(tmp, path)
