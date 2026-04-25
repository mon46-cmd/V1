"""Tests for `downloader.universe`.

Offline tests construct synthetic ticker + instruments frames and
assert each rejection reason fires independently, and that the final
ranking respects ``turnover_24h`` descending. A LIVE end-to-end test
exercises :func:`build_universe` against Bybit and asserts we get
exactly ``cfg.universe_size`` rows with loose filters.
"""
from __future__ import annotations

import asyncio
import os
import socket
from pathlib import Path

import pandas as pd
import pytest

from core.config import load_config
from downloader.universe import (
    REJ_AGE_TOO_YOUNG,
    REJ_EXCLUDED_SUBSTRING,
    REJ_EXCLUDED_SYMBOL,
    REJ_NO_QUOTE,
    REJ_NOT_LINEAR,
    REJ_NOT_TRADING,
    REJ_PRICE_TOO_LOW,
    REJ_SPREAD_TOO_WIDE,
    REJ_TURNOVER_TOO_LOW,
    REJ_WRONG_QUOTE,
    UNIVERSE_COLUMNS,
    build_universe,
    filter_universe,
    load_universe,
    save_universe,
)


# ---------------- fixtures ----------------------------------------------
_NOW_MS = 1_713_657_600_000  # 2024-04-21 00:00 UTC (deterministic)


def _inst_row(
    symbol: str,
    *,
    status: str = "Trading",
    quote: str = "USDT",
    launch_ms: int = _NOW_MS - 365 * 86_400_000,
    tick: str = "0.01",
    qty_step: str = "0.001",
    min_qty: str = "0.001",
    contract_type: str = "LinearPerpetual",
) -> dict[str, object]:
    return {
        "symbol": symbol,
        "status": status,
        "quoteCoin": quote,
        "baseCoin": symbol.replace(quote, ""),
        "contractType": contract_type,
        "launchTime": pd.Timestamp(launch_ms, unit="ms", tz="UTC"),
        "priceFilter.tickSize": tick,
        "lotSizeFilter.qtyStep": qty_step,
        "lotSizeFilter.minOrderQty": min_qty,
    }


def _ticker(
    symbol: str,
    *,
    price: float = 100.0,
    bid: float = 99.99,
    ask: float = 100.01,
    turnover: float = 50_000_000.0,
    volume: float = 500_000.0,
    oi: float = 1_000.0,
    oi_value: float = 100_000.0,
    mark: float = 100.0,
    index: float = 100.0,
    funding: float = 0.0001,
    change_pct: float = 0.01,
) -> dict[str, object]:
    return {
        "symbol": symbol,
        "price": price,
        "bid": bid,
        "ask": ask,
        "mark_price": mark,
        "index_price": index,
        "volume_24h": volume,
        "turnover_24h": turnover,
        "open_interest": oi,
        "open_interest_value": oi_value,
        "funding_rate": funding,
        "price_change_24h_pct": change_pct,
        "next_funding_ms": 0,
        "high_24h": price * 1.02,
        "low_24h": price * 0.98,
    }


def _instruments_df(rows: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(rows)


# ---------------- offline unit tests ------------------------------------
def test_happy_path_returns_topn_sorted_by_turnover(tmp_data_root: Path) -> None:
    cfg = load_config()
    tickers = [
        _ticker("BTCUSDT", turnover=9e9),
        _ticker("ETHUSDT", turnover=5e9),
        _ticker("SOLUSDT", turnover=1e9),
        _ticker("XRPUSDT", turnover=5e8),
    ]
    inst = _instruments_df([_inst_row(t["symbol"]) for t in tickers])
    survivors, rej = filter_universe(tickers, inst, cfg, now_ms=_NOW_MS, size=3)
    assert list(survivors["symbol"]) == ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    assert list(survivors.columns) == list(UNIVERSE_COLUMNS)
    assert rej.empty
    # Spread bps computed from bid/ask.
    row = survivors.iloc[0]
    assert pytest.approx(row["spread_bps"], rel=1e-6) == (100.01 - 99.99) / 100.0 * 10_000.0


def test_rejects_not_trading(tmp_data_root: Path) -> None:
    cfg = load_config()
    tickers = [_ticker("BTCUSDT")]
    inst = _instruments_df([_inst_row("BTCUSDT", status="PreLaunch")])
    survivors, rej = filter_universe(tickers, inst, cfg, now_ms=_NOW_MS)
    assert survivors.empty
    assert list(rej["reason"]) == [REJ_NOT_TRADING]


def test_rejects_wrong_quote(tmp_data_root: Path) -> None:
    cfg = load_config()
    tickers = [_ticker("BTCUSDC")]
    inst = _instruments_df([_inst_row("BTCUSDC", quote="USDC")])
    _, rej = filter_universe(tickers, inst, cfg, now_ms=_NOW_MS)
    assert rej.iloc[0]["reason"] == REJ_WRONG_QUOTE


def test_rejects_excluded_symbol_and_substring(tmp_data_root: Path) -> None:
    cfg = load_config()
    tickers = [_ticker("USDCUSDT"), _ticker("1000PEPE-USDT", ask=1.0, bid=0.999)]
    inst = _instruments_df([_inst_row("USDCUSDT"), _inst_row("1000PEPE-USDT")])
    _, rej = filter_universe(tickers, inst, cfg, now_ms=_NOW_MS)
    reasons = sorted(rej["reason"].tolist())
    assert REJ_EXCLUDED_SYMBOL in reasons
    assert REJ_EXCLUDED_SUBSTRING in reasons


def test_rejects_young_listing(tmp_data_root: Path) -> None:
    cfg = load_config()
    tickers = [_ticker("NEWCOINUSDT")]
    inst = _instruments_df([_inst_row(
        "NEWCOINUSDT", launch_ms=_NOW_MS - 5 * 86_400_000,
    )])
    _, rej = filter_universe(tickers, inst, cfg, now_ms=_NOW_MS, min_listing_age_days=30)
    assert rej.iloc[0]["reason"] == REJ_AGE_TOO_YOUNG


def test_rejects_low_turnover(tmp_data_root: Path) -> None:
    cfg = load_config()
    tickers = [_ticker("THINUSDT", turnover=1_000.0)]
    inst = _instruments_df([_inst_row("THINUSDT")])
    _, rej = filter_universe(tickers, inst, cfg, now_ms=_NOW_MS, min_turnover_usd_24h=1_000_000.0)
    assert rej.iloc[0]["reason"] == REJ_TURNOVER_TOO_LOW


def test_rejects_low_price(tmp_data_root: Path) -> None:
    cfg = load_config()
    tickers = [_ticker("DUSTUSDT", price=0.0, bid=0.0, ask=0.0)]
    inst = _instruments_df([_inst_row("DUSTUSDT")])
    _, rej = filter_universe(tickers, inst, cfg, now_ms=_NOW_MS, min_price_usd=0.0001)
    assert rej.iloc[0]["reason"] == REJ_PRICE_TOO_LOW


def test_rejects_wide_spread(tmp_data_root: Path) -> None:
    cfg = load_config()
    tickers = [_ticker("WIDEUSDT", price=100.0, bid=99.0, ask=101.0)]  # ~200 bps
    inst = _instruments_df([_inst_row("WIDEUSDT")])
    _, rej = filter_universe(tickers, inst, cfg, now_ms=_NOW_MS, max_spread_bps=10.0)
    assert rej.iloc[0]["reason"] == REJ_SPREAD_TOO_WIDE


def test_rejects_no_quote(tmp_data_root: Path) -> None:
    cfg = load_config()
    tickers = [_ticker("NOBIDUSDT", bid=0.0, ask=0.0)]
    inst = _instruments_df([_inst_row("NOBIDUSDT")])
    _, rej = filter_universe(tickers, inst, cfg, now_ms=_NOW_MS)
    assert rej.iloc[0]["reason"] == REJ_NO_QUOTE


def test_rejects_missing_instrument_row(tmp_data_root: Path) -> None:
    cfg = load_config()
    tickers = [_ticker("GHOSTUSDT")]
    inst = _instruments_df([])  # no rows
    _, rej = filter_universe(tickers, inst, cfg, now_ms=_NOW_MS)
    assert rej.iloc[0]["reason"] == REJ_NOT_LINEAR


def test_save_and_load_roundtrip(tmp_data_root: Path) -> None:
    cfg = load_config()
    tickers = [_ticker("BTCUSDT", turnover=9e9), _ticker("ETHUSDT", turnover=5e9)]
    inst = _instruments_df([_inst_row("BTCUSDT"), _inst_row("ETHUSDT")])
    survivors, rejections = filter_universe(tickers, inst, cfg, now_ms=_NOW_MS)

    path = save_universe(survivors, "run-test", cfg, rejections=rejections)
    assert path.exists()
    back = load_universe("run-test", cfg)
    assert back is not None
    assert list(back["symbol"]) == list(survivors["symbol"])
    # Rejections file is optional; empty rej should not create it.
    assert not (cfg.run_root / "run-test" / "universe_rejections.parquet").exists()


# ---------------- live end-to-end test ---------------------------------
def _bybit_reachable() -> bool:
    if os.getenv("BYBIT_OFFLINE", "").lower() in ("1", "true", "yes"):
        return False
    try:
        with socket.create_connection(("api.bybit.com", 443), timeout=3.0):
            return True
    except OSError:
        return False


@pytest.mark.skipif(not _bybit_reachable(), reason="Bybit REST unreachable or BYBIT_OFFLINE")
@pytest.mark.timeout(60)
def test_build_universe_live_returns_exact_size(tmp_data_root: Path) -> None:
    cfg = load_config()
    # Loosen filters so we are guaranteed to fill the universe.
    survivors, rej = asyncio.run(build_universe(
        cfg,
        size=10,
        min_turnover_usd_24h=1_000_000.0,   # 1M$ floor
        max_spread_bps=50.0,
        min_listing_age_days=7,
        min_price_usd=1e-6,
    ))
    assert len(survivors) == 10
    assert survivors["turnover_24h"].is_monotonic_decreasing
    assert survivors["spread_bps"].max() <= 50.0
    assert (survivors["turnover_24h"] >= 1_000_000.0).all()
    # BTCUSDT should always clear loose filters.
    assert "BTCUSDT" in set(survivors["symbol"])
    # Every row has non-null tick_size and qty_step.
    assert survivors["tick_size"].notna().all()
    assert survivors["qty_step"].notna().all()
    # Rejection frame is well-shaped.
    if not rej.empty:
        assert set(rej.columns) == {"symbol", "reason", "detail"}
