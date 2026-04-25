"""LIVE end-to-end test for the Phase 4 feature pipeline.

Fetches real Bybit data for BTCUSDT / ETHUSDT (15m, 1h, 4h OHLCV +
funding + OI + mark + index), runs ``compute("snapshot", bundle)``
and asserts the schema + numeric sanity on the last row.
"""
from __future__ import annotations

import os
import socket

import numpy as np
import pandas as pd
import pytest


pytestmark = [
    pytest.mark.skipif(
        os.getenv("BYBIT_OFFLINE", "").lower() in ("1", "true", "yes"),
        reason="BYBIT_OFFLINE is set",
    ),
]


def _has_internet(host: str = "api.bybit.com", port: int = 443, timeout: float = 3.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


if not _has_internet():  # pragma: no cover
    pytestmark.append(pytest.mark.skip(reason="api.bybit.com unreachable"))


async def _fetch_bundle(rest, symbol: str, ref_df_15m: pd.DataFrame | None = None):
    from features import SymbolBundle

    end = pd.Timestamp.utcnow().tz_localize(None).tz_localize("UTC").floor("1min")
    # ~5 days of 15m gives us 480 bars > 200 bar VP warmup.
    start_15 = end - pd.Timedelta(days=6)
    start_1h = end - pd.Timedelta(days=20)
    start_4h = end - pd.Timedelta(days=80)
    start_fund = end - pd.Timedelta(days=30)
    start_oi = end - pd.Timedelta(days=6)

    k15 = await rest.klines(symbol, "15", start_15, end)
    k1h = await rest.klines(symbol, "60", start_1h, end)
    k4h = await rest.klines(symbol, "240", start_4h, end)
    funding = await rest.funding(symbol, start_fund, end)
    oi = await rest.open_interest(symbol, "1h", start_oi, end)
    mk = await rest.mark_klines(symbol, "15", start_15, end)
    ix = await rest.index_klines(symbol, "15", start_15, end)

    def _idx(df):
        if df.empty:
            return df
        return df.set_index("timestamp")

    return SymbolBundle(
        symbol=symbol,
        base_15m=_idx(k15),
        bars_1h=_idx(k1h),
        bars_4h=_idx(k4h),
        funding=_idx(funding).rename(columns={"funding_rate": "funding_rate"}) if not funding.empty else None,
        oi=_idx(oi).rename(columns={"open_interest": "oi"}) if not oi.empty else None,
        mark_15m=_idx(mk) if not mk.empty else None,
        index_15m=_idx(ix) if not ix.empty else None,
        ref_15m=ref_df_15m,
    )


async def test_pipeline_live_btcusdt_schema_and_nans(tmp_data_root) -> None:  # noqa: ARG001
    from core.config import load_config
    from downloader.http import HttpClient
    from downloader.rest import RestClient
    from features import SNAPSHOT_COLUMNS, FeatureConfig, compute

    cfg_bybit = load_config()
    async with HttpClient(cfg_bybit) as http:
        rest = RestClient(http, cfg_bybit)
        btc_bundle = await _fetch_bundle(rest, "BTCUSDT")
        # For BTC, ref is BTC itself -- self-beta trivially 1.0.
        btc_bundle.ref_15m = btc_bundle.base_15m

    fcfg = FeatureConfig()
    out = compute("snapshot", btc_bundle, cfg=fcfg)

    # Schema
    assert list(out.columns) == list(SNAPSHOT_COLUMNS), "column contract drift"
    assert len(out) >= 400, f"expected >=400 bars, got {len(out)}"

    # Last row sanity
    last = out.iloc[-1]
    assert last["symbol"] == "BTCUSDT"
    for c in ("close", "atr_14", "rv_20", "rsi_14", "ema_50_dist",
              "macd_hist", "bb_width", "adx_14",
              "utc_hour_sin", "utc_hour_cos",
              "is_funding_minute", "time_to_next_funding_sec",
              "trend_score_mtf"):
        v = last[c]
        assert isinstance(v, (int, float, np.floating))
        assert not np.isnan(float(v)), f"{c} is NaN on live last row"

    # Ranges
    assert 0.0 <= float(last["rsi_14"]) <= 100.0
    assert float(last["atr_14"]) > 0
    assert float(last["close"]) > 1000  # BTC is never < $1k
    # MTF attached something
    for c in ("h1_rsi_14", "h4_rsi_14", "h1_ema_50_dist", "h4_ema_50_dist"):
        assert c in out.columns


async def test_pipeline_live_eth_with_btc_reference(tmp_data_root) -> None:  # noqa: ARG001
    from core.config import load_config
    from downloader.http import HttpClient
    from downloader.rest import RestClient
    from features import FeatureConfig, compute

    cfg_bybit = load_config()
    async with HttpClient(cfg_bybit) as http:
        rest = RestClient(http, cfg_bybit)
        btc_bundle = await _fetch_bundle(rest, "BTCUSDT")
        eth_bundle = await _fetch_bundle(rest, "ETHUSDT", ref_df_15m=btc_bundle.base_15m)

    out = compute("snapshot", eth_bundle, cfg=FeatureConfig())
    last = out.iloc[-1]
    beta = float(last["beta_btc_100"])
    corr = float(last["corr_btc_100"])
    assert np.isfinite(beta), "beta should be finite with real BTC ref"
    assert np.isfinite(corr), "corr should be finite with real BTC ref"
    # ETH is strongly (positively) correlated with BTC historically.
    assert 0.3 <= corr <= 1.0, f"eth/btc corr out of expected range: {corr}"
    assert 0.2 <= beta <= 3.0, f"eth/btc beta out of expected range: {beta}"
