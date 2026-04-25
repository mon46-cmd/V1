"""LIVE end-to-end test for Phase 5 snapshot builder.

Fetches real Bybit data for 3 symbols (BTC/ETH/SOL), runs the full
``build_snapshot`` -> peer/cluster pipeline, and validates the
schema + persistence + cross-sectional outputs.

Skipped offline (BYBIT_OFFLINE=1) or when api.bybit.com is unreachable.
"""
from __future__ import annotations

import os
import socket
from pathlib import Path

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


async def test_build_snapshot_live_three_symbols(tmp_path: Path):
    from core.config import load_config
    from downloader.http import HttpClient
    from downloader.rest import RestClient
    from features import (
        SNAPSHOT_COLUMNS,
        FeatureConfig,
        build_snapshot,
        save_snapshot,
    )

    cfg = load_config()
    fcfg = FeatureConfig()
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    now = pd.Timestamp.utcnow().tz_localize(None).tz_localize("UTC").floor("1min")

    async with HttpClient(cfg) as http:
        rest = RestClient(http, cfg)
        snap, report = await build_snapshot(symbols, rest, fcfg, now=now, concurrency=3)

    # Schema contract
    assert list(snap.columns) == list(SNAPSHOT_COLUMNS), \
        f"schema drift: missing={set(SNAPSHOT_COLUMNS)-set(snap.columns)}"
    assert report.n_built == 3, f"only built {report.n_built}/{report.n_requested}: {report.failures}"
    assert report.n_failed == 0
    assert len(snap) == 3
    assert set(snap["symbol"]) == set(symbols)

    # Sanity: latest BTC close > $1000
    btc = snap.loc[snap["symbol"] == "BTCUSDT"].iloc[0]
    assert btc["close"] > 1_000.0
    # ATR percent in a sane range (0.05% .. 30%)
    assert 0.0005 < btc["atr_14_pct"] < 30.0
    # RSI in [0, 100]
    assert 0.0 <= btc["rsi_14"] <= 100.0
    # BTC's relative strength vs itself should be 0
    assert btc["rs_vs_btc_24h"] == pytest.approx(0.0)

    # Peer layer must be populated
    assert snap["cluster_id"].notna().all()
    assert snap["rank_turnover_24h"].notna().all()
    # turnover ranks 1..3 (no ties expected at this size)
    assert sorted(snap["rank_turnover_24h"].astype(int).tolist()) == [1, 2, 3]
    # cluster_size sums match n_symbols
    sizes = snap.groupby("cluster_id")["symbol"].count().sum()
    assert sizes == 3

    # Persist + roundtrip
    paths = save_snapshot(snap, "live-test", fcfg, runs_root=tmp_path, report=report)
    assert paths["parquet"].exists()
    rt = pd.read_parquet(paths["parquet"])
    assert list(rt.columns) == list(SNAPSHOT_COLUMNS)
    assert len(rt) == 3
