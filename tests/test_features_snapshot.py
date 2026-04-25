"""Offline tests for `features.snapshot` (Phase 5).

Uses a fake REST client that returns pre-generated synthetic frames
for 3 symbols. Validates the full build -> peer -> save pipeline
without hitting the network.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from features import (
    SNAPSHOT_COLUMNS,
    FeatureConfig,
    SymbolBundle,
    build_snapshot,
    build_snapshot_for_symbol,
    fetch_symbol_bundle,
    save_snapshot,
)


def _ohlcv(n: int, freq: str, seed: int = 1, base: float = 100.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2024-02-01", periods=n, freq=freq, tz="UTC")
    steps = rng.normal(0, 0.002, size=n)
    close = base * np.exp(np.cumsum(steps))
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) * (1.0 + np.abs(rng.normal(0, 0.001, size=n)))
    low = np.minimum(open_, close) * (1.0 - np.abs(rng.normal(0, 0.001, size=n)))
    vol = rng.uniform(10.0, 500.0, size=n)
    df = pd.DataFrame({"timestamp": ts, "open": open_, "high": high,
                       "low": low, "close": close, "volume": vol,
                       "turnover": close * vol})
    return df


class _FakeRest:
    """Minimal Bybit RestClient stand-in returning synthetic data."""

    def __init__(self, *, n15: int = 400, base_map: dict[str, float] | None = None):
        self.n15 = n15
        self.base_map = base_map or {"BTCUSDT": 30_000.0, "ETHUSDT": 2_000.0, "SOLUSDT": 120.0}

    def _seed(self, symbol: str) -> int:
        return abs(hash(symbol)) % (2**31)

    async def klines(self, symbol, interval, start, end):
        base = self.base_map.get(symbol, 50.0)
        freq_map = {"15": "15min", "60": "1h", "240": "4h"}
        freq = freq_map.get(str(interval), "15min")
        return _ohlcv(self.n15 if freq == "15min" else max(120, self.n15 // 4),
                      freq, seed=self._seed(symbol + interval), base=base)

    async def mark_klines(self, symbol, interval, start, end):
        df = await self.klines(symbol, interval, start, end)
        df = df[["timestamp", "close"]].copy()
        df["close"] = df["close"] * (1.0 + 1e-4)
        return df

    async def index_klines(self, symbol, interval, start, end):
        df = await self.klines(symbol, interval, start, end)
        return df[["timestamp", "close"]].copy()

    async def funding(self, symbol, start, end):
        ts = pd.date_range(end - pd.Timedelta(days=5), end, freq="8h", tz="UTC")
        rng = np.random.default_rng(self._seed(symbol + "fund"))
        return pd.DataFrame({"timestamp": ts,
                             "funding_rate": rng.uniform(-2e-4, 4e-4, len(ts))})

    async def open_interest(self, symbol, interval, start, end):
        ts = pd.date_range(end - pd.Timedelta(days=5), end, freq="1h", tz="UTC")
        rng = np.random.default_rng(self._seed(symbol + "oi"))
        oi = 1e6 + np.cumsum(rng.normal(0, 50.0, len(ts)))
        return pd.DataFrame({"timestamp": ts, "open_interest": oi})


def _bundle_from_fake(rest: _FakeRest, symbol: str, *, ref_15m=None) -> SymbolBundle:
    import asyncio
    now = pd.Timestamp("2024-02-20", tz="UTC")
    return asyncio.get_event_loop().run_until_complete(
        fetch_symbol_bundle(rest, symbol, now=now, reference_15m=ref_15m)
    )


@pytest.mark.asyncio
async def test_build_snapshot_for_symbol_produces_single_row():
    rest = _FakeRest(n15=500)
    now = pd.Timestamp("2024-02-20", tz="UTC")
    bundle = await fetch_symbol_bundle(rest, "BTCUSDT", now=now, reference_15m=None)
    bundle.ref_15m = bundle.base_15m  # self-ref
    row = build_snapshot_for_symbol(bundle, FeatureConfig())
    assert len(row) == 1
    assert row["symbol"].iloc[0] == "BTCUSDT"
    # Schema: every SNAPSHOT_COLUMNS name is present.
    for c in SNAPSHOT_COLUMNS:
        assert c in row.columns, f"missing {c}"


@pytest.mark.asyncio
async def test_build_snapshot_multi_symbol_and_peer_layer():
    rest = _FakeRest(n15=500)
    now = pd.Timestamp("2024-02-20", tz="UTC")
    snap, report = await build_snapshot(
        ["BTCUSDT", "ETHUSDT", "SOLUSDT"], rest, FeatureConfig(),
        now=now, concurrency=3,
    )
    # Column contract
    assert list(snap.columns) == list(SNAPSHOT_COLUMNS)
    assert report.n_built == 3
    assert report.n_failed == 0
    # Peer columns populated
    for c in ("rank_turnover_24h", "cluster_id", "cluster_leader",
              "rs_vs_btc_24h"):
        assert c in snap.columns
    # BTC vs itself = 0 relative strength
    btc_row = snap.loc[snap["symbol"] == "BTCUSDT"].iloc[0]
    assert btc_row["rs_vs_btc_24h"] == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_save_snapshot_writes_parquet_json_meta(tmp_path: Path):
    rest = _FakeRest(n15=500)
    now = pd.Timestamp("2024-02-20", tz="UTC")
    snap, report = await build_snapshot(
        ["BTCUSDT", "ETHUSDT", "SOLUSDT"], rest, FeatureConfig(),
        now=now, concurrency=3,
    )
    paths = save_snapshot(snap, "test-run", FeatureConfig(),
                          runs_root=tmp_path, report=report)
    assert paths["parquet"].exists()
    assert paths["json"].exists()
    assert paths["meta"].exists()

    roundtrip = pd.read_parquet(paths["parquet"])
    assert list(roundtrip.columns) == list(SNAPSHOT_COLUMNS)
    assert len(roundtrip) == 3

    meta = json.loads(paths["meta"].read_text())
    assert meta["feature_version"] == FeatureConfig().version
    assert meta["n_rows"] == 3
    assert meta["n_built"] == 3


@pytest.mark.asyncio
async def test_atr_pct_rank_96_populated():
    """Regression: atr_14_pct lives in Layer 1 but compute_context only
    received Layer 4 -> atr_pct_rank_96 was NaN for every symbol."""
    rest = _FakeRest(n15=500)
    now = pd.Timestamp("2024-02-20", tz="UTC")
    snap, _ = await build_snapshot(
        ["BTCUSDT", "ETHUSDT", "SOLUSDT"], rest, FeatureConfig(),
        now=now, concurrency=3,
    )
    assert "atr_pct_rank_96" in snap.columns
    assert snap["atr_pct_rank_96"].notna().all(), \
        "atr_pct_rank_96 should be populated on the last bar after 500 15m bars"
    assert (snap["atr_pct_rank_96"] >= 0).all()
    assert (snap["atr_pct_rank_96"] <= 1).all()
    assert "bb_width_rank_96" in snap.columns
    assert snap["bb_width_rank_96"].notna().all()


@pytest.mark.asyncio
async def test_save_snapshot_handles_nat_timestamp(tmp_path: Path):
    """Regression: pd.NaT is NOT an instance of pd.Timestamp, the old
    branch fell through and orjson raised TypeError on NaTType."""
    rest = _FakeRest(n15=500)
    now = pd.Timestamp("2024-02-20", tz="UTC")
    snap, report = await build_snapshot(
        ["BTCUSDT", "ETHUSDT"], rest, FeatureConfig(),
        now=now, concurrency=2,
    )
    snap = snap.copy()
    snap.loc[snap.index[0], "timestamp"] = pd.NaT
    paths = save_snapshot(snap, "test-nat", FeatureConfig(),
                          runs_root=tmp_path, report=report)
    payload = json.loads(paths["json"].read_text())
    assert payload[0]["timestamp"] is None
    assert payload[1]["timestamp"] is not None
