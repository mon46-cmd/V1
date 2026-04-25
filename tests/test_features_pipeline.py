"""End-to-end snapshot pipeline test with a synthetic bundle."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from features import (
    DASHBOARD_COLUMNS,
    REGISTRY,
    SNAPSHOT_COLUMNS,
    FeatureConfig,
    SymbolBundle,
    compute,
    profile_columns,
)


def _ohlcv(n: int, freq: str, *, seed: int = 11, start: str = "2024-01-01") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ts = pd.date_range(start, periods=n, freq=freq, tz="UTC")
    steps = rng.normal(0, 0.002, size=n)
    close = 30_000.0 * np.exp(np.cumsum(steps))
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) * (1.0 + np.abs(rng.normal(0, 0.001, size=n)))
    low = np.minimum(open_, close) * (1.0 - np.abs(rng.normal(0, 0.001, size=n)))
    vol = rng.uniform(10.0, 200.0, size=n)
    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": vol},
        index=ts,
    )
    df.index.name = "timestamp"
    return df


def _make_bundle(n: int = 400) -> SymbolBundle:
    base = _ohlcv(n, "15min", seed=1)
    h1 = _ohlcv(n // 4 + 10, "1h", seed=2)
    h4 = _ohlcv(n // 16 + 10, "4h", seed=3)
    ref = _ohlcv(n, "15min", seed=9)

    # Funding -- sparse 8h, aligned on settle hours.
    fund_idx = pd.date_range(base.index[0].floor("8h"), base.index[-1], freq="8h", tz="UTC")
    fund = pd.DataFrame({"funding_rate": np.linspace(-0.0002, 0.0004, len(fund_idx))}, index=fund_idx)
    fund.index.name = "timestamp"

    # OI -- hourly.
    oi_idx = pd.date_range(base.index[0], base.index[-1], freq="1h", tz="UTC")
    rng = np.random.default_rng(5)
    oi = pd.DataFrame({"oi": 1_000_000.0 + np.cumsum(rng.normal(0, 100.0, len(oi_idx)))},
                      index=oi_idx)
    oi.index.name = "timestamp"

    # Mark / index -- 15m.
    mk = base[["close"]] * (1.0 + 1e-4)
    ix = base[["close"]].copy()

    return SymbolBundle(
        symbol="BTCUSDT",
        base_15m=base,
        bars_1h=h1,
        bars_4h=h4,
        funding=fund,
        oi=oi,
        mark_15m=mk,
        index_15m=ix,
        ref_15m=ref,
    )


def test_snapshot_profile_columns_exact():
    bundle = _make_bundle(400)
    cfg = FeatureConfig()
    out = compute("snapshot", bundle, cfg=cfg)
    assert list(out.columns) == list(SNAPSHOT_COLUMNS)
    assert len(out) == 400


def test_snapshot_last_row_core_not_nan():
    bundle = _make_bundle(400)
    out = compute("snapshot", bundle)
    last = out.iloc[-1]
    core_always = [
        "symbol", "timestamp", "open", "high", "low", "close", "volume",
        "atr_14", "rv_20", "rsi_14", "ema_50_dist", "macd_hist",
        "utc_hour_sin", "utc_hour_cos",
        "is_funding_minute", "time_to_next_funding_sec",
        "trend_score_mtf",
    ]
    for c in core_always:
        v = last[c]
        if isinstance(v, float):
            assert not np.isnan(v), f"{c} is NaN"


def test_dashboard_profile_superset_of_snapshot():
    bundle = _make_bundle(400)
    out = compute("dashboard", bundle)
    # Dashboard should contain every registry-declared feature name.
    for name in REGISTRY.names():
        assert name in out.columns, name


def test_profile_columns_unknown_raises():
    with pytest.raises(ValueError):
        profile_columns("not-a-profile")


def test_compute_is_pure():
    """Re-running on the same bundle must give identical output."""
    bundle = _make_bundle(300)
    a = compute("snapshot", bundle).drop(columns=["timestamp"])
    b = compute("snapshot", bundle).drop(columns=["timestamp"])
    pd.testing.assert_frame_equal(a, b)
