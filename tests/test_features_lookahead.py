"""No-lookahead contract: truncating the tail must not move earlier values.

Run against each layer independently so a single broken feature is
easy to localise.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from features import (
    FeatureConfig,
    check_no_lookahead,
    compute_layer1,
    compute_layer4,
    compute_layer5,
    compute_layer8,
)


def _ohlcv(n: int, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
    steps = rng.normal(0, 0.003, size=n)
    close = 100.0 * np.exp(np.cumsum(steps))
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) * (1.0 + np.abs(rng.normal(0, 0.001, size=n)))
    low = np.minimum(open_, close) * (1.0 - np.abs(rng.normal(0, 0.001, size=n)))
    vol = rng.uniform(10.0, 500.0, size=n)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": vol},
        index=ts,
    )


def _assert_all_ok(reports):
    bad = [r for r in reports if not r.ok]
    assert not bad, "\n".join(f"{r.feature}: {r.detail}" for r in bad)


def test_layer1_no_lookahead():
    cfg = FeatureConfig()
    df = _ohlcv(120)
    names = ["ret", "log_ret", "rv_20", "atr_14", "atr_14_pct",
             "parkinson_20", "garman_klass_20", "yang_zhang_20"]
    reports = check_no_lookahead(lambda d: compute_layer1(d, cfg), df, names)
    _assert_all_ok(reports)


def test_layer4_no_lookahead():
    cfg = FeatureConfig()
    df = _ohlcv(200)
    names = ["rsi_14", "macd_hist", "bb_width", "bb_pct_b",
             "ema_8_dist", "ema_21_dist", "ema_50_dist",
             "adx_14", "plus_di_14", "minus_di_14",
             "vwap_rolling_20_dist", "obv", "obv_slope_20",
             "supertrend_dir"]
    reports = check_no_lookahead(lambda d: compute_layer4(d, cfg), df, names)
    _assert_all_ok(reports)


def test_layer5_no_lookahead():
    cfg = FeatureConfig()
    df = _ohlcv(150)
    rng = np.random.default_rng(7)
    df["funding_rate"] = np.linspace(-1e-4, 3e-4, len(df))
    df["oi"] = 1_000_000.0 + np.cumsum(rng.normal(0, 50.0, len(df)))
    df["mark_price"] = df["close"] * (1.0 + 1e-4)
    df["index_price"] = df["close"]
    names = ["funding_z_20", "oi_chg_1h", "oi_chg_pct_1h",
             "oi_z_50", "basis_bps", "basis_z_50"]
    reports = check_no_lookahead(lambda d: compute_layer5(d, cfg), df, names)
    _assert_all_ok(reports)


def test_layer8_no_lookahead():
    cfg = FeatureConfig()
    df = _ohlcv(260)
    names = ["poc_price_200", "vah_price_200", "val_price_200",
             "poc_dist", "vah_dist", "val_dist", "value_area_width_200"]
    reports = check_no_lookahead(lambda d: compute_layer8(d, cfg), df, names)
    _assert_all_ok(reports)
