"""Layer 1 sanity tests: closed-form checks on synthetic bars."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from features import FeatureConfig, compute_layer1


def _make_ohlc(n: int = 60, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
    steps = rng.normal(0, 0.002, size=n)
    close = 100.0 * np.exp(np.cumsum(steps))
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) * (1.0 + np.abs(rng.normal(0, 0.001, size=n)))
    low = np.minimum(open_, close) * (1.0 - np.abs(rng.normal(0, 0.001, size=n)))
    vol = rng.uniform(100.0, 1000.0, size=n)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": vol},
        index=ts,
    )


def test_layer1_columns():
    cfg = FeatureConfig()
    df = _make_ohlc(50)
    out = compute_layer1(df, cfg)
    for c in ("ret", "log_ret", "rv_20", "atr_14", "atr_14_pct",
              "parkinson_20", "garman_klass_20", "yang_zhang_20"):
        assert c in out.columns, c
    assert len(out) == len(df)


def test_layer1_log_ret_matches_numpy():
    cfg = FeatureConfig()
    df = _make_ohlc(30)
    out = compute_layer1(df, cfg)
    expected = np.log(df["close"].to_numpy()[1:] / df["close"].to_numpy()[:-1])
    np.testing.assert_allclose(out["log_ret"].to_numpy()[1:], expected, atol=1e-12)


def test_layer1_atr_wilder_matches_reference():
    """Hand-compute Wilder ATR and compare."""
    cfg = FeatureConfig()
    df = _make_ohlc(80)
    out = compute_layer1(df, cfg)

    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)  # matches production path (skipna=True on row 0)
    alpha = 1.0 / cfg.window_atr
    ref = tr.ewm(alpha=alpha, adjust=False, min_periods=cfg.window_atr).mean().to_numpy()
    got = out["atr_14"].to_numpy()
    mask = ~np.isnan(ref)
    np.testing.assert_allclose(got[mask], ref[mask], atol=1e-12)


def test_layer1_parkinson_closed_form():
    cfg = FeatureConfig()
    df = _make_ohlc(40)
    out = compute_layer1(df, cfg)
    hl = np.log(df["high"] / df["low"])
    ref = np.sqrt((hl ** 2).rolling(cfg.window_parkinson, min_periods=cfg.window_parkinson).mean()
                  / (4.0 * math.log(2.0)))
    # ignore NaN positions
    a = out["parkinson_20"].dropna().to_numpy()
    b = ref.dropna().to_numpy()
    np.testing.assert_allclose(a, b, atol=1e-12)


def test_layer1_yang_zhang_k_is_finite():
    cfg = FeatureConfig()
    df = _make_ohlc(60)
    out = compute_layer1(df, cfg)
    yz = out["yang_zhang_20"].dropna()
    assert len(yz) > 0
    assert np.isfinite(yz.to_numpy()).all()
    assert (yz.to_numpy() >= 0).all()


def test_layer1_no_lookahead():
    """Truncating the last bar must not change earlier values."""
    cfg = FeatureConfig()
    df = _make_ohlc(80)
    full = compute_layer1(df, cfg)
    trunc = compute_layer1(df.iloc[:-1], cfg)
    for c in full.columns:
        a = full[c].iloc[:-1].to_numpy(dtype="float64")
        b = trunc[c].to_numpy(dtype="float64")
        mask = ~(np.isnan(a) | np.isnan(b))
        if mask.any():
            np.testing.assert_allclose(a[mask], b[mask], atol=1e-12)
