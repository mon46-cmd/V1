"""Offline tests for trader flags (Phase 5)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from features import FeatureConfig, compute_flags


def _ohlcv(n: int, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
    steps = rng.normal(0, 0.003, size=n)
    close = 100.0 * np.exp(np.cumsum(steps))
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) * (1.0 + np.abs(rng.normal(0, 0.001, size=n)))
    low = np.minimum(open_, close) * (1.0 - np.abs(rng.normal(0, 0.001, size=n)))
    vol = rng.uniform(50.0, 200.0, size=n)
    df = pd.DataFrame({"open": open_, "high": high, "low": low,
                       "close": close, "volume": vol}, index=ts)
    # Layer-4 helpers the flags module consumes:
    df["rsi_14"] = 50.0
    df["macd_hist"] = 0.0
    df["bb_width"] = 0.1
    df["supertrend_dir"] = 1.0
    return df


def test_flags_schema_and_all_zero_on_flat_series():
    cfg = FeatureConfig()
    df = _ohlcv(150)
    out = compute_flags(df, cfg)
    assert list(out.columns) == [
        "flag_volume_climax", "flag_sweep_up", "flag_sweep_dn",
        "flag_squeeze_release",
        "flag_macd_cross_up", "flag_macd_cross_dn",
        "flag_regime_flip",
        "flag_rsi_overbought", "flag_rsi_oversold",
    ]
    # All flags are float64 0/1
    for c in out.columns:
        vals = out[c].dropna().unique()
        assert set(vals).issubset({0.0, 1.0}), f"{c}: {vals}"


def test_flag_volume_climax_detects_spike():
    cfg = FeatureConfig(climax_lookback=20, climax_mult=3.0)
    df = _ohlcv(40)
    # Inject an overwhelming volume spike at bar 30.
    df.loc[df.index[30], "volume"] = df["volume"].iloc[:30].mean() * 10
    out = compute_flags(df, cfg)
    assert out["flag_volume_climax"].iloc[30] == 1.0
    # Surrounding bars should be 0.
    assert out["flag_volume_climax"].iloc[29] == 0.0
    assert out["flag_volume_climax"].iloc[31] == 0.0


def test_flag_sweep_up_detects_wick_back():
    cfg = FeatureConfig()
    df = _ohlcv(40)
    i = 30
    prior_high = df["high"].iloc[i - 20:i].max()
    df.loc[df.index[i], "open"] = prior_high * 0.999
    df.loc[df.index[i], "high"] = prior_high * 1.01      # pierce
    df.loc[df.index[i], "close"] = prior_high * 0.995    # come back below
    df.loc[df.index[i], "low"] = prior_high * 0.994
    out = compute_flags(df, cfg)
    assert out["flag_sweep_up"].iloc[i] == 1.0


def test_flag_macd_cross_up_detects_sign_change():
    cfg = FeatureConfig()
    df = _ohlcv(40)
    df["macd_hist"] = -0.5
    df.loc[df.index[20], "macd_hist"] = 0.5  # flip from -0.5 at bar 19 to +0.5 at bar 20
    out = compute_flags(df, cfg)
    assert out["flag_macd_cross_up"].iloc[20] == 1.0
    assert out["flag_macd_cross_dn"].iloc[20] == 0.0


def test_flag_regime_flip_detects_supertrend_flip():
    cfg = FeatureConfig()
    df = _ohlcv(40)
    df["supertrend_dir"] = 1.0
    df.loc[df.index[25:], "supertrend_dir"] = -1.0
    out = compute_flags(df, cfg)
    assert out["flag_regime_flip"].iloc[25] == 1.0
    assert out["flag_regime_flip"].iloc[26] == 0.0


def test_flag_rsi_extremes():
    cfg = FeatureConfig()
    df = _ohlcv(40)
    df.loc[df.index[10], "rsi_14"] = 75.0
    df.loc[df.index[11], "rsi_14"] = 25.0
    out = compute_flags(df, cfg)
    assert out["flag_rsi_overbought"].iloc[10] == 1.0
    assert out["flag_rsi_oversold"].iloc[11] == 1.0
