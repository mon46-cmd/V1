"""Layer 5 -- derivatives context (funding, open interest, basis).

The inputs here do **not** live on the raw OHLCV kline -- they come
from the separate Bybit REST endpoints (`funding`, `open_interest`) and
from mark + index klines. The caller attaches them upstream with
``features.align.attach_asof`` (backward) so no lookahead is possible.
This module then derives the ratios / z-scores.

If a required input column is missing the output column is filled with
NaN, so the pipeline can still run with partial bundles.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from features.config import FeatureConfig


def compute_layer5(df: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    idx = df.index
    out = pd.DataFrame(index=idx)

    # Funding ---------------------------------------------------------
    fr = _col(df, "funding_rate")
    out["funding_rate"] = fr
    out["funding_z_20"] = _zscore(fr, cfg.window_funding_z)
    out["funding_annualized"] = fr * 3.0 * 365.0  # 3 settlements/day

    # OI --------------------------------------------------------------
    oi = _col(df, "oi")
    out["oi"] = oi
    out["oi_chg_1h"] = oi - oi.shift(cfg.oi_chg_bars_1h)
    out["oi_chg_24h"] = oi - oi.shift(cfg.oi_chg_bars_24h)
    out["oi_chg_pct_1h"] = (oi - oi.shift(cfg.oi_chg_bars_1h)) / oi.shift(cfg.oi_chg_bars_1h).replace(0, np.nan) * 100.0
    out["oi_chg_pct_24h"] = (oi - oi.shift(cfg.oi_chg_bars_24h)) / oi.shift(cfg.oi_chg_bars_24h).replace(0, np.nan) * 100.0
    out["oi_z_50"] = _zscore(oi, cfg.window_oi_z)

    # Basis -----------------------------------------------------------
    mark = _col(df, "mark_price")
    index_p = _col(df, "index_price")
    basis = (mark - index_p) / index_p.replace(0, np.nan) * 10_000.0
    out["basis_bps"] = basis
    out["basis_z_50"] = _zscore(basis, cfg.window_basis_z)

    return out


def _col(df: pd.DataFrame, name: str) -> pd.Series:
    if name in df.columns:
        return df[name].astype("float64")
    return pd.Series(np.nan, index=df.index, dtype="float64")


def _zscore(s: pd.Series, n: int) -> pd.Series:
    mean = s.rolling(n, min_periods=n).mean()
    std = s.rolling(n, min_periods=n).std(ddof=1).replace(0, np.nan)
    return (s - mean) / std
