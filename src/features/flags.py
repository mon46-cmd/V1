"""Trader-style binary/ordinal flags for the snapshot row.

All flags are computed from the already-computed Layer 1 / Layer 4
columns plus raw OHLCV. They live on the 15m base frame (Tier A) and
fire at bar close. None of them use look-ahead data.

Flags emitted:

- ``flag_volume_climax``   -- last bar's volume > ``cfg.climax_mult`` x
  rolling mean of the prior ``cfg.climax_lookback`` bars.
- ``flag_sweep_up``        -- last bar's high pierced the prior N-bar
  high but the close came back inside (bull-trap / liquidity sweep
  of resting stops).
- ``flag_sweep_dn``        -- mirror image on the downside.
- ``flag_squeeze_release`` -- BB width moved from the bottom decile of
  the trailing 100 bars to above the median in one bar.
- ``flag_macd_cross_up``   -- ``macd_hist`` flipped from <=0 to >0 on
  the last closed bar.
- ``flag_macd_cross_dn``   -- mirror image.
- ``flag_regime_flip``     -- ``supertrend_dir`` flipped sign on the
  last closed bar.
- ``flag_rsi_overbought``  -- ``rsi_14`` >= 70 on the last bar.
- ``flag_rsi_oversold``    -- ``rsi_14`` <= 30 on the last bar.

All outputs are float64 {0.0, 1.0} so they merge cleanly into the
numeric feature frame and ship to the LLM as plain 0/1 ints.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from features.config import FeatureConfig


FLAG_COLUMNS: tuple[str, ...] = (
    "flag_volume_climax",
    "flag_sweep_up",
    "flag_sweep_dn",
    "flag_squeeze_release",
    "flag_macd_cross_up",
    "flag_macd_cross_dn",
    "flag_regime_flip",
    "flag_rsi_overbought",
    "flag_rsi_oversold",
)


def compute_flags(df: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    """Compute trader flags from OHLCV + Layer 1/4 columns.

    ``df`` must already contain ``rsi_14``, ``macd_hist``, ``bb_width``
    and ``supertrend_dir`` (all Layer 4 outputs). Missing columns are
    handled gracefully -- the associated flags are set to 0.0.
    """
    out = pd.DataFrame(index=df.index)
    n = len(df)

    # Volume climax -------------------------------------------------
    vol = df["volume"].astype("float64") if "volume" in df.columns else pd.Series(0.0, index=df.index)
    lb = cfg.climax_lookback
    rolling_mean = vol.shift(1).rolling(lb, min_periods=lb).mean()
    out["flag_volume_climax"] = (vol > cfg.climax_mult * rolling_mean).astype("float64").fillna(0.0)

    # Liquidity sweeps ----------------------------------------------
    sw = cfg.sweep_lookback
    if {"high", "low", "close", "open"}.issubset(df.columns):
        prior_high = df["high"].shift(1).rolling(sw, min_periods=sw).max()
        prior_low = df["low"].shift(1).rolling(sw, min_periods=sw).min()
        up = (df["high"] > prior_high) & (df["close"] < prior_high) & (df["close"] < df["open"])
        dn = (df["low"] < prior_low) & (df["close"] > prior_low) & (df["close"] > df["open"])
        out["flag_sweep_up"] = up.astype("float64").fillna(0.0)
        out["flag_sweep_dn"] = dn.astype("float64").fillna(0.0)
    else:
        out["flag_sweep_up"] = 0.0
        out["flag_sweep_dn"] = 0.0

    # Squeeze release ----------------------------------------------
    if "bb_width" in df.columns:
        sr = cfg.squeeze_lookback
        bbw = df["bb_width"].astype("float64")
        prior = bbw.shift(1)
        q10 = prior.rolling(sr, min_periods=sr).quantile(0.10)
        q50 = prior.rolling(sr, min_periods=sr).quantile(0.50)
        was_tight = (prior <= q10)
        now_wide = (bbw > q50)
        out["flag_squeeze_release"] = (was_tight & now_wide).astype("float64").fillna(0.0)
    else:
        out["flag_squeeze_release"] = 0.0

    # MACD cross ----------------------------------------------------
    if "macd_hist" in df.columns:
        h = df["macd_hist"].astype("float64")
        prev = h.shift(1)
        out["flag_macd_cross_up"] = ((prev <= 0) & (h > 0)).astype("float64").fillna(0.0)
        out["flag_macd_cross_dn"] = ((prev >= 0) & (h < 0)).astype("float64").fillna(0.0)
    else:
        out["flag_macd_cross_up"] = 0.0
        out["flag_macd_cross_dn"] = 0.0

    # Regime flip ---------------------------------------------------
    if "supertrend_dir" in df.columns:
        st = df["supertrend_dir"].astype("float64")
        out["flag_regime_flip"] = (np.sign(st) != np.sign(st.shift(1))).astype("float64")
        out["flag_regime_flip"] = out["flag_regime_flip"].fillna(0.0)
    else:
        out["flag_regime_flip"] = 0.0

    # RSI extremes --------------------------------------------------
    if "rsi_14" in df.columns:
        r = df["rsi_14"].astype("float64")
        out["flag_rsi_overbought"] = (r >= cfg.rsi_overbought).astype("float64").fillna(0.0)
        out["flag_rsi_oversold"] = (r <= cfg.rsi_oversold).astype("float64").fillna(0.0)
    else:
        out["flag_rsi_overbought"] = 0.0
        out["flag_rsi_oversold"] = 0.0

    # Ensure stable column order / size.
    for c in FLAG_COLUMNS:
        if c not in out.columns:
            out[c] = 0.0
    return out[list(FLAG_COLUMNS)]
