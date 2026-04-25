"""Layer 12 -- composite features that mix multiple layers.

Currently a single feature, ``trend_score_mtf``: mean of the sign of
the 15m / 1h / 4h EMA-50 distance, producing a discrete score in
{-1, -0.67, -0.33, 0, 0.33, 0.67, 1} (with NaN entries contributing
zero when at least one timeframe is available).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from features.config import FeatureConfig


def compute_layer12(df: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:  # noqa: ARG001
    out = pd.DataFrame(index=df.index)
    cols = ["ema_50_dist", "h1_ema_50_dist", "h4_ema_50_dist"]
    sub = pd.DataFrame({c: df[c] if c in df.columns else np.nan for c in cols}, index=df.index)
    signs = np.sign(sub)
    score = signs.mean(axis=1, skipna=True)
    out["trend_score_mtf"] = score
    return out
