"""Context features (Phase 5): multi-horizon returns, relative volume,
position-in-range, rolling percentile ranks.

All bar-level (Tier A), no lookahead. At base 15m:
- 1h = 4 bars, 4h = 16 bars, 24h = 96 bars.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from features.config import FeatureConfig


def compute_context(df: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:  # noqa: ARG001
    """Return a frame with the context columns aligned to ``df.index``.

    ``df`` must carry OHLCV; if ``atr_14_pct`` / ``bb_width`` are
    present their 96-bar percentile ranks are computed too, else
    those columns are NaN.
    """
    n = len(df)
    out = pd.DataFrame(index=df.index)
    close = df["close"].astype("float64")
    high = df["high"].astype("float64") if "high" in df.columns else close
    low = df["low"].astype("float64") if "low" in df.columns else close
    volume = df["volume"].astype("float64") if "volume" in df.columns else pd.Series(0.0, index=df.index)

    out["ret_1h"] = close / close.shift(4) - 1.0
    out["ret_4h"] = close / close.shift(16) - 1.0
    out["ret_24h"] = close / close.shift(96) - 1.0

    # Turnover 24h (USDT perp -> close * volume is a decent proxy).
    turnover_bar = close * volume
    out["turnover_24h"] = turnover_bar.rolling(96, min_periods=96).sum()

    # Relative volume (last bar vs 20-bar avg, excluding current).
    vol_avg = volume.shift(1).rolling(20, min_periods=20).mean()
    out["rel_volume_20"] = volume / vol_avg.replace(0, np.nan)

    # Position of close within the 24h (96-bar) high-low range.
    hi_96 = high.rolling(96, min_periods=96).max()
    lo_96 = low.rolling(96, min_periods=96).min()
    span = (hi_96 - lo_96).replace(0, np.nan)
    out["hi_lo_24h_pos"] = (close - lo_96) / span

    # Rolling 96-bar percentile ranks (in [0, 1], last bar = position).
    if "atr_14_pct" in df.columns:
        out["atr_pct_rank_96"] = _pct_rank_rolling(df["atr_14_pct"].astype("float64"), 96)
    else:
        out["atr_pct_rank_96"] = np.nan
    if "bb_width" in df.columns:
        out["bb_width_rank_96"] = _pct_rank_rolling(df["bb_width"].astype("float64"), 96)
    else:
        out["bb_width_rank_96"] = np.nan
    return out


def _pct_rank_rolling(s: pd.Series, n: int) -> pd.Series:
    """Percentile rank of the latest value within the trailing ``n`` bars.

    Returns a value in [0, 1]. 1.0 = current is the max, 0.0 = min.
    NaN until ``n`` bars of warm-up and wherever the current value is NaN.
    """
    def _r(a: np.ndarray) -> float:
        last = a[-1]
        if not np.isfinite(last):
            return np.nan
        finite = a[np.isfinite(a)]
        if len(finite) < 2:
            return np.nan
        # fraction of values <= last
        return float((finite <= last).sum() / len(finite))

    return s.rolling(n, min_periods=n).apply(_r, raw=True)
