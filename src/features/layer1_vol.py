"""Layer 1 -- returns and realized-volatility features.

All functions take a DataFrame indexed by ``timestamp`` (UTC) with
OHLCV columns present, and return a new DataFrame containing **only**
the layer's output columns (aligned on the same index). They are
pure: no I/O, no global state, no mutation of the input.

No-lookahead contract: every row uses only data with index <= that row.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from features.config import FeatureConfig


def compute_layer1(df: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    """Return a frame with Layer-1 columns aligned to ``df.index``."""
    close = df["close"].astype("float64")
    high = df["high"].astype("float64")
    low = df["low"].astype("float64")
    openp = df["open"].astype("float64") if "open" in df.columns else close

    # Returns
    prev_close = close.shift(1)
    ret = close / prev_close - 1.0
    log_ret = np.log(close / prev_close)

    rv = log_ret.rolling(cfg.window_rv, min_periods=cfg.window_rv).std(ddof=1)

    # Wilder ATR -------------------------------------------------------
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = _wilder(tr, cfg.window_atr)
    atr_pct = atr / close * 100.0

    # Parkinson --------------------------------------------------------
    hl = np.log(high / low)
    parkinson_var = (hl ** 2).rolling(cfg.window_parkinson, min_periods=cfg.window_parkinson).mean()
    parkinson = np.sqrt(parkinson_var / (4.0 * math.log(2.0)))

    # Garman-Klass -----------------------------------------------------
    rs_gk = 0.5 * hl ** 2 - (2.0 * math.log(2.0) - 1.0) * np.log(close / openp) ** 2
    gk_var = rs_gk.rolling(cfg.window_garman_klass, min_periods=cfg.window_garman_klass).mean()
    garman_klass = np.sqrt(gk_var.clip(lower=0.0))

    # Yang-Zhang -------------------------------------------------------
    yz = _yang_zhang(openp, high, low, close, cfg.window_yang_zhang)

    out = pd.DataFrame({
        "ret": ret,
        "log_ret": log_ret,
        "rv_20": rv,
        "atr_14": atr,
        "atr_14_pct": atr_pct,
        "parkinson_20": parkinson,
        "garman_klass_20": garman_klass,
        "yang_zhang_20": yz,
    }, index=df.index)
    return out


# ---- helpers ----------------------------------------------------------
def _wilder(s: pd.Series, n: int) -> pd.Series:
    """Wilder smoothing (a.k.a. RMA): EMA with alpha = 1/n."""
    return s.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def _yang_zhang(o: pd.Series, h: pd.Series, l: pd.Series, c: pd.Series, n: int) -> pd.Series:
    """Yang-Zhang (2000) OHLC volatility estimator.

    sigma_yz^2 = sigma_o^2 + k * sigma_c^2 + (1-k) * sigma_rs^2
    with k = 0.34 / (1.34 + (n+1)/(n-1)).
    """
    if n < 2:
        return pd.Series(np.nan, index=c.index, dtype="float64")
    prev_c = c.shift(1)
    overnight = np.log(o / prev_c)
    open_to_close = np.log(c / o)
    # Rogers-Satchell.
    rs = np.log(h / c) * np.log(h / o) + np.log(l / c) * np.log(l / o)

    sigma_o = overnight.rolling(n, min_periods=n).var(ddof=1)
    sigma_c = open_to_close.rolling(n, min_periods=n).var(ddof=1)
    sigma_rs = rs.rolling(n, min_periods=n).mean()

    k = 0.34 / (1.34 + (n + 1.0) / (n - 1.0))
    var = sigma_o + k * sigma_c + (1.0 - k) * sigma_rs
    return np.sqrt(var.clip(lower=0.0))
