"""Layer 10 -- cross-asset features vs. the reference symbol (BTCUSDT).

We take a window of rolling log-returns for the target symbol and the
reference, and derive:

- ``beta_btc_100`` -- covariance / variance (classic beta).
- ``corr_btc_100`` -- Pearson correlation.
- ``residual_vs_btc_100`` -- target_ret - beta * ref_ret (this-bar).

The caller is expected to align the reference's ``log_ret`` onto the
target's index **before** calling :func:`compute_layer10` -- typically
via ``merge_asof(direction='backward')`` since the reference bar at T
is always available at T.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from features.config import FeatureConfig


def compute_layer10(
    df: pd.DataFrame,
    cfg: FeatureConfig,
    *,
    ref_log_ret_col: str = "ref_log_ret",
) -> pd.DataFrame:
    idx = df.index
    out = pd.DataFrame(index=idx)
    if "log_ret" not in df.columns or ref_log_ret_col not in df.columns:
        for c in ("beta_btc_100", "corr_btc_100", "residual_vs_btc_100"):
            out[c] = np.nan
        return out

    y = df["log_ret"].astype("float64")
    x = df[ref_log_ret_col].astype("float64")
    n = cfg.window_beta

    cov = y.rolling(n, min_periods=n).cov(x)
    var_x = x.rolling(n, min_periods=n).var(ddof=1).replace(0, np.nan)
    beta = cov / var_x
    corr = y.rolling(n, min_periods=n).corr(x)
    resid = y - beta * x

    out["beta_btc_100"] = beta
    out["corr_btc_100"] = corr
    out["residual_vs_btc_100"] = resid
    return out
