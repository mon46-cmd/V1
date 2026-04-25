"""Layer 3 -- regime statistics computed on returns.

Wraps the Rust ``des_core`` kernels (Phase 6) with a numpy fallback
so tests still run on a fresh checkout where the wheel hasn't been
built yet. Outputs:

- ``hurst_100``: rolling Hurst exponent (R/S) over 100 bars of log returns.
- ``vr_2_100``: Lo-MacKinlay variance ratio at q=2 over 100 bars.
- ``acf1_50``: lag-1 autocorrelation of returns over 50 bars.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from features.config import FeatureConfig

try:
    import des_core as _rust  # type: ignore
except Exception:  # pragma: no cover
    _rust = None


def compute_layer3(df: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    if "log_ret" in df.columns:
        ret = df["log_ret"].to_numpy(dtype="float64")
    elif "ret" in df.columns:
        ret = df["ret"].to_numpy(dtype="float64")
    else:
        close = df["close"].to_numpy(dtype="float64")
        ret = np.concatenate([[np.nan], np.diff(np.log(close))])

    n = len(ret)
    out = pd.DataFrame(index=df.index)
    out["hurst_100"] = _rolling_hurst(ret, cfg.hurst_window)
    out["vr_2_100"] = _rolling_vr(ret, cfg.hurst_window, 2)
    out["acf1_50"] = _rolling_acf1(ret, cfg.acf_window)
    return out


def _safe(arr: np.ndarray) -> list[float]:
    # Pass NaN/Inf through as-is. The Rust rolling kernels return NaN
    # for any window that contains a non-finite value, which matches
    # the Python reference. Zero-filling here would silently bias the
    # first valid window when ``log_ret`` carries its natural leading
    # NaN (from diff(log(close))).
    return arr.astype("float64", copy=False).tolist()


def _rolling_hurst(ret: np.ndarray, window: int) -> np.ndarray:
    if _rust is not None:
        return np.asarray(_rust.rolling_hurst(_safe(ret), window), dtype="float64")
    return _py_rolling_hurst(ret, window)


def _rolling_vr(ret: np.ndarray, window: int, q: int) -> np.ndarray:
    if _rust is not None:
        return np.asarray(_rust.rolling_variance_ratio(_safe(ret), window, q), dtype="float64")
    return _py_rolling_vr(ret, window, q)


def _rolling_acf1(x: np.ndarray, window: int) -> np.ndarray:
    if _rust is not None:
        return np.asarray(_rust.rolling_acf1(_safe(x), window), dtype="float64")
    return _py_rolling_acf1(x, window)


# ---- Python fallbacks (also used as the parity reference) -----------
def _py_hurst_rs(returns: np.ndarray) -> float:
    n = returns.size
    if n < 16 or not np.all(np.isfinite(returns)):
        return float("nan")
    sizes: list[int] = []
    k = 8
    while k <= n:
        sizes.append(k)
        k = int(np.ceil(k * 1.7))
    if not sizes or sizes[-1] != n:
        sizes.append(n)
    log_n: list[float] = []
    log_rs: list[float] = []
    for m in sizes:
        rs = _py_mean_rs(returns, m)
        if np.isfinite(rs) and rs > 0:
            log_n.append(np.log(m))
            log_rs.append(np.log(rs))
    if len(log_n) < 3:
        return float("nan")
    a = np.asarray(log_n)
    b = np.asarray(log_rs)
    am = a.mean()
    bm = b.mean()
    den = ((a - am) ** 2).sum()
    if den <= 0:
        return float("nan")
    return float(((a - am) * (b - bm)).sum() / den)


def _py_mean_rs(returns: np.ndarray, m: int) -> float:
    n = returns.size
    if m < 4 or m > n:
        return float("nan")
    chunks = n // m
    if chunks == 0:
        return float("nan")
    acc = []
    for c in range(chunks):
        s = returns[c * m: c * m + m]
        mean = s.mean()
        d = s - mean
        cum = np.cumsum(d)
        rng = float(cum.max() - cum.min())
        std = float(np.sqrt((d * d).mean()))
        if std > 0 and np.isfinite(rng):
            acc.append(rng / std)
    return float(np.mean(acc)) if acc else float("nan")


def _py_rolling_hurst(ret: np.ndarray, window: int) -> np.ndarray:
    n = ret.size
    out = np.full(n, np.nan)
    if window < 16 or window > n:
        return out
    for i in range(window - 1, n):
        out[i] = _py_hurst_rs(ret[i - window + 1: i + 1])
    return out


def _py_variance_ratio(returns: np.ndarray, q: int) -> float:
    n = returns.size
    if q < 2 or n < q * 2:
        return float("nan")
    var1 = float(returns.var(ddof=1))
    if var1 <= 0:
        return float("nan")
    sums = np.convolve(returns, np.ones(q), mode="valid")
    var_q = float(sums.var(ddof=1))
    return var_q / (q * var1)


def _py_rolling_vr(ret: np.ndarray, window: int, q: int) -> np.ndarray:
    n = ret.size
    out = np.full(n, np.nan)
    if window < q * 2 or window > n:
        return out
    for i in range(window - 1, n):
        out[i] = _py_variance_ratio(ret[i - window + 1: i + 1], q)
    return out


def _py_rolling_acf1(x: np.ndarray, window: int) -> np.ndarray:
    n = x.size
    out = np.full(n, np.nan)
    if window < 4 or window > n:
        return out
    for i in range(window - 1, n):
        s = x[i - window + 1: i + 1]
        m = s.mean()
        d = s - m
        c0 = float((d * d).mean())
        if c0 <= 0:
            continue
        out[i] = float((d[1:] * d[:-1]).sum() / window / c0)
    return out
