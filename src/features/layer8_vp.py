"""Layer 8 -- bar-approximated volume profile.

For each bar, look at the trailing ``cfg.window_vp`` bars, bin the
price range into ``cfg.vp_num_bins`` cells, and assign each bar's
volume uniformly across the cells it touched (``[low, high]``). From
the resulting histogram derive:

- POC (point of control): price-bin mid with max volume.
- VAH / VAL: edges of the smallest contiguous window around POC
  containing ``cfg.vp_value_area`` (default 70%) of the total volume.

Phase 6: when ``des_core`` is importable we delegate the hot loop to
the Rust kernel for ~10-30x speedup; otherwise the pure-Python path
below is used (kept for fallback + parity testing).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from features.config import FeatureConfig

try:
    import des_core as _rust  # type: ignore
except Exception:  # pragma: no cover
    _rust = None


def compute_layer8(df: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    n = len(df)
    win = cfg.window_vp
    bins = cfg.vp_num_bins
    area = cfg.vp_value_area

    high = df["high"].to_numpy(dtype="float64")
    low = df["low"].to_numpy(dtype="float64")
    vol = df["volume"].to_numpy(dtype="float64") if "volume" in df.columns else np.zeros(n)

    if _rust is not None and n >= win:
        poc_l, vah_l, val_l = _rust.rolling_volume_profile(
            np.where(np.isfinite(low), low, 0.0).tolist(),
            np.where(np.isfinite(high), high, 0.0).tolist(),
            np.where(np.isfinite(vol), vol, 0.0).tolist(),
            win, bins, area,
        )
        poc = np.asarray(poc_l, dtype="float64")
        vah = np.asarray(vah_l, dtype="float64")
        val = np.asarray(val_l, dtype="float64")
    else:
        poc, vah, val = _py_rolling_vp(low, high, vol, win, bins, area)

    poc_s = pd.Series(poc, index=df.index, dtype="float64")
    vah_s = pd.Series(vah, index=df.index, dtype="float64")
    val_s = pd.Series(val, index=df.index, dtype="float64")
    close_s = df["close"].astype("float64")

    out = pd.DataFrame(index=df.index)
    out["poc_price_200"] = poc_s
    out["vah_price_200"] = vah_s
    out["val_price_200"] = val_s
    out["poc_dist"] = (close_s - poc_s) / close_s.replace(0, np.nan) * 100.0
    out["vah_dist"] = (close_s - vah_s) / close_s.replace(0, np.nan) * 100.0
    out["val_dist"] = (close_s - val_s) / close_s.replace(0, np.nan) * 100.0
    out["value_area_width_200"] = (vah_s - val_s) / poc_s.replace(0, np.nan)
    return out


def _py_rolling_vp(low, high, vol, win, bins, area):
    n = len(low)
    poc = np.full(n, np.nan)
    vah = np.full(n, np.nan)
    val = np.full(n, np.nan)
    for i in range(win - 1, n):
        lo_w = low[i - win + 1: i + 1]
        hi_w = high[i - win + 1: i + 1]
        vol_w = vol[i - win + 1: i + 1]

        p_lo = float(np.nanmin(lo_w))
        p_hi = float(np.nanmax(hi_w))
        if not np.isfinite(p_lo) or not np.isfinite(p_hi) or p_hi <= p_lo:
            continue

        bin_edges = np.linspace(p_lo, p_hi, bins + 1)
        bin_mid = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        hist = _build_hist(lo_w, hi_w, vol_w, bin_edges)
        if hist.sum() <= 0:
            continue

        poc_idx = int(np.argmax(hist))
        poc[i] = float(bin_mid[poc_idx])
        vah_i, val_i = _value_area(hist, poc_idx, area)
        vah[i] = float(bin_mid[vah_i])
        val[i] = float(bin_mid[val_i])
    return poc, vah, val


def _build_hist(
    low: np.ndarray, high: np.ndarray, vol: np.ndarray, edges: np.ndarray,
) -> np.ndarray:
    """Spread each bar's volume uniformly across the bins it overlaps."""
    nbins = len(edges) - 1
    hist = np.zeros(nbins, dtype="float64")
    for l, h, v in zip(low, high, vol):
        if not np.isfinite(l) or not np.isfinite(h) or v <= 0 or h < l:
            continue
        # Bar spans [l, h]; find overlapping bins.
        lo_idx = int(np.clip(np.searchsorted(edges, l, side="right") - 1, 0, nbins - 1))
        hi_idx = int(np.clip(np.searchsorted(edges, h, side="left"), 0, nbins - 1))
        if hi_idx < lo_idx:
            hi_idx = lo_idx
        span = edges[hi_idx + 1] - edges[lo_idx] if hi_idx + 1 < len(edges) else edges[-1] - edges[lo_idx]
        if span <= 0:
            hist[lo_idx] += v
            continue
        for b in range(lo_idx, hi_idx + 1):
            lo_b = max(edges[b], l)
            hi_b = min(edges[b + 1], h)
            overlap = max(0.0, hi_b - lo_b)
            bar_span = max(h - l, (edges[1] - edges[0]))
            hist[b] += v * (overlap / bar_span)
    return hist


def _value_area(hist: np.ndarray, poc_idx: int, area: float) -> tuple[int, int]:
    """Expand outward from POC until ``area`` fraction of volume is covered."""
    total = hist.sum()
    target = total * area
    lo = hi = poc_idx
    accum = hist[poc_idx]
    n = len(hist)
    while accum < target and (lo > 0 or hi < n - 1):
        up_next = hist[hi + 1] if hi < n - 1 else -1.0
        dn_next = hist[lo - 1] if lo > 0 else -1.0
        if up_next < 0 and dn_next < 0:
            break
        if up_next >= dn_next:
            hi += 1
            accum += hist[hi]
        else:
            lo -= 1
            accum += hist[lo]
    return hi, lo
