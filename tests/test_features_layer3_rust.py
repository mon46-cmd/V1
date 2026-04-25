"""Phase 6 -- Rust kernel parity tests vs the pure-Python references.

Skipped if ``des_core`` is not importable. When present, asserts that
each kernel matches the Python reference within 1e-10 on random inputs,
and that the Hurst kernel handles 2048 points well under 1ms.
"""
from __future__ import annotations

import time

import numpy as np
import pytest

try:
    import des_core as rust  # type: ignore
except Exception:  # pragma: no cover
    rust = None

from features.layer3_regime import (
    _py_hurst_rs,
    _py_rolling_acf1,
    _py_rolling_hurst,
    _py_rolling_vr,
    _py_variance_ratio,
)
from features.layer8_vp import _py_rolling_vp


pytestmark = pytest.mark.skipif(rust is None, reason="des_core not built")


def _rng_returns(n: int, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 0.01, size=n)


def test_hurst_matches_reference():
    x = _rng_returns(1024, seed=42).tolist()
    rs = rust.hurst_rs(x)
    py = _py_hurst_rs(np.asarray(x))
    assert abs(rs - py) < 1e-10, (rs, py)


def test_acf_matches_reference():
    x = _rng_returns(500, seed=7)
    rs = np.asarray(rust.acf(x.tolist(), 5))
    # build a reference acf the same way layer3 does
    n = x.size
    mean = x.mean()
    d = x - mean
    c0 = (d * d).sum() / n
    py = np.array([(d[lag:] * d[:n - lag]).sum() / n / c0 if lag else 1.0
                   for lag in range(6)])
    assert np.max(np.abs(rs - py)) < 1e-10


def test_variance_ratio_matches_reference():
    x = _rng_returns(2000, seed=3).tolist()
    for q in (2, 4, 8):
        rs = rust.variance_ratio(x, q)
        py = _py_variance_ratio(np.asarray(x), q)
        assert abs(rs - py) < 1e-10, (q, rs, py)


def test_rolling_hurst_matches_reference():
    x = _rng_returns(400, seed=11)
    rs = np.asarray(rust.rolling_hurst(x.tolist(), 100))
    py = _py_rolling_hurst(x, 100)
    mask = np.isfinite(rs) & np.isfinite(py)
    assert mask.sum() > 100
    assert np.max(np.abs(rs[mask] - py[mask])) < 1e-10


def test_rolling_vr_matches_reference():
    x = _rng_returns(400, seed=21)
    rs = np.asarray(rust.rolling_variance_ratio(x.tolist(), 100, 2))
    py = _py_rolling_vr(x, 100, 2)
    mask = np.isfinite(rs) & np.isfinite(py)
    assert mask.sum() > 100
    assert np.max(np.abs(rs[mask] - py[mask])) < 1e-10


def test_rolling_acf1_matches_reference():
    x = _rng_returns(400, seed=31)
    rs = np.asarray(rust.rolling_acf1(x.tolist(), 50))
    py = _py_rolling_acf1(x, 50)
    mask = np.isfinite(rs) & np.isfinite(py)
    assert mask.sum() > 100
    assert np.max(np.abs(rs[mask] - py[mask])) < 1e-10


def test_hurst_speed_2048():
    x = _rng_returns(2048, seed=99).tolist()
    # warmup + measure
    rust.hurst_rs(x)
    iters = 50
    t0 = time.perf_counter()
    for _ in range(iters):
        rust.hurst_rs(x)
    dt_per_call_ms = (time.perf_counter() - t0) * 1000 / iters
    assert dt_per_call_ms < 1.0, f"hurst {dt_per_call_ms:.3f}ms > 1ms target"


def test_rolling_volume_profile_matches_reference():
    rng = np.random.default_rng(0)
    n = 350
    base = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.002, n)))
    high = base * (1 + rng.uniform(0.0005, 0.005, n))
    low = base * (1 - rng.uniform(0.0005, 0.005, n))
    vol = rng.uniform(50, 500, n)

    win, bins, area = 200, 64, 0.70
    rs_poc, rs_vah, rs_val = rust.rolling_volume_profile(
        low.tolist(), high.tolist(), vol.tolist(), win, bins, area,
    )
    py_poc, py_vah, py_val = _py_rolling_vp(low, high, vol, win, bins, area)

    rs_poc = np.asarray(rs_poc); rs_vah = np.asarray(rs_vah); rs_val = np.asarray(rs_val)
    mask = np.isfinite(rs_poc) & np.isfinite(py_poc)
    assert mask.sum() > 50
    # POC midpoints can differ by exactly one bin width when a bar lies
    # on a bin edge (Python uses searchsorted, Rust uses floor).
    bin_width = (high.max() - low.min()) / bins
    assert np.max(np.abs(rs_poc[mask] - py_poc[mask])) < 1.5 * bin_width
    assert np.max(np.abs(rs_vah[mask] - py_vah[mask])) < 1.5 * bin_width
    assert np.max(np.abs(rs_val[mask] - py_val[mask])) < 1.5 * bin_width


def test_amihud_basic():
    rng = np.random.default_rng(0)
    n = 200
    ret = rng.normal(0, 0.01, n)
    dv = rng.uniform(1e6, 1e7, n)
    out = np.asarray(rust.rolling_amihud(ret.tolist(), dv.tolist(), 50))
    assert np.isnan(out[:49]).all()
    assert np.isfinite(out[49:]).all()
    assert (out[49:] >= 0).all()


def test_kyle_lambda_basic():
    rng = np.random.default_rng(0)
    n = 200
    sv = rng.normal(0, 1000, n)
    # ret correlated with signed_volume
    ret = 1e-5 * sv + rng.normal(0, 1e-4, n)
    out = np.asarray(rust.rolling_kyle_lambda(ret.tolist(), sv.tolist(), 50))
    assert np.isfinite(out[49:]).all()


def test_rolls_spread_basic():
    rng = np.random.default_rng(0)
    close = 100.0 + np.cumsum(rng.normal(0, 0.5, 200))
    out = np.asarray(rust.rolling_rolls_spread(close.tolist(), 50))
    # First valid index is window+1=51 (needs both lagged and current windows).
    assert np.isfinite(out[51:]).all()
    assert (out[51:] >= 0).all()
