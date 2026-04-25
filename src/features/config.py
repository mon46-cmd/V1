"""Feature engine configuration (versioned).

Changing any field in :class:`FeatureConfig` bumps the stored parquet
identity — downstream caches keyed on ``cfg.version`` are invalidated
automatically.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FeatureConfig:
    """Frozen knobs for the bar-level (Tier A) feature pipeline."""

    # Identity: bump when formulas / windows change.
    version: str = "v3-2026-04-25"

    # Layer 1 (vol)
    window_rv: int = 20
    window_atr: int = 14
    window_parkinson: int = 20
    window_garman_klass: int = 20
    window_yang_zhang: int = 20

    # Layer 4 (classical TA)
    window_rsi: int = 14
    macd_windows: tuple[int, int, int] = (12, 26, 9)
    window_bb: int = 20
    bb_stdev_mult: float = 2.0
    ema_windows: tuple[int, int, int] = (8, 21, 50)
    window_adx: int = 14
    window_vwap: int = 20
    window_obv_slope: int = 20
    supertrend_atr_window: int = 10
    supertrend_mult: float = 3.0

    # Layer 5 (derivatives)
    window_funding_z: int = 20
    oi_chg_bars_1h: int = 4       # base_tf=15m -> 4 bars = 1h
    oi_chg_bars_24h: int = 96     # base_tf=15m -> 96 bars = 24h
    window_oi_z: int = 50
    window_basis_z: int = 50

    # Layer 8 (volume profile)
    window_vp: int = 200
    vp_value_area: float = 0.70
    vp_num_bins: int = 64

    # Layer 3 (regime statistics, Phase 6)
    hurst_window: int = 100
    acf_window: int = 50
    vr_q: int = 2

    # Layer 10 (cross-asset)
    window_beta: int = 100
    beta_reference: str = "BTCUSDT"

    # Layer 11 (calendar)
    funding_settle_hours_utc: tuple[int, int, int] = (0, 8, 16)
    funding_window_minutes: int = 5

    # Layer: trader flags (Phase 5)
    climax_lookback: int = 20
    climax_mult: float = 3.0
    sweep_lookback: int = 20
    squeeze_lookback: int = 100
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0

    # Cross-section / peer layer (Phase 5)
    peer_cluster_k: int = 5
    peer_cluster_features: tuple[str, ...] = (
        "ret_24h", "atr_14_pct", "oi_chg_pct_24h", "funding_rate", "rs_vs_btc_24h",
    )
    peer_cluster_max_iter: int = 50
    peer_cluster_seed: int = 42

    # Concurrency
    snapshot_concurrency: int = 6


def feature_cache_key(cfg: FeatureConfig, symbol: str, interval: str) -> str:
    """Stable cache key for a (symbol, interval) feature frame."""
    return f"{cfg.version}|{symbol}|{interval}"
