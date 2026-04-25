"""Declarative registry of every feature produced by the Tier A engine.

Each entry records:

- ``name``      -- final column name in the output frame.
- ``layer``     -- 1..12 per the architecture doc.
- ``tier``      -- "A" (bar-level, always computed) or "B" (tick, later).
- ``inputs``    -- list of input column names required on the bar frame.
- ``warmup``    -- minimum bar count before the feature is non-NaN.
- ``rust``      -- True if a Rust kernel implementation is preferred /
                   expected (Python fallback still exists).
- ``window``    -- numeric knob used by the feature (for docs + tests).
- ``formula``   -- one-line human description.

The registry is the single source of truth for which columns belong to
which profile; :func:`profile_columns` returns the subset consumed by a
given pipeline profile.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class FeatureEntry:
    name: str
    layer: int
    tier: str
    inputs: tuple[str, ...]
    warmup: int
    rust: bool = False
    window: Any = None
    formula: str = ""

    def dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "layer": self.layer,
            "tier": self.tier,
            "inputs": list(self.inputs),
            "warmup": self.warmup,
            "rust": self.rust,
            "window": self.window,
            "formula": self.formula,
        }


@dataclass(frozen=True)
class Registry:
    entries: tuple[FeatureEntry, ...] = field(default_factory=tuple)

    def names(self) -> tuple[str, ...]:
        return tuple(e.name for e in self.entries)

    def by_layer(self, layer: int) -> tuple[FeatureEntry, ...]:
        return tuple(e for e in self.entries if e.layer == layer)

    def by_name(self, name: str) -> FeatureEntry | None:
        for e in self.entries:
            if e.name == name:
                return e
        return None

    def by_tier(self, tier: str) -> tuple[FeatureEntry, ...]:
        return tuple(e for e in self.entries if e.tier == tier)


# ---- helpers ----------------------------------------------------------
OHLCV = ("open", "high", "low", "close", "volume")


def _l1() -> list[FeatureEntry]:
    return [
        FeatureEntry("ret", 1, "A", ("close",), 1, False, None, "close_t/close_{t-1} - 1"),
        FeatureEntry("log_ret", 1, "A", ("close",), 1, False, None, "ln(close_t/close_{t-1})"),
        FeatureEntry("rv_20", 1, "A", ("close",), 20, False, 20, "stdev(log_ret, 20)"),
        FeatureEntry("atr_14", 1, "A", ("high", "low", "close"), 14, False, 14, "Wilder ATR"),
        FeatureEntry("atr_14_pct", 1, "A", ("high", "low", "close"), 14, False, 14, "atr/close*100"),
        FeatureEntry("parkinson_20", 1, "A", ("high", "low"), 20, False, 20, "Parkinson HL vol"),
        FeatureEntry("garman_klass_20", 1, "A", OHLCV[:4], 20, False, 20, "Garman-Klass OHLC vol"),
        FeatureEntry("yang_zhang_20", 1, "A", OHLCV[:4], 20, False, 20, "Yang-Zhang OHLC vol"),
    ]


def _l4() -> list[FeatureEntry]:
    return [
        FeatureEntry("rsi_14", 4, "A", ("close",), 14, False, 14, "Wilder RSI"),
        FeatureEntry("macd", 4, "A", ("close",), 26, False, (12, 26), "EMA(12)-EMA(26)"),
        FeatureEntry("macd_signal", 4, "A", ("close",), 35, False, (12, 26, 9), "EMA(9,MACD)"),
        FeatureEntry("macd_hist", 4, "A", ("close",), 35, False, (12, 26, 9), "MACD - signal"),
        FeatureEntry("bb_mid", 4, "A", ("close",), 20, False, 20, "SMA(20)"),
        FeatureEntry("bb_upper", 4, "A", ("close",), 20, False, 20, "SMA(20)+2*std"),
        FeatureEntry("bb_lower", 4, "A", ("close",), 20, False, 20, "SMA(20)-2*std"),
        FeatureEntry("bb_width", 4, "A", ("close",), 20, False, 20, "(up-lo)/mid"),
        FeatureEntry("bb_pct_b", 4, "A", ("close",), 20, False, 20, "(close-lo)/(up-lo)"),
        FeatureEntry("ema_8", 4, "A", ("close",), 8, False, 8, "EMA(8)"),
        FeatureEntry("ema_21", 4, "A", ("close",), 21, False, 21, "EMA(21)"),
        FeatureEntry("ema_50", 4, "A", ("close",), 50, False, 50, "EMA(50)"),
        FeatureEntry("ema_8_dist", 4, "A", ("close",), 8, False, 8, "(c-ema)/c*100"),
        FeatureEntry("ema_21_dist", 4, "A", ("close",), 21, False, 21, "(c-ema)/c*100"),
        FeatureEntry("ema_50_dist", 4, "A", ("close",), 50, False, 50, "(c-ema)/c*100"),
        FeatureEntry("adx_14", 4, "A", ("high", "low", "close"), 14, False, 14, "Wilder ADX"),
        FeatureEntry("plus_di_14", 4, "A", ("high", "low", "close"), 14, False, 14, "+DI"),
        FeatureEntry("minus_di_14", 4, "A", ("high", "low", "close"), 14, False, 14, "-DI"),
        FeatureEntry("vwap_rolling_20_dist", 4, "A", ("high", "low", "close", "volume"), 20, False, 20, "(c-vwap)/vwap*100"),
        FeatureEntry("obv", 4, "A", ("close", "volume"), 1, False, None, "cumsum(vol*sign(dc))"),
        FeatureEntry("obv_slope_20", 4, "A", ("close", "volume"), 20, False, 20, "slope(OBV, 20)"),
        FeatureEntry("supertrend", 4, "A", ("high", "low", "close"), 10, False, 10, "ATR bands"),
        FeatureEntry("supertrend_dir", 4, "A", ("high", "low", "close"), 10, False, 10, "+/-1"),
    ]


def _l5() -> list[FeatureEntry]:
    return [
        FeatureEntry("funding_rate", 5, "A", ("funding_rate",), 0, False, None, "asof merge"),
        FeatureEntry("funding_z_20", 5, "A", ("funding_rate",), 20, False, 20, "z(fr, 20)"),
        FeatureEntry("funding_annualized", 5, "A", ("funding_rate",), 0, False, None, "fr*3*365"),
        FeatureEntry("oi", 5, "A", ("oi",), 0, False, None, "asof merge"),
        FeatureEntry("oi_chg_1h", 5, "A", ("oi",), 4, False, 4, "oi - oi_{t-4}"),
        FeatureEntry("oi_chg_24h", 5, "A", ("oi",), 96, False, 96, "oi - oi_{t-96}"),
        FeatureEntry("oi_chg_pct_1h", 5, "A", ("oi",), 4, False, 4, "(oi-oi_{t-4})/oi_{t-4}*100"),
        FeatureEntry("oi_chg_pct_24h", 5, "A", ("oi",), 96, False, 96, "(oi-oi_{t-96})/oi_{t-96}*100"),
        FeatureEntry("oi_z_50", 5, "A", ("oi",), 50, False, 50, "z(oi,50)"),
        FeatureEntry("basis_bps", 5, "A", ("mark_price", "index_price"), 0, False, None, "(m-i)/i*1e4"),
        FeatureEntry("basis_z_50", 5, "A", ("mark_price", "index_price"), 50, False, 50, "z(basis,50)"),
    ]


def _l8() -> list[FeatureEntry]:
    return [
        FeatureEntry("poc_price_200", 8, "A", ("high", "low", "close", "volume"), 200, True, 200, "volume-profile POC"),
        FeatureEntry("poc_dist", 8, "A", ("close",), 200, True, 200, "(c-poc)/c*100"),
        FeatureEntry("vah_price_200", 8, "A", ("high", "low", "close", "volume"), 200, True, 200, "70% cum high"),
        FeatureEntry("val_price_200", 8, "A", ("high", "low", "close", "volume"), 200, True, 200, "70% cum low"),
        FeatureEntry("vah_dist", 8, "A", ("close",), 200, True, 200, "(c-vah)/c*100"),
        FeatureEntry("val_dist", 8, "A", ("close",), 200, True, 200, "(c-val)/c*100"),
        FeatureEntry("value_area_width_200", 8, "A", ("high", "low", "close", "volume"), 200, True, 200, "(vah-val)/poc"),
    ]


MTF_ATTACH: tuple[str, ...] = (
    "close", "log_ret", "rv_20", "atr_14_pct",
    "rsi_14", "macd_hist", "adx_14", "ema_50_dist", "bb_pct_b",
    "funding_rate", "basis_bps", "oi_z_50",
    "poc_dist", "vah_dist", "val_dist",
)


def _l9() -> list[FeatureEntry]:
    out: list[FeatureEntry] = []
    for base in MTF_ATTACH:
        out.append(FeatureEntry(f"h1_{base}", 9, "A", (base,), 4, False, "1h", f"asof-backward 1h {base}"))
        out.append(FeatureEntry(f"h4_{base}", 9, "A", (base,), 16, False, "4h", f"asof-backward 4h {base}"))
    return out


def _l10() -> list[FeatureEntry]:
    return [
        FeatureEntry("beta_btc_100", 10, "A", ("log_ret",), 100, False, 100, "cov/var vs BTC"),
        FeatureEntry("corr_btc_100", 10, "A", ("log_ret",), 100, False, 100, "corr vs BTC"),
        FeatureEntry("residual_vs_btc_100", 10, "A", ("log_ret",), 100, False, 100, "r - beta*r_btc"),
    ]


def _l11() -> list[FeatureEntry]:
    return [
        FeatureEntry("utc_hour_sin", 11, "A", ("timestamp",), 0, False, None, "sin(2pi*h/24)"),
        FeatureEntry("utc_hour_cos", 11, "A", ("timestamp",), 0, False, None, "cos(2pi*h/24)"),
        FeatureEntry("utc_weekday_sin", 11, "A", ("timestamp",), 0, False, None, "sin(2pi*d/7)"),
        FeatureEntry("utc_weekday_cos", 11, "A", ("timestamp",), 0, False, None, "cos(2pi*d/7)"),
        FeatureEntry("is_funding_minute", 11, "A", ("timestamp",), 0, False, None, "+/-5m from 0/8/16 UTC"),
        FeatureEntry("time_to_next_funding_sec", 11, "A", ("timestamp",), 0, False, None, "countdown"),
        FeatureEntry("is_us_hours", 11, "A", ("timestamp",), 0, False, None, "14-22 UTC"),
        FeatureEntry("is_eu_hours", 11, "A", ("timestamp",), 0, False, None, "08-17 UTC"),
        FeatureEntry("is_asia_hours", 11, "A", ("timestamp",), 0, False, None, "22-07 UTC"),
    ]


def _l12() -> list[FeatureEntry]:
    return [
        FeatureEntry("trend_score_mtf", 12, "A", ("ema_50_dist", "h1_ema_50_dist", "h4_ema_50_dist"),
                     200, False, None, "mean(sign(ema_50_dist) across 15m/1h/4h)"),
    ]


def _flags() -> list[FeatureEntry]:
    """Trader flags (Phase 5). Layer 13 = 'flags'."""
    return [
        FeatureEntry("flag_volume_climax", 13, "A", ("volume",), 20, False, 20, "v > climax_mult * avg(v,20)"),
        FeatureEntry("flag_sweep_up", 13, "A", ("high", "close"), 20, False, 20, "wick above prior high, close back"),
        FeatureEntry("flag_sweep_dn", 13, "A", ("low", "close"), 20, False, 20, "wick below prior low, close back"),
        FeatureEntry("flag_squeeze_release", 13, "A", ("bb_width",), 100, False, 100, "BBW: q10 -> >q50"),
        FeatureEntry("flag_macd_cross_up", 13, "A", ("macd_hist",), 35, False, None, "hist <=0 -> >0"),
        FeatureEntry("flag_macd_cross_dn", 13, "A", ("macd_hist",), 35, False, None, "hist >=0 -> <0"),
        FeatureEntry("flag_regime_flip", 13, "A", ("supertrend_dir",), 10, False, None, "supertrend dir flip"),
        FeatureEntry("flag_rsi_overbought", 13, "A", ("rsi_14",), 14, False, None, "rsi >= 70"),
        FeatureEntry("flag_rsi_oversold", 13, "A", ("rsi_14",), 14, False, None, "rsi <= 30"),
    ]


def _context() -> list[FeatureEntry]:
    """Bar-level context columns needed for the snapshot + peer layer."""
    return [
        FeatureEntry("ret_1h", 14, "A", ("close",), 4, False, 4, "close_t / close_{t-4} - 1"),
        FeatureEntry("ret_4h", 14, "A", ("close",), 16, False, 16, "close_t / close_{t-16} - 1"),
        FeatureEntry("ret_24h", 14, "A", ("close",), 96, False, 96, "close_t / close_{t-96} - 1"),
        FeatureEntry("turnover_24h", 14, "A", ("close", "volume"), 96, False, 96, "sum(close*vol, 96)"),
        FeatureEntry("rel_volume_20", 14, "A", ("volume",), 20, False, 20, "vol_t / avg(vol, 20)"),
        FeatureEntry("hi_lo_24h_pos", 14, "A", ("close", "high", "low"), 96, False, 96, "(close-lo)/(hi-lo) over 96 bars"),
        FeatureEntry("atr_pct_rank_96", 14, "A", ("atr_14_pct",), 96, False, 96, "percentile rank of atr_14_pct in 96"),
        FeatureEntry("bb_width_rank_96", 14, "A", ("bb_width",), 96, False, 96, "percentile rank of bb_width in 96"),
    ]


def _l3() -> list[FeatureEntry]:
    """Regime statistics (Phase 6, Rust-backed)."""
    return [
        FeatureEntry("hurst_100", 3, "A", ("log_ret",), 100, False, 100, "rolling Hurst (R/S) over 100 bars"),
        FeatureEntry("vr_2_100", 3, "A", ("log_ret",), 100, False, 100, "Lo-MacKinlay VR(q=2) over 100 bars"),
        FeatureEntry("acf1_50", 3, "A", ("log_ret",), 50, False, 50, "lag-1 autocorr over 50 bars"),
    ]


def build_registry() -> Registry:
    entries: list[FeatureEntry] = []
    for fn in (_l1, _l3, _l4, _l5, _l8, _l9, _l10, _l11, _l12, _flags, _context):
        entries.extend(fn())
    return Registry(entries=tuple(entries))


REGISTRY = build_registry()


# ---- profiles ---------------------------------------------------------
SNAPSHOT_COLUMNS: tuple[str, ...] = (
    "symbol", "timestamp", "open", "high", "low", "close", "volume",
    # Context (Phase 5)
    "ret_1h", "ret_4h", "ret_24h", "turnover_24h",
    "rel_volume_20", "hi_lo_24h_pos",
    "atr_pct_rank_96", "bb_width_rank_96",
    # Layer 1
    "atr_14", "atr_14_pct", "rv_20", "parkinson_20", "yang_zhang_20",
    # Layer 3 (regime, Rust)
    "hurst_100", "vr_2_100", "acf1_50",
    # Layer 4
    "rsi_14", "ema_21_dist", "ema_50_dist", "macd_hist",
    "bb_width", "bb_pct_b", "adx_14", "supertrend_dir",
    # Layer 5
    "funding_rate", "funding_z_20", "funding_annualized",
    "oi_chg_1h", "oi_chg_24h", "oi_chg_pct_1h", "oi_chg_pct_24h", "oi_z_50",
    "basis_bps", "basis_z_50",
    # Layer 8
    "poc_dist", "vah_dist", "val_dist", "value_area_width_200",
    # Layer 9 (MTF)
    "h1_rsi_14", "h4_rsi_14",
    "h1_ema_50_dist", "h4_ema_50_dist",
    "h1_atr_14_pct", "h4_atr_14_pct",
    "h1_rv_20", "h4_rv_20",
    "h1_adx_14", "h4_adx_14",
    "h1_macd_hist", "h4_macd_hist",
    # Layer 10
    "beta_btc_100", "corr_btc_100",
    # Layer 11
    "utc_hour_sin", "utc_hour_cos",
    "is_funding_minute", "time_to_next_funding_sec",
    # Layer 12
    "trend_score_mtf",
    # Flags (Phase 5)
    "flag_volume_climax", "flag_sweep_up", "flag_sweep_dn",
    "flag_squeeze_release",
    "flag_macd_cross_up", "flag_macd_cross_dn",
    "flag_regime_flip",
    "flag_rsi_overbought", "flag_rsi_oversold",
    # Peer / cross-section (Phase 5, filled by features.peer)
    "rank_ret_24h", "pct_rank_ret_24h",
    "rank_atr_pct", "pct_rank_atr_pct",
    "rank_turnover_24h", "pct_rank_turnover_24h",
    "rank_oi_chg_pct_24h", "pct_rank_oi_chg_pct_24h",
    "rank_funding_rate", "pct_rank_funding_rate",
    "rs_vs_btc_24h", "rs_vs_eth_24h",
    "cluster_id", "cluster_size", "cluster_leader",
    "cluster_avg_ret_24h", "cluster_avg_funding_rate",
    "dist_to_centroid",
)


DASHBOARD_COLUMNS: tuple[str, ...] = tuple(
    ["symbol", "timestamp"]
    + ["open", "high", "low", "close", "volume"]
    + list(REGISTRY.names())
)


PROFILES: dict[str, tuple[str, ...]] = {
    "snapshot": SNAPSHOT_COLUMNS,
    "dashboard": DASHBOARD_COLUMNS,
}


def profile_columns(profile: str) -> tuple[str, ...]:
    if profile not in PROFILES:
        raise ValueError(f"unknown profile {profile!r}; known: {list(PROFILES)}")
    return PROFILES[profile]
