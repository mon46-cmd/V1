"""Bar-level feature pipeline.

Usage::

    from features import compute, SymbolBundle
    bundle = SymbolBundle(symbol="BTCUSDT", base_15m=df_15m,
                          bars_1h=df_1h, bars_4h=df_4h,
                          funding=fund, oi=oi,
                          mark=mk, index=ix, ref_15m=btc_15m)
    feat = compute("snapshot", bundle, cfg=FeatureConfig())

The pipeline is synchronous and pure. It does not talk to the network;
the caller fetches the bundle upstream.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from features.align import attach_asof, attach_mtf
from features.config import FeatureConfig
from features.context import compute_context
from features.flags import compute_flags
from features.layer1_vol import compute_layer1
from features.layer3_regime import compute_layer3
from features.layer4_tech import compute_layer4
from features.layer5_derivs import compute_layer5
from features.layer8_vp import compute_layer8
from features.layer10_xasset import compute_layer10
from features.layer11_calendar import compute_layer11
from features.layer12_composite import compute_layer12
from features.registry import MTF_ATTACH, profile_columns


@dataclass
class SymbolBundle:
    """Inputs for one symbol's feature computation.

    All frames must be indexed by a tz-aware UTC ``timestamp`` column
    promoted to the index, except where noted.
    """
    symbol: str
    base_15m: pd.DataFrame
    bars_1h: pd.DataFrame | None = None
    bars_4h: pd.DataFrame | None = None
    funding: pd.DataFrame | None = None       # cols: funding_rate
    oi: pd.DataFrame | None = None            # cols: oi (a.k.a. open_interest)
    mark_15m: pd.DataFrame | None = None      # cols: close (mark)
    index_15m: pd.DataFrame | None = None     # cols: close (index)
    ref_15m: pd.DataFrame | None = None       # BTC base 15m (for xasset)


def compute(profile: str, bundle: SymbolBundle, *, cfg: FeatureConfig | None = None) -> pd.DataFrame:
    """Run the pipeline and return only the columns for ``profile``."""
    cfg = cfg or FeatureConfig()
    all_cols = _compute_all(bundle, cfg)
    wanted = profile_columns(profile)
    # Keep only columns we can actually produce; fill missing with NaN.
    present = [c for c in wanted if c in all_cols.columns]
    missing = [c for c in wanted if c not in all_cols.columns]
    out = all_cols[present].copy()
    for c in missing:
        out[c] = float("nan")
    return out[list(wanted)]


def _compute_all(bundle: SymbolBundle, cfg: FeatureConfig) -> pd.DataFrame:
    base = _as_indexed(bundle.base_15m)

    # Layer 1 + Layer 4 from OHLCV.
    l1 = compute_layer1(base, cfg)
    l4 = compute_layer4(base, cfg)

    # Layer 3 (regime stats) needs log_ret from Layer 1.
    l3 = compute_layer3(pd.concat([base[["close"]], l1[["log_ret"]]], axis=1), cfg)

    # Layer 5 needs funding / oi / mark / index attached by asof.
    ctx = base.copy()
    if bundle.funding is not None:
        ctx = attach_asof(ctx, _as_indexed(bundle.funding), ["funding_rate"])
    if bundle.oi is not None:
        oi_ind = _as_indexed(bundle.oi)
        col = "oi" if "oi" in oi_ind.columns else ("open_interest" if "open_interest" in oi_ind.columns else None)
        if col is not None:
            ctx = attach_asof(ctx, oi_ind.rename(columns={col: "oi"}), ["oi"])
    if bundle.mark_15m is not None and bundle.index_15m is not None:
        mk = _as_indexed(bundle.mark_15m)
        ix = _as_indexed(bundle.index_15m)
        if "close" in mk.columns:
            ctx = attach_asof(ctx, mk.rename(columns={"close": "mark_price"}), ["mark_price"])
        if "close" in ix.columns:
            ctx = attach_asof(ctx, ix.rename(columns={"close": "index_price"}), ["index_price"])
    l5 = compute_layer5(ctx, cfg)

    # Layer 8.
    l8 = compute_layer8(base, cfg)

    # Context (multi-horizon returns, rel_volume, pct-ranks).
    # Needs atr_14_pct (Layer 1) and bb_width (Layer 4) for the percentile ranks.
    l_for_ctx = pd.concat(
        [base[["open", "high", "low", "close", "volume"]], l1[["atr_14_pct"]], l4],
        axis=1,
    )
    lctx = compute_context(l_for_ctx, cfg)

    # Trader flags (need Layer 4 columns).
    lflags = compute_flags(l_for_ctx, cfg)

    # Stitch the 15m native layers first so MTF has full feature set.
    bars15 = pd.concat([base[["open", "high", "low", "close", "volume"]], l1, l3, l4, l5, l8, lctx, lflags], axis=1)

    # Higher-TF feature frames (computed with the same pipeline, sans MTF/xasset/composite).
    l1_1h = l4_1h = l5_1h = l8_1h = None
    bars_1h = _as_indexed(bundle.bars_1h) if bundle.bars_1h is not None else None
    bars_4h = _as_indexed(bundle.bars_4h) if bundle.bars_4h is not None else None
    h1_feat = _higher_tf_features(bars_1h, cfg) if bars_1h is not None else None
    h4_feat = _higher_tf_features(bars_4h, cfg) if bars_4h is not None else None

    # Layer 9 MTF attach.
    bars15_mtf = attach_mtf(bars15, h1_feat, h4_feat, MTF_ATTACH)

    # Layer 10 cross-asset (needs aligned ref log_ret).
    if bundle.ref_15m is not None:
        ref = _as_indexed(bundle.ref_15m)
        ref_l1 = compute_layer1(ref, cfg)[["log_ret"]].rename(columns={"log_ret": "ref_log_ret"})
        bars15_mtf = attach_asof(bars15_mtf, ref_l1, ["ref_log_ret"])
    l10 = compute_layer10(bars15_mtf, cfg)

    # Layer 11 calendar + Layer 12 composite.
    l11 = compute_layer11(base, cfg)
    tmp_for_l12 = pd.concat([bars15_mtf[["ema_50_dist"]] if "ema_50_dist" in bars15_mtf.columns else pd.DataFrame(index=bars15_mtf.index),
                             bars15_mtf[["h1_ema_50_dist"]] if "h1_ema_50_dist" in bars15_mtf.columns else pd.DataFrame(index=bars15_mtf.index),
                             bars15_mtf[["h4_ema_50_dist"]] if "h4_ema_50_dist" in bars15_mtf.columns else pd.DataFrame(index=bars15_mtf.index)],
                            axis=1)
    l12 = compute_layer12(tmp_for_l12, cfg)

    # Drop ref_log_ret helper column before returning.
    final = bars15_mtf.drop(columns=["ref_log_ret"], errors="ignore")
    final = pd.concat([final, l10, l11, l12], axis=1)

    # Attach symbol + timestamp columns.
    final.insert(0, "symbol", bundle.symbol)
    # Move timestamp from index to explicit column (keep index too).
    final["timestamp"] = final.index
    # Deduplicate columns (can happen if l1 is already in bars15).
    final = final.loc[:, ~final.columns.duplicated(keep="last")]
    return final


def _higher_tf_features(bars: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    """Compute the subset of features consumed by MTF attach."""
    l1 = compute_layer1(bars, cfg)
    l4 = compute_layer4(bars, cfg)
    l5_cols = pd.DataFrame(index=bars.index)
    l8 = compute_layer8(bars, cfg)
    out = pd.concat([bars[["close"]], l1, l4, l5_cols, l8], axis=1)
    out = out.loc[:, ~out.columns.duplicated(keep="last")]
    return out


def _as_indexed(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure a tz-aware UTC DatetimeIndex named 'timestamp'."""
    if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
        return df
    if "timestamp" in df.columns:
        return df.set_index("timestamp")
    raise ValueError("expected tz-aware DatetimeIndex or a 'timestamp' column")
