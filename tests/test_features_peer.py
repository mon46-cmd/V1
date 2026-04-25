"""Offline tests for peer / cross-section features (Phase 5)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from features import FeatureConfig, compute_peer


def _cross_section() -> pd.DataFrame:
    # 6 symbols with deliberately different profiles.
    rows = [
        {"symbol": "BTCUSDT", "ret_24h": 0.02,  "atr_14_pct": 1.2, "turnover_24h": 5e9,
         "oi_chg_pct_24h": 1.0,  "funding_rate": 0.00010},
        {"symbol": "ETHUSDT", "ret_24h": 0.03,  "atr_14_pct": 1.5, "turnover_24h": 2e9,
         "oi_chg_pct_24h": 2.5,  "funding_rate": 0.00015},
        {"symbol": "SOLUSDT", "ret_24h": 0.08,  "atr_14_pct": 3.0, "turnover_24h": 6e8,
         "oi_chg_pct_24h": 8.0,  "funding_rate": 0.00030},
        {"symbol": "AVAXUSDT", "ret_24h": 0.07, "atr_14_pct": 2.8, "turnover_24h": 3e8,
         "oi_chg_pct_24h": 7.5,  "funding_rate": 0.00025},
        {"symbol": "DOGEUSDT", "ret_24h": -0.04, "atr_14_pct": 2.1, "turnover_24h": 4e8,
         "oi_chg_pct_24h": -3.0, "funding_rate": -0.00020},
        {"symbol": "PEPEUSDT", "ret_24h": -0.05, "atr_14_pct": 4.0, "turnover_24h": 2e8,
         "oi_chg_pct_24h": -2.5, "funding_rate": -0.00025},
    ]
    return pd.DataFrame(rows)


def test_peer_ranks_and_rs_reference():
    cfg = FeatureConfig()
    cs = _cross_section()
    out, report = compute_peer(cs, cfg)

    # Ranks: turnover -- BTC is highest (rank 1).
    btc_idx = out.index[out["symbol"] == "BTCUSDT"][0]
    assert int(out.loc[btc_idx, "rank_turnover_24h"]) == 1
    # ret_24h rank -- SOL (+0.08) is highest.
    sol_idx = out.index[out["symbol"] == "SOLUSDT"][0]
    assert int(out.loc[sol_idx, "rank_ret_24h"]) == 1

    # Relative strength vs BTC
    assert out.loc[sol_idx, "rs_vs_btc_24h"] == pytest.approx(0.08 - 0.02)
    # BTC vs itself = 0
    assert out.loc[btc_idx, "rs_vs_btc_24h"] == pytest.approx(0.0)

    # Report sanity
    assert report.n_symbols == 6
    assert report.btc_ret_24h == pytest.approx(0.02)
    assert report.eth_ret_24h == pytest.approx(0.03)


def test_peer_clusters_group_similar_symbols():
    cfg = FeatureConfig(peer_cluster_k=3, peer_cluster_seed=42)
    cs = _cross_section()
    out, report = compute_peer(cs, cfg)

    assert "cluster_id" in out.columns
    # SOL and AVAX are the most alike (high vol, strong gain, OI up) -- should share a cluster.
    sol = out.loc[out["symbol"] == "SOLUSDT", "cluster_id"].iloc[0]
    avax = out.loc[out["symbol"] == "AVAXUSDT", "cluster_id"].iloc[0]
    assert sol == avax, "SOL and AVAX should cluster together"

    # DOGE and PEPE (negative ret, negative OI) -- likely same cluster too.
    doge = out.loc[out["symbol"] == "DOGEUSDT", "cluster_id"].iloc[0]
    pepe = out.loc[out["symbol"] == "PEPEUSDT", "cluster_id"].iloc[0]
    assert doge == pepe, "DOGE and PEPE should cluster together"

    # BTC and the memecoins must NOT share a cluster.
    btc = out.loc[out["symbol"] == "BTCUSDT", "cluster_id"].iloc[0]
    assert btc != doge

    # Sizes and leaders are self-consistent.
    for cid, size in report.cluster_sizes.items():
        mask = out["cluster_id"] == cid
        assert int(mask.sum()) == size


def test_peer_cluster_leader_is_highest_turnover_in_cluster():
    cfg = FeatureConfig(peer_cluster_k=2, peer_cluster_seed=7)
    cs = _cross_section()
    out, _report = compute_peer(cs, cfg)

    for cid, group in out.groupby("cluster_id"):
        leader = group["cluster_leader"].iloc[0]
        top = group.loc[group["turnover_24h"].idxmax(), "symbol"]
        assert leader == top


def test_peer_dist_to_centroid_nonneg():
    cfg = FeatureConfig(peer_cluster_k=3)
    cs = _cross_section()
    out, _ = compute_peer(cs, cfg)
    assert (out["dist_to_centroid"] >= 0).all()


def test_peer_handles_missing_columns_gracefully():
    """A stripped cross-section with only symbol + turnover still clusters."""
    cfg = FeatureConfig(peer_cluster_k=2)
    cs = pd.DataFrame([
        {"symbol": "A", "turnover_24h": 100.0},
        {"symbol": "B", "turnover_24h": 200.0},
        {"symbol": "C", "turnover_24h": 300.0},
    ])
    out, _ = compute_peer(cs, cfg)
    # Ranks exist; cluster_id exists (default 0 when insufficient features).
    assert "rank_turnover_24h" in out.columns
    assert "cluster_id" in out.columns


def test_peer_requires_symbol_column():
    cfg = FeatureConfig()
    with pytest.raises(ValueError):
        compute_peer(pd.DataFrame({"ret_24h": [0.01, 0.02]}), cfg)
