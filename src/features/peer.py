"""Cross-sectional ("peer") features.

Given the last-row snapshot of every symbol in the universe, this
module adds columns that let an LLM reason about the universe as a
*whole*:

- ``rank_*`` -- integer rank (1 = best/highest) within the universe on
  the metric, plus a ``pct_rank_*`` in [0, 1].
- ``rs_vs_btc_24h`` / ``rs_vs_eth_24h`` -- return of the symbol minus
  the reference's (relative strength).
- ``cluster_id`` -- KMeans label on a z-scored feature vector.
- ``cluster_size`` / ``cluster_leader`` -- peers and the highest-turnover
  symbol in the cluster.
- ``cluster_avg_ret_24h`` / ``cluster_avg_funding`` -- peer-group means.
- ``dist_to_centroid`` -- Euclidean distance in z-score space to the
  cluster's own centroid (lower = more typical peer).

All functions are pure and numpy-only. KMeans is a minimal in-house
implementation to avoid a scikit-learn dependency.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from features.config import FeatureConfig


RANK_COLUMNS: tuple[str, ...] = (
    "rank_ret_24h", "pct_rank_ret_24h",
    "rank_atr_pct", "pct_rank_atr_pct",
    "rank_turnover_24h", "pct_rank_turnover_24h",
    "rank_oi_chg_pct_24h", "pct_rank_oi_chg_pct_24h",
    "rank_funding_rate", "pct_rank_funding_rate",
    "rs_vs_btc_24h", "rs_vs_eth_24h",
)

CLUSTER_COLUMNS: tuple[str, ...] = (
    "cluster_id", "cluster_size", "cluster_leader",
    "cluster_avg_ret_24h", "cluster_avg_funding_rate",
    "dist_to_centroid",
)

PEER_COLUMNS: tuple[str, ...] = RANK_COLUMNS + CLUSTER_COLUMNS


@dataclass(frozen=True)
class PeerReport:
    n_symbols: int
    n_clusters: int
    btc_ret_24h: float
    eth_ret_24h: float
    cluster_sizes: dict[int, int]


def compute_peer(
    rows: pd.DataFrame,
    cfg: FeatureConfig,
) -> tuple[pd.DataFrame, PeerReport]:
    """Add peer-rank, relative-strength and cluster columns.

    ``rows`` is a *cross-section* DataFrame with one row per symbol
    (typically the last closed 15m bar for each). It must include
    ``symbol`` and ideally ``ret_24h``, ``atr_14_pct``, ``turnover_24h``,
    ``oi_chg_pct_24h``, ``funding_rate``. The return value is a new
    DataFrame with the same index and extra peer columns appended.
    """
    if "symbol" not in rows.columns:
        raise ValueError("compute_peer requires a 'symbol' column")
    df = rows.copy().reset_index(drop=True)
    n = len(df)

    # Ranks (higher metric -> rank 1) ----------------------------------
    _rank_col(df, "ret_24h", "rank_ret_24h", "pct_rank_ret_24h", higher_is_better=True)
    _rank_col(df, "atr_14_pct", "rank_atr_pct", "pct_rank_atr_pct", higher_is_better=True)
    _rank_col(df, "turnover_24h", "rank_turnover_24h", "pct_rank_turnover_24h", higher_is_better=True)
    _rank_col(df, "oi_chg_pct_24h", "rank_oi_chg_pct_24h", "pct_rank_oi_chg_pct_24h", higher_is_better=True)
    _rank_col(df, "funding_rate", "rank_funding_rate", "pct_rank_funding_rate", higher_is_better=True)

    # Relative strength vs BTC / ETH -----------------------------------
    btc_ret = _ref_ret(df, "BTCUSDT")
    eth_ret = _ref_ret(df, "ETHUSDT")
    base_ret = df["ret_24h"] if "ret_24h" in df.columns else pd.Series(np.nan, index=df.index)
    df["rs_vs_btc_24h"] = (base_ret - btc_ret).astype("float64") if not np.isnan(btc_ret) else np.nan
    df["rs_vs_eth_24h"] = (base_ret - eth_ret).astype("float64") if not np.isnan(eth_ret) else np.nan

    # Clustering on z-scored feature vector ----------------------------
    feat_cols = [c for c in cfg.peer_cluster_features if c in df.columns]
    if n >= 2 and len(feat_cols) >= 2:
        X = df[feat_cols].to_numpy(dtype="float64")
        X = _impute_median(X)
        Z = _zscore(X)
        k = min(cfg.peer_cluster_k, max(1, n - 1))
        labels, centroids = _kmeans(Z, k=k, max_iter=cfg.peer_cluster_max_iter, seed=cfg.peer_cluster_seed)
        df["cluster_id"] = labels.astype("int64")
        # Distance to own centroid
        deltas = Z - centroids[labels]
        df["dist_to_centroid"] = np.linalg.norm(deltas, axis=1)
    else:
        df["cluster_id"] = 0
        df["dist_to_centroid"] = 0.0

    # Per-cluster aggregates ------------------------------------------
    sizes: dict[int, int] = {}
    leaders: dict[int, str] = {}
    avg_ret: dict[int, float] = {}
    avg_fund: dict[int, float] = {}
    for cid, group in df.groupby("cluster_id"):
        sizes[int(cid)] = len(group)
        if "turnover_24h" in group.columns and group["turnover_24h"].notna().any():
            leaders[int(cid)] = str(group.loc[group["turnover_24h"].idxmax(), "symbol"])
        else:
            leaders[int(cid)] = str(group["symbol"].iloc[0])
        avg_ret[int(cid)] = float(group["ret_24h"].mean()) if "ret_24h" in group.columns else np.nan
        avg_fund[int(cid)] = float(group["funding_rate"].mean()) if "funding_rate" in group.columns else np.nan

    df["cluster_size"] = df["cluster_id"].map(sizes).astype("int64")
    df["cluster_leader"] = df["cluster_id"].map(leaders).astype("string")
    df["cluster_avg_ret_24h"] = df["cluster_id"].map(avg_ret).astype("float64")
    df["cluster_avg_funding_rate"] = df["cluster_id"].map(avg_fund).astype("float64")

    report = PeerReport(
        n_symbols=n,
        n_clusters=len(sizes),
        btc_ret_24h=float(btc_ret) if not np.isnan(btc_ret) else float("nan"),
        eth_ret_24h=float(eth_ret) if not np.isnan(eth_ret) else float("nan"),
        cluster_sizes={int(k): int(v) for k, v in sizes.items()},
    )
    return df, report


# ---- helpers ----------------------------------------------------------
def _rank_col(
    df: pd.DataFrame,
    src: str,
    rank_name: str,
    pct_name: str,
    *,
    higher_is_better: bool,
) -> None:
    if src not in df.columns:
        df[rank_name] = np.nan
        df[pct_name] = np.nan
        return
    s = df[src].astype("float64")
    if higher_is_better:
        ranked = s.rank(ascending=False, method="min", na_option="keep")
        pct = s.rank(ascending=True, method="average", pct=True, na_option="keep")
    else:
        ranked = s.rank(ascending=True, method="min", na_option="keep")
        pct = s.rank(ascending=False, method="average", pct=True, na_option="keep")
    df[rank_name] = ranked.astype("float64")
    df[pct_name] = pct.astype("float64")


def _ref_ret(df: pd.DataFrame, symbol: str) -> float:
    if "ret_24h" not in df.columns:
        return float("nan")
    hit = df[df["symbol"] == symbol]
    if hit.empty:
        return float("nan")
    v = hit["ret_24h"].iloc[0]
    return float(v) if pd.notna(v) else float("nan")


def _impute_median(X: np.ndarray) -> np.ndarray:
    X = X.copy()
    for j in range(X.shape[1]):
        col = X[:, j]
        m = np.nanmedian(col) if np.isfinite(col).any() else 0.0
        mask = ~np.isfinite(col)
        col[mask] = m if np.isfinite(m) else 0.0
        X[:, j] = col
    return X


def _zscore(X: np.ndarray) -> np.ndarray:
    mu = X.mean(axis=0)
    sigma = X.std(axis=0, ddof=0)
    sigma = np.where(sigma < 1e-12, 1.0, sigma)
    return (X - mu) / sigma


def _kmeans(
    X: np.ndarray, *, k: int, max_iter: int, seed: int, tol: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """Minimal KMeans with k-means++ init, numpy-only.

    Returns (labels, centroids). Shapes: labels (n,), centroids (k, d).
    """
    n, d = X.shape
    if k >= n:
        # Degenerate: each point is its own cluster.
        return np.arange(n, dtype=np.int64) % k, X[:k].copy() if k <= n else np.vstack([X, np.zeros((k - n, d))])

    rng = np.random.default_rng(seed)
    # k-means++ init
    centers = np.empty((k, d), dtype="float64")
    idx0 = int(rng.integers(n))
    centers[0] = X[idx0]
    for i in range(1, k):
        d2 = np.min(((X[:, None, :] - centers[:i][None, :, :]) ** 2).sum(axis=2), axis=1)
        total = d2.sum()
        if total <= 0:
            centers[i] = X[int(rng.integers(n))]
            continue
        probs = d2 / total
        idx = int(rng.choice(n, p=probs))
        centers[i] = X[idx]

    labels = np.zeros(n, dtype=np.int64)
    for _ in range(max_iter):
        # Assign
        dists = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        new_labels = np.argmin(dists, axis=1)
        if np.array_equal(new_labels, labels):
            labels = new_labels
            break
        labels = new_labels
        # Update
        new_centers = centers.copy()
        for c in range(k):
            mask = labels == c
            if mask.any():
                new_centers[c] = X[mask].mean(axis=0)
        shift = float(np.linalg.norm(new_centers - centers))
        centers = new_centers
        if shift < tol:
            break
    return labels, centers
