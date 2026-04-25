"""Snapshot builder (Phase 5).

Given a universe of symbols, fetch the required bundles, run the
Tier A feature pipeline per symbol with bounded concurrency, then
apply the cross-sectional peer layer (ranks + clusters + relative
strength) and persist the result as parquet + JSON.

Public surface:

- ``fetch_symbol_bundle(rest, symbol, *, now, reference_15m)`` - pulls
  the REST pieces (15m/1h/4h klines, funding, OI, mark, index) and
  returns a ``SymbolBundle``. Handles missing/empty endpoints.
- ``build_snapshot_for_symbol(bundle, cfg)`` - pure; returns a single
  1-row DataFrame keyed on ``symbol`` with all SNAPSHOT_COLUMNS that
  are computable from the bundle alone (peer columns are NaN here).
- ``build_snapshot(symbols, rest, cfg, *, concurrency)`` - fetches
  bundles concurrently, computes per-symbol rows, then runs
  ``features.peer.compute_peer`` to fill peer columns.
- ``save_snapshot(df, run_id, cfg, *, report=None)`` - writes
  ``<run_root>/<run_id>/snapshot.parquet`` and a matching JSON via
  the existing ``ParquetCache`` atomic path. Also emits a compact
  ``snapshot.json`` with the SNAPSHOT_COLUMNS subset for LLM input.
"""
from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import orjson
import pandas as pd

from features.config import FeatureConfig
from features.peer import PEER_COLUMNS, PeerReport, compute_peer
from features.pipeline import SymbolBundle, compute
from features.registry import SNAPSHOT_COLUMNS


@dataclass(frozen=True)
class SnapshotReport:
    run_id: str
    as_of_ms: int
    n_requested: int
    n_built: int
    n_failed: int
    failures: tuple[tuple[str, str], ...]  # (symbol, error)
    peer: PeerReport | None


# ---- fetch ------------------------------------------------------------
async def fetch_symbol_bundle(
    rest,
    symbol: str,
    *,
    now: pd.Timestamp,
    reference_15m: pd.DataFrame | None = None,
) -> SymbolBundle:
    """Fetch the REST inputs needed by the feature pipeline.

    Uses ~6 days of 15m (480 bars > 200-bar VP warmup), ~20 days of 1h,
    ~80 days of 4h, 30 days of funding and 6 days of OI hourly.
    """
    end = now.floor("1min")
    start_15 = end - pd.Timedelta(days=6)
    start_1h = end - pd.Timedelta(days=20)
    start_4h = end - pd.Timedelta(days=80)
    start_fund = end - pd.Timedelta(days=30)
    start_oi = end - pd.Timedelta(days=6)

    k15, k1h, k4h, funding, oi, mk, ix = await asyncio.gather(
        rest.klines(symbol, "15", start_15, end),
        rest.klines(symbol, "60", start_1h, end),
        rest.klines(symbol, "240", start_4h, end),
        rest.funding(symbol, start_fund, end),
        rest.open_interest(symbol, "1h", start_oi, end),
        rest.mark_klines(symbol, "15", start_15, end),
        rest.index_klines(symbol, "15", start_15, end),
    )

    def _idx(df: pd.DataFrame) -> pd.DataFrame | None:
        if df is None or df.empty:
            return None
        if df.index.name == "timestamp" and isinstance(df.index, pd.DatetimeIndex):
            return df
        if "timestamp" in df.columns:
            return df.set_index("timestamp")
        return None

    oi_frame = _idx(oi)
    if oi_frame is not None and "open_interest" in oi_frame.columns:
        oi_frame = oi_frame.rename(columns={"open_interest": "oi"})

    return SymbolBundle(
        symbol=symbol,
        base_15m=_idx(k15),
        bars_1h=_idx(k1h),
        bars_4h=_idx(k4h),
        funding=_idx(funding),
        oi=oi_frame,
        mark_15m=_idx(mk),
        index_15m=_idx(ix),
        ref_15m=reference_15m,
    )


# ---- per-symbol ------------------------------------------------------
def build_snapshot_for_symbol(bundle: SymbolBundle, cfg: FeatureConfig) -> pd.DataFrame:
    """Compute the Tier A features for ``bundle`` and return the **last
    closed 15m bar** as a 1-row DataFrame matching ``SNAPSHOT_COLUMNS``.

    Peer columns are filled with NaN here; the cross-sectional pass
    in :func:`build_snapshot` fills them.
    """
    frame = compute("snapshot", bundle, cfg=cfg)
    if frame.empty:
        raise ValueError(f"empty frame for {bundle.symbol!r}")
    last = frame.iloc[[-1]].copy()
    # timestamp was promoted to a column by `compute`; normalize.
    if "timestamp" in last.columns and not isinstance(last["timestamp"].iloc[0], pd.Timestamp):
        last["timestamp"] = pd.to_datetime(last["timestamp"], utc=True)
    last.reset_index(drop=True, inplace=True)
    return last


# ---- build snapshot for a universe ------------------------------------
async def build_snapshot(
    symbols: Sequence[str],
    rest,
    cfg: FeatureConfig,
    *,
    now: pd.Timestamp | None = None,
    concurrency: int | None = None,
    reference_symbol: str = "BTCUSDT",
) -> tuple[pd.DataFrame, SnapshotReport]:
    """Fetch + compute + peer-aggregate.

    Returns (snapshot DataFrame with SNAPSHOT_COLUMNS, report).
    """
    now = now or pd.Timestamp.utcnow().tz_localize(None).tz_localize("UTC")
    conc = concurrency or cfg.snapshot_concurrency
    symbols_unique: list[str] = list(dict.fromkeys(symbols))

    # First fetch the reference (BTCUSDT) so every other bundle can
    # carry it as ref_15m for the xasset layer.
    ref_bundle: SymbolBundle | None = None
    if reference_symbol in symbols_unique:
        ref_bundle = await fetch_symbol_bundle(rest, reference_symbol, now=now, reference_15m=None)
    else:
        try:
            ref_bundle = await fetch_symbol_bundle(rest, reference_symbol, now=now, reference_15m=None)
        except Exception:  # noqa: BLE001
            ref_bundle = None
    ref_15m = ref_bundle.base_15m if (ref_bundle is not None) else None

    sem = asyncio.Semaphore(conc)
    rows: list[pd.DataFrame] = []
    failures: list[tuple[str, str]] = []

    async def _worker(sym: str) -> None:
        async with sem:
            try:
                if sym == reference_symbol and ref_bundle is not None:
                    bundle = ref_bundle
                    # Reference's xasset vs itself => trivially (1.0, 1.0).
                    bundle.ref_15m = bundle.base_15m
                else:
                    bundle = await fetch_symbol_bundle(rest, sym, now=now, reference_15m=ref_15m)
                row = build_snapshot_for_symbol(bundle, cfg)
                rows.append(row)
            except Exception as exc:  # noqa: BLE001
                failures.append((sym, f"{type(exc).__name__}: {exc}"))

    await asyncio.gather(*[_worker(s) for s in symbols_unique])

    if not rows:
        empty = pd.DataFrame(columns=list(SNAPSHOT_COLUMNS))
        report = SnapshotReport(
            run_id="",
            as_of_ms=int(now.timestamp() * 1000),
            n_requested=len(symbols_unique),
            n_built=0,
            n_failed=len(failures),
            failures=tuple(failures),
            peer=None,
        )
        return empty, report

    cross = pd.concat(rows, ignore_index=True)

    # Peer layer: fills PEER_COLUMNS on every row from the cross-section.
    peered, peer_report = compute_peer(cross, cfg)

    # Enforce exact column contract (fill missing with NaN, drop extras).
    out = _enforce_columns(peered, SNAPSHOT_COLUMNS)
    report = SnapshotReport(
        run_id="",
        as_of_ms=int(now.timestamp() * 1000),
        n_requested=len(symbols_unique),
        n_built=len(out),
        n_failed=len(failures),
        failures=tuple(failures),
        peer=peer_report,
    )
    return out, report


def _enforce_columns(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    cols = list(cols)
    out = pd.DataFrame(index=df.index)
    for c in cols:
        if c in df.columns:
            out[c] = df[c]
        else:
            out[c] = np.nan
    return out


# ---- persistence ------------------------------------------------------
def save_snapshot(
    df: pd.DataFrame,
    run_id: str,
    cfg: FeatureConfig,
    *,
    runs_root: Path | str | None = None,
    report: SnapshotReport | None = None,
) -> dict[str, Path]:
    """Write ``snapshot.parquet`` + ``snapshot.json`` atomically.

    ``runs_root`` defaults to ``DATA_ROOT/runs`` (DATA_ROOT env var or
    ``./data``). Also writes a compact ``snapshot_meta.json`` with the
    feature version + report.
    """
    runs_root = Path(runs_root) if runs_root else Path(os.getenv("DATA_ROOT", "data")) / "runs"
    run_dir = Path(runs_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = run_dir / "snapshot.parquet"
    json_path = run_dir / "snapshot.json"
    meta_path = run_dir / "snapshot_meta.json"

    # Parquet (atomic).
    tmp_parq = parquet_path.with_suffix(".parquet.tmp")
    df.to_parquet(tmp_parq, index=False)
    os.replace(tmp_parq, parquet_path)

    # JSON (compact, list-of-dicts, numpy-safe).
    records = _records_for_json(df)
    tmp_json = json_path.with_suffix(".json.tmp")
    tmp_json.write_bytes(orjson.dumps(records, option=orjson.OPT_SERIALIZE_NUMPY))
    os.replace(tmp_json, json_path)

    # Metadata
    meta = {
        "feature_version": cfg.version,
        "snapshot_columns": list(SNAPSHOT_COLUMNS),
        "peer_columns": list(PEER_COLUMNS),
        "n_rows": int(len(df)),
        "run_id": run_id,
    }
    if report is not None:
        meta.update({
            "as_of_ms": report.as_of_ms,
            "n_requested": report.n_requested,
            "n_built": report.n_built,
            "n_failed": report.n_failed,
            "failures": [list(f) for f in report.failures],
            "peer": {
                "n_symbols": report.peer.n_symbols,
                "n_clusters": report.peer.n_clusters,
                "btc_ret_24h": report.peer.btc_ret_24h,
                "eth_ret_24h": report.peer.eth_ret_24h,
                "cluster_sizes": {str(k): int(v) for k, v in report.peer.cluster_sizes.items()},
            } if report.peer is not None else None,
        })
    tmp_meta = meta_path.with_suffix(".json.tmp")
    tmp_meta.write_bytes(orjson.dumps(meta, option=orjson.OPT_SERIALIZE_NUMPY))
    os.replace(tmp_meta, meta_path)

    return {"parquet": parquet_path, "json": json_path, "meta": meta_path}


def _records_for_json(df: pd.DataFrame) -> list[dict]:
    records: list[dict] = []
    for _, row in df.iterrows():
        rec: dict = {}
        for col, val in row.items():
            # NaT / NaN / None first (pd.NaT is NOT an instance of pd.Timestamp).
            if val is None or (val is not val) or (isinstance(val, float) and val != val):
                rec[col] = None
            elif isinstance(val, pd.Timestamp):
                rec[col] = val.isoformat()
            elif pd.isna(val):  # catches pd.NaT and pandas NA scalars
                rec[col] = None
            elif isinstance(val, np.integer):
                rec[col] = int(val)
            elif isinstance(val, np.floating):
                rec[col] = float(val)
            else:
                rec[col] = val
        records.append(rec)
    return records
