"""Alignment helpers: asof attach, MTF stitch, reference-asset join."""
from __future__ import annotations

from typing import Iterable

import pandas as pd


def attach_asof(
    base: pd.DataFrame,
    extra: pd.DataFrame,
    columns: Iterable[str],
    *,
    direction: str = "backward",
    tolerance: pd.Timedelta | None = None,
) -> pd.DataFrame:
    """Attach ``columns`` from ``extra`` onto ``base`` using ``merge_asof``.

    Both frames are expected to be sorted by a tz-aware UTC DatetimeIndex
    named ``timestamp``. The result is indexed identically to ``base``.
    """
    cols = [c for c in columns if c in extra.columns]
    if not cols:
        return base.copy()
    left = base.reset_index().rename(columns={"index": "timestamp"}) \
        if base.index.name != "timestamp" else base.reset_index()
    right = extra[cols].reset_index()
    if right.columns[0] != "timestamp":
        right = right.rename(columns={right.columns[0]: "timestamp"})
    merged = pd.merge_asof(
        left.sort_values("timestamp"),
        right.sort_values("timestamp"),
        on="timestamp",
        direction=direction,
        tolerance=tolerance,
    )
    return merged.set_index("timestamp")


def attach_mtf(
    base_15m: pd.DataFrame,
    features_1h: pd.DataFrame | None,
    features_4h: pd.DataFrame | None,
    names: Iterable[str],
) -> pd.DataFrame:
    """For each ``name``, attach ``h1_<name>`` and ``h4_<name>`` columns.

    Uses merge_asof ``backward`` so the 15m row at T sees only the most
    recently *closed* higher-TF bar (no lookahead).
    """
    out = base_15m.copy()
    for name in names:
        if features_1h is not None and name in features_1h.columns:
            h1 = features_1h[[name]].rename(columns={name: f"h1_{name}"})
            out = _merge_asof_indexed(out, h1)
        if features_4h is not None and name in features_4h.columns:
            h4 = features_4h[[name]].rename(columns={name: f"h4_{name}"})
            out = _merge_asof_indexed(out, h4)
    return out


def _merge_asof_indexed(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    """merge_asof on a tz-aware DatetimeIndex, preserving the left index."""
    l = left.reset_index()
    r = right.reset_index()
    ts_name = l.columns[0]
    merged = pd.merge_asof(
        l.sort_values(ts_name),
        r.sort_values(r.columns[0]).rename(columns={r.columns[0]: ts_name}),
        on=ts_name,
        direction="backward",
    )
    return merged.set_index(ts_name)
