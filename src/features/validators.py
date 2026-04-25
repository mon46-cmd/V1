"""Feature validators: no-lookahead, NaN / warmup profile, schema check."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from features.registry import REGISTRY


@dataclass(frozen=True)
class LookaheadReport:
    feature: str
    ok: bool
    detail: str = ""


@dataclass(frozen=True)
class FeatureQualityReport:
    rows: int
    cols: int
    nan_fraction: dict[str, float] = field(default_factory=dict)
    missing_columns: tuple[str, ...] = ()
    lookahead_violations: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, object]:
        return {
            "rows": self.rows,
            "cols": self.cols,
            "nan_fraction": dict(self.nan_fraction),
            "missing_columns": list(self.missing_columns),
            "lookahead_violations": list(self.lookahead_violations),
        }


def check_no_lookahead(
    compute_fn,
    df: pd.DataFrame,
    feature_names: list[str],
    *,
    atol: float = 1e-10,
) -> list[LookaheadReport]:
    """Assert that dropping the last bar does not change feature values
    at earlier timestamps.

    ``compute_fn(df)`` must return a DataFrame with columns that are a
    superset of ``feature_names`` and indexed identically to ``df``.
    """
    full = compute_fn(df)
    truncated = compute_fn(df.iloc[:-1])
    reports: list[LookaheadReport] = []
    for name in feature_names:
        if name not in full.columns or name not in truncated.columns:
            reports.append(LookaheadReport(name, False, "missing column"))
            continue
        a = full[name].iloc[:-1]
        b = truncated[name]
        try:
            ok = _series_equal(a, b, atol=atol)
        except Exception as exc:  # noqa: BLE001
            reports.append(LookaheadReport(name, False, f"compare error: {exc}"))
            continue
        reports.append(LookaheadReport(name, ok, "" if ok else "value changed after truncation"))
    return reports


def _series_equal(a: pd.Series, b: pd.Series, *, atol: float) -> bool:
    if len(a) != len(b):
        return False
    av = a.to_numpy(dtype="float64")
    bv = b.to_numpy(dtype="float64")
    # NaN positions must match.
    nan_a = np.isnan(av)
    nan_b = np.isnan(bv)
    if not np.array_equal(nan_a, nan_b):
        return False
    mask = ~nan_a
    return bool(np.allclose(av[mask], bv[mask], atol=atol, rtol=0))


def profile_frame(df: pd.DataFrame, expected: list[str]) -> FeatureQualityReport:
    nan_frac = {c: float(df[c].isna().mean()) for c in df.columns if c in expected}
    missing = tuple(c for c in expected if c not in df.columns)
    return FeatureQualityReport(
        rows=len(df),
        cols=len(df.columns),
        nan_fraction=nan_frac,
        missing_columns=missing,
    )


def registry_warmup(feature_names: list[str]) -> dict[str, int]:
    """Return the declared warmup for each requested feature."""
    out: dict[str, int] = {}
    for name in feature_names:
        entry = REGISTRY.by_name(name)
        if entry is not None:
            out[name] = int(entry.warmup)
    return out
