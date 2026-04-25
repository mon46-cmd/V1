"""Layer 11 -- calendar & seasonality features.

All derived purely from the tz-aware UTC bar timestamp. No inputs,
so warmup is 0.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from features.config import FeatureConfig


def compute_layer11(df: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    ts = df.index
    if not isinstance(ts, pd.DatetimeIndex) or ts.tz is None:
        raise ValueError("Layer 11 expects a tz-aware DatetimeIndex (UTC)")

    hour = ts.hour.to_numpy(dtype="float64")
    weekday = ts.weekday.to_numpy(dtype="float64")

    out = pd.DataFrame(index=df.index)
    out["utc_hour_sin"] = np.sin(2 * math.pi * hour / 24.0)
    out["utc_hour_cos"] = np.cos(2 * math.pi * hour / 24.0)
    out["utc_weekday_sin"] = np.sin(2 * math.pi * weekday / 7.0)
    out["utc_weekday_cos"] = np.cos(2 * math.pi * weekday / 7.0)

    # Session windows (UTC).
    out["is_us_hours"] = ((hour >= 14) & (hour < 22)).astype("float64")
    out["is_eu_hours"] = ((hour >= 8) & (hour < 17)).astype("float64")
    out["is_asia_hours"] = ((hour >= 22) | (hour < 7)).astype("float64")

    # Funding cadence: 8h settlements at the configured hours (UTC).
    minute = ts.minute.to_numpy(dtype="float64")
    total_min = hour * 60.0 + minute
    settle_minutes = np.array(sorted({h * 60 for h in cfg.funding_settle_hours_utc}), dtype="float64")
    # Time-to-next-settlement in seconds (wraps to +24h).
    day_min = 24 * 60
    diffs = (settle_minutes[None, :] - total_min[:, None]) % day_min
    to_next_min = diffs.min(axis=1)
    out["time_to_next_funding_sec"] = to_next_min * 60.0

    # Inside +/- funding_window_minutes of any settlement (wrap-aware).
    # Distance to nearest settlement = min(diff, day_min - diff).
    wrap = np.minimum(diffs, day_min - diffs).min(axis=1)
    out["is_funding_minute"] = (wrap <= cfg.funding_window_minutes).astype("float64")
    return out
