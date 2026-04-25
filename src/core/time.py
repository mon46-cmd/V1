"""Time helpers. Single source for 'now'.

Tests monkeypatch `now_utc` to freeze the clock.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd


def now_utc() -> pd.Timestamp:
    """Return current UTC instant as a tz-aware pandas Timestamp."""
    return pd.Timestamp(datetime.now(timezone.utc))


def to_utc(ts: pd.Timestamp | datetime | str | int | float) -> pd.Timestamp:
    """Coerce an arbitrary timestamp to a tz-aware UTC pandas Timestamp.

    Integers and floats are interpreted as milliseconds since epoch
    (Bybit's convention).
    """
    if isinstance(ts, (int, float)):
        t = pd.Timestamp(int(ts), unit="ms", tz="UTC")
        return t
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    else:
        t = t.tz_convert("UTC")
    return t
