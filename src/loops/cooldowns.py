"""Persistent cooldown store.

Keeps per-symbol last-trigger state on disk so the gate can resume
after a restart. Format: a single JSON file at
``<run_root>/<run_id>/cooldowns.json`` with the shape::

    {
      "BTCUSDT": {"last_bar_ts": "2026-04-25T12:30:00Z",
                  "last_close": 65000.0,
                  "last_flag": "flag_volume_climax"},
      ...
    }

Writes are atomic (``.tmp`` + ``os.replace``). The store also tracks
``bars_since`` lazily: since 15m bars are deterministic-spaced, we
compute it from ``(now_bar_ts - last_bar_ts) / 15min`` at read time.
"""
from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import pandas as pd

from .triggers import CooldownState, TriggerDecision

_BAR_15M = pd.Timedelta(minutes=15)


@dataclass(frozen=True)
class CooldownEntry:
    last_bar_ts: pd.Timestamp
    last_close: float
    last_flag: str

    def to_json(self) -> dict:
        return {
            "last_bar_ts": self.last_bar_ts.isoformat(),
            "last_close": float(self.last_close),
            "last_flag": self.last_flag,
        }

    @classmethod
    def from_json(cls, d: Mapping) -> "CooldownEntry":
        ts = pd.Timestamp(d["last_bar_ts"])
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        return cls(
            last_bar_ts=ts,
            last_close=float(d["last_close"]),
            last_flag=str(d.get("last_flag", "")),
        )


@dataclass
class CooldownStore:
    """Atomic JSON-backed cooldown table. Thread-safe for single process."""

    path: Path
    _data: dict[str, CooldownEntry] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @classmethod
    def load(cls, path: Path) -> "CooldownStore":
        store = cls(path=Path(path))
        if store.path.exists():
            try:
                raw = json.loads(store.path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                raw = {}
            for sym, payload in raw.items():
                try:
                    store._data[sym] = CooldownEntry.from_json(payload)
                except (KeyError, ValueError, TypeError):
                    continue
        return store

    def save(self) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(self.path.suffix + ".tmp")
            payload = {sym: e.to_json() for sym, e in self._data.items()}
            tmp.write_text(json.dumps(payload, indent=2, sort_keys=True),
                           encoding="utf-8")
            os.replace(tmp, self.path)

    def state_for(self, symbol: str, *, now_bar_ts: pd.Timestamp | None = None,
                  ) -> CooldownState:
        entry = self._data.get(symbol)
        if entry is None:
            return CooldownState()
        if now_bar_ts is None:
            return CooldownState(last_bar_ts=entry.last_bar_ts,
                                 last_close=entry.last_close,
                                 bars_since=10**9)
        if now_bar_ts.tzinfo is None:
            now_bar_ts = now_bar_ts.tz_localize("UTC")
        delta = now_bar_ts - entry.last_bar_ts
        if delta < pd.Timedelta(0):
            bars_since = 0
        else:
            bars_since = int(delta // _BAR_15M)
        return CooldownState(last_bar_ts=entry.last_bar_ts,
                             last_close=entry.last_close,
                             bars_since=bars_since)

    def record(self, decision: TriggerDecision) -> None:
        if not decision.fired:
            return
        if decision.bar_ts is None or decision.close is None or decision.flag is None:
            return
        with self._lock:
            self._data[decision.symbol] = CooldownEntry(
                last_bar_ts=decision.bar_ts,
                last_close=float(decision.close),
                last_flag=decision.flag,
            )

    def __contains__(self, symbol: str) -> bool:
        return symbol in self._data

    def __len__(self) -> int:
        return len(self._data)

    def symbols(self) -> list[str]:
        return list(self._data.keys())
