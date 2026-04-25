"""Daily USD budget tracker.

State is persisted at ``data/runs/<run_id>/budget.json`` so a crash
does not blow the budget on restart. The tracker is process-local;
multiple workers should not share a budget file.

Day boundaries are UTC. The first call of a new UTC day resets the
counter automatically.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock


@dataclass
class BudgetTracker:
    """Tracks USD spend with a UTC-day window."""

    daily_cap_usd: float
    state_path: Path | None = None
    _spent_usd: float = 0.0
    _day: str = ""  # YYYY-MM-DD UTC
    _lock: Lock = field(default_factory=Lock, repr=False, compare=False)

    def __post_init__(self) -> None:
        self._day = _utc_day()
        if self.state_path is not None and self.state_path.exists():
            try:
                data = json.loads(self.state_path.read_text(encoding="utf-8"))
                if data.get("day") == self._day:
                    self._spent_usd = float(data.get("spent_usd", 0.0))
            except Exception:  # noqa: BLE001
                # Corrupt state -> start fresh, but keep the file.
                self._spent_usd = 0.0

    @property
    def spent_usd(self) -> float:
        with self._lock:
            self._maybe_roll()
            return self._spent_usd

    @property
    def remaining_usd(self) -> float:
        return max(0.0, self.daily_cap_usd - self.spent_usd)

    def can_afford(self, cost_estimate_usd: float) -> bool:
        return cost_estimate_usd <= self.remaining_usd + 1e-9

    def charge(self, cost_usd: float) -> float:
        """Add ``cost_usd`` to the running total, persist, return new total."""
        with self._lock:
            self._maybe_roll()
            self._spent_usd += max(0.0, float(cost_usd))
            self._persist()
            return self._spent_usd

    def reset(self) -> None:
        with self._lock:
            self._spent_usd = 0.0
            self._day = _utc_day()
            self._persist()

    # ---- internal -----------------------------------------------------
    def _maybe_roll(self) -> None:
        today = _utc_day()
        if today != self._day:
            self._day = today
            self._spent_usd = 0.0
            self._persist()

    def _persist(self) -> None:
        if self.state_path is None:
            return
        payload = {"day": self._day, "spent_usd": round(self._spent_usd, 6),
                   "cap_usd": self.daily_cap_usd}
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        try:
            tmp.write_text(json.dumps(payload), encoding="utf-8")
            os.replace(tmp, self.state_path)
        except OSError:
            # Best-effort cleanup of orphan .tmp on failure.
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
            raise


def _utc_day() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
