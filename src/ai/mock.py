"""Offline mock router used by tests and dry-run mode.

Looks up canned responses under ``tests/fixtures/ai_responses/`` keyed
by ``call_type`` and ``symbol`` (or ``"_universe"`` for watchlist).

Falls back to a built-in deterministic stub when no fixture file
exists, so the system stays offline-runnable on a fresh checkout.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .prompts import PROMPT_VERSION

_DEFAULT_USAGE = {"prompt_tokens": 1200, "completion_tokens": 350, "total_tokens": 1550}


@dataclass
class MockRouter:
    """Returns canned responses for offline / dry-run paths."""

    fixtures_root: Path | None = None

    def watchlist(self, *, symbols: list[str], as_of: str | None = None) -> dict:
        as_of = as_of or _now_iso()
        cached = self._load_fixture("watchlist", "_universe")
        if cached is not None:
            return self._wrap(cached)
        # Built-in stub: pick first 3 symbols, alternate sides.
        sides = ("long", "short", "long")
        sels = []
        for sym, side in zip(symbols[:3], sides):
            sels.append({
                "symbol": sym,
                "side": side,
                "expected_move_pct": 8.0 if side == "long" else -8.0,
                "confidence": 0.55,
                "thesis": f"Mock thesis for {sym}",
                "key_confluences": ["mock-flag"],
                "catalysts": [],
                "social_pulse": {
                    "sentiment": "neutral",
                    "sentiment_score": 0.0,
                    "attention_delta": "normal",
                    "shill_risk": "low",
                    "notable_handles": [],
                    "notes": "mock",
                },
                "risks": ["mock-risk"],
            })
        body = {
            "prompt_version": PROMPT_VERSION,
            "as_of": as_of,
            "market_regime": "chop",
            "regime_evidence": ["mock evidence"],
            "reasoning": ["mock reason 1", "mock reason 2", "mock reason 3"],
            "selections": sels,
            "discarded_pumps": [],
            "notes": "mock router",
        }
        return self._wrap(body)

    def deep(self, *, symbol: str, mark_price: float | None = None) -> dict:
        cached = self._load_fixture("deep", symbol)
        if cached is not None:
            return self._wrap(cached)
        mark = float(mark_price) if mark_price else 100.0
        sl = mark * 0.985
        tp1 = mark * 1.03
        tp2 = mark * 1.06
        body = {
            "prompt_version": PROMPT_VERSION,
            "symbol": symbol,
            "action": "long",
            "entry": mark,
            "entry_trigger": None,
            "activation_kind": None,
            "stop_loss": sl,
            "take_profit_1": tp1,
            "take_profit_2": tp2,
            "time_horizon_bars": 24,
            "confidence": 0.6,
            "expected_move_pct": 6.0,
            "reasoning": ["mock confluence", "mock regime", "mock flow"],
            "key_confluences": ["mock-flag"],
            "rationale": "mock rationale",
            "invalidation": "close below stop_loss",
        }
        return self._wrap(body)

    def review(self, *, symbol: str, trigger_reason: str = "drawdown",
               mark_price: float | None = None,
               stop_loss: float | None = None,
               side: str = "long") -> dict:
        """Canned review response. Defaults to ``tighten_stop`` so callers
        can exercise the SL-update path without wiring a fixture."""
        cached = self._load_fixture("review", symbol)
        if cached is not None:
            return self._wrap(cached)
        # Default: tighten the stop ~30% closer to the mark.
        if mark_price is not None and stop_loss is not None:
            if side == "long":
                new_sl = stop_loss + 0.3 * (float(mark_price) - float(stop_loss))
            else:
                new_sl = stop_loss - 0.3 * (float(stop_loss) - float(mark_price))
        else:
            new_sl = stop_loss
        body = {
            "prompt_version": PROMPT_VERSION,
            "symbol": symbol,
            "action": "tighten_stop" if new_sl is not None else "hold",
            "new_stop_loss": new_sl,
            "confidence": 0.55,
            "rationale": f"mock review: trigger={trigger_reason}",
            "reasoning": [f"mock-{trigger_reason}"],
        }
        return self._wrap(body)

    # ---- internal -----------------------------------------------------
    def _load_fixture(self, call_type: str, key: str) -> dict | None:
        if self.fixtures_root is None:
            return None
        path = self.fixtures_root / call_type / f"{key}.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return None

    def _wrap(self, body: dict) -> dict:
        """Format the response as if it came from OpenRouter."""
        import orjson
        text = orjson.dumps(body).decode("utf-8")
        return {
            "raw_text": text,
            "parsed": body,
            "http_status": 200,
            "latency_ms": 1,
            "json_valid": True,
            "schema_valid": True,
            "repair_retries": 0,
            "usage": dict(_DEFAULT_USAGE),
            "model": "mock/mock-1",
        }


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()
