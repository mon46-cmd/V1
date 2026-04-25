"""Tests for ai/client.py via the offline MockRouter path.

These never touch the network. We toggle ``ai_dry_run`` so the
client routes everything through ``MockRouter``.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pandas as pd
import pytest

from ai.audit import AuditWriter
from ai.budget import BudgetTracker
from ai.client import AIClient
from ai.mock import MockRouter
from ai.prompts import PROMPT_VERSION
from ai.schemas import DeepSignal, WatchlistResponse


def _cfg(tmp_path: Path):
    """Build a Config that points at tmp_path and forces offline mode."""
    from core.config import load_config
    import os

    os.environ["DATA_ROOT"] = str(tmp_path)
    os.environ["AI_DRY_RUN"] = "1"
    return load_config()


def _snap_df() -> pd.DataFrame:
    return pd.DataFrame([
        {"symbol": "BTCUSDT", "close": 60000.0, "ret_24h": 0.02, "rsi_14": 55.0},
        {"symbol": "ETHUSDT", "close": 3000.0,  "ret_24h": 0.04, "rsi_14": 62.0},
        {"symbol": "SOLUSDT", "close": 140.0,   "ret_24h": -0.01,"rsi_14": 48.0},
    ])


@pytest.mark.asyncio
async def test_watchlist_offline_returns_validated(tmp_path: Path, monkeypatch):
    cfg = _cfg(tmp_path)
    run_dir = cfg.run_root / "TEST"
    audit = AuditWriter(run_dir=run_dir)
    budget = BudgetTracker(daily_cap_usd=1.0, state_path=run_dir / "budget.json")

    client = AIClient(cfg, budget=budget, audit=audit)
    assert client._offline is True

    resp = await client.chat_watchlist(_snap_df(), as_of="2026-04-25T00:00:00Z")
    assert isinstance(resp, WatchlistResponse)
    assert resp.prompt_version == PROMPT_VERSION
    assert len(resp.selections) >= 1
    assert (run_dir / "prompts.jsonl").exists()
    line = (run_dir / "prompts.jsonl").read_text(encoding="utf-8").strip()
    rec = json.loads(line)
    assert rec["call_type"] == "watchlist"
    assert rec["decision"]["type"] == "watchlist"
    # Sidecars written.
    assert any((run_dir / "prompts").glob("*.req.json"))


@pytest.mark.asyncio
async def test_deep_offline_consistency_pass(tmp_path: Path):
    cfg = _cfg(tmp_path)
    client = AIClient(cfg)
    sig = await client.chat_deep("BTCUSDT", payload={
        "as_of": "t",
        "trigger": {"flag": "flag_volume_climax", "mark_price": 100.0},
        "bars_15m": [{"close": 100.0}],
    })
    assert isinstance(sig, DeepSignal)
    assert sig.symbol == "BTCUSDT"
    # MockRouter builds an internally-consistent long with R:R >= 2 -> not forced flat.
    assert sig.action == "long"


@pytest.mark.asyncio
async def test_deep_inconsistent_response_forces_flat(tmp_path: Path):
    cfg = _cfg(tmp_path)

    # Build a custom mock that returns a long with R:R < 2 and SL on wrong side.
    class BadMock(MockRouter):
        def deep(self, *, symbol, mark_price=None):
            body = {
                "prompt_version": PROMPT_VERSION,
                "symbol": symbol,
                "action": "long",
                "entry": 100.0,
                "stop_loss": 105.0,        # WRONG side
                "take_profit_1": 100.5,     # tiny reward
                "take_profit_2": 101.0,
                "time_horizon_bars": 24,
                "confidence": 0.7,
                "expected_move_pct": 5.0,
                "reasoning": ["x", "y", "z"],
                "rationale": "bad",
                "invalidation": "n/a",
            }
            return self._wrap(body)

    client = AIClient(cfg, mock=BadMock())
    sig = await client.chat_deep("ETHUSDT", payload={
        "as_of": "t",
        "trigger": {"mark_price": 100.0},
        "bars_15m": [{"close": 100.0}],
    })
    # >= 2 consistency violations -> client forces flat.
    assert sig.action == "flat"


@pytest.mark.asyncio
async def test_budget_exhausted_returns_synthetic_flat(tmp_path: Path):
    cfg = _cfg(tmp_path)
    run_dir = cfg.run_root / "BUDGET"
    audit = AuditWriter(run_dir=run_dir)
    # Cap at zero -> immediately exhausted.
    budget = BudgetTracker(daily_cap_usd=0.0, state_path=run_dir / "budget.json")
    client = AIClient(cfg, budget=budget, audit=audit)

    sig = await client.chat_deep("BTCUSDT", payload={
        "as_of": "t",
        "trigger": {"mark_price": 100.0},
        "bars_15m": [{"close": 100.0}],
    })
    assert sig.action == "flat"
    # Warning must be logged.
    warn_path = run_dir / "ai_warnings.jsonl"
    assert warn_path.exists()
    content = warn_path.read_text(encoding="utf-8")
    assert "budget_exhausted" in content


def test_audit_redacts_secrets(tmp_path: Path):
    audit = AuditWriter(run_dir=tmp_path / "redact")
    audit.write_call(
        call_id="abc123",
        call_type="watchlist",
        model="x-ai/grok-4",
        prompt_version=PROMPT_VERSION,
        symbol=None,
        request={
            "system": "sys",
            "user": "Bearer sk-supersecretkey1234567890ABCD please",
            "headers": {"Authorization": "Bearer sk-LIVEKEY1234567890ABCD"},
            "temperature": 0.2,
        },
        response={"raw_text": "ok", "latency_ms": 10, "http_status": 200, "json_valid": True},
        decision={"type": "watchlist"},
    )
    line = (tmp_path / "redact" / "prompts.jsonl").read_text(encoding="utf-8")
    assert "sk-supersecretkey" not in line
    assert "sk-LIVEKEY" not in line
    req_files = list((tmp_path / "redact" / "prompts").glob("*.req.json"))
    assert req_files
    req = req_files[0].read_text(encoding="utf-8")
    assert "sk-supersecretkey" not in req
    assert "sk-LIVEKEY" not in req
    assert "***" in req


def test_budget_persists_and_rolls(tmp_path: Path, monkeypatch):
    state = tmp_path / "b.json"
    b = BudgetTracker(daily_cap_usd=1.0, state_path=state)
    b.charge(0.30)
    assert state.exists()
    # Re-load picks up state.
    b2 = BudgetTracker(daily_cap_usd=1.0, state_path=state)
    assert abs(b2.spent_usd - 0.30) < 1e-9
    # Force a day rollover.
    b2._day = "1999-01-01"
    b2._maybe_roll()
    assert b2.spent_usd == 0.0


def test_cost_usd_known_and_unknown_models():
    from ai.prices import cost_usd
    # grok-4: 3.0 prompt, 15.0 completion per 1M.
    cost = cost_usd("x-ai/grok-4", 1_000_000, 1_000_000)
    assert abs(cost - (3.0 + 15.0)) < 1e-9
    # Unknown model -> default (3.0/15.0).
    cost2 = cost_usd("does/not/exist", 1_000_000, 0)
    assert abs(cost2 - 3.0) < 1e-9
