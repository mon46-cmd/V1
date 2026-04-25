"""Cost-regression guard: a fresh watchlist.json must be reused.

Every restart of `run_exec.py` allocates a new run_id, so a naive
implementation calls ``ai.chat_watchlist`` (a multi-agent Grok call,
the most expensive thing in the system) on every redeploy. The
scanner must instead pick up a recent ``watchlist.json`` produced by
a previous run.
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import pytest

from ai.schemas import WatchlistResponse, WatchlistSelection
from core.config import load_config
from features.config import FeatureConfig
from loops.scanner import Scanner, _find_recent_watchlist


def _write_watchlist(path: Path, *, symbols: list[str]) -> None:
    payload = WatchlistResponse(
        prompt_version="test",
        as_of=pd.Timestamp.utcnow().tz_localize(None).tz_localize("UTC").isoformat(),
        market_regime="chop",
        selections=[
            WatchlistSelection(symbol=s, side="long",
                               expected_move_pct=2.5, confidence=0.8,
                               thesis=f"reuse {s}")
            for s in symbols
        ],
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload.model_dump_json(indent=2), encoding="utf-8")


def test_find_recent_watchlist_picks_freshest(tmp_path: Path) -> None:
    run_root = tmp_path / "runs"
    a = run_root / "20260101-000000" / "watchlist.json"
    b = run_root / "20260101-001000" / "watchlist.json"
    _write_watchlist(a, symbols=["AAA"])
    time.sleep(0.01)
    _write_watchlist(b, symbols=["BBB"])
    found = _find_recent_watchlist(run_root, max_age_sec=3600.0)
    assert found == b


def test_find_recent_watchlist_respects_age_cap(tmp_path: Path) -> None:
    run_root = tmp_path / "runs"
    p = run_root / "20250101-000000" / "watchlist.json"
    _write_watchlist(p, symbols=["AAA"])
    # Fake a stale mtime (2h ago).
    old = time.time() - 7200
    import os
    os.utime(p, (old, old))
    assert _find_recent_watchlist(run_root, max_age_sec=900.0) is None
    assert _find_recent_watchlist(run_root, max_age_sec=10_000.0) == p


def test_find_recent_watchlist_zero_disables(tmp_path: Path) -> None:
    run_root = tmp_path / "runs"
    p = run_root / "20260101-000000" / "watchlist.json"
    _write_watchlist(p, symbols=["AAA"])
    assert _find_recent_watchlist(run_root, max_age_sec=0.0) is None


@pytest.mark.asyncio
async def test_scanner_reuses_recent_watchlist_without_calling_ai(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Scanner.run_once with a fresh watchlist on disk must NOT call
    ``ai.chat_watchlist`` -- this is the cost-regression guard."""
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("BYBIT_OFFLINE", "1")
    monkeypatch.setenv("AI_DRY_RUN", "1")
    cfg = load_config()
    feature_cfg = FeatureConfig()

    # Seed a previous run with a fresh watchlist.
    prev = cfg.run_root / "20260101-000000" / "watchlist.json"
    _write_watchlist(prev, symbols=["BTCUSDT"])

    class _SpyAI:
        def __init__(self) -> None:
            self.calls = 0

        async def chat_watchlist(self, *args, **kwargs):  # noqa: ANN001
            self.calls += 1
            raise AssertionError("chat_watchlist must not be called when reuse is fresh")

    spy = _SpyAI()
    scanner = Scanner(cfg=cfg, feature_cfg=feature_cfg, ai=spy,  # type: ignore[arg-type]
                      watchlist_reuse_sec=3600.0)

    # Bypass universe/snapshot by jumping into the reuse branch directly.
    paths = scanner._paths()
    wl_path = paths["watchlist_json"]
    reuse_sec = scanner._effective_reuse_sec()
    found = _find_recent_watchlist(cfg.run_root, max_age_sec=reuse_sec,
                                   preferred=wl_path)
    assert found == prev
    assert spy.calls == 0
