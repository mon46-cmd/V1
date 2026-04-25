"""Scanner loop: universe + snapshot + watchlist + trigger gate.

This is the Phase 8 orchestration entry point. The full deep-LLM call
chain lands in Phase 11; for now the scanner:

1. Builds (or reuses) a universe.
2. Builds a fresh snapshot for that universe.
3. Calls ``ai.AIClient.chat_watchlist`` to produce ``watchlist.json``.
4. For every shortlisted symbol, evaluates the trigger gate against
   the most recent 15m snapshot row.
5. Persists fired triggers to ``triggers.jsonl`` and the cooldown
   state to ``cooldowns.json`` under the run directory.

The deep call is optional: a caller may pass ``deep_callback`` to fire
the deep prompt themselves. The scanner does NOT call it by default
because Phase 8 only owns the gate, not execution.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from ai import AIClient, WatchlistResponse
from core.config import Config
from core.ids import run_id as _new_run_id
from core.paths import run_dir
from downloader.http import HttpClient
from downloader.rest import RestClient
from downloader.universe import build_universe, save_universe
from features.config import FeatureConfig
from features.snapshot import build_snapshot, save_snapshot

from .cooldowns import CooldownStore
from .triggers import TriggerDecision, detect_trigger

log = logging.getLogger(__name__)

DeepCallback = Callable[[str, dict, TriggerDecision], Awaitable[None]]


@dataclass
class ScannerResult:
    run_id: str
    n_universe: int
    n_snapshot: int
    n_watchlist: int
    decisions: list[TriggerDecision] = field(default_factory=list)

    @property
    def n_fired(self) -> int:
        return sum(1 for d in self.decisions if d.fired)


def _row_for_symbol(snap_df: pd.DataFrame, symbol: str) -> dict | None:
    if snap_df.empty:
        return None
    sub = snap_df[snap_df["symbol"] == symbol]
    if sub.empty:
        return None
    return sub.iloc[-1].to_dict()


def _append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, default=str, sort_keys=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def _decision_to_record(d: TriggerDecision) -> dict:
    return {
        "symbol": d.symbol,
        "bar_ts": d.bar_ts.isoformat() if d.bar_ts is not None else None,
        "decision": d.decision,
        "fired": d.fired,
        "flag": d.flag,
        "close": d.close,
        "atr_pct": d.atr_pct,
        "move_pct": d.move_pct,
        "threshold_pct": d.threshold_pct,
        "bars_elapsed": d.bars_elapsed,
        "reason": d.reason,
    }


@dataclass
class Scanner:
    """Run-once and run-forever orchestrator.

    Pass ``ai`` (an :class:`AIClient`); pass ``deep_callback`` to react
    to fired triggers (Phase 11 wires this to ``chat_deep`` + intent
    queue). ``run_id`` is allocated on first iteration if omitted.
    """

    cfg: Config
    feature_cfg: FeatureConfig
    ai: AIClient
    deep_callback: DeepCallback | None = None
    run_id: str | None = None

    _store: CooldownStore | None = None

    def _ensure_run(self) -> str:
        if self.run_id is None:
            self.run_id = _new_run_id()
        d = run_dir(self.cfg, self.run_id)
        d.mkdir(parents=True, exist_ok=True)
        if self._store is None:
            self._store = CooldownStore.load(d / "cooldowns.json")
        return self.run_id

    def _paths(self) -> dict[str, Path]:
        rid = self._ensure_run()
        d = run_dir(self.cfg, rid)
        return {
            "run_dir": d,
            "watchlist_json": d / "watchlist.json",
            "triggers_jsonl": d / "triggers.jsonl",
            "cooldowns_json": d / "cooldowns.json",
        }

    async def run_once(self) -> ScannerResult:
        rid = self._ensure_run()
        paths = self._paths()
        log.info("scanner.run_once start run_id=%s", rid)

        # 1. Universe.
        survivors, rejections = await build_universe(self.cfg)
        if survivors.empty:
            log.warning("scanner.run_once empty universe")
            return ScannerResult(run_id=rid, n_universe=0, n_snapshot=0, n_watchlist=0)
        save_universe(survivors, rid, self.cfg, rejections=rejections)
        symbols = list(survivors["symbol"].astype(str))
        log.info("scanner.run_once universe n=%d", len(symbols))

        # 2. Snapshot.
        async with HttpClient(self.cfg) as http:
            rest = RestClient(http, self.cfg)
            snap_df, snap_report = await build_snapshot(
                symbols, rest, self.feature_cfg,
            )
        save_snapshot(snap_df, rid, self.feature_cfg, runs_root=self.cfg.run_root,
                      report=snap_report)
        log.info("scanner.run_once snapshot built=%d failed=%d",
                 snap_report.n_built, snap_report.n_failed)

        # 3. Watchlist (LLM A).
        as_of = pd.Timestamp.utcnow().tz_localize(None).tz_localize("UTC").isoformat()
        wl: WatchlistResponse = await self.ai.chat_watchlist(snap_df, as_of=as_of)
        wl_path = paths["watchlist_json"]
        wl_path.parent.mkdir(parents=True, exist_ok=True)
        wl_path.write_text(wl.model_dump_json(indent=2), encoding="utf-8")
        watch_symbols = [s.symbol for s in wl.shortlist]
        log.info("scanner.run_once watchlist n=%d", len(watch_symbols))

        # 4. Trigger gate per shortlisted symbol.
        decisions: list[TriggerDecision] = []
        assert self._store is not None
        for sym in watch_symbols:
            row = _row_for_symbol(snap_df, sym)
            if row is None:
                decisions.append(TriggerDecision(symbol=sym, bar_ts=None,
                                                 decision="no_bar",
                                                 reason="symbol missing from snapshot"))
                continue
            bar_ts = pd.Timestamp(row.get("timestamp")) if row.get("timestamp") is not None else None
            state = self._store.state_for(sym, now_bar_ts=bar_ts)
            d = detect_trigger(symbol=sym, bar=row, state=state, cfg=self.cfg)
            decisions.append(d)
            _append_jsonl(paths["triggers_jsonl"], _decision_to_record(d))
            if d.fired:
                self._store.record(d)
                log.info("trigger fired symbol=%s flag=%s decision=%s", sym, d.flag, d.decision)
                if self.deep_callback is not None:
                    try:
                        await self.deep_callback(sym, row, d)
                    except Exception:  # noqa: BLE001
                        log.exception("deep_callback failed symbol=%s", sym)

        # 5. Persist cooldown state.
        self._store.save()

        return ScannerResult(
            run_id=rid,
            n_universe=len(symbols),
            n_snapshot=int(snap_report.n_built),
            n_watchlist=len(watch_symbols),
            decisions=decisions,
        )

    async def run_forever(self, interval_sec: int = 1800) -> None:
        """Loop ``run_once`` every ``interval_sec`` (default 30 min)."""
        while True:
            try:
                await self.run_once()
            except Exception:  # noqa: BLE001
                log.exception("scanner.run_once crashed")
            await asyncio.sleep(max(60, int(interval_sec)))


__all__ = ["Scanner", "ScannerResult", "DeepCallback"]
