"""Phase 11 - live exec loop driver.

Runs the full pipeline in a single process:

    Scanner.run_forever()            (every 30 min)
        -> ExecLoop.on_trigger       (deep LLM + intent submit)
    ActivationWatcher.run(ws_feed)   (per tick + book)
        -> ExecLoop.emit_event       (broker.open_from_intent)
    bar_feed                         (per closed 15m bar)
        -> ExecLoop.on_bar           (broker.on_bar fills)

The websocket adapter is intentionally minimal in this Phase 11
revision: a thin async generator that wraps ``downloader.ws.WsClient``
and yields :class:`Tick` / :class:`BookTop` items the watcher already
understands. A real bar feed is left out -- live exec only emits
broker fills when the snapshot builder rolls a new 15m bar, so we
reuse the scanner's snapshot pass to drive ``on_bar`` for watchlist
positions.

Phase 11 is a paper-only loop. No real orders are placed.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_SRC = _HERE.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from ai import AIClient, AuditWriter, BudgetTracker  # noqa: E402
from core.config import load_config  # noqa: E402
from core.ids import run_id as _new_run_id  # noqa: E402
from core.paths import run_dir  # noqa: E402
from features.config import FeatureConfig  # noqa: E402
from loops.exec import ExecConfig, ExecLoop  # noqa: E402

log = logging.getLogger("run_exec")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the live (paper) exec loop.")
    p.add_argument("--equity", type=float, default=10_000.0)
    p.add_argument("--scanner-interval", type=int, default=1800,
                   help="seconds between scanner.run_once() iterations")
    p.add_argument("--dry-run", action="store_true",
                   help="force MockRouter for the AI client")
    p.add_argument("--run-id", type=str, default=None,
                   help="resume an existing run_id (replays fills.jsonl)")
    p.add_argument("--stop-poll-sec", type=float, default=10.0,
                   help="seconds between STOP-file checks")
    return p.parse_args(argv)


async def _stop_file_watcher(exec_loop, *, poll_sec: float) -> None:
    """Background task: when ``data/STOP`` appears, log loudly and let
    the main coroutine exit on its next idle check.
    """
    while True:
        if exec_loop.stop_requested():
            log.warning("STOP file detected at %s -- shutting down",
                        exec_loop.stop_file_path())
            return
        await asyncio.sleep(max(1.0, poll_sec))


async def _run(args: argparse.Namespace) -> int:
    if args.dry_run:
        os.environ.setdefault("AI_DRY_RUN", "1")

    cfg = load_config()
    feature_cfg = FeatureConfig()
    rid = args.run_id or _new_run_id()
    rdir = run_dir(cfg, rid)
    rdir.mkdir(parents=True, exist_ok=True)
    audit = AuditWriter(run_dir=rdir)
    budget = BudgetTracker(daily_cap_usd=cfg.ai_budget_usd_per_day,
                           state_path=rdir / "budget.json")
    ai = AIClient(cfg=cfg, budget=budget, audit=audit)
    exec_loop = ExecLoop.build(
        cfg=cfg, feature_cfg=feature_cfg, ai=ai,
        run_id=rid,
        exec_cfg=ExecConfig(starting_equity_usd=args.equity),
    )
    if exec_loop.stop_requested():
        log.error("refusing to start: STOP file present at %s",
                  exec_loop.stop_file_path())
        return 2

    scanner = exec_loop.make_scanner()
    log.info("run_exec start run_id=%s interval=%ds (drop %s to stop)",
             exec_loop.run_id, args.scanner_interval,
             exec_loop.stop_file_path())

    # Race: scanner.run_forever vs STOP-file watcher. Whichever finishes
    # first cancels the other. Scanner loop drives triggers ->
    # on_trigger; the WS-driven activation/bar feed lands in a later
    # phase.
    stop_task = asyncio.create_task(
        _stop_file_watcher(exec_loop, poll_sec=args.stop_poll_sec))
    scan_task = asyncio.create_task(
        scanner.run_forever(interval_sec=args.scanner_interval))
    try:
        done, pending = await asyncio.wait(
            {stop_task, scan_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        for t in pending:
            t.cancel()
        for t in pending:
            try:
                await t
            except asyncio.CancelledError:
                pass
    finally:
        # Persist a final state snapshot regardless of exit path.
        try:
            exec_loop._save_state()
        except Exception:  # noqa: BLE001
            log.exception("final save_state failed")
    return 0 if not exec_loop.stop_requested() else 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=os.environ.get("LOGLEVEL", "INFO"),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    args = _parse_args(argv)
    try:
        return asyncio.run(_run(args))
    except KeyboardInterrupt:
        log.info("run_exec interrupted")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
