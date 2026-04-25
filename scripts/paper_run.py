"""Phase 11 - paper run with a static watchlist (offline-friendly).

Drives the exec loop without calling the watchlist LLM. Intended for
smoke tests and short paper sessions where you want to skip the
universe + snapshot + Prompt-A pipeline and just exercise the deep
prompt + intent + broker path.

Usage::

    python scripts/paper_run.py BTCUSDT ETHUSDT --equity 10000 --dry-run

In ``--dry-run`` mode (or when ``BYBIT_OFFLINE=1``) the AI client uses
the canned ``MockRouter``; otherwise live OpenRouter calls are made.

The script does NOT poll real ticks/bars in this revision -- that
plumbing lives in :mod:`scripts.run_exec`. ``paper_run.py`` is a
one-shot: build a synthetic trigger from the latest 15m close for
each watchlist symbol, run it through ``ExecLoop.on_trigger``, and
print the queued intent + sizing decisions.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

# Make ``src/`` importable when invoked from anywhere.
_HERE = Path(__file__).resolve().parent
_SRC = _HERE.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import pandas as pd  # noqa: E402

from ai import AIClient, AuditWriter, BudgetTracker  # noqa: E402
from core.config import load_config  # noqa: E402
from core.ids import run_id as _new_run_id  # noqa: E402
from core.paths import run_dir  # noqa: E402
from core.time import now_utc  # noqa: E402
from features.config import FeatureConfig  # noqa: E402
from loops.exec import ExecConfig, ExecLoop  # noqa: E402
from loops.triggers import TriggerDecision  # noqa: E402

log = logging.getLogger("paper_run")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Paper run with a static watchlist.")
    p.add_argument("symbols", nargs="+", help="symbols (e.g. BTCUSDT ETHUSDT)")
    p.add_argument("--equity", type=float, default=10_000.0)
    p.add_argument("--dry-run", action="store_true",
                   help="force MockRouter for the AI client")
    p.add_argument("--mark", type=float, default=100.0,
                   help="synthetic mark price used to seed each trigger")
    return p.parse_args(argv)


async def _run(args: argparse.Namespace) -> int:
    if args.dry_run:
        os.environ.setdefault("AI_DRY_RUN", "1")

    cfg = load_config()
    feature_cfg = FeatureConfig()
    rid = _new_run_id()
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
    log.info("paper_run start run_id=%s symbols=%s",
             exec_loop.run_id, args.symbols)

    now = now_utc()
    submitted = 0
    for sym in args.symbols:
        bar = {
            "symbol": sym,
            "timestamp": now,
            "close": float(args.mark),
            "atr_pct": 0.01,
        }
        decision = TriggerDecision(
            symbol=sym, bar_ts=now, decision="fresh",
            flag="flag_volume_climax", close=float(args.mark),
            atr_pct=0.01, move_pct=0.0, threshold_pct=0.0,
            bars_elapsed=999, reason="paper_run",
        )
        intent = await exec_loop.on_trigger(sym, bar, decision)
        if intent is None:
            print(f"  {sym}: rejected / flat")
            continue
        submitted += 1
        print(f"  {sym}: queued id={intent.intent_id} qty={intent.qty:.6f} "
              f"side={intent.side} entry={intent.entry} sl={intent.stop_loss} "
              f"tp1={intent.take_profit_1}")

    print(f"\nrun_id={exec_loop.run_id}  submitted={submitted}")
    print(f"intents -> {exec_loop._intents_path}")
    print(f"fills   -> {exec_loop._fills_path}")
    print(f"state   -> {exec_loop._portfolio_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=os.environ.get("LOGLEVEL", "INFO"),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    args = _parse_args(argv)
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
