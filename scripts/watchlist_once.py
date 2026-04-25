"""CLI: build one snapshot, ask the AI for a watchlist, save the result.

Usage (from V5/)::

    ./.venv/Scripts/python.exe scripts/watchlist_once.py --size 15
    ./.venv/Scripts/python.exe scripts/watchlist_once.py --dry-run
    ./.venv/Scripts/python.exe scripts/watchlist_once.py --no-save

Behavior:
- ``--dry-run`` forces ``AI_DRY_RUN=1`` and routes the call through
  ``MockRouter`` (no network, no spend).
- Without ``--dry-run`` the live OpenRouter endpoint is used IF
  ``OPENROUTER_API_KEY`` is set AND ``BYBIT_OFFLINE=0``. Otherwise
  the mock fires automatically (failsafe).
- The watchlist is written to
  ``data/runs/<run_id>/watchlist.json`` and an audit record is
  appended to ``prompts.jsonl`` next to it.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path
from time import perf_counter

import pandas as pd

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a snapshot then ask the AI for a watchlist.")
    p.add_argument("--size", type=int, default=15, help="universe size (top-N)")
    p.add_argument("--concurrency", type=int, default=None)
    p.add_argument("--min-turnover", type=float, default=25_000_000.0)
    p.add_argument("--max-spread-bps", type=float, default=10.0)
    p.add_argument("--min-age-days", type=int, default=30)
    p.add_argument("--symbols", nargs="*", default=None, help="bypass universe; use these symbols")
    p.add_argument("--run-id", type=str, default=None)
    p.add_argument("--no-save", action="store_true", help="do not write watchlist.json or audit record")
    p.add_argument("--dry-run", action="store_true", help="force MockRouter; no network spend")
    return p.parse_args()


async def _main() -> int:
    args = _cli()
    if args.dry_run:
        os.environ["AI_DRY_RUN"] = "1"

    from core.config import load_config
    from downloader.http import HttpClient
    from downloader.rest import RestClient
    from downloader.universe import filter_universe
    from features import FeatureConfig, build_snapshot, save_snapshot
    from ai import AIClient, AuditWriter, BudgetTracker

    cfg = load_config()
    fcfg = FeatureConfig()

    async with HttpClient(cfg) as http:
        rest = RestClient(http, cfg)

        if args.symbols:
            symbols = list(args.symbols)
        else:
            tickers = await rest.tickers()
            instruments = await rest.instruments()
            uni, _ = filter_universe(
                tickers, instruments, cfg,
                size=args.size,
                min_turnover_usd_24h=args.min_turnover,
                max_spread_bps=args.max_spread_bps,
                min_listing_age_days=args.min_age_days,
            )
            if uni.empty:
                print("universe empty; aborting", file=sys.stderr)
                return 2
            symbols = uni["symbol"].tolist()

        t0 = perf_counter()
        print(f"[watchlist] snapshot for {len(symbols)} symbols...")
        snap, report = await build_snapshot(symbols, rest, fcfg, concurrency=args.concurrency)
        dt = perf_counter() - t0

    print(f"[watchlist] snapshot built in {dt:.1f}s ({report.n_built}/{report.n_requested})")

    run_id = args.run_id or pd.Timestamp.now(tz="UTC").strftime("%Y%m%d-%H%M%S")
    run_dir = cfg.run_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    if not args.no_save:
        save_snapshot(snap, run_id, fcfg, report=report)

    budget = BudgetTracker(
        daily_cap_usd=cfg.ai_budget_usd_per_day,
        state_path=(cfg.data_root / "state" / "budget.json") if not args.no_save else None,
    )
    audit = AuditWriter(run_dir=run_dir) if not args.no_save else None

    client = AIClient(cfg, budget=budget, audit=audit)

    as_of = pd.Timestamp.now(tz="UTC").isoformat()
    print(f"[watchlist] calling AI ({client._offline and 'offline/mock' or 'live OpenRouter'})...")
    t0 = perf_counter()
    resp = await client.chat_watchlist(snap, as_of=as_of)
    dt = perf_counter() - t0

    print(f"[watchlist] response in {dt*1000:.0f} ms  regime={resp.market_regime}  "
          f"selections={len(resp.selections)}")
    for sel in resp.selections:
        print(f"  {sel.side.upper():5s} {sel.symbol:12s} "
              f"emp={sel.expected_move_pct:+.1f}%  conf={sel.confidence:.2f}  "
              f"thesis={sel.thesis[:80]}")
    if resp.discarded_pumps:
        print(f"  discarded pumps: {', '.join(resp.discarded_pumps)}")

    print(f"\n[budget] spent ${budget.spent_usd:.4f} / ${budget.daily_cap_usd:.2f}  "
          f"(remaining ${budget.remaining_usd:.4f})")

    if not args.no_save:
        out = run_dir / "watchlist.json"
        out.write_text(resp.model_dump_json(indent=2), encoding="utf-8")
        print(f"\n[saved] {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
