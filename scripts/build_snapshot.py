"""One-shot snapshot builder.

Usage (from V5/):

    ./.venv/Scripts/python.exe scripts/build_snapshot.py --size 10

Flow:
1. Build the universe (Phase 3) to pick the top-N liquid USDT perps.
2. Fetch per-symbol bundles + compute Tier A features + apply the
   peer layer (ranks + clusters).
3. Save parquet + json under data/runs/<run_id>/.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from time import perf_counter

import pandas as pd

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a multi-symbol Tier A feature snapshot.")
    p.add_argument("--size", type=int, default=30, help="universe size (top-N by turnover)")
    p.add_argument("--concurrency", type=int, default=None, help="override FeatureConfig.snapshot_concurrency")
    p.add_argument("--min-turnover", type=float, default=25_000_000.0, help="min 24h turnover USD")
    p.add_argument("--max-spread-bps", type=float, default=10.0)
    p.add_argument("--min-age-days", type=int, default=30)
    p.add_argument("--run-id", type=str, default=None, help="override run id (default = UTC YYYYMMDD-HHMMSS)")
    p.add_argument("--no-save", action="store_true")
    p.add_argument("--symbols", nargs="*", default=None, help="bypass universe; use these symbols")
    return p.parse_args()


async def _main() -> int:
    args = _cli()
    from core.config import load_config
    from downloader.http import HttpClient
    from downloader.rest import RestClient
    from downloader.universe import build_universe
    from features import FeatureConfig, build_snapshot, save_snapshot

    cfg = load_config()
    fcfg = FeatureConfig()

    async with HttpClient(cfg) as http:
        rest = RestClient(http, cfg)

        if args.symbols:
            symbols = list(args.symbols)
        else:
            tickers = await rest.tickers()
            instruments = await rest.instruments()
            from downloader.universe import filter_universe
            uni, _rej = filter_universe(
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
        print(f"[snapshot] fetching + computing for {len(symbols)} symbols...")
        snap, report = await build_snapshot(
            symbols, rest, fcfg, concurrency=args.concurrency,
        )
        dt = perf_counter() - t0

    print(f"[snapshot] built {report.n_built}/{report.n_requested} rows in {dt:.1f}s "
          f"(failures={report.n_failed})")
    if report.failures:
        for s, e in report.failures[:5]:
            print(f"  FAIL {s}: {e}")

    if report.peer is not None:
        p = report.peer
        print(f"[peer] n={p.n_symbols}  clusters={p.n_clusters}  "
              f"btc_ret_24h={p.btc_ret_24h:+.3%}  eth_ret_24h={p.eth_ret_24h:+.3%}")

    # Compact preview
    preview_cols = [
        "symbol", "close", "ret_24h", "atr_14_pct",
        "rsi_14", "funding_rate", "oi_chg_pct_24h",
        "trend_score_mtf", "cluster_id", "cluster_leader",
        "rank_turnover_24h",
    ]
    print("\n[snapshot] preview:")
    cols_present = [c for c in preview_cols if c in snap.columns]
    with pd.option_context("display.max_rows", 40, "display.width", 160,
                           "display.float_format", lambda x: f"{x:.4f}"):
        print(snap[cols_present].to_string(index=False))

    if not args.no_save:
        run_id = args.run_id or pd.Timestamp.utcnow().strftime("%Y%m%d-%H%M%S")
        paths = save_snapshot(snap, run_id, fcfg, report=report)
        print(f"\n[saved] {paths['parquet']}")
        print(f"[saved] {paths['json']}")
        print(f"[saved] {paths['meta']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
