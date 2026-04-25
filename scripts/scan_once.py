"""CLI: build one universe snapshot and print / save it.

Usage::

    python scripts/scan_once.py                       # defaults
    python scripts/scan_once.py --size 10 --show-rejections
    python scripts/scan_once.py --min-turnover 5e6 --max-spread-bps 20 --no-save

Exit codes:
    0   success, non-empty universe
    2   universe came back empty (filters too tight, or live fetch failed)
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

# Let ``python scripts/scan_once.py`` work without installing the package.
_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import pandas as pd  # noqa: E402

from core.config import load_config  # noqa: E402
from core.ids import run_id as make_run_id  # noqa: E402
from core.logging import configure as configure_logging  # noqa: E402
from downloader.universe import build_universe, save_universe  # noqa: E402


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build one Bybit universe snapshot.")
    p.add_argument("--size", type=int, default=None, help="top-N survivors (default: cfg.universe_size)")
    p.add_argument("--min-turnover", type=float, default=None, help="min 24h USD turnover")
    p.add_argument("--max-spread-bps", type=float, default=None, help="max bid/ask spread in bps")
    p.add_argument("--min-age", type=int, default=None, help="min listing age in days")
    p.add_argument("--min-price", type=float, default=None, help="min last-trade price (USD)")
    p.add_argument("--show-rejections", action="store_true", help="print rejection counts by reason")
    p.add_argument("--no-save", action="store_true", help="skip writing universe.parquet")
    p.add_argument("--run-id", default=None, help="override run id (default: now_utc)")
    return p.parse_args(argv)


async def _main_async(args: argparse.Namespace) -> int:
    cfg = load_config()
    configure_logging(cfg, process="scan_once")

    survivors, rejections = await build_universe(
        cfg,
        size=args.size,
        min_turnover_usd_24h=args.min_turnover,
        max_spread_bps=args.max_spread_bps,
        min_listing_age_days=args.min_age,
        min_price_usd=args.min_price,
    )

    print(f"\nuniverse: {len(survivors)} symbols (candidates before filter: "
          f"{len(survivors) + len(rejections)})")
    if not survivors.empty:
        preview = survivors[["symbol", "price", "spread_bps", "turnover_24h", "age_days"]].copy()
        preview["turnover_24h"] = preview["turnover_24h"].map(lambda v: f"{v:>14,.0f}")
        preview["spread_bps"] = preview["spread_bps"].map(lambda v: f"{v:6.2f}")
        preview["age_days"] = preview["age_days"].map(lambda v: f"{v:6.0f}")
        print(preview.to_string(index=False))

    if args.show_rejections and not rejections.empty:
        print("\nrejections by reason:")
        counts = rejections["reason"].value_counts()
        for reason, n in counts.items():
            print(f"  {reason:26s} {n:>5d}")

    if survivors.empty:
        print("\n[!] universe is empty; loosen filters or check connectivity", file=sys.stderr)
        return 2

    if not args.no_save:
        rid = args.run_id or make_run_id()
        path = save_universe(survivors, rid, cfg, rejections=rejections)
        print(f"\nsaved -> {path}")

    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        return asyncio.run(_main_async(args))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
