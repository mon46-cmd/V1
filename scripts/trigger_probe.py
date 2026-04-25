"""Replay cached 15m bars against the trigger gate.

Offline-only. Reads a 15m kline parquet from the local cache and
runs ``detect_trigger`` bar-by-bar, printing the resulting ledger.

Useful for sanity-checking the cooldown logic on a real symbol
before/after tuning ``cooldown_bypass_atr_mult`` or
``cooldown_bypass_floor_pct``.

Usage:

    python scripts/trigger_probe.py --symbol BTCUSDT \
        [--from 2026-04-20] [--to 2026-04-25] [--limit 200]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# Make src/ importable when running as a plain script.
_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from core.config import load_config  # noqa: E402
from features.config import FeatureConfig  # noqa: E402
from features.pipeline import compute  # noqa: E402
from features.snapshot import SymbolBundle  # noqa: E402
from loops.cooldowns import CooldownStore  # noqa: E402
from loops.triggers import detect_trigger  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Replay 15m bars against the trigger gate.")
    p.add_argument("--symbol", required=True)
    p.add_argument("--from", dest="dt_from", default=None,
                   help="UTC start (e.g. 2026-04-20). Default: all cached bars.")
    p.add_argument("--to", dest="dt_to", default=None,
                   help="UTC end (inclusive).")
    p.add_argument("--limit", type=int, default=0,
                   help="Print at most N decisions (0 = all).")
    p.add_argument("--state-file", default=None,
                   help="Optional cooldowns.json path; default = ephemeral.")
    return p.parse_args()


def _load_klines(cfg, symbol: str) -> pd.DataFrame:
    """Read cached 15m klines for ``symbol`` from the parquet cache."""
    # ParquetCache layout from Phase 1: cache/klines/<SYM>/<interval>.parquet
    candidates = [
        cfg.cache_root / "klines" / symbol / "15.parquet",
        cfg.cache_root / "klines_15" / f"{symbol}.parquet",
    ]
    for path in candidates:
        if path.exists():
            return pd.read_parquet(path)
    raise FileNotFoundError(
        f"No 15m kline cache for {symbol}. Tried: "
        + ", ".join(str(p) for p in candidates)
    )


def main() -> int:
    args = _parse_args()
    cfg = load_config()
    fcfg = FeatureConfig()

    klines = _load_klines(cfg, args.symbol)
    if args.dt_from:
        klines = klines[klines["timestamp"] >= pd.Timestamp(args.dt_from, tz="UTC")]
    if args.dt_to:
        klines = klines[klines["timestamp"] <= pd.Timestamp(args.dt_to, tz="UTC")]
    if klines.empty:
        print(f"no bars for {args.symbol} in range")
        return 2

    # Build a minimal SymbolBundle and compute features bar-by-bar.
    # We use the full series so trigger flags get their proper warmup.
    bundle = SymbolBundle(
        symbol=args.symbol,
        base_15m=klines.set_index("timestamp"),
        bars_1h=None, bars_4h=None,
        funding=None, oi=None,
        mark_15m=None, index_15m=None, ref_15m=None,
    )
    feats = compute("snapshot", bundle, cfg=fcfg)
    if "timestamp" not in feats.columns:
        feats = feats.reset_index().rename(columns={"index": "timestamp"})

    state_path = Path(args.state_file) if args.state_file else None
    if state_path:
        store = CooldownStore.load(state_path)
    else:
        store = CooldownStore(path=Path("/dev/null"))

    rows = feats.to_dict(orient="records")
    decisions = []
    for row in rows:
        bar_ts = pd.Timestamp(row.get("timestamp"))
        state = store.state_for(args.symbol, now_bar_ts=bar_ts)
        d = detect_trigger(symbol=args.symbol, bar=row, state=state, cfg=cfg)
        decisions.append(d)
        if d.fired:
            store.record(d)
        if args.limit and len(decisions) >= args.limit:
            break

    # Print a compact ledger; only show non-trivial decisions.
    interesting = [d for d in decisions
                   if d.decision not in ("no_flag", "no_bar")]
    print(f"{args.symbol}: {len(decisions)} bars, {len(interesting)} non-trivial")
    print(f"{'bar_ts':<25} {'decision':<18} {'flag':<22} {'close':>12} {'reason'}")
    for d in interesting:
        ts = d.bar_ts.isoformat() if d.bar_ts is not None else "-"
        flag = d.flag or "-"
        close = f"{d.close:.6g}" if d.close is not None else "-"
        print(f"{ts:<25} {d.decision:<18} {flag:<22} {close:>12} {d.reason}")

    fired = sum(1 for d in decisions if d.fired)
    print(f"-- fired: {fired} / {len(decisions)} --")
    if state_path:
        store.save()
        print(f"state saved -> {state_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
