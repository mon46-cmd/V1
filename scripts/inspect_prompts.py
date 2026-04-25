"""Render the three AI prompts on a real snapshot and dump them to disk.

Usage (offline, no network spend):

    ./.venv/Scripts/python.exe scripts/inspect_prompts.py --size 12

Output (under data/runs/<run_id>/prompt_review/):
    watchlist.system.txt
    watchlist.user.json
    watchlist.user.preview.json    # human-readable trimmed view
    deep.system.txt
    deep.user.json
    SUMMARY.md                      # token sizes, sample row, observations
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from time import perf_counter

import pandas as pd

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inspect rendered AI prompts on real snapshot data.")
    p.add_argument("--size", type=int, default=12)
    p.add_argument("--symbols", nargs="*", default=None)
    p.add_argument("--symbol-deep", type=str, default=None,
                   help="symbol to render Prompt B against (default: first watchlist row)")
    p.add_argument("--deep-top", type=int, default=5,
                   help="render Prompt B individually for the top N snapshot rows (default 5)")
    p.add_argument("--out", type=str, default=None,
                   help="output dir (default: data/runs/PROMPT_REVIEW_<ts>/)")
    return p.parse_args()


def _approx_tokens(s: str) -> int:
    return max(1, len(s) // 4)


async def _build_snapshot(args):
    from core.config import load_config
    from downloader.http import HttpClient
    from downloader.rest import RestClient
    from downloader.universe import filter_universe
    from features import FeatureConfig, build_snapshot

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
                tickers, instruments, cfg, size=args.size,
                min_turnover_usd_24h=25_000_000.0,
                max_spread_bps=10.0, min_listing_age_days=30,
            )
            symbols = uni["symbol"].tolist()
        snap, report = await build_snapshot(symbols, rest, fcfg)
    return cfg, fcfg, snap, report, symbols


def _trim_for_payload(snap: pd.DataFrame) -> pd.DataFrame:
    """Trim using the shared ai.prompts helper (column drop + smart rounding)."""
    from ai.prompts import payload_trim_for_llm
    return payload_trim_for_llm(snap)


def _render_deep_payload(row: pd.Series, snap: pd.DataFrame) -> dict:
    """Build a Prompt B payload that mirrors what runtime would send.

    Bars are emitted in compact COLUMNAR form (parallel arrays). Inspector
    has only the current snapshot row, so we synthesise realistic windows by
    jittering the live values around the close. Schema + length are what
    matter -- runtime will replace these arrays with real bars.
    """
    import numpy as np

    sym = row["symbol"]
    mark = float(row.get("close", 0.0))
    rsi = float(row.get("rsi_14", 50.0))
    atr_pct = float(row.get("atr_14_pct", 0.5))
    rel_vol = float(row.get("rel_volume_20", 1.0))
    h1_rsi = float(row.get("h1_rsi_14", 50.0))
    h4_rsi = float(row.get("h4_rsi_14", 50.0))
    h1_atr = float(row.get("h1_atr_14_pct", 0.5))
    h4_atr = float(row.get("h4_atr_14_pct", 1.0))
    rng = np.random.default_rng(seed=hash(sym) & 0xFFFF)

    # ---- Histories: 32x 15m (8h), 24x 1h (1d), 12x 4h (2d) ----
    def _walk(n: int, scale_pct: float) -> list[float]:
        w = np.cumsum(rng.normal(0.0, scale_pct / 100.0 / 4.0, n))
        return (mark * (1.0 + w - w[-1])).round(6).tolist()

    n15, n1h, n4h = 32, 24, 12
    closes_15m = _walk(n15, atr_pct)
    rsis_15m = np.clip(rsi + rng.normal(0, 3, n15), 0, 100).round(2).tolist()
    atrs_15m = np.clip(atr_pct + rng.normal(0, atr_pct * 0.1, n15), 0, None).round(4).tolist()
    rvs_15m = np.clip(rel_vol + rng.normal(0, 0.25, n15), 0, None).round(3).tolist()
    vols_15m = np.clip(rng.lognormal(mean=10.0, sigma=0.6, size=n15), 0, None).round(0).tolist()
    base_t = pd.Timestamp.now(tz="UTC").floor("15min") - pd.Timedelta(minutes=15 * n15)
    ts_15m = [(base_t + pd.Timedelta(minutes=15 * i)).isoformat() for i in range(n15)]
    bars_15m = {
        "t": ts_15m, "c": closes_15m, "v": vols_15m,
        "rsi": rsis_15m, "atr_pct": atrs_15m, "rel_vol": rvs_15m,
    }

    closes_1h = _walk(n1h, h1_atr)
    rsis_1h = np.clip(h1_rsi + rng.normal(0, 4, n1h), 0, 100).round(2).tolist()
    atrs_1h = np.clip(h1_atr + rng.normal(0, h1_atr * 0.1, n1h), 0, None).round(4).tolist()
    base_t1 = pd.Timestamp.now(tz="UTC").floor("1h") - pd.Timedelta(hours=n1h)
    ts_1h = [(base_t1 + pd.Timedelta(hours=i)).isoformat() for i in range(n1h)]
    bars_1h = {"t": ts_1h, "c": closes_1h, "rsi": rsis_1h, "atr_pct": atrs_1h}

    closes_4h = _walk(n4h, h4_atr)
    rsis_4h = np.clip(h4_rsi + rng.normal(0, 4, n4h), 0, 100).round(2).tolist()
    atrs_4h = np.clip(h4_atr + rng.normal(0, h4_atr * 0.1, n4h), 0, None).round(4).tolist()
    base_t4 = pd.Timestamp.now(tz="UTC").floor("4h") - pd.Timedelta(hours=4 * n4h)
    ts_4h = [(base_t4 + pd.Timedelta(hours=4 * i)).isoformat() for i in range(n4h)]
    bars_4h = {"t": ts_4h, "c": closes_4h, "rsi": rsis_4h, "atr_pct": atrs_4h}

    trigger = {
        "flag": "flag_volume_climax",
        "mark_price": mark,
        "atr_14_pct": atr_pct,
        "rel_volume_20": rel_vol,
        "as_of": pd.Timestamp.now(tz="UTC").isoformat(),
    }
    deriv = {
        "funding_rate": float(row.get("funding_rate", 0.0)),
        "funding_z_20": float(row.get("funding_z_20", 0.0)),
        "time_to_next_funding_sec": int(row.get("time_to_next_funding_sec", 9999) or 9999),
        "oi_chg_pct_24h": float(row.get("oi_chg_pct_24h", 0.0)),
        "oi_z_50": float(row.get("oi_z_50", 0.0)),
        "basis_bps": float(row.get("basis_bps", 0.0)),
    }
    mtf = {
        "trend_score_mtf": float(row.get("trend_score_mtf", 0.0)),
        "h1_rsi_14": h1_rsi,
        "h4_rsi_14": h4_rsi,
        "h1_ema_50_dist": float(row.get("h1_ema_50_dist", 0.0)),
        "h4_ema_50_dist": float(row.get("h4_ema_50_dist", 0.0)),
        "h1_atr_14_pct": h1_atr,
        "h4_atr_14_pct": h4_atr,
    }
    vp = {
        "poc_dist": float(row.get("poc_dist", 0.0)),
        "vah_dist": float(row.get("vah_dist", 0.0)),
        "val_dist": float(row.get("val_dist", 0.0)),
        "value_area_width_200": float(row.get("value_area_width_200", 0.0)),
    }
    flow = {
        "rs_vs_btc_24h": (None if sym == "BTCUSDT" else float(row.get("rs_vs_btc_24h", 0.0))),
        "rs_vs_eth_24h": float(row.get("rs_vs_eth_24h", 0.0)),
        "beta_btc_100": (None if sym == "BTCUSDT" else float(row.get("beta_btc_100", 1.0))),
        "corr_btc_100": (None if sym == "BTCUSDT" else float(row.get("corr_btc_100", 0.0))),
        "cluster_id": int(row.get("cluster_id", 0)),
        "cluster_leader": str(row.get("cluster_leader", "")),
    }
    snapshot_row = {
        k: (None if (isinstance(row.get(k), float) and pd.isna(row.get(k))) else row.get(k))
        for k in (
            "symbol", "close", "ret_1h", "ret_4h", "ret_24h",
            "rsi_14", "atr_14_pct", "atr_pct_rank_96",
            "bb_pct_b", "bb_width_rank_96", "adx_14", "ema_50_dist",
            "hurst_100", "vr_2_100", "acf1_50",
            "rel_volume_20", "trend_score_mtf", "supertrend_dir",
            "macd_hist", "hi_lo_24h_pos",
            "flag_volume_climax", "flag_sweep_up", "flag_sweep_dn",
            "flag_squeeze_release", "flag_macd_cross_up", "flag_macd_cross_dn",
            "flag_regime_flip", "flag_rsi_overbought", "flag_rsi_oversold",
        ) if k in row.index
    }
    symbol_meta = {
        "tick_size": 0.01,
        "min_qty": 0.001,
        "symbol_base": str(sym).replace("USDT", ""),
    }

    # ---- Levels (synthesised from the 15m walk) ----
    arr = np.array(closes_15m)
    levels = {
        "intraday_high": float(round(arr.max(), 6)),
        "intraday_low": float(round(arr.min(), 6)),
        "swing_high_24h": float(round(arr.max() * 1.01, 6)),
        "swing_low_24h": float(round(arr.min() * 0.99, 6)),
        "prior_day_high": float(round(arr.max() * 1.015, 6)),
        "prior_day_low": float(round(arr.min() * 0.985, 6)),
        "prior_day_close": float(round(mark * 0.995, 6)),
    }

    # ---- Liquidity ----
    liquidity = {
        "turnover_24h": float(row.get("turnover_24h", 0.0) or 0.0),
        "value_area_width_200": float(row.get("value_area_width_200", 0.0) or 0.0),
        "rel_volume_20": rel_vol,
        "atr_14_pct": atr_pct,
    }

    # ---- Market context (BTC + ETH refs from snap) ----
    def _ref(sym_ref: str) -> dict:
        sub = snap[snap["symbol"] == sym_ref]
        if sub.empty:
            return {}
        r = sub.iloc[0]
        return {
            "close": float(r.get("close", 0.0)),
            "ret_1h": float(r.get("ret_1h", 0.0) or 0.0),
            "ret_4h": float(r.get("ret_4h", 0.0) or 0.0),
            "ret_24h": float(r.get("ret_24h", 0.0) or 0.0),
            "rsi_14": float(r.get("rsi_14", 50.0)),
            "atr_14_pct": float(r.get("atr_14_pct", 0.5)),
            "trend_score_mtf": float(r.get("trend_score_mtf", 0.0) or 0.0),
            "funding_rate": float(r.get("funding_rate", 0.0) or 0.0),
            "oi_chg_pct_24h": float(r.get("oi_chg_pct_24h", 0.0) or 0.0),
        }

    top_movers = (
        snap[["symbol", "ret_24h", "atr_pct_rank_96", "oi_chg_pct_24h"]]
        .dropna(subset=["ret_24h"])
        .sort_values("ret_24h", key=lambda s: s.abs(), ascending=False)
        .head(8)
        .round(4)
        .to_dict(orient="records")
    )
    market_context = {
        "btc": _ref("BTCUSDT"),
        "eth": _ref("ETHUSDT"),
        "universe_size": int(len(snap)),
        "top_movers_abs_24h": top_movers,
    }

    # ---- Peer context (cluster siblings) ----
    cluster_id = int(row.get("cluster_id", 0))
    peers_df = snap[(snap["cluster_id"] == cluster_id) & (snap["symbol"] != sym)]
    peer_cols = ["symbol", "ret_24h", "rsi_14", "atr_14_pct", "atr_pct_rank_96",
                 "trend_score_mtf", "oi_chg_pct_24h", "funding_rate"]
    peer_cols = [c for c in peer_cols if c in peers_df.columns]
    peer_context = {
        "cluster_id": cluster_id,
        "cluster_leader": str(row.get("cluster_leader", "")),
        "peers": peers_df[peer_cols].head(8).round(4).to_dict(orient="records") if peer_cols else [],
    }

    # ---- Session context ----
    now = pd.Timestamp.now(tz="UTC")
    funding_in = int(row.get("time_to_next_funding_sec", 0) or 0)
    session_context = {
        "as_of": now.isoformat(),
        "utc_hour": int(now.hour),
        "utc_weekday": now.day_name(),
        "is_weekend": bool(now.weekday() >= 5),
        "next_funding_in_min": round(funding_in / 60.0, 1) if funding_in else None,
    }

    # ---- Social context (stub: empty findings, populated shape) ----
    sym_base = str(sym).replace("USDT", "")
    social_context = {
        "as_of": now.isoformat(),
        "sentiment": "unknown",
        "sentiment_score": None,
        "attention_delta": "normal",
        "shill_risk": "unknown",
        "notable_handles": [],
        "catalysts": [],
        "twitter_search_terms": [
            f"${sym_base}",
            f"#{sym_base}",
            f"{sym_base} listing OR delisted OR exploit OR hack OR partnership OR unlock OR airdrop OR ETF",
        ],
        "notes": "watchlist analyst did not surface social findings; run discovery sweep.",
    }

    # ---- Field glossary (only keys actually present) ----
    from ai.prompts import FIELD_GLOSSARY
    keys_present: set[str] = set()
    for blk in (snapshot_row, deriv, mtf, vp, flow, liquidity):
        keys_present.update(blk.keys())
    field_glossary = {k: FIELD_GLOSSARY[k] for k in keys_present if k in FIELD_GLOSSARY}

    return {
        "symbol": sym,
        "as_of": now.isoformat(),
        "trigger": trigger,
        "snapshot_row": snapshot_row,
        "bars_15m": bars_15m,
        "bars_1h": bars_1h,
        "bars_4h": bars_4h,
        "deriv": deriv,
        "flow": flow,
        "book": {},
        "vp": vp,
        "mtf": mtf,
        "levels": levels,
        "liquidity": liquidity,
        "market_context": market_context,
        "peer_context": peer_context,
        "session_context": session_context,
        "symbol_meta": symbol_meta,
        "social_context": social_context,
        "field_glossary": field_glossary,
    }


def _main_render_block_marker():  # placeholder anchor (unused)
    pass


async def _main() -> int:
    args = _cli()
    cfg, fcfg, snap, report, symbols = await _build_snapshot(args)
    print(f"[snapshot] {report.n_built}/{report.n_requested} rows  cols={len(snap.columns)}")

    # Output directory.
    ts = pd.Timestamp.now(tz="UTC").strftime("%Y%m%d-%H%M%S")
    if args.out:
        out_dir = Path(args.out)
    else:
        out_dir = cfg.run_root / f"PROMPT_REVIEW_{ts}" / "prompt_review"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[review] writing to {out_dir}")

    from ai.prompts import (
        PROMPT_VERSION,
        render_deep_prompt,
        render_watchlist_prompt,
    )

    # ---- Watchlist (full snapshot) ----
    snap_trim = _trim_for_payload(snap)
    sys_w, user_w = render_watchlist_prompt(
        rows=snap_trim, as_of=pd.Timestamp.now(tz="UTC").isoformat(),
    )
    (out_dir / "watchlist.system.txt").write_text(sys_w, encoding="utf-8")
    (out_dir / "watchlist.user.json").write_text(user_w, encoding="utf-8")

    # Pretty preview: first 3 rows + glossary keys.
    parsed = json.loads(user_w)
    preview = {
        "prompt_version": parsed["prompt_version"],
        "as_of": parsed["as_of"],
        "universe_size": parsed["universe_size"],
        "field_glossary_keys": sorted(parsed["field_glossary"].keys()),
        "n_glossary_entries": len(parsed["field_glossary"]),
        "row_columns": sorted(parsed["rows"][0].keys()) if parsed["rows"] else [],
        "n_row_columns": len(parsed["rows"][0].keys()) if parsed["rows"] else 0,
        "first_3_rows": parsed["rows"][:3],
    }
    (out_dir / "watchlist.user.preview.json").write_text(
        json.dumps(preview, indent=2), encoding="utf-8",
    )

    # ---- Deep (top N symbols, one prompt per symbol) ----
    if args.symbol_deep:
        deep_syms = [args.symbol_deep]
    else:
        deep_syms = snap["symbol"].head(max(1, args.deep_top)).tolist()

    deep_rows_summary: list[tuple[str, str, str]] = []
    for idx, deep_sym in enumerate(deep_syms, start=1):
        sub = snap[snap["symbol"] == deep_sym]
        if sub.empty:
            print(f"[deep] {deep_sym} not in snapshot, skipping")
            continue
        deep_row = sub.iloc[0]
        deep_payload = _render_deep_payload(deep_row, snap)
        sys_d, user_d = render_deep_prompt(**deep_payload)

        stem = f"deep_{idx:02d}_{deep_sym}"
        (out_dir / f"{stem}.system.txt").write_text(sys_d, encoding="utf-8")
        (out_dir / f"{stem}.user.json").write_text(user_d, encoding="utf-8")
        # Pretty pre-formatted JSON for human review (.txt extension).
        try:
            pretty = json.dumps(json.loads(user_d), indent=2, ensure_ascii=False)
        except Exception:
            pretty = user_d
        (out_dir / f"{stem}.user.txt").write_text(pretty, encoding="utf-8")
        deep_rows_summary.append((deep_sym, sys_d, user_d))
        print(f"[deep] {idx}/{len(deep_syms)} {deep_sym} -> {stem}.{{system.txt,user.json,user.txt}}")

    # Also write a watchlist.user.txt pretty version for symmetry.
    (out_dir / "watchlist.user.txt").write_text(
        json.dumps(parsed, indent=2, ensure_ascii=False), encoding="utf-8",
    )

    # Back-compat: keep deep.system.txt / deep.user.json pointing at the first row.
    if deep_rows_summary:
        first_sym, first_sys, first_user = deep_rows_summary[0]
        (out_dir / "deep.system.txt").write_text(first_sys, encoding="utf-8")
        (out_dir / "deep.user.json").write_text(first_user, encoding="utf-8")

    # ---- Summary ----
    summary = []
    summary.append(f"# Prompt review summary  ({PROMPT_VERSION})\n")
    summary.append(f"- universe size: **{len(snap)}** symbols")
    summary.append(f"- snapshot columns: **{len(snap.columns)}** (after trim: **{len(snap_trim.columns)}**)")
    summary.append(f"- deep symbols: {', '.join(f'`{s}`' for s, _, _ in deep_rows_summary)}")
    summary.append("")
    summary.append("## Sizes (chars / approx tokens)\n")
    summary.append("| call | system chars | system ~tok | user chars | user ~tok | total ~tok |")
    summary.append("|---|---:|---:|---:|---:|---:|")
    st, ut = _approx_tokens(sys_w), _approx_tokens(user_w)
    summary.append(f"| watchlist | {len(sys_w)} | {st} | {len(user_w)} | {ut} | {st + ut} |")
    for sym, sysstr, usrstr in deep_rows_summary:
        st, ut = _approx_tokens(sysstr), _approx_tokens(usrstr)
        summary.append(
            f"| deep:{sym} | {len(sysstr)} | {st} | {len(usrstr)} | {ut} | {st + ut} |"
        )
    summary.append("")
    summary.append("## Watchlist row schema (sample first row)\n")
    if parsed["rows"]:
        sample = parsed["rows"][0]
        summary.append("```json")
        summary.append(json.dumps(sample, indent=2))
        summary.append("```")
    summary.append("")
    summary.append("## Glossary keys carried\n")
    summary.append(", ".join(f"`{k}`" for k in sorted(parsed["field_glossary"].keys())))

    (out_dir / "SUMMARY.md").write_text("\n".join(summary), encoding="utf-8")

    print()
    print(f"[done] {out_dir / 'SUMMARY.md'}")
    print()
    # Echo the summary table to stdout for instant review.
    print("\n".join(summary[:18]))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
