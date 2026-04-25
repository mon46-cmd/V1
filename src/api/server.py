"""Phase 13 - read-only HTTP API + static dashboard.

FastAPI app on ``127.0.0.1:8765`` (default) that exposes the run
artefacts already on disk plus a live SSE stream tailing the active
run's audit files. Nothing here writes to the orchestrator state; the
API is purely a read view, safe to run alongside scanner/exec.

Routes:

  GET  /api/runs                       list run ids (newest first)
  GET  /api/runs/{run_id}/universe     universe.json contents
  GET  /api/runs/{run_id}/snapshot     snapshot.json (capped)
  GET  /api/runs/{run_id}/prompts      prompts.jsonl (sanitized, capped)
  GET  /api/runs/{run_id}/watchlist    watchlist.json
  GET  /api/runs/{run_id}/triggers     triggers.jsonl tail
  GET  /api/runs/{run_id}/intents      intents.jsonl tail
  GET  /api/runs/{run_id}/fills        fills.jsonl
  GET  /api/runs/{run_id}/reviews      reviews.jsonl
  GET  /api/portfolio                  active portfolio.json
  GET  /api/metrics                    7d summary of fills/budget
  GET  /api/live                       SSE stream of new tail lines
  GET  /                               static dashboard

The "active" run is the most recently modified run directory under
``cfg.run_root``. Override with ``?run_id=...`` on routes that take it.
"""
from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import (
    FileResponse,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles

from core.config import Config, load_config

from . import charts as chartdata

log = logging.getLogger(__name__)

# Maximum rows returned for the "tail-style" endpoints (prompts /
# triggers / intents / fills / reviews). The dashboard shows the most
# recent activity; deep history goes through the artefact files
# directly (parquet / jsonl).
DEFAULT_LIMIT = 500
MAX_LIMIT = 5000

# Fields stripped from prompts.jsonl before returning. The full prompt
# text can be quite large and is not needed for the operator view.
_PROMPT_SANITISE_FIELDS = ("system", "user", "raw_text")


# ---------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------
def create_app(cfg: Config | None = None) -> FastAPI:
    cfg = cfg or load_config()
    app = FastAPI(
        title="V5 paper orchestrator",
        version="0.1.0",
        docs_url="/api/docs",
        redoc_url=None,
    )
    app.state.cfg = cfg

    # ---- discovery helpers -------------------------------------------
    def _list_runs() -> list[dict]:
        root = cfg.run_root
        if not root.exists():
            return []
        out: list[dict] = []
        for p in root.iterdir():
            if not p.is_dir():
                continue
            try:
                mtime = p.stat().st_mtime
            except OSError:
                continue
            out.append({"run_id": p.name, "mtime": mtime,
                        "path": str(p)})
        out.sort(key=lambda r: r["mtime"], reverse=True)
        return out

    def _active_run_id() -> str | None:
        runs = _list_runs()
        return runs[0]["run_id"] if runs else None

    def _resolve_run(run_id: str | None) -> Path:
        rid = run_id or _active_run_id()
        if rid is None:
            raise HTTPException(status_code=404, detail="no runs found")
        # Defence in depth: run_id comes from the URL. Reject any value
        # that is not a bare basename so ``..`` / absolute paths /
        # NUL bytes can never escape ``cfg.run_root``.
        if (not rid or "/" in rid or "\\" in rid or rid in (".", "..")
                or "\x00" in rid or rid != Path(rid).name):
            raise HTTPException(status_code=400,
                                detail="invalid run_id")
        d = cfg.run_root / rid
        try:
            resolved = d.resolve(strict=False)
            root_resolved = cfg.run_root.resolve(strict=False)
        except OSError:
            raise HTTPException(status_code=404,
                                detail=f"run {rid!r} not found") from None
        if root_resolved not in resolved.parents and resolved != root_resolved:
            raise HTTPException(status_code=400,
                                detail="invalid run_id")
        if not d.exists() or not d.is_dir():
            raise HTTPException(status_code=404,
                                detail=f"run {rid!r} not found")
        return d

    # ---- routes -------------------------------------------------------
    def _health_payload() -> dict:
        return {
            "status": "ok",
            "version": "0.1.0",
            "data_root": str(cfg.data_root),
            "run_root": str(cfg.run_root),
            "active_run": _active_run_id(),
            "ts": pd.Timestamp.now(tz="UTC").isoformat(),
        }

    @app.get("/api/health")
    def health() -> dict:
        return _health_payload()

    @app.get("/health")
    def health_compat() -> dict:
        return _health_payload()

    @app.get("/api/runs")
    def list_runs() -> dict:
        runs = _list_runs()
        for r in runs:
            r.pop("path", None)
            r["mtime_iso"] = pd.Timestamp(r["mtime"], unit="s",
                                          tz="UTC").isoformat()
        return {"runs": runs, "active": _active_run_id()}

    @app.get("/api/runs/{run_id}/universe")
    def get_universe(run_id: str) -> Any:
        return _read_json(_resolve_run(run_id) / "universe.json", default=[])

    @app.get("/api/runs/{run_id}/snapshot")
    def get_snapshot(
        run_id: str,
        limit: int = Query(DEFAULT_LIMIT, ge=1, le=MAX_LIMIT),
    ) -> Any:
        data = _read_json(_resolve_run(run_id) / "snapshot.json", default=[])
        if isinstance(data, list):
            return data[:limit]
        return data

    @app.get("/api/runs/{run_id}/prompts")
    def get_prompts(
        run_id: str,
        limit: int = Query(DEFAULT_LIMIT, ge=1, le=MAX_LIMIT),
        call_type: str | None = None,
    ) -> dict:
        rows = _tail_jsonl(_resolve_run(run_id) / "prompts.jsonl", limit=limit)
        sanitised = [_sanitise_prompt(r) for r in rows]
        if call_type:
            sanitised = [r for r in sanitised if r.get("call_type") == call_type]
        return {"rows": sanitised, "count": len(sanitised)}

    @app.get("/api/runs/{run_id}/watchlist")
    def get_watchlist(run_id: str) -> Any:
        return _read_json(_resolve_run(run_id) / "watchlist.json", default={})

    @app.get("/api/runs/{run_id}/triggers")
    def get_triggers(
        run_id: str,
        limit: int = Query(DEFAULT_LIMIT, ge=1, le=MAX_LIMIT),
    ) -> dict:
        rows = _tail_jsonl(_resolve_run(run_id) / "triggers.jsonl", limit=limit)
        return {"rows": rows, "count": len(rows)}

    @app.get("/api/runs/{run_id}/intents")
    def get_intents(
        run_id: str,
        limit: int = Query(DEFAULT_LIMIT, ge=1, le=MAX_LIMIT),
    ) -> dict:
        rows = _tail_jsonl(_resolve_run(run_id) / "intents.jsonl", limit=limit)
        return {"rows": rows, "count": len(rows)}

    @app.get("/api/runs/{run_id}/fills")
    def get_fills(
        run_id: str,
        limit: int = Query(MAX_LIMIT, ge=1, le=MAX_LIMIT),
    ) -> dict:
        rows = _tail_jsonl(_resolve_run(run_id) / "fills.jsonl", limit=limit)
        return {"rows": rows, "count": len(rows)}

    @app.get("/api/runs/{run_id}/reviews")
    def get_reviews(
        run_id: str,
        limit: int = Query(DEFAULT_LIMIT, ge=1, le=MAX_LIMIT),
    ) -> dict:
        rows = _tail_jsonl(_resolve_run(run_id) / "reviews.jsonl", limit=limit)
        return {"rows": rows, "count": len(rows)}

    @app.get("/api/portfolio")
    def get_portfolio(run_id: str | None = None) -> Any:
        d = _resolve_run(run_id)
        return _read_json(d / "portfolio.json", default={})

    @app.get("/api/metrics")
    def get_metrics(
        days: int = Query(7, ge=1, le=30),
        run_id: str | None = None,
    ) -> dict:
        d = _resolve_run(run_id)
        return _build_metrics(d, days=days)

    @app.get("/api/equity_curve")
    def get_equity_curve(run_id: str | None = None) -> dict:
        d = _resolve_run(run_id)
        return _build_equity_curve(d)

    @app.get("/api/performance/by_symbol")
    def perf_by_symbol(run_id: str | None = None) -> dict:
        d = _resolve_run(run_id)
        return {"rows": _perf_by_symbol(d)}

    @app.get("/api/performance/by_day")
    def perf_by_day(
        run_id: str | None = None,
        days: int = Query(30, ge=1, le=365),
    ) -> dict:
        d = _resolve_run(run_id)
        return {"rows": _perf_by_day(d, days=days)}

    @app.get("/api/ai/usage")
    def ai_usage(
        run_id: str | None = None,
        days: int = Query(7, ge=1, le=90),
    ) -> dict:
        d = _resolve_run(run_id)
        return _ai_usage(d, days=days)

    @app.get("/api/positions")
    def positions(run_id: str | None = None) -> dict:
        d = _resolve_run(run_id)
        portfolio = _read_json(d / "portfolio.json", default={})
        return {"rows": portfolio.get("open_positions") or []}

    # ---- chart / per-symbol routes -----------------------------------
    @app.get("/api/symbols")
    def list_symbols(run_id: str | None = None) -> dict:
        d = _resolve_run(run_id)
        return {"rows": _collect_symbols(d, cache_root=cfg.cache_root)}

    @app.get("/api/candles")
    def get_candles(
        symbol: str,
        tf: str = "15",
        limit: int = Query(500, ge=10, le=2000),
    ) -> dict:
        try:
            return chartdata.candles_payload(
                cfg.cache_root, symbol, tf, limit=limit,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/symbol_events")
    def get_symbol_events(
        symbol: str,
        run_id: str | None = None,
    ) -> dict:
        d = _resolve_run(run_id)
        try:
            sym = chartdata._safe_symbol(symbol)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return _symbol_events(d, sym)

    @app.get("/api/ai/calls")
    def ai_calls(
        run_id: str | None = None,
        symbol: str | None = None,
        call_type: str | None = None,
        limit: int = Query(MAX_LIMIT, ge=1, le=MAX_LIMIT),
    ) -> dict:
        d = _resolve_run(run_id)
        return {"rows": _ai_calls_enriched(
            d, symbol=symbol, call_type=call_type, limit=limit,
        )}

    @app.get("/api/live")
    async def live_stream(
        request: Request,
        run_id: str | None = None,
        interval: float = Query(1.0, ge=0.25, le=10.0),
    ) -> StreamingResponse:
        d = _resolve_run(run_id)
        gen = _live_event_source(d, request=request,
                                 poll_interval_sec=interval)
        return StreamingResponse(gen, media_type="text/event-stream")

    # ---- static dashboard --------------------------------------------
    static_dir = Path(__file__).resolve().parent / "static"
    if static_dir.is_dir():
        app.mount("/static", StaticFiles(directory=static_dir), name="static")

        @app.get("/")
        def index() -> FileResponse:
            return FileResponse(static_dir / "index.html")

    return app


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _read_json(path: Path, *, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        log.warning("malformed JSON at %s", path)
        return default


def _tail_jsonl(path: Path, *, limit: int) -> list[dict]:
    """Return the last ``limit`` valid JSON records from a JSONL file.

    Large append-only audit files can grow to hundreds of MB. We read
    backward in 256 KB blocks and stop as soon as we have ``limit + 1``
    newline boundaries, so memory + I/O stays bounded regardless of
    file size. Malformed lines are silently skipped.
    """
    if not path.exists():
        return []
    try:
        size = path.stat().st_size
    except OSError:
        return []
    if size == 0:
        return []
    BLOCK = 256 * 1024
    needed = max(int(limit), 1) + 1
    chunks: list[bytes] = []
    newlines = 0
    try:
        with open(path, "rb") as f:
            pos = size
            while pos > 0 and newlines < needed:
                step = min(BLOCK, pos)
                pos -= step
                f.seek(pos)
                blk = f.read(step)
                chunks.append(blk)
                newlines += blk.count(b"\n")
    except OSError:
        return []
    raw = b"".join(reversed(chunks))
    text = raw.decode("utf-8", errors="replace")
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) > limit:
        lines = lines[-limit:]
    out: list[dict] = []
    for line in lines:
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _sanitise_prompt(row: dict) -> dict:
    """Strip large prompt-body fields from a prompts.jsonl row.

    Returns a shallow copy. ``request`` and ``response`` sub-dicts get
    their text fields trimmed; everything else is kept as-is.
    """
    out: dict[str, Any] = dict(row)
    req = out.get("request")
    if isinstance(req, dict):
        out["request"] = {k: v for k, v in req.items()
                          if k not in _PROMPT_SANITISE_FIELDS}
    resp = out.get("response")
    if isinstance(resp, dict):
        out["response"] = {k: v for k, v in resp.items()
                           if k not in _PROMPT_SANITISE_FIELDS}
    return out


def _build_metrics(run_dir: Path, *, days: int) -> dict:
    """Aggregate fills + budget into a small dashboard summary."""
    fills = _tail_jsonl(run_dir / "fills.jsonl", limit=MAX_LIMIT)
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days)
    n_entries = 0
    n_closes = 0
    n_winners = 0
    n_losers = 0
    realized = 0.0
    fees = 0.0
    closes_by_kind: dict[str, int] = {}
    for f in fills:
        ts_raw = f.get("ts")
        if not ts_raw:
            continue
        try:
            ts = pd.Timestamp(ts_raw)
        except (ValueError, TypeError):
            continue
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        if ts < cutoff:
            continue
        kind = f.get("kind", "")
        fees += float(f.get("fee_usd") or 0.0)
        if kind == "entry":
            n_entries += 1
            continue
        pnl = float(f.get("pnl_usd") or 0.0)
        realized += pnl
        if kind in {"stop", "tp2", "time", "manual", "funding"}:
            n_closes += 1
            closes_by_kind[kind] = closes_by_kind.get(kind, 0) + 1
            if pnl > 0:
                n_winners += 1
            elif pnl < 0:
                n_losers += 1

    win_rate = (n_winners / n_closes) if n_closes > 0 else 0.0

    budget = _read_json(run_dir / "budget.json", default={})
    portfolio = _read_json(run_dir / "portfolio.json", default={})

    return {
        "window_days": days,
        "fills_window": {
            "entries": n_entries,
            "closes": n_closes,
            "closes_by_kind": closes_by_kind,
            "winners": n_winners,
            "losers": n_losers,
            "win_rate": round(win_rate, 4),
            "realized_pnl_usd": round(realized, 4),
            "fees_paid_usd": round(fees, 4),
        },
        "budget": budget,
        "portfolio": {
            "as_of": portfolio.get("as_of"),
            "equity_usd": portfolio.get("equity_usd"),
            "cash_usd": portfolio.get("cash_usd"),
            "open_positions": len(portfolio.get("open_positions") or []),
            "loser_streak": portfolio.get("loser_streak"),
            "risk_multiplier": portfolio.get("risk_multiplier"),
        },
    }


def _build_equity_curve(run_dir: Path) -> dict:
    """Build a (ts, equity_usd, realized_usd) timeseries from fills.

    The exact starting equity is unknown without the config so we
    anchor at zero realized PnL and let the dashboard offset against
    ``portfolio.json::equity_usd`` if needed.
    """
    fills = _tail_jsonl(run_dir / "fills.jsonl", limit=MAX_LIMIT)
    cumulative = 0.0
    fees_cum = 0.0
    points: list[dict] = []
    for f in fills:
        ts = f.get("ts")
        kind = f.get("kind", "")
        fees_cum += float(f.get("fee_usd") or 0.0)
        if kind != "entry":
            cumulative += float(f.get("pnl_usd") or 0.0)
        points.append({
            "ts": ts,
            "kind": kind,
            "symbol": f.get("symbol"),
            "realized_usd": round(cumulative, 4),
            "fees_usd": round(fees_cum, 4),
            "pnl_usd": float(f.get("pnl_usd") or 0.0),
        })
    return {"points": points, "count": len(points)}


# ---------------------------------------------------------------------
# Performance / AI aggregations
# ---------------------------------------------------------------------
_CLOSE_KINDS = frozenset({"stop", "tp1", "tp2", "time", "manual", "funding"})


def _perf_by_symbol(run_dir: Path) -> list[dict]:
    """Per-symbol aggregate of realized PnL, win-rate, fees, trade counts."""
    fills = _tail_jsonl(run_dir / "fills.jsonl", limit=MAX_LIMIT)
    by: dict[str, dict[str, float]] = {}
    for f in fills:
        sym = f.get("symbol") or "?"
        kind = f.get("kind", "")
        row = by.setdefault(sym, {
            "symbol": sym, "entries": 0, "closes": 0,
            "winners": 0, "losers": 0,
            "realized_usd": 0.0, "fees_usd": 0.0, "volume_usd": 0.0,
        })
        row["fees_usd"] += float(f.get("fee_usd") or 0.0)
        price = float(f.get("price") or 0.0)
        qty = float(f.get("qty") or 0.0)
        row["volume_usd"] += abs(price * qty)
        if kind == "entry":
            row["entries"] += 1
        elif kind in {"stop", "tp2", "time", "manual", "funding"}:
            pnl = float(f.get("pnl_usd") or 0.0)
            row["realized_usd"] += pnl
            row["closes"] += 1
            if pnl > 0:
                row["winners"] += 1
            elif pnl < 0:
                row["losers"] += 1
        elif kind == "tp1":
            row["realized_usd"] += float(f.get("pnl_usd") or 0.0)
    out: list[dict] = []
    for row in by.values():
        wr = (row["winners"] / row["closes"]) if row["closes"] else 0.0
        row["win_rate"] = round(wr, 4)
        for k in ("realized_usd", "fees_usd", "volume_usd"):
            row[k] = round(row[k], 4)
        out.append(row)
    out.sort(key=lambda r: r["realized_usd"], reverse=True)
    return out


def _perf_by_day(run_dir: Path, *, days: int) -> list[dict]:
    """Daily realized PnL + fees + close count, last ``days`` UTC days."""
    fills = _tail_jsonl(run_dir / "fills.jsonl", limit=MAX_LIMIT)
    cutoff = pd.Timestamp.now(tz="UTC").normalize() - pd.Timedelta(days=days - 1)
    by: dict[str, dict[str, float]] = {}
    for f in fills:
        ts_raw = f.get("ts")
        if not ts_raw:
            continue
        try:
            ts = pd.Timestamp(ts_raw)
        except (ValueError, TypeError):
            continue
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        if ts < cutoff:
            continue
        day = ts.strftime("%Y-%m-%d")
        row = by.setdefault(day, {
            "day": day, "realized_usd": 0.0, "fees_usd": 0.0,
            "closes": 0, "entries": 0,
        })
        row["fees_usd"] += float(f.get("fee_usd") or 0.0)
        kind = f.get("kind", "")
        if kind == "entry":
            row["entries"] += 1
        elif kind in {"stop", "tp1", "tp2", "time", "manual", "funding"}:
            row["realized_usd"] += float(f.get("pnl_usd") or 0.0)
            if kind != "tp1":
                row["closes"] += 1

    # Fill in zero-rows for missing days so the chart x-axis is dense.
    today = pd.Timestamp.now(tz="UTC").normalize()
    out: list[dict] = []
    for i in range(days):
        day = (cutoff + pd.Timedelta(days=i)).strftime("%Y-%m-%d")
        if day in by:
            row = by[day]
        else:
            row = {"day": day, "realized_usd": 0.0, "fees_usd": 0.0,
                   "closes": 0, "entries": 0}
        for k in ("realized_usd", "fees_usd"):
            row[k] = round(row[k], 4)
        out.append(row)
        if (cutoff + pd.Timedelta(days=i)) > today:
            break
    return out


def _ai_usage(run_dir: Path, *, days: int) -> dict:
    """Aggregate prompts.jsonl into per-call-type totals + a daily series."""
    rows = _tail_jsonl(run_dir / "prompts.jsonl", limit=MAX_LIMIT)
    cutoff = pd.Timestamp.now(tz="UTC").normalize() - pd.Timedelta(days=days - 1)
    totals: dict[str, dict[str, float]] = {}
    daily: dict[str, dict[str, float]] = {}
    grand_cost = 0.0
    grand_calls = 0
    grand_tokens_in = 0
    grand_tokens_out = 0

    for r in rows:
        ct = r.get("call_type") or r.get("type") or "unknown"
        usage = r.get("usage") or (r.get("response") or {}).get("usage") or {}
        cost = float(r.get("cost_usd") or r.get("cost") or 0.0)
        tin = int(usage.get("prompt_tokens") or 0)
        tout = int(usage.get("completion_tokens") or 0)

        t = totals.setdefault(ct, {
            "call_type": ct, "calls": 0, "cost_usd": 0.0,
            "tokens_in": 0, "tokens_out": 0,
        })
        t["calls"] += 1
        t["cost_usd"] += cost
        t["tokens_in"] += tin
        t["tokens_out"] += tout

        grand_calls += 1
        grand_cost += cost
        grand_tokens_in += tin
        grand_tokens_out += tout

        ts_raw = r.get("ts")
        if not ts_raw:
            continue
        try:
            ts = pd.Timestamp(ts_raw)
        except (ValueError, TypeError):
            continue
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        if ts < cutoff:
            continue
        day = ts.strftime("%Y-%m-%d")
        d = daily.setdefault(day, {"day": day, "calls": 0,
                                   "cost_usd": 0.0, "by_type": {}})
        d["calls"] += 1
        d["cost_usd"] += cost
        d["by_type"][ct] = d["by_type"].get(ct, 0) + 1

    # Round totals.
    by_type = []
    for t in totals.values():
        t["cost_usd"] = round(t["cost_usd"], 6)
        by_type.append(t)
    by_type.sort(key=lambda r: r["cost_usd"], reverse=True)

    today = pd.Timestamp.now(tz="UTC").normalize()
    series: list[dict] = []
    for i in range(days):
        day_ts = cutoff + pd.Timedelta(days=i)
        day = day_ts.strftime("%Y-%m-%d")
        if day in daily:
            row = daily[day]
            row["cost_usd"] = round(row["cost_usd"], 6)
        else:
            row = {"day": day, "calls": 0, "cost_usd": 0.0, "by_type": {}}
        series.append(row)
        if day_ts > today:
            break

    budget = _read_json(run_dir / "budget.json", default={})
    return {
        "totals": {
            "calls": grand_calls,
            "cost_usd": round(grand_cost, 6),
            "tokens_in": grand_tokens_in,
            "tokens_out": grand_tokens_out,
        },
        "by_type": by_type,
        "series": series,
        "budget": budget,
    }


# ---------------------------------------------------------------------
# Per-symbol aggregation (chart view)
# ---------------------------------------------------------------------
def _ts_to_unix(value: Any) -> int | None:
    if value is None:
        return None
    try:
        ts = pd.Timestamp(value)
    except (ValueError, TypeError):
        return None
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return int(ts.value // 1_000_000_000)


def _collect_symbols(run_dir: Path, *, cache_root: Path) -> list[dict]:
    """Union of symbols seen across triggers/fills/intents/prompts/portfolio.

    Returns one row per symbol with activity counters and a flag telling
    the UI whether a 15m candle parquet exists in the cache (so the
    frontend can render the chart icon).
    """
    counters: dict[str, dict[str, int]] = {}

    def bump(sym: str | None, key: str) -> None:
        if not sym:
            return
        row = counters.setdefault(sym, {
            "symbol": sym, "triggers": 0, "fills": 0, "intents": 0,
            "ai_calls": 0, "reviews": 0, "open_position": 0,
        })
        row[key] += 1

    for r in _tail_jsonl(run_dir / "triggers.jsonl", limit=MAX_LIMIT):
        bump(r.get("symbol"), "triggers")
    for r in _tail_jsonl(run_dir / "fills.jsonl", limit=MAX_LIMIT):
        bump(r.get("symbol"), "fills")
    for r in _tail_jsonl(run_dir / "intents.jsonl", limit=MAX_LIMIT):
        bump(r.get("symbol"), "intents")
    for r in _tail_jsonl(run_dir / "prompts.jsonl", limit=MAX_LIMIT):
        bump(r.get("symbol"), "ai_calls")
        # Watchlist responses don't have a top-level symbol; pull from
        # the decision selections so universe symbols still surface.
        decision = r.get("decision") or {}
        for sel in decision.get("selections") or []:
            bump(sel.get("symbol"), "ai_calls")
    for r in _tail_jsonl(run_dir / "reviews.jsonl", limit=MAX_LIMIT):
        bump(r.get("symbol"), "reviews")

    portfolio = _read_json(run_dir / "portfolio.json", default={})
    for p in portfolio.get("open_positions") or []:
        sym = p.get("symbol")
        bump(sym, "open_position")

    out: list[dict] = []
    for row in counters.values():
        sym = row["symbol"]
        try:
            cp = chartdata.candle_path(cache_root, sym, "15")
            row["has_candles"] = cp.exists()
        except ValueError:
            row["has_candles"] = False
        out.append(row)
    out.sort(
        key=lambda r: (
            -r["open_position"], -r["fills"], -r["triggers"], r["symbol"],
        )
    )
    return out


def _decision_summary(call_type: str, decision: Any) -> dict:
    """Flatten the most useful fields from a decision payload.

    Mirrors the schemas from ``ai/schemas.py`` (WatchlistResponse,
    DeepSignal, ReviewResponse). Missing fields default to ``None``.
    """
    out: dict[str, Any] = {
        "action": None, "confidence": None, "reasoning": [],
        "key_confluences": [], "entry": None, "stop_loss": None,
        "take_profit_1": None, "take_profit_2": None,
        "time_horizon_bars": None, "expected_move_pct": None,
        "rationale": None, "invalidation": None,
        "market_regime": None, "regime_evidence": [],
        "selections": [], "discarded_pumps": [],
        "hook_reason": None, "new_stop_loss": None, "new_tp2": None,
    }
    if not isinstance(decision, dict):
        return out
    for k in (
        "action", "confidence", "reasoning", "key_confluences",
        "entry", "stop_loss", "take_profit_1", "take_profit_2",
        "time_horizon_bars", "expected_move_pct",
        "rationale", "invalidation",
        "market_regime", "regime_evidence",
        "discarded_pumps", "hook_reason",
        "new_stop_loss", "new_tp2",
    ):
        if k in decision:
            out[k] = decision[k]
    sels = decision.get("selections") or []
    if isinstance(sels, list):
        out["selections"] = [
            {
                "symbol": s.get("symbol"),
                "side": s.get("side"),
                "confidence": s.get("confidence"),
                "expected_move_pct": s.get("expected_move_pct"),
                "thesis": s.get("thesis"),
            }
            for s in sels if isinstance(s, dict)
        ]
    return out


def _ai_calls_enriched(
    run_dir: Path, *, symbol: str | None, call_type: str | None,
    limit: int,
) -> list[dict]:
    """Sanitised prompts with ``decision_summary`` flattened on top."""
    rows = _tail_jsonl(run_dir / "prompts.jsonl", limit=limit)
    out: list[dict] = []
    sym_u = symbol.upper() if symbol else None
    for r in rows:
        if call_type and r.get("call_type") != call_type:
            continue
        ct = r.get("call_type") or "unknown"
        decision = r.get("decision") or {}
        summary = _decision_summary(ct, decision)
        if sym_u:
            row_sym = (r.get("symbol") or "").upper()
            in_selections = any(
                (sel.get("symbol") or "").upper() == sym_u
                for sel in summary["selections"]
            )
            if row_sym != sym_u and not in_selections:
                continue
        usage = r.get("usage") or (r.get("response") or {}).get("usage") or {}
        cost = r.get("cost_usd") or r.get("cost")
        if cost is None:
            cost = (r.get("response") or {}).get("cost_usd")
        out.append({
            "ts": r.get("ts"),
            "call_id": r.get("call_id"),
            "call_type": ct,
            "model": r.get("model"),
            "symbol": r.get("symbol"),
            "cost_usd": cost,
            "tokens_in": usage.get("prompt_tokens"),
            "tokens_out": usage.get("completion_tokens"),
            "latency_ms": (r.get("response") or {}).get("latency_ms"),
            "decision_summary": summary,
            "decision": decision,
        })
    return out


def _symbol_events(run_dir: Path, symbol: str) -> dict:
    """Filter triggers / fills / intents / reviews / prompts for one symbol.

    All timestamps are converted to int unix-seconds for the chart
    front-end so it can drop markers directly onto the candle series.
    """
    sym_u = symbol.upper()

    def _is_sym(r: dict) -> bool:
        return (r.get("symbol") or "").upper() == sym_u

    triggers_raw = _tail_jsonl(run_dir / "triggers.jsonl", limit=MAX_LIMIT)
    triggers = []
    for r in triggers_raw:
        if not _is_sym(r):
            continue
        triggers.append({
            "time": _ts_to_unix(r.get("bar_ts") or r.get("ts")),
            "fired": bool(r.get("fired")),
            "flag": r.get("flag"),
            "decision": r.get("decision"),
            "close": r.get("close"),
            "atr_pct": r.get("atr_pct"),
            "move_pct": r.get("move_pct"),
            "reason": r.get("reason"),
        })

    fills_raw = _tail_jsonl(run_dir / "fills.jsonl", limit=MAX_LIMIT)
    fills = []
    for r in fills_raw:
        if not _is_sym(r):
            continue
        fills.append({
            "time": _ts_to_unix(r.get("bar_ts") or r.get("ts")),
            "kind": r.get("kind"),
            "side": r.get("side"),
            "price": r.get("price"),
            "qty": r.get("qty"),
            "pnl_usd": r.get("pnl_usd"),
            "fee_usd": r.get("fee_usd"),
            "reason": r.get("reason"),
        })

    intents_raw = _tail_jsonl(run_dir / "intents.jsonl", limit=MAX_LIMIT)
    intents = []
    for r in intents_raw:
        if not _is_sym(r):
            continue
        intents.append({
            "time": _ts_to_unix(r.get("ts")),
            "intent_id": r.get("intent_id"),
            "status": r.get("status"),
            "side": r.get("side"),
            "entry": r.get("entry"),
            "stop_loss": r.get("stop_loss"),
            "take_profit_1": r.get("take_profit_1"),
            "take_profit_2": r.get("take_profit_2"),
            "activation_kind": r.get("activation_kind"),
            "confidence": r.get("confidence"),
            "tag": r.get("tag"),
        })

    reviews_raw = _tail_jsonl(run_dir / "reviews.jsonl", limit=MAX_LIMIT)
    reviews = []
    for r in reviews_raw:
        if not _is_sym(r):
            continue
        reviews.append({
            "time": _ts_to_unix(r.get("ts")),
            "action": r.get("action") or (r.get("decision") or {}).get("action"),
            "confidence": r.get("confidence"),
            "rationale": r.get("rationale") or (r.get("decision") or {}).get("rationale"),
            "hook_reason": r.get("hook_reason"),
            "new_stop_loss": r.get("new_stop_loss"),
            "new_tp2": r.get("new_tp2"),
        })

    ai_calls = _ai_calls_enriched(run_dir, symbol=sym_u,
                                  call_type=None, limit=MAX_LIMIT)
    ai_markers = []
    for c in ai_calls:
        ai_markers.append({
            "time": _ts_to_unix(c.get("ts")),
            "call_type": c.get("call_type"),
            "action": c.get("decision_summary", {}).get("action"),
            "confidence": c.get("decision_summary", {}).get("confidence"),
            "cost_usd": c.get("cost_usd"),
            "model": c.get("model"),
            "call_id": c.get("call_id"),
        })

    return {
        "symbol": sym_u,
        "triggers": triggers,
        "fills": fills,
        "intents": intents,
        "reviews": reviews,
        "ai_calls": ai_markers,
    }


# ---------------------------------------------------------------------
# SSE live stream
# ---------------------------------------------------------------------
_LIVE_FILES = (
    ("triggers", "triggers.jsonl"),
    ("intents", "intents.jsonl"),
    ("fills", "fills.jsonl"),
    ("reviews", "reviews.jsonl"),
)


async def _live_event_source(
    run_dir: Path,
    *,
    request: Request,
    poll_interval_sec: float,
) -> AsyncIterator[str]:
    """Tail every JSONL audit file in ``run_dir`` and yield SSE events.

    Disconnects cleanly when the client closes the request. Each
    delivered chunk is one SSE record::

        event: <kind>
        data: <one JSONL line>

    """
    offsets: dict[str, int] = {}
    for kind, name in _LIVE_FILES:
        p = run_dir / name
        offsets[kind] = p.stat().st_size if p.exists() else 0

    # Send a hello so the client immediately knows the connection is
    # live (some proxies hold buffers otherwise).
    yield _sse("hello", json.dumps({
        "run_id": run_dir.name,
        "watching": [k for k, _ in _LIVE_FILES],
        "ts": pd.Timestamp.now(tz="UTC").isoformat(),
    }))

    while True:
        if await request.is_disconnected():
            return
        any_yield = False
        for kind, name in _LIVE_FILES:
            p = run_dir / name
            if not p.exists():
                continue
            try:
                size = p.stat().st_size
            except OSError:
                continue
            start = offsets.get(kind, 0)
            if size <= start:
                continue
            try:
                with p.open("r", encoding="utf-8") as f:
                    f.seek(start)
                    chunk = f.read()
                    offsets[kind] = f.tell()
            except OSError:
                continue
            for line in chunk.splitlines():
                line = line.strip()
                if not line:
                    continue
                # Validate the line is JSON before sending so a half-
                # written record does not break the stream.
                try:
                    json.loads(line)
                except json.JSONDecodeError:
                    continue
                yield _sse(kind, line)
                any_yield = True
        if not any_yield:
            # Keep-alive ping: SSE comments are ignored by the client
            # but keep proxies from closing the connection.
            yield ": keep-alive\n\n"
        await asyncio.sleep(poll_interval_sec)


def _sse(event: str, data: str) -> str:
    return f"event: {event}\ndata: {data}\n\n"


__all__ = ["create_app"]
