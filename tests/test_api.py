"""Phase 13 - read-only HTTP API + dashboard tests.

All tests use a fixture run tree under ``tmp_data_root`` and the
FastAPI ``TestClient`` so the suite stays fully offline.
"""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.server import _live_event_source, _sanitise_prompt, create_app
from core.config import load_config


# ---------------------------------------------------------------------
# Fixture run tree
# ---------------------------------------------------------------------
def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


@pytest.fixture
def run_tree(tmp_data_root: Path) -> tuple[Path, str]:
    """Build a minimal but realistic run directory.

    Returns (run_dir_path, run_id).
    """
    run_id = "20260101T000000Z-test"
    rdir = tmp_data_root / "runs" / run_id
    rdir.mkdir(parents=True)

    _write_json(rdir / "universe.json", [
        {"symbol": "BTCUSDT", "score": 0.9},
        {"symbol": "ETHUSDT", "score": 0.7},
    ])
    _write_json(rdir / "snapshot.json", [
        {"symbol": "BTCUSDT", "rsi_14": 55.1, "atr_14": 120.0},
        {"symbol": "ETHUSDT", "rsi_14": 48.2, "atr_14": 9.0},
    ])
    _write_json(rdir / "watchlist.json", {
        "as_of": "2026-01-01T00:05:00Z",
        "symbols": ["BTCUSDT", "SOLUSDT"],
        "rationale": "trend continuation",
    })
    _write_json(rdir / "portfolio.json", {
        "as_of": "2026-01-01T00:30:00Z",
        "equity_usd": 10500.0,
        "cash_usd": 9000.0,
        "realized_pnl_usd": 500.0,
        "fees_paid_usd": 12.5,
        "open_positions": [
            {"position_id": "p1", "symbol": "BTCUSDT", "side": "long",
             "qty": 0.1, "entry_price": 50000.0, "stop_loss": 49000.0,
             "tp1": 50500.0, "tp2": 51000.0,
             "opened_at": "2026-01-01T00:10:00Z"},
        ],
        "closed_positions_24h": 1,
        "watchlist": ["BTCUSDT"],
        "risk_multiplier": 1.0,
        "loser_streak": 0,
    })
    _write_json(rdir / "budget.json", {
        "date": "2026-01-01",
        "spent_usd": 0.42,
        "cap_usd": 5.0,
        "calls": 7,
    })

    now = pd.Timestamp.now(tz="UTC").floor("s")
    _write_jsonl(rdir / "fills.jsonl", [
        {"ts": (now - pd.Timedelta(minutes=20)).isoformat(),
         "position_id": "p0", "symbol": "BTCUSDT", "side": "long",
         "kind": "entry", "price": 49500.0, "qty": 0.1,
         "fee_usd": 2.0, "pnl_usd": 0.0},
        {"ts": (now - pd.Timedelta(minutes=10)).isoformat(),
         "position_id": "p0", "symbol": "BTCUSDT", "side": "long",
         "kind": "tp2", "price": 50500.0, "qty": 0.1,
         "fee_usd": 2.5, "pnl_usd": 100.0},
        {"ts": (now - pd.Timedelta(minutes=5)).isoformat(),
         "position_id": "p1", "symbol": "BTCUSDT", "side": "long",
         "kind": "entry", "price": 50000.0, "qty": 0.1,
         "fee_usd": 2.0, "pnl_usd": 0.0},
    ])
    _write_jsonl(rdir / "triggers.jsonl", [
        {"ts": now.isoformat(), "symbol": "BTCUSDT",
         "kind": "breakout", "price": 50000.0},
    ])
    _write_jsonl(rdir / "intents.jsonl", [
        {"ts": now.isoformat(), "intent_id": "i1",
         "symbol": "BTCUSDT", "side": "long",
         "size_usd": 5000.0, "stop_loss": 49000.0},
    ])
    _write_json(rdir / "prompts" / "call-watchlist.req.json", {
        "system": "Choose up to five pairs with the cleanest momentum and liquidity.",
        "user": "[{\"symbol\":\"BTCUSDT\"},{\"symbol\":\"SOLUSDT\"}]",
        "headers": {"authorization": "***"},
    })
    _write_json(rdir / "prompts" / "call-watchlist.resp.json", {
        "raw_text": "{\"market_regime\":\"risk-on\",\"selections\":[{\"symbol\":\"BTCUSDT\",\"side\":\"long\"}]}",
        "parsed": {"market_regime": "risk-on",
                   "selections": [{"symbol": "BTCUSDT", "side": "long"}]},
    })
    _write_json(rdir / "prompts" / "call-deep.req.json", {
        "system": "Decide whether to trade BTCUSDT on the current trigger.",
        "user": "{\"symbol\":\"BTCUSDT\",\"trigger\":\"breakout\"}",
    })
    _write_json(rdir / "prompts" / "call-deep.resp.json", {
        "raw_text": "{\"action\":\"long\",\"confidence\":0.75}",
        "parsed": {"action": "long", "confidence": 0.75},
    })
    _write_jsonl(rdir / "prompts.jsonl", [
        {"ts": now.isoformat(), "call_id": "call-watchlist", "call_type": "watchlist",
         "model": "x-ai/grok-4.20",
         "request": {"system": "SECRET-SYSTEM", "user": "SECRET-USER",
                     "schema": "watchlist"},
         "response": {"raw_text": "SECRET-RAW",
                      "decision": {"symbols": ["BTCUSDT"]}},
         "decision": {"market_regime": "risk-on",
                      "selections": [{"symbol": "BTCUSDT", "side": "long",
                                      "confidence": 0.8, "thesis": "trend"}]},
         "usage": {"prompt_tokens": 200, "completion_tokens": 30},
         "cost_usd": 0.001},
        {"ts": now.isoformat(), "call_id": "call-trigger", "call_type": "trigger",
         "model": "x-ai/grok-4.20",
         "request": {"system": "S", "user": "U"},
         "response": {"decision": {"action": "enter"}},
         "usage": {"prompt_tokens": 100, "completion_tokens": 20},
         "cost_usd": 0.0005},
        {"ts": now.isoformat(), "call_id": "call-deep", "call_type": "deep",
         "symbol": "BTCUSDT", "model": "x-ai/grok-4.20",
         "request": {"system": "S", "user": "U"},
         "response": {"latency_ms": 100},
         "decision": {"action": "long", "confidence": 0.75,
                      "reasoning": ["bullish divergence", "EMA stack"],
                      "key_confluences": ["RSI oversold"],
                      "entry": 50000.0, "stop_loss": 49000.0,
                      "take_profit_1": 51000.0,
                      "time_horizon_bars": 24},
         "usage": {"prompt_tokens": 500, "completion_tokens": 80},
         "cost_usd": 0.003},
    ])
    _write_jsonl(rdir / "reviews.jsonl", [
        {"ts": now.isoformat(), "position_id": "p1",
         "symbol": "BTCUSDT",
         "trigger_reason": "tp1", "action": "tighten_stop",
         "new_stop_loss": 49500.0},
    ])
    return rdir, run_id


@pytest.fixture
def client(run_tree: tuple[Path, str]) -> TestClient:
    cfg = load_config()
    app = create_app(cfg)
    return TestClient(app)


# ---------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------
def test_health(client: TestClient, run_tree) -> None:
    _, rid = run_tree
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["active_run"] == rid


def test_health_compat_alias(client: TestClient, run_tree) -> None:
    _, rid = run_tree
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["active_run"] == rid


def test_list_runs(client: TestClient, run_tree) -> None:
    _, rid = run_tree
    r = client.get("/api/runs")
    assert r.status_code == 200
    body = r.json()
    assert body["active"] == rid
    assert any(run["run_id"] == rid for run in body["runs"])


def test_universe(client: TestClient, run_tree) -> None:
    _, rid = run_tree
    r = client.get(f"/api/runs/{rid}/universe")
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body, list)
    assert body[0]["symbol"] == "BTCUSDT"


def test_snapshot(client: TestClient, run_tree) -> None:
    _, rid = run_tree
    r = client.get(f"/api/runs/{rid}/snapshot?limit=10")
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body, list)
    assert {"symbol", "rsi_14"}.issubset(body[0].keys())


def test_watchlist(client: TestClient, run_tree) -> None:
    _, rid = run_tree
    r = client.get(f"/api/runs/{rid}/watchlist")
    assert r.status_code == 200
    assert r.json()["symbols"] == ["BTCUSDT", "SOLUSDT"]


def test_prompts_sanitised(client: TestClient, run_tree) -> None:
    _, rid = run_tree
    r = client.get(f"/api/runs/{rid}/prompts")
    assert r.status_code == 200
    rows = r.json()["rows"]
    assert len(rows) == 3
    for row in rows:
        # raw text fields must be stripped
        assert "system" not in (row.get("request") or {})
        assert "user" not in (row.get("request") or {})
        assert "raw_text" not in (row.get("response") or {})
        # but metadata is kept
        assert row["model"] == "x-ai/grok-4.20"
        assert row["usage"]["prompt_tokens"] >= 1


def test_prompts_filter_by_type(client: TestClient, run_tree) -> None:
    _, rid = run_tree
    r = client.get(f"/api/runs/{rid}/prompts?call_type=watchlist")
    assert r.status_code == 200
    rows = r.json()["rows"]
    assert len(rows) == 1
    assert rows[0]["call_type"] == "watchlist"


def test_triggers_intents_fills_reviews(client: TestClient, run_tree) -> None:
    _, rid = run_tree
    for path, expected in [
        (f"/api/runs/{rid}/triggers", 1),
        (f"/api/runs/{rid}/intents", 1),
        (f"/api/runs/{rid}/fills", 3),
        (f"/api/runs/{rid}/reviews", 1),
    ]:
        r = client.get(path)
        assert r.status_code == 200, path
        body = r.json()
        assert body["count"] == expected, path
        assert isinstance(body["rows"], list)


def test_portfolio(client: TestClient) -> None:
    r = client.get("/api/portfolio")
    assert r.status_code == 200
    body = r.json()
    assert body["equity_usd"] == 10500.0
    assert len(body["open_positions"]) == 1


def test_positions_route(client: TestClient) -> None:
    r = client.get("/api/positions")
    assert r.status_code == 200
    body = r.json()
    assert len(body["rows"]) == 1
    assert body["rows"][0]["symbol"] == "BTCUSDT"


def test_perf_by_symbol(client: TestClient) -> None:
    r = client.get("/api/performance/by_symbol")
    assert r.status_code == 200
    rows = r.json()["rows"]
    assert len(rows) == 1
    btc = rows[0]
    assert btc["symbol"] == "BTCUSDT"
    assert btc["entries"] == 2
    assert btc["closes"] == 1
    assert btc["winners"] == 1
    assert btc["realized_usd"] == 100.0
    assert btc["win_rate"] == 1.0


def test_perf_by_day(client: TestClient) -> None:
    r = client.get("/api/performance/by_day?days=7")
    assert r.status_code == 200
    rows = r.json()["rows"]
    # Window includes today; non-empty.
    assert len(rows) >= 1
    total = sum(row["realized_usd"] for row in rows)
    assert total == 100.0


def test_ai_usage(client: TestClient) -> None:
    r = client.get("/api/ai/usage?days=7")
    assert r.status_code == 200
    body = r.json()
    assert body["totals"]["calls"] == 3
    # 0.001 + 0.0005 + 0.003 = 0.0045
    assert abs(body["totals"]["cost_usd"] - 0.0045) < 1e-9
    assert body["totals"]["tokens_in"] == 800
    assert body["totals"]["tokens_out"] == 130
    types = {row["call_type"] for row in body["by_type"]}
    assert types == {"watchlist", "trigger", "deep"}
    assert body["budget"]["spent_usd"] == 0.42
    # Series covers the 7-day window.
    assert len(body["series"]) >= 1


def test_metrics(client: TestClient) -> None:
    r = client.get("/api/metrics")
    assert r.status_code == 200
    body = r.json()
    win = body["fills_window"]
    assert win["entries"] == 2
    assert win["closes"] == 1
    assert win["winners"] == 1
    assert win["losers"] == 0
    assert win["win_rate"] == 1.0
    assert win["realized_pnl_usd"] == 100.0
    assert body["budget"]["spent_usd"] == 0.42
    assert body["portfolio"]["open_positions"] == 1


def test_equity_curve(client: TestClient) -> None:
    r = client.get("/api/equity_curve")
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 3
    last = body["points"][-1]
    assert last["realized_usd"] == 100.0


def test_unknown_run_returns_404(client: TestClient) -> None:
    r = client.get("/api/runs/does-not-exist/universe")
    assert r.status_code == 404


def test_static_index(client: TestClient) -> None:
    r = client.get("/")
    assert r.status_code == 200
    assert "<html" in r.text.lower()


def test_static_assets(client: TestClient) -> None:
    for path in ("/static/app.js", "/static/style.css"):
        r = client.get(path)
        assert r.status_code == 200, path


def test_sanitise_prompt_drops_secret_fields() -> None:
    row = {
        "ts": "2026-01-01T00:00:00Z",
        "request": {"system": "SECRET", "user": "PII", "schema": "watchlist"},
        "response": {"raw_text": "RAW", "decision": "ok"},
        "model": "m",
    }
    out = _sanitise_prompt(row)
    assert "system" not in out["request"]
    assert "user" not in out["request"]
    assert out["request"]["schema"] == "watchlist"
    assert "raw_text" not in out["response"]
    assert out["response"]["decision"] == "ok"


def test_live_sse_emits_new_lines(run_tree) -> None:
    """Drive the async generator directly so we don't need the
    TestClient streaming machinery (which blocks until completion)."""
    rdir, _ = run_tree
    triggers_path = rdir / "triggers.jsonl"

    class _FakeRequest:
        def __init__(self) -> None:
            self._disconnected = False

        async def is_disconnected(self) -> bool:
            return self._disconnected

    async def _drive() -> list[str]:
        req = _FakeRequest()
        gen = _live_event_source(rdir, request=req, poll_interval_sec=0.05)

        # First frame is the synthetic hello.
        hello = await gen.__anext__()
        assert hello.startswith("event: hello")

        # Append a new line AFTER the generator captured offsets.
        with triggers_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": "2026-01-01T00:31:00Z",
                                 "symbol": "ETHUSDT",
                                 "kind": "breakout",
                                 "price": 2500.0}) + "\n")

        events: list[str] = []
        deadline = asyncio.get_event_loop().time() + 3.0
        while asyncio.get_event_loop().time() < deadline:
            chunk = await gen.__anext__()
            events.append(chunk)
            if "event: triggers" in chunk:
                break
        # Tell the generator the client disconnected and let it exit.
        req._disconnected = True
        try:
            await asyncio.wait_for(gen.aclose(), timeout=1.0)
        except asyncio.TimeoutError:
            pass
        return events

    events = asyncio.run(_drive())
    assert any("event: triggers" in e for e in events)
    assert any("ETHUSDT" in e for e in events)


# ---------------------------------------------------------------------
# Phase 13 v3: chart / per-symbol routes
# ---------------------------------------------------------------------
def _write_candles(cache_root: Path, symbol: str, interval: str,
                   bars: int = 120) -> Path:
    """Write a synthetic candle parquet matching the production layout."""
    p = (cache_root / "klines" / symbol / interval
         / f"{symbol}_klines_{interval}.parquet")
    p.parent.mkdir(parents=True, exist_ok=True)
    end = pd.Timestamp.now(tz="UTC").floor("min")
    idx = pd.date_range(end - pd.Timedelta(minutes=15 * (bars - 1)),
                        end, periods=bars, tz="UTC")
    base = 50000.0
    closes = base + (pd.Series(range(bars)) * 5.0)
    df = pd.DataFrame({
        "timestamp": idx,
        "open": closes - 2,
        "high": closes + 10,
        "low": closes - 10,
        "close": closes,
        "volume": [10.0] * bars,
        "turnover": [500000.0] * bars,
    })
    # Inject one volume climax bar so `compute_flag_markers` returns a hit.
    df.loc[df.index[-5], "volume"] = 200.0
    df.to_parquet(p, index=False)
    return p


def test_symbols_endpoint(client: TestClient) -> None:
    r = client.get("/api/symbols")
    assert r.status_code == 200
    rows = r.json()["rows"]
    assert any(row["symbol"] == "BTCUSDT" for row in rows)
    assert any(row["symbol"] == "SOLUSDT" for row in rows)
    btc = next(row for row in rows if row["symbol"] == "BTCUSDT")
    assert btc["fills"] == 3
    assert btc["triggers"] == 1
    assert btc["ai_calls"] >= 2  # deep + watchlist selection
    assert btc["open_position"] == 1
    sol = next(row for row in rows if row["symbol"] == "SOLUSDT")
    assert sol["watchlist"] == 1


def test_candles_missing(client: TestClient) -> None:
    r = client.get("/api/candles?symbol=ZZZUSDT&tf=15&limit=200")
    assert r.status_code == 200
    body = r.json()
    assert body["symbol"] == "ZZZUSDT"
    assert body["rows"] == []
    assert "volume_climax" in body["flags_available"]


def test_candles_with_data(client: TestClient, tmp_data_root: Path) -> None:
    cfg = load_config()
    _write_candles(cfg.cache_root, "BTCUSDT", "15", bars=120)
    r = client.get("/api/candles?symbol=BTCUSDT&tf=15&limit=100")
    assert r.status_code == 200
    body = r.json()
    assert body["symbol"] == "BTCUSDT"
    assert len(body["rows"]) == 100
    row = body["rows"][0]
    assert {"time", "open", "high", "low", "close", "volume"} <= row.keys()
    assert isinstance(row["time"], int)
    inds = body["indicators"]
    assert "ema_21" in inds and len(inds["ema_21"]) == 100
    # The injected high-volume bar should produce a climax marker.
    assert len(body["flags"]["volume_climax"]) >= 1


def test_candles_invalid_symbol(client: TestClient) -> None:
    r = client.get("/api/candles?symbol=bad-sym&tf=15")
    assert r.status_code == 400


def test_candles_invalid_tf(client: TestClient) -> None:
    r = client.get("/api/candles?symbol=BTCUSDT&tf=foo")
    assert r.status_code == 400


def test_symbol_events(client: TestClient) -> None:
    r = client.get("/api/symbol_events?symbol=BTCUSDT")
    assert r.status_code == 200
    body = r.json()
    assert body["symbol"] == "BTCUSDT"
    assert len(body["fills"]) == 3
    assert len(body["triggers"]) == 1
    assert len(body["intents"]) == 1
    assert len(body["reviews"]) == 1
    assert any(c["call_type"] == "deep" for c in body["ai_calls"])
    # Times are unix seconds (or None).
    for f in body["fills"]:
        assert f["time"] is None or isinstance(f["time"], int)


def test_symbol_events_unknown(client: TestClient) -> None:
    r = client.get("/api/symbol_events?symbol=ZZZUSDT")
    assert r.status_code == 200
    body = r.json()
    assert body["symbol"] == "ZZZUSDT"
    assert body["fills"] == [] and body["triggers"] == []


def test_ai_calls_enriched(client: TestClient) -> None:
    r = client.get("/api/ai/calls")
    assert r.status_code == 200
    rows = r.json()["rows"]
    assert len(rows) == 3
    # Decision summary should be flattened on every row.
    deep = next(r for r in rows if r["call_type"] == "deep")
    ds = deep["decision_summary"]
    assert ds["action"] == "long"
    assert ds["confidence"] == 0.75
    assert "bullish divergence" in ds["reasoning"]
    assert ds["entry"] == 50000.0
    assert deep["tokens_in"] == 500
    # Watchlist selections preserved.
    wl = next(r for r in rows if r["call_type"] == "watchlist")
    assert wl["decision_summary"]["market_regime"] == "risk-on"
    assert wl["decision_summary"]["selections"][0]["symbol"] == "BTCUSDT"


def test_ai_call_detail_returns_full_sidecars(client: TestClient, run_tree) -> None:
    _, rid = run_tree
    r = client.get(f"/api/ai/calls/call-watchlist?run_id={rid}")
    assert r.status_code == 200
    body = r.json()
    assert body["call_id"] == "call-watchlist"
    assert body["request"]["system"].startswith("Choose up to five pairs")
    assert body["request"]["headers"]["authorization"] == "***"
    assert "BTCUSDT" in body["response"]["raw_text"]


def test_ai_call_detail_unknown_returns_404(client: TestClient, run_tree) -> None:
    _, rid = run_tree
    r = client.get(f"/api/ai/calls/does-not-exist?run_id={rid}")
    assert r.status_code == 404


def test_ai_calls_filter_symbol(client: TestClient) -> None:
    r = client.get("/api/ai/calls?symbol=BTCUSDT")
    assert r.status_code == 200
    rows = r.json()["rows"]
    # deep (symbol=BTC) + watchlist (selections include BTC) = 2
    assert len(rows) == 2
    types = {row["call_type"] for row in rows}
    assert types == {"deep", "watchlist"}


def test_ai_calls_filter_call_type(client: TestClient) -> None:
    r = client.get("/api/ai/calls?call_type=deep")
    assert r.status_code == 200
    rows = r.json()["rows"]
    assert len(rows) == 1
    assert rows[0]["call_type"] == "deep"
