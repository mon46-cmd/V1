"""Tests for ai/prompts.py.

Render-only checks. No network. Verify glossary inclusion, rubric
present, JSON-only instruction, char-cap sanity.
"""
from __future__ import annotations

import json

import pandas as pd

from ai.prompts import (
    FIELD_GLOSSARY,
    PROMPT_VERSION,
    render_deep_prompt,
    render_watchlist_prompt,
)


def _fake_rows(n: int = 6) -> list[dict]:
    rows = []
    for i in range(n):
        rows.append({
            "symbol": f"X{i}USDT",
            "close": 100.0 + i,
            "ret_24h": 0.05 - i * 0.01,
            "atr_14_pct": 1.5 + i * 0.1,
            "rsi_14": 55.0 + i,
            "funding_rate": 0.0001 * (i - 3),
            "trend_score_mtf": 0.33,
            "atr_pct_rank_96": 0.6,
        })
    return rows


class TestWatchlistPrompt:
    def test_renders_two_strings(self):
        sys_, user = render_watchlist_prompt(rows=_fake_rows(), as_of="2026-04-25T00:00:00Z")
        assert isinstance(sys_, str) and isinstance(user, str)
        assert sys_ and user

    def test_system_has_rubric_and_json_only(self):
        sys_, _ = render_watchlist_prompt(rows=_fake_rows(), as_of="t")
        assert "Scoring rubric" in sys_
        # JSON output schema must be inlined.
        assert "Output JSON shape" in sys_
        assert "no extra keys" in sys_

    def test_user_is_valid_json_with_glossary(self):
        _, user = render_watchlist_prompt(rows=_fake_rows(), as_of="t")
        payload = json.loads(user)
        assert payload["prompt_version"] == PROMPT_VERSION
        assert "rows" in payload and len(payload["rows"]) == 6
        assert "field_glossary" in payload
        # rsi_14 is in the rows -> must be in the glossary subset.
        assert "rsi_14" in payload["field_glossary"]
        assert payload["field_glossary"]["rsi_14"] == FIELD_GLOSSARY["rsi_14"]

    def test_dataframe_input_works(self):
        df = pd.DataFrame(_fake_rows())
        sys_, user = render_watchlist_prompt(rows=df, as_of="t")
        assert isinstance(sys_, str)
        payload = json.loads(user)
        assert len(payload["rows"]) == 6

    def test_size_under_token_budget(self):
        # 30 rows * ~25 fields each must remain well under our ~80k char cap.
        rows = _fake_rows(30)
        _, user = render_watchlist_prompt(rows=rows, as_of="t")
        assert len(user) < 80_000


class TestDeepPrompt:
    def test_renders_with_minimal_payload(self):
        sys_, user = render_deep_prompt(
            symbol="BTCUSDT",
            as_of="t",
            trigger={"flag": "flag_volume_climax", "mark_price": 100.0},
            bars_15m=[{"c": 100.0, "rsi": 55.0}],
        )
        assert "Risk policy" in sys_
        assert "Output JSON shape" in sys_
        payload = json.loads(user)
        assert payload["symbol"] == "BTCUSDT"
        assert payload["prompt_version"] == PROMPT_VERSION
        # bars_15m list-of-dicts gets normalised to columnar form.
        assert isinstance(payload["history"]["bars_15m"], dict)
        assert payload["history"]["bars_15m"]["c"] == [100.0]
