"""Async OpenRouter client with budget, audit, and offline fallback.

Public coroutines:

- ``chat_watchlist(snap_df, *, as_of)`` -> ``WatchlistResponse``
- ``chat_deep(symbol, payload)``        -> ``DeepSignal``

If ``cfg.bybit_offline`` or ``cfg.ai_dry_run`` is true, or
``cfg.openrouter_api_key`` is empty, all calls go through the
``MockRouter``. Otherwise the live OpenRouter chat-completions endpoint
is used with ``response_format={"type": "json_object"}``.
"""
from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiohttp
import orjson
import pandas as pd
from pydantic import ValidationError

from core.config import Config
from .audit import AuditWriter
from .budget import BudgetTracker
from .mock import MockRouter
from .prices import cost_usd
from .prompts import (
    PROMPT_VERSION,
    render_deep_prompt,
    render_review_prompt,
    render_watchlist_prompt,
)
from .schemas import DeepSignal, ReviewResponse, WatchlistResponse


@dataclass
class _Call:
    call_id: str
    call_type: str
    model: str
    symbol: str | None
    system: str
    user: str
    temperature: float = 0.2
    max_tokens: int = 1500


class AIClient:
    """Routes chat calls to OpenRouter or the MockRouter."""

    def __init__(
        self,
        cfg: Config,
        *,
        budget: BudgetTracker | None = None,
        audit: AuditWriter | None = None,
        mock: MockRouter | None = None,
    ) -> None:
        self.cfg = cfg
        self.budget = budget
        self.audit = audit
        self._offline = (
            cfg.bybit_offline
            or cfg.ai_dry_run
            or not cfg.openrouter_api_key
        )
        if self._offline:
            self.mock = mock or MockRouter(
                fixtures_root=cfg.repo_root / "tests" / "fixtures" / "ai_responses",
            )
        else:
            self.mock = None
        # One concurrent live request per call type.
        self._sem = {
            "watchlist": asyncio.Semaphore(1),
            "deep": asyncio.Semaphore(2),
            "review": asyncio.Semaphore(2),
        }

    # ---- public coroutines -------------------------------------------
    async def chat_watchlist(
        self,
        snap_df: pd.DataFrame,
        *,
        as_of: str,
    ) -> WatchlistResponse:
        rows = snap_df.to_dict(orient="records") if isinstance(snap_df, pd.DataFrame) else list(snap_df)
        system, user = render_watchlist_prompt(rows=rows, as_of=as_of)
        call = _Call(
            call_id=_new_call_id(),
            call_type="watchlist",
            model=self.cfg.model_watchlist,
            symbol=None,
            system=system,
            user=user,
            temperature=0.3,
            max_tokens=2200,
        )
        return await self._run(call, schema=WatchlistResponse, mock_fn=lambda: self.mock.watchlist(  # type: ignore[union-attr]
            symbols=[r.get("symbol", "") for r in rows], as_of=as_of,
        ))

    async def chat_deep(self, symbol: str, payload: dict) -> DeepSignal:
        system, user = render_deep_prompt(symbol=symbol, **payload)
        call = _Call(
            call_id=_new_call_id(),
            call_type="deep",
            model=self.cfg.model_deep,
            symbol=symbol,
            system=system,
            user=user,
            temperature=0.2,
            max_tokens=1200,
        )
        mark = payload.get("trigger", {}).get("mark_price") if payload else None
        signal = await self._run(call, schema=DeepSignal, mock_fn=lambda: self.mock.deep(  # type: ignore[union-attr]
            symbol=symbol, mark_price=mark,
        ))
        # Defensive post-processing: enforce consistency.
        warnings = signal.check_consistency(mark_price=mark)
        if warnings and self.audit is not None:
            self.audit.write_warning(
                call_id=call.call_id, symbol=symbol,
                kind="consistency", message="; ".join(warnings),
                context={"action": signal.action},
            )
        if len(warnings) >= 2 and signal.action != "flat":
            signal = signal.model_copy(update={"action": "flat"})
        return signal

    async def chat_review(
        self,
        symbol: str,
        payload: dict,
    ) -> ReviewResponse:
        """Position review (Prompt C). Fired by exec-loop hooks
        (drawdown, regime-flip, funding-approach, TP1)."""
        system, user = render_review_prompt(symbol=symbol, **payload)
        call = _Call(
            call_id=_new_call_id(),
            call_type="review",
            model=self.cfg.model_review,
            symbol=symbol,
            system=system,
            user=user,
            temperature=0.2,
            max_tokens=900,
        )
        pos = payload.get("position", {}) or {}
        return await self._run(call, schema=ReviewResponse, mock_fn=lambda: self.mock.review(  # type: ignore[union-attr]
            symbol=symbol,
            trigger_reason=payload.get("trigger_reason", "drawdown"),
            mark_price=pos.get("mark_price"),
            stop_loss=pos.get("stop_loss"),
            side=pos.get("side", "long"),
        ))

    # ---- internal -----------------------------------------------------
    async def _run(self, call: _Call, *, schema, mock_fn):
        # Budget pre-check (rough estimate based on system+user chars).
        est_cost = self._estimate_cost(call)
        if self.budget is not None and not self.budget.can_afford(est_cost):
            if self.audit is not None:
                self.audit.write_warning(
                    call_id=call.call_id, symbol=call.symbol,
                    kind="budget_exhausted",
                    message=f"need ~${est_cost:.4f}, remaining ${self.budget.remaining_usd:.4f}",
                )
            return self._synthetic_flat(call, schema)

        async with self._sem[call.call_type]:
            response = await self._dispatch(call, mock_fn=mock_fn)

        parsed_obj, schema_ok, schema_err = self._validate(response.get("parsed"), schema, call)
        response["schema_valid"] = schema_ok

        # One repair retry if schema invalid AND we are live.
        if not schema_ok and not self._offline:
            repair_response = await self._repair(call, response, schema_err)
            if repair_response is not None:
                response = repair_response
                parsed_obj, schema_ok, schema_err = self._validate(response.get("parsed"), schema, call)
                response["schema_valid"] = schema_ok
                response["repair_retries"] = response.get("repair_retries", 0) + 1

        if not schema_ok:
            if self.audit is not None:
                self.audit.write_warning(
                    call_id=call.call_id, symbol=call.symbol,
                    kind="schema_invalid",
                    message=str(schema_err),
                )
            parsed_obj = self._synthetic_flat(call, schema)

        # Charge budget on the actual usage we got back (live path).
        usage = response.get("usage") or {}
        cost = cost_usd(call.model, int(usage.get("prompt_tokens", 0)),
                        int(usage.get("completion_tokens", 0)))
        cumulative = None
        remaining = None
        if self.budget is not None and not self._offline:
            cumulative = self.budget.charge(cost)
            remaining = self.budget.remaining_usd

        if self.audit is not None:
            decision = self._decision_summary(parsed_obj)
            self.audit.write_call(
                call_id=call.call_id,
                call_type=call.call_type,
                model=call.model,
                prompt_version=PROMPT_VERSION,
                symbol=call.symbol,
                request={
                    "system": call.system,
                    "user": call.user,
                    "temperature": call.temperature,
                    "response_format": {"type": "json_object"},
                },
                response=response,
                decision=decision,
                usage=usage,
                cost_usd=cost,
                cumulative_usd=cumulative,
                budget_remaining_usd=remaining,
            )
        return parsed_obj

    async def _dispatch(self, call: _Call, *, mock_fn) -> dict:
        if self._offline:
            return mock_fn()
        return await self._post(call)

    async def _post(self, call: _Call, *, override_user: str | None = None) -> dict:
        url = self.cfg.openrouter_base_url.rstrip("/") + "/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.cfg.openrouter_api_key}",
            "Content-Type": "application/json",
            "X-Title": self.cfg.openrouter_title,
        }
        if self.cfg.openrouter_referer:
            headers["HTTP-Referer"] = self.cfg.openrouter_referer
        body = {
            "model": call.model,
            "temperature": call.temperature,
            "max_tokens": call.max_tokens,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": call.system},
                {"role": "user", "content": override_user or call.user},
            ],
        }

        timeout = aiohttp.ClientTimeout(total=self.cfg.ai_timeout_sec)
        t0 = time.perf_counter()
        async with aiohttp.ClientSession(timeout=timeout) as sess:
            async with sess.post(url, headers=headers, data=orjson.dumps(body)) as resp:
                status = resp.status
                text = await resp.text()
        latency_ms = int((time.perf_counter() - t0) * 1000)

        # Parse OpenRouter envelope.
        raw_text = ""
        usage: dict[str, Any] = {}
        try:
            env = json.loads(text)
            choices = env.get("choices", [])
            if choices:
                raw_text = choices[0].get("message", {}).get("content", "") or ""
            usage = env.get("usage", {}) or {}
        except Exception:  # noqa: BLE001
            raw_text = ""

        parsed = None
        json_valid = False
        try:
            parsed = json.loads(raw_text) if raw_text else None
            json_valid = parsed is not None
        except Exception:  # noqa: BLE001
            json_valid = False

        return {
            "raw_text": raw_text,
            "parsed": parsed,
            "http_status": status,
            "latency_ms": latency_ms,
            "json_valid": json_valid,
            "schema_valid": False,
            "repair_retries": 0,
            "usage": usage,
            "model": call.model,
        }

    async def _repair(self, call: _Call, prev: dict, err: str | None) -> dict | None:
        """One repair attempt: ask the model to fix its prior JSON."""
        prev_text = prev.get("raw_text", "")
        repair_user = (
            "Your previous response did NOT validate against the required JSON schema.\n"
            f"Validation error: {err}\n\n"
            "Return ONLY a corrected JSON object that satisfies the schema. "
            "Do not add commentary.\n\n"
            "Previous response:\n" + prev_text
        )
        try:
            return await self._post(call, override_user=repair_user)
        except Exception:  # noqa: BLE001
            return None

    def _estimate_cost(self, call: _Call) -> float:
        # Char/4 -> token approximation.
        prompt_tokens = (len(call.system) + len(call.user)) // 4
        completion_tokens = call.max_tokens
        return cost_usd(call.model, prompt_tokens, completion_tokens)

    def _validate(self, parsed: Any, schema, call: _Call):
        if parsed is None:
            return None, False, "parsed JSON is None"
        try:
            obj = schema.model_validate(parsed)
            return obj, True, None
        except ValidationError as exc:
            return None, False, exc.errors()[:3]

    def _synthetic_flat(self, call: _Call, schema):
        if schema is DeepSignal:
            return DeepSignal(
                prompt_version=PROMPT_VERSION,
                symbol=call.symbol or "",
                action="flat",
                confidence=0.0,
                rationale="synthetic-flat: response invalid or budget exhausted",
                invalidation="n/a",
            )
        if schema is WatchlistResponse:
            return WatchlistResponse(
                prompt_version=PROMPT_VERSION,
                as_of="",
                market_regime="unknown",
                reasoning=["synthetic-flat watchlist (response invalid or budget exhausted)"] * 3,
                selections=[],
                notes="synthetic-flat",
            )
        if schema is ReviewResponse:
            return ReviewResponse(
                prompt_version=PROMPT_VERSION,
                symbol=call.symbol or "",
                action="hold",
                confidence=0.0,
                rationale="synthetic-hold: response invalid or budget exhausted",
            )
        raise TypeError(f"unknown schema {schema!r}")

    def _decision_summary(self, obj: Any) -> dict:
        if isinstance(obj, WatchlistResponse):
            return {
                "type": "watchlist",
                "regime": obj.market_regime,
                "n_selections": len(obj.selections),
                "symbols": [s.symbol for s in obj.selections],
            }
        if isinstance(obj, DeepSignal):
            return {
                "type": "deep",
                "action": obj.action,
                "symbol": obj.symbol,
                "entry": obj.entry,
                "stop_loss": obj.stop_loss,
                "tp1": obj.take_profit_1,
                "confidence": obj.confidence,
            }
        if isinstance(obj, ReviewResponse):
            return {
                "type": "review",
                "action": obj.action,
                "symbol": obj.symbol,
                "new_stop_loss": obj.new_stop_loss,
                "confidence": obj.confidence,
            }
        return {"type": "unknown"}


def _new_call_id() -> str:
    return uuid.uuid4().hex[:16]
