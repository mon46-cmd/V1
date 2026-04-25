"""Audit-log writer for AI calls.

For every call we append one JSON line to ``prompts.jsonl`` and
write two sidecars (``<call_id>.req.json`` and ``.resp.json``) under
``prompts/`` so the line file stays light.

Secrets (Authorization header values, API keys) are stripped before
anything is written.
"""
from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

import orjson

_SECRET_KEYS = {"authorization", "x-api-key", "api-key", "openrouter_api_key"}
_SECRET_VALUE_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_\-]{16,}"),
    re.compile(r"Bearer\s+[A-Za-z0-9_\-\.]+", re.IGNORECASE),
]


def _mask(s: str) -> str:
    if not isinstance(s, str):
        return s
    out = s
    for p in _SECRET_VALUE_PATTERNS:
        out = p.sub("***", out)
    return out


def _redact(obj: Any) -> Any:
    """Recursively strip secret-looking fields/values from a dict."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if isinstance(k, str) and k.lower() in _SECRET_KEYS:
                out[k] = "***"
            else:
                out[k] = _redact(v)
        return out
    if isinstance(obj, list):
        return [_redact(x) for x in obj]
    if isinstance(obj, str):
        return _mask(obj)
    return obj


def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


@dataclass
class AuditWriter:
    """One writer per run. Thread-safe append to ``prompts.jsonl``."""

    run_dir: Path
    _lock: Lock = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self._lock = Lock()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "prompts").mkdir(parents=True, exist_ok=True)

    @property
    def line_path(self) -> Path:
        return self.run_dir / "prompts.jsonl"

    @property
    def warnings_path(self) -> Path:
        return self.run_dir / "ai_warnings.jsonl"

    def write_call(
        self,
        *,
        call_id: str,
        call_type: str,
        model: str,
        prompt_version: str,
        symbol: str | None,
        request: dict,
        response: dict,
        decision: dict,
        usage: dict | None = None,
        cost_usd: float | None = None,
        cumulative_usd: float | None = None,
        budget_remaining_usd: float | None = None,
    ) -> None:
        """Append one record + write sidecar files."""
        ts = datetime.now(tz=timezone.utc).isoformat()
        # Sidecars carry the full bodies (redacted).
        req_clean = _redact(request)
        resp_clean = _redact(response)
        req_path = self.run_dir / "prompts" / f"{call_id}.req.json"
        resp_path = self.run_dir / "prompts" / f"{call_id}.resp.json"
        _atomic_write_json(req_path, req_clean)
        _atomic_write_json(resp_path, resp_clean)

        raw_text = response.get("raw_text", "") if isinstance(response, dict) else ""
        record = {
            "ts": ts,
            "call_id": call_id,
            "call_type": call_type,
            "model": model,
            "prompt_version": prompt_version,
            "symbol": symbol,
            "request": {
                "system_chars": len(request.get("system", "")) if isinstance(request, dict) else 0,
                "user_chars": len(request.get("user", "")) if isinstance(request, dict) else 0,
                "temperature": request.get("temperature") if isinstance(request, dict) else None,
                "response_format": request.get("response_format") if isinstance(request, dict) else None,
            },
            "response": {
                "latency_ms": response.get("latency_ms"),
                "http_status": response.get("http_status"),
                "raw_text_chars": len(raw_text),
                "json_valid": response.get("json_valid", False),
                "schema_valid": response.get("schema_valid", False),
                "repair_retries": response.get("repair_retries", 0),
                "usage": usage,
                "cost_usd": cost_usd,
                "cumulative_usd": cumulative_usd,
                "budget_remaining_usd": budget_remaining_usd,
                "content_sha256": sha256_hex(raw_text),
            },
            "decision": decision,
        }
        line = orjson.dumps(record).decode("utf-8") + "\n"
        with self._lock:
            with open(self.line_path, "a", encoding="utf-8") as f:
                f.write(line)

    def write_warning(self, *, call_id: str, symbol: str | None,
                      kind: str, message: str, context: dict | None = None) -> None:
        rec = {
            "ts": datetime.now(tz=timezone.utc).isoformat(),
            "call_id": call_id,
            "symbol": symbol,
            "kind": kind,
            "message": message,
            "context": context or {},
        }
        line = orjson.dumps(rec).decode("utf-8") + "\n"
        with self._lock:
            with open(self.warnings_path, "a", encoding="utf-8") as f:
                f.write(line)


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2 | orjson.OPT_SERIALIZE_NUMPY))
    os.replace(tmp, path)
