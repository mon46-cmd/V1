"""Structured JSON logging.

One call per process at startup:

    from core import load_config
    from core.logging import configure
    cfg = load_config()
    configure(cfg, process="scanner")

Emits JSON lines to `data/logs/<process>.log` (daily rotation, 14-day
retention) and a console stream at WARNING for operator feedback.
Secrets are never logged; upstream code is responsible for not passing
secret strings to the logger, but `SafeFilter` below strips obvious
Authorization headers just in case.
"""
from __future__ import annotations

import json
import logging
import logging.handlers
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import Config


_SECRET_PATTERNS = [
    (re.compile(r"(Authorization['\"]?\s*[:=]\s*['\"]?)[^'\"\s,}]+", re.IGNORECASE), r"\1***"),
    (re.compile(r"(Bearer\s+)[A-Za-z0-9\-_.=]+"), r"\1***"),
    (re.compile(r"sk-[A-Za-z0-9\-_]{10,}"), "sk-***"),
]


def _mask(msg: str) -> str:
    for pat, repl in _SECRET_PATTERNS:
        msg = pat.sub(repl, msg)
    return msg


class SafeFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            record.msg = _mask(str(record.msg))
        except Exception:
            pass
        return True


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S") + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        # Attach any extra fields put on the record.
        for k, v in record.__dict__.items():
            if k in {"args", "msg", "levelname", "levelno", "pathname", "filename",
                     "module", "exc_info", "exc_text", "stack_info", "lineno",
                     "funcName", "created", "msecs", "relativeCreated", "thread",
                     "threadName", "processName", "process", "name", "message"}:
                continue
            try:
                json.dumps(v)
                payload[k] = v
            except (TypeError, ValueError):
                payload[k] = repr(v)
        return json.dumps(payload, ensure_ascii=False)


def configure(cfg: "Config", process: str) -> None:
    """Install the JSON file handler and a WARNING console handler.

    Safe to call more than once; previous handlers on the root logger
    are removed first.
    """
    root = logging.getLogger()
    root.setLevel(cfg.log_level)

    for h in list(root.handlers):
        root.removeHandler(h)

    cfg.log_root.mkdir(parents=True, exist_ok=True)
    log_file = cfg.log_root / f"{process}.log"

    file_h = logging.handlers.TimedRotatingFileHandler(
        log_file, when="midnight", backupCount=14, utc=True, encoding="utf-8",
    )
    file_h.setFormatter(JsonFormatter())
    file_h.addFilter(SafeFilter())
    file_h.setLevel(cfg.log_level)
    root.addHandler(file_h)

    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    console.addFilter(SafeFilter())
    console.setLevel("WARNING")
    root.addHandler(console)
