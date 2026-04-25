"""Filesystem helpers for the data tree.

All writes elsewhere in the codebase assume these directories exist.
`ensure_dirs(cfg)` is idempotent and safe to call repeatedly.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import Config


def ensure_dirs(cfg: "Config") -> None:
    """Create the full data directory tree if missing."""
    for p in (
        cfg.data_root,
        cfg.cache_root,
        cfg.feature_root,
        cfg.run_root,
        cfg.log_root,
    ):
        Path(p).mkdir(parents=True, exist_ok=True)


def run_dir(cfg: "Config", run_id: str) -> Path:
    """Return (creating if needed) the directory for a given run id."""
    d = cfg.run_root / run_id
    d.mkdir(parents=True, exist_ok=True)
    return d
