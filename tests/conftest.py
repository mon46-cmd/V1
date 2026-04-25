"""Shared test fixtures and path setup."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Ensure `src/` is importable without installing the package.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture
def tmp_data_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point DATA_ROOT at an isolated tmpdir for the duration of a test."""
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    # Make sure a stale live flag doesn't leak into tests.
    for k in ("BYBIT_OFFLINE", "AI_DRY_RUN", "OPENROUTER_LIVE"):
        monkeypatch.delenv(k, raising=False)
    return tmp_path
