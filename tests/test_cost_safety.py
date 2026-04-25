"""Cost-safety: AI client kill switch + scanner singleton lock."""
from __future__ import annotations

from pathlib import Path

import pytest

from ai.client import AIClient
from core.config import Config, load_config
from core.lock import LockBusy, file_lock


def test_ai_kill_switch_forces_offline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-fake-real-looking-key")
    monkeypatch.delenv("BYBIT_OFFLINE", raising=False)
    monkeypatch.delenv("AI_DRY_RUN", raising=False)
    monkeypatch.setenv("AI_KILL_SWITCH", "1")
    cfg = load_config()
    client = AIClient(cfg=cfg)
    assert client._offline is True
    assert client.mock is not None


def test_ai_kill_switch_off_uses_live_when_key_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-fake-real-looking-key")
    monkeypatch.delenv("BYBIT_OFFLINE", raising=False)
    monkeypatch.delenv("AI_DRY_RUN", raising=False)
    monkeypatch.delenv("AI_KILL_SWITCH", raising=False)
    cfg = load_config()
    client = AIClient(cfg=cfg)
    assert client._offline is False


def test_file_lock_rejects_concurrent_holder(tmp_path: Path) -> None:
    lock = tmp_path / "exec.lock"
    with file_lock(lock):
        with pytest.raises(LockBusy):
            with file_lock(lock):
                pass


def test_file_lock_releases_on_exit(tmp_path: Path) -> None:
    lock = tmp_path / "exec.lock"
    with file_lock(lock):
        pass
    # Should be reacquirable.
    with file_lock(lock):
        pass
