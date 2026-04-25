"""Phase 0 smoke tests: imports, config, paths, time, ids, logging."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
import pytest


def test_package_imports():
    import core
    import downloader  # noqa: F401
    import features  # noqa: F401
    import ai  # noqa: F401
    import portfolio  # noqa: F401
    import api  # noqa: F401
    import loops  # noqa: F401

    assert hasattr(core, "load_config")
    assert hasattr(core, "run_id")
    assert hasattr(core, "ulid")
    assert hasattr(core, "now_utc")


def test_load_config_builds_tree(tmp_data_root: Path):
    from core import load_config

    cfg = load_config()
    assert cfg.data_root == tmp_data_root
    for p in (cfg.cache_root, cfg.feature_root, cfg.run_root, cfg.log_root):
        assert p.exists() and p.is_dir()
    # Immutable.
    with pytest.raises(Exception):
        cfg.universe_size = 999  # type: ignore[misc]


def test_defaults_match_plan(tmp_data_root: Path):
    from core import load_config

    cfg = load_config()
    assert cfg.universe_size == 30
    assert cfg.watch_interval == "15"
    assert cfg.higher_tfs == ("60", "240")
    assert cfg.prompt_cooldown_candles == 3
    assert cfg.per_trade_risk_pct == 0.01
    assert cfg.max_concurrent_positions == 3
    assert "flag_volume_climax" in cfg.trigger_flags


def test_now_utc_is_tz_aware():
    from core import now_utc

    t = now_utc()
    assert isinstance(t, pd.Timestamp)
    assert t.tz is not None
    assert str(t.tz) == "UTC"


def test_to_utc_coerces_ms_epoch():
    from core.time import to_utc

    t = to_utc(1_700_000_000_000)  # ms epoch
    assert t.tz is not None
    assert t.year == 2023


def test_run_id_format():
    from core import run_id

    rid = run_id(pd.Timestamp("2026-04-24T16:34:12Z"))
    assert rid == "20260424T163412Z"


def test_ulid_unique_and_sortable():
    from core import ulid

    ids = sorted({ulid() for _ in range(256)})
    assert len(ids) == 256
    assert all(len(i) == 26 for i in ids)


def test_logging_writes_json(tmp_data_root: Path):
    from core import load_config
    from core.logging import configure

    cfg = load_config()
    configure(cfg, process="test")

    log = logging.getLogger("v5.test")
    log.info("hello %s", "world", extra={"symbol": "BTCUSDT"})
    log.warning("Authorization: Bearer sk-should-be-masked")

    # Flush handlers.
    for h in logging.getLogger().handlers:
        h.flush()

    log_file = cfg.log_root / "test.log"
    assert log_file.exists()
    lines = [ln for ln in log_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert lines, "log file is empty"

    last = json.loads(lines[-1])
    assert last["level"] == "WARNING"
    # Secret masking.
    assert "sk-should-be-masked" not in last["msg"]
    assert "***" in last["msg"]

    first_info = next(json.loads(ln) for ln in lines if json.loads(ln)["level"] == "INFO")
    assert first_info["msg"] == "hello world"
    assert first_info.get("symbol") == "BTCUSDT"


def test_rust_extension_optional():
    """des_core is optional in phase 0; if present, it must expose version()."""
    try:
        import des_core  # type: ignore[import-not-found]
    except ImportError:
        pytest.skip("des_core not built yet (run setup.ps1 -Dev)")
    assert des_core.version()
