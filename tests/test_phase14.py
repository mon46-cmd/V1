"""Phase 14 - operator hardening tests.

Covers:

- ``scripts/health_check`` probes (snapshot age, portfolio age, budget,
  disk, services-skipped path, JSON output, exit code mapping).
- API path-traversal hardening of ``run_id``.
- ``_tail_jsonl`` reverse-block reader on a large file.
- Linux-style ``DATA_ROOT`` (absolute POSIX path) is accepted by
  ``Config`` round-trip.
"""
from __future__ import annotations

import importlib
import json
import os
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def _import_health():
    # Re-import so ``load_config`` picks up the patched env each test.
    if "health_check" in sys.modules:
        del sys.modules["health_check"]
    return importlib.import_module("health_check")


# ---------------------------------------------------------------------
# Health check unit tests
# ---------------------------------------------------------------------
def _make_args(**over):
    import argparse
    ns = argparse.Namespace(
        services=[],
        api_url="",
        api_timeout=2.0,
        max_snapshot_min=45.0,
        max_portfolio_min=90.0,
        max_budget_frac=0.95,
        min_disk_gb=0.001,  # tmpfs may be tiny in CI
        log_window_min=60.0,
        max_log_errors=1,
        format="text",
        allow_root=False,
    )
    for k, v in over.items():
        setattr(ns, k, v)
    return ns


def test_health_snapshot_pass(tmp_data_root: Path):
    from core.config import load_config
    cfg = load_config()
    rdir = cfg.run_root / "20260425T000000Z"
    rdir.mkdir(parents=True)
    (rdir / "snapshot.parquet").write_bytes(b"\x00")
    (rdir / "portfolio.json").write_text("{}")
    h = _import_health()
    res = h.run_probes(_make_args(), cfg)
    by_name = {r.name: r for r in res}
    assert by_name["snapshot_age"].status == h.STATUS_PASS
    assert by_name["portfolio_age"].status == h.STATUS_PASS
    assert by_name["api"].status == h.STATUS_PASS  # skipped
    assert by_name["budget"].status == h.STATUS_PASS  # missing budget.json
    assert h.overall_status(res) in (h.STATUS_PASS, h.STATUS_WARN)


def test_health_snapshot_fail_when_stale(tmp_data_root: Path):
    from core.config import load_config
    cfg = load_config()
    rdir = cfg.run_root / "20260425T000000Z"
    rdir.mkdir(parents=True)
    snap = rdir / "snapshot.parquet"
    snap.write_bytes(b"\x00")
    # Backdate by 90 minutes.
    old = time.time() - 90 * 60
    os.utime(snap, (old, old))
    h = _import_health()
    res = h.run_probes(_make_args(max_snapshot_min=45.0), cfg)
    snap_probe = next(r for r in res if r.name == "snapshot_age")
    assert snap_probe.status == h.STATUS_FAIL
    assert h.status_to_exit(h.overall_status(res)) == 2


def test_health_no_snapshot_fails(tmp_data_root: Path):
    from core.config import load_config
    cfg = load_config()
    h = _import_health()
    res = h.run_probes(_make_args(), cfg)
    snap_probe = next(r for r in res if r.name == "snapshot_age")
    assert snap_probe.status == h.STATUS_FAIL


def test_health_budget_warn_at_threshold(tmp_data_root: Path,
                                         monkeypatch: pytest.MonkeyPatch):
    from core.config import load_config
    cfg = load_config()
    rdir = cfg.run_root / "20260425T010000Z"
    rdir.mkdir(parents=True)
    (rdir / "snapshot.parquet").write_bytes(b"\x00")
    # 80% of cap (3 USD default) -> WARN window starts at 0.8 * 0.95.
    (rdir / "budget.json").write_text(json.dumps({"spent_usd": 2.5}))
    h = _import_health()
    res = h.run_probes(_make_args(), cfg)
    bp = next(r for r in res if r.name == "budget")
    assert bp.status in (h.STATUS_WARN, h.STATUS_FAIL)


def test_health_budget_fail_above_cap(tmp_data_root: Path):
    from core.config import load_config
    cfg = load_config()
    rdir = cfg.run_root / "20260425T020000Z"
    rdir.mkdir(parents=True)
    (rdir / "snapshot.parquet").write_bytes(b"\x00")
    (rdir / "budget.json").write_text(json.dumps({"spent_usd": 100.0}))
    h = _import_health()
    res = h.run_probes(_make_args(), cfg)
    bp = next(r for r in res if r.name == "budget")
    assert bp.status == h.STATUS_FAIL


def test_health_json_output_parses(tmp_data_root: Path):
    from core.config import load_config
    cfg = load_config()
    rdir = cfg.run_root / "20260425T030000Z"
    rdir.mkdir(parents=True)
    (rdir / "snapshot.parquet").write_bytes(b"\x00")
    h = _import_health()
    res = h.run_probes(_make_args(format="json"), cfg)
    payload = json.loads(h.render_json(res, h.overall_status(res)))
    assert payload["overall"] in (h.STATUS_PASS, h.STATUS_WARN, h.STATUS_FAIL)
    assert isinstance(payload["probes"], list)
    assert all("name" in p and "status" in p for p in payload["probes"])


def test_health_status_to_exit_mapping():
    h = _import_health()
    assert h.status_to_exit(h.STATUS_PASS) == 0
    assert h.status_to_exit(h.STATUS_WARN) == 1
    assert h.status_to_exit(h.STATUS_FAIL) == 2


def test_health_main_exits_nonzero_when_no_snapshot(tmp_data_root: Path,
                                                    capsys):
    h = _import_health()
    code = h.main(["--format", "json"])
    assert code in (1, 2)
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload["overall"] in (h.STATUS_WARN, h.STATUS_FAIL)


# ---------------------------------------------------------------------
# API hardening
# ---------------------------------------------------------------------
def test_api_rejects_path_traversal_run_id(tmp_data_root: Path):
    from fastapi.testclient import TestClient
    from api.server import create_app
    from core.config import load_config
    cfg = load_config()
    # Create a real run so the active-run path works.
    rdir = cfg.run_root / "20260425T040000Z"
    rdir.mkdir(parents=True)
    (rdir / "universe.json").write_text("[]")

    app = create_app(cfg)
    client = TestClient(app)

    # Valid baseline.
    r = client.get(f"/api/runs/{rdir.name}/universe")
    assert r.status_code == 200

    # Traversal attempts must be 400, not 200/500.
    for evil in ("..", "../etc", "..\\windows",
                 "20260425T040000Z/../../etc", ".",
                 "with\\back"):
        r = client.get(f"/api/runs/{evil}/universe")
        assert r.status_code in (400, 404), (
            f"{evil!r} -> {r.status_code} {r.text}"
        )


def test_api_run_id_validation_via_query(tmp_data_root: Path):
    from fastapi.testclient import TestClient
    from api.server import create_app
    from core.config import load_config
    cfg = load_config()
    rdir = cfg.run_root / "20260425T050000Z"
    rdir.mkdir(parents=True)
    (rdir / "portfolio.json").write_text("{}")
    app = create_app(cfg)
    client = TestClient(app)

    r = client.get("/api/portfolio", params={"run_id": ".."})
    assert r.status_code == 400
    r = client.get("/api/portfolio", params={"run_id": rdir.name})
    assert r.status_code == 200


def test_tail_jsonl_reverse_block_reader(tmp_path: Path):
    from api.server import _tail_jsonl
    p = tmp_path / "big.jsonl"
    # Write 5000 records at ~80 bytes each = ~400 KB. Crosses several
    # 256-KB blocks of the reverse reader.
    with open(p, "w", encoding="utf-8") as f:
        for i in range(5000):
            f.write(json.dumps({"i": i, "msg": "x" * 50}) + "\n")
    rows = _tail_jsonl(p, limit=10)
    assert len(rows) == 10
    assert [r["i"] for r in rows] == list(range(4990, 5000))
    rows100 = _tail_jsonl(p, limit=100)
    assert len(rows100) == 100
    assert rows100[-1]["i"] == 4999


def test_tail_jsonl_skips_malformed(tmp_path: Path):
    from api.server import _tail_jsonl
    p = tmp_path / "mixed.jsonl"
    p.write_text(
        '{"a":1}\n' "not json\n" '{"a":2}\n' "\n" '{"a":3}\n',
        encoding="utf-8",
    )
    rows = _tail_jsonl(p, limit=10)
    assert [r["a"] for r in rows] == [1, 2, 3]


# ---------------------------------------------------------------------
# Linux paths smoke (Config)
# ---------------------------------------------------------------------
def test_config_accepts_posix_data_root(tmp_path: Path,
                                        monkeypatch: pytest.MonkeyPatch):
    posix_root = tmp_path / "v5-data"
    monkeypatch.setenv("DATA_ROOT", str(posix_root))
    for k in ("BYBIT_OFFLINE", "AI_DRY_RUN", "OPENROUTER_LIVE"):
        monkeypatch.delenv(k, raising=False)
    # Re-import so dotenv side effects don't poison.
    from core.config import load_config
    cfg = load_config()
    assert cfg.data_root == posix_root
    for sub in (cfg.cache_root, cfg.feature_root,
                cfg.run_root, cfg.log_root):
        assert sub.exists() and sub.is_dir()
        # All sub-paths live inside data_root.
        assert posix_root in sub.parents or sub == posix_root


# ---------------------------------------------------------------------
# Deploy artefacts present + parseable
# ---------------------------------------------------------------------
def test_deploy_artefacts_present():
    base = ROOT / "deploy"
    expected = [
        base / "deploy.md",
        base / "logrotate.v5",
        base / "systemd" / "v5-trader.service",
        base / "systemd" / "v5-api.service",
        base / "systemd" / "v5-health.service",
        base / "systemd" / "v5-health.timer",
    ]
    for p in expected:
        assert p.is_file(), f"missing deploy artefact: {p}"


def test_systemd_units_have_required_sections():
    base = ROOT / "deploy" / "systemd"
    for unit in ("v5-trader.service", "v5-api.service",
                 "v5-health.service"):
        text = (base / unit).read_text(encoding="utf-8")
        assert "[Unit]" in text and "[Service]" in text, unit
        # Hardening flags we deliberately ship.
        assert "NoNewPrivileges=yes" in text, unit
        assert "ProtectSystem=" in text, unit
        # Loopback / no privileged ops.
        assert "ExecStart=" in text, unit
    timer = (base / "v5-health.timer").read_text(encoding="utf-8")
    assert "[Timer]" in timer
    assert "OnUnitActiveSec=" in timer
