"""Phase 14 - operator health-check.

A single binary intended to be run from cron / systemd-timer that
validates the V5 paper-orchestrator is alive on a VPS.

Checks performed (each one is a named "probe" with a pass/warn/fail
status). The overall exit code follows the worst probe:

    0  -> all probes PASS
    1  -> at least one probe WARN, no FAIL
    2  -> at least one probe FAIL
    3  -> the health-checker itself crashed before it could probe

Probes:

    services        - all configured systemd --user services are
                      ``active (running)``. Disabled by default; enable
                      with ``--services NAME [NAME ...]``.
    snapshot_age    - the most recent ``snapshot.parquet`` (or
                      ``snapshot.json``) under ``cfg.run_root`` is
                      younger than ``--max-snapshot-min`` minutes
                      (default 45).
    portfolio_age   - the active run's ``portfolio.json`` is younger
                      than ``--max-portfolio-min`` minutes (default 90).
    api             - ``GET /api/health`` on ``--api-url`` returns
                      ``status=ok`` within ``--api-timeout`` seconds.
                      Skipped when ``--api-url`` is empty.
    budget          - the active run's ``budget.json`` ``spent_usd``
                      is below ``--max-budget-frac`` of
                      ``ai_budget_usd_per_day`` (default 0.95).
    disk            - ``cfg.data_root`` filesystem has at least
                      ``--min-disk-gb`` free GB (default 1.0).
    log_errors      - the most recent service logs contain fewer than
                      ``--max-log-errors`` ERROR-level entries in the
                      last ``--log-window-min`` minutes (default
                      1 / 60). Best-effort: if the journal is not
                      reachable the probe WARNs rather than FAILs.

Output formats:

    --format text   (default) human-readable single line per probe
    --format json   one JSON object on stdout (operator-friendly, also
                    safe to pipe into jq / Prometheus textfile)

Exit codes are designed so a cron job can just::

    0 1 * * *  /opt/v5/.venv/bin/python /opt/v5/scripts/health_check.py \\
                  --services v5-trader v5-api \\
                  --api-url http://127.0.0.1:8765 \\
                  || systemd-cat -t v5-health -p err

without parsing stdout.

Security: no probe sends credentials; ``--api-url`` is loopback-only by
default. The tool refuses to run as root unless ``--allow-root`` is
passed (defence in depth - cron should drop to a service account).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

_HERE = Path(__file__).resolve().parent
_SRC = _HERE.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from core.config import Config, load_config  # noqa: E402

log = logging.getLogger("health_check")

# Probe statuses. Lower is better (PASS < WARN < FAIL).
STATUS_PASS = "PASS"
STATUS_WARN = "WARN"
STATUS_FAIL = "FAIL"
_STATUS_RANK = {STATUS_PASS: 0, STATUS_WARN: 1, STATUS_FAIL: 2}


@dataclass
class ProbeResult:
    name: str
    status: str
    detail: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # Trim empty extras.
        if not d["extra"]:
            d.pop("extra")
        return d


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _now_ts() -> float:
    return time.time()


def _newest_match(root: Path, names: Iterable[str]) -> Path | None:
    """Return the freshest file under ``root`` whose basename matches."""
    if not root.exists():
        return None
    best: tuple[float, Path] | None = None
    name_set = set(names)
    # ``rglob`` over a small run tree is cheap (a handful of files per
    # run, dozens of runs in a long-lived deployment).
    for p in root.rglob("*"):
        if not p.is_file() or p.name not in name_set:
            continue
        try:
            mtime = p.stat().st_mtime
        except OSError:
            continue
        if best is None or mtime > best[0]:
            best = (mtime, p)
    return best[1] if best else None


def _active_run_dir(cfg: Config) -> Path | None:
    root = cfg.run_root
    if not root.exists():
        return None
    candidates: list[tuple[float, Path]] = []
    for p in root.iterdir():
        if not p.is_dir():
            continue
        try:
            candidates.append((p.stat().st_mtime, p))
        except OSError:
            continue
    if not candidates:
        return None
    candidates.sort(key=lambda r: r[0], reverse=True)
    return candidates[0][1]


def _file_age_min(path: Path) -> float:
    return (_now_ts() - path.stat().st_mtime) / 60.0


# ----------------------------------------------------------------------
# Probes
# ----------------------------------------------------------------------
def probe_services(names: list[str]) -> list[ProbeResult]:
    """Each service ``--user`` ``ActiveState`` must be ``active``."""
    out: list[ProbeResult] = []
    if not names:
        return out
    systemctl = shutil.which("systemctl")
    if systemctl is None:
        for n in names:
            out.append(ProbeResult(
                name=f"service:{n}", status=STATUS_WARN,
                detail="systemctl not on PATH (not a systemd host)"))
        return out
    for n in names:
        try:
            r = subprocess.run(
                [systemctl, "--user", "is-active", n],
                capture_output=True, text=True, timeout=5.0, check=False,
            )
            state = (r.stdout or "").strip() or (r.stderr or "").strip()
            if state == "active":
                out.append(ProbeResult(
                    name=f"service:{n}", status=STATUS_PASS,
                    detail="active (running)"))
            else:
                out.append(ProbeResult(
                    name=f"service:{n}", status=STATUS_FAIL,
                    detail=f"state={state!r}"))
        except (subprocess.TimeoutExpired, OSError) as exc:
            out.append(ProbeResult(
                name=f"service:{n}", status=STATUS_FAIL,
                detail=f"{type(exc).__name__}: {exc}"))
    return out


def probe_snapshot_age(cfg: Config, *, max_minutes: float) -> ProbeResult:
    snap = _newest_match(cfg.run_root, ("snapshot.parquet", "snapshot.json"))
    if snap is None:
        return ProbeResult(
            name="snapshot_age", status=STATUS_FAIL,
            detail="no snapshot found under run_root",
            extra={"run_root": str(cfg.run_root)},
        )
    age_min = _file_age_min(snap)
    detail = f"{snap.name} age={age_min:.1f}m (limit={max_minutes:.0f}m)"
    extra = {"path": str(snap), "age_minutes": round(age_min, 2)}
    if age_min > max_minutes:
        return ProbeResult(name="snapshot_age", status=STATUS_FAIL,
                           detail=detail, extra=extra)
    if age_min > max_minutes * 0.8:
        return ProbeResult(name="snapshot_age", status=STATUS_WARN,
                           detail=detail, extra=extra)
    return ProbeResult(name="snapshot_age", status=STATUS_PASS,
                       detail=detail, extra=extra)


def probe_portfolio_age(cfg: Config, *, max_minutes: float) -> ProbeResult:
    rdir = _active_run_dir(cfg)
    if rdir is None:
        return ProbeResult(name="portfolio_age", status=STATUS_WARN,
                           detail="no active run directory")
    pfile = rdir / "portfolio.json"
    if not pfile.exists():
        return ProbeResult(name="portfolio_age", status=STATUS_WARN,
                           detail=f"missing {pfile.name}",
                           extra={"run_id": rdir.name})
    age_min = _file_age_min(pfile)
    detail = f"portfolio.json age={age_min:.1f}m (limit={max_minutes:.0f}m)"
    extra = {"run_id": rdir.name, "age_minutes": round(age_min, 2)}
    if age_min > max_minutes:
        return ProbeResult(name="portfolio_age", status=STATUS_FAIL,
                           detail=detail, extra=extra)
    return ProbeResult(name="portfolio_age", status=STATUS_PASS,
                       detail=detail, extra=extra)


def probe_api(url: str, *, timeout_sec: float) -> ProbeResult:
    if not url:
        return ProbeResult(name="api", status=STATUS_PASS,
                           detail="skipped (no --api-url)")
    target = url.rstrip("/") + "/api/health"
    try:
        req = urllib.request.Request(target,
                                     headers={"User-Agent": "v5-health/1"})
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:  # noqa: S310 (loopback)
            if resp.status != 200:
                return ProbeResult(name="api", status=STATUS_FAIL,
                                   detail=f"http {resp.status}",
                                   extra={"url": target})
            body = resp.read(64 * 1024)
        data = json.loads(body.decode("utf-8", errors="replace"))
        if data.get("status") != "ok":
            return ProbeResult(name="api", status=STATUS_FAIL,
                               detail=f"payload={data}",
                               extra={"url": target})
        return ProbeResult(name="api", status=STATUS_PASS,
                           detail=f"ok active_run={data.get('active_run')}",
                           extra={"url": target,
                                  "active_run": data.get("active_run")})
    except (urllib.error.URLError, OSError, socket.timeout,
            json.JSONDecodeError) as exc:
        return ProbeResult(name="api", status=STATUS_FAIL,
                           detail=f"{type(exc).__name__}: {exc}",
                           extra={"url": target})


def probe_budget(cfg: Config, *, max_frac: float) -> ProbeResult:
    cap = float(getattr(cfg, "ai_budget_usd_per_day", 0.0) or 0.0)
    rdir = _active_run_dir(cfg)
    if rdir is None:
        return ProbeResult(name="budget", status=STATUS_WARN,
                           detail="no active run directory")
    bfile = rdir / "budget.json"
    if not bfile.exists():
        return ProbeResult(name="budget", status=STATUS_PASS,
                           detail="no budget file yet (cold start)")
    try:
        data = json.loads(bfile.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return ProbeResult(name="budget", status=STATUS_WARN,
                           detail=f"unreadable: {exc}")
    spent = float(data.get("spent_usd", 0.0))
    extra = {"spent_usd": round(spent, 4),
             "cap_usd": round(cap, 4),
             "run_id": rdir.name}
    if cap <= 0.0:
        return ProbeResult(name="budget", status=STATUS_PASS,
                           detail=f"spent=${spent:.3f} (cap unset)",
                           extra=extra)
    frac = spent / cap
    extra["frac"] = round(frac, 4)
    detail = f"spent=${spent:.3f}/${cap:.2f} ({frac * 100:.1f}%)"
    if frac >= max_frac:
        return ProbeResult(name="budget", status=STATUS_FAIL,
                           detail=detail, extra=extra)
    if frac >= max_frac * 0.8:
        return ProbeResult(name="budget", status=STATUS_WARN,
                           detail=detail, extra=extra)
    return ProbeResult(name="budget", status=STATUS_PASS,
                       detail=detail, extra=extra)


def probe_disk(cfg: Config, *, min_free_gb: float) -> ProbeResult:
    target = cfg.data_root if cfg.data_root.exists() else cfg.repo_root
    try:
        usage = shutil.disk_usage(str(target))
    except OSError as exc:
        return ProbeResult(name="disk", status=STATUS_FAIL,
                           detail=f"{type(exc).__name__}: {exc}")
    free_gb = usage.free / (1024 ** 3)
    extra = {"free_gb": round(free_gb, 2), "path": str(target)}
    detail = f"free={free_gb:.2f} GB at {target} (min={min_free_gb} GB)"
    if free_gb < min_free_gb:
        return ProbeResult(name="disk", status=STATUS_FAIL,
                           detail=detail, extra=extra)
    if free_gb < min_free_gb * 1.5:
        return ProbeResult(name="disk", status=STATUS_WARN,
                           detail=detail, extra=extra)
    return ProbeResult(name="disk", status=STATUS_PASS,
                       detail=detail, extra=extra)


def probe_log_errors(
    *,
    services: list[str],
    window_min: float,
    max_errors: int,
) -> ProbeResult:
    if not services:
        return ProbeResult(name="log_errors", status=STATUS_PASS,
                           detail="skipped (no --services)")
    journalctl = shutil.which("journalctl")
    if journalctl is None:
        return ProbeResult(name="log_errors", status=STATUS_WARN,
                           detail="journalctl not on PATH")
    cmd = [journalctl, "--user", "--no-pager", "--since",
           f"{int(window_min)}min ago", "-p", "err"]
    for n in services:
        cmd.extend(["-u", n])
    try:
        r = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=10.0, check=False)
    except (subprocess.TimeoutExpired, OSError) as exc:
        return ProbeResult(name="log_errors", status=STATUS_WARN,
                           detail=f"journalctl: {type(exc).__name__}: {exc}")
    lines = [ln for ln in (r.stdout or "").splitlines() if ln.strip()]
    n = len(lines)
    extra = {"count": n, "window_min": window_min,
             "services": list(services)}
    detail = f"errors={n} in last {int(window_min)}m (limit={max_errors})"
    if n > max_errors:
        return ProbeResult(name="log_errors", status=STATUS_FAIL,
                           detail=detail, extra=extra)
    if n == max_errors and max_errors > 0:
        return ProbeResult(name="log_errors", status=STATUS_WARN,
                           detail=detail, extra=extra)
    return ProbeResult(name="log_errors", status=STATUS_PASS,
                       detail=detail, extra=extra)


# ----------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------
def run_probes(args: argparse.Namespace, cfg: Config) -> list[ProbeResult]:
    results: list[ProbeResult] = []
    if args.services:
        results.extend(probe_services(args.services))
    results.append(probe_snapshot_age(
        cfg, max_minutes=args.max_snapshot_min))
    results.append(probe_portfolio_age(
        cfg, max_minutes=args.max_portfolio_min))
    results.append(probe_api(args.api_url, timeout_sec=args.api_timeout))
    results.append(probe_budget(cfg, max_frac=args.max_budget_frac))
    results.append(probe_disk(cfg, min_free_gb=args.min_disk_gb))
    if args.services:
        results.append(probe_log_errors(
            services=args.services,
            window_min=args.log_window_min,
            max_errors=args.max_log_errors,
        ))
    return results


def overall_status(results: list[ProbeResult]) -> str:
    rank = max((_STATUS_RANK[r.status] for r in results), default=0)
    for k, v in _STATUS_RANK.items():
        if v == rank:
            return k
    return STATUS_PASS


def status_to_exit(status: str) -> int:
    return {STATUS_PASS: 0, STATUS_WARN: 1, STATUS_FAIL: 2}.get(status, 2)


def render_text(results: list[ProbeResult], overall: str) -> str:
    lines = [f"v5-health: overall={overall}"]
    for r in results:
        lines.append(f"  [{r.status}] {r.name}: {r.detail}")
    return "\n".join(lines)


def render_json(results: list[ProbeResult], overall: str) -> str:
    payload = {
        "overall": overall,
        "ts": time.time(),
        "probes": [r.to_dict() for r in results],
    }
    return json.dumps(payload, indent=2, sort_keys=False)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="V5 paper-orchestrator health check")
    p.add_argument("--services", nargs="*", default=[],
                   help="systemd --user unit names to verify")
    p.add_argument("--api-url", default="",
                   help="dashboard URL (e.g. http://127.0.0.1:8765); "
                        "empty disables the probe")
    p.add_argument("--api-timeout", type=float, default=5.0)
    p.add_argument("--max-snapshot-min", type=float, default=45.0)
    p.add_argument("--max-portfolio-min", type=float, default=90.0)
    p.add_argument("--max-budget-frac", type=float, default=0.95)
    p.add_argument("--min-disk-gb", type=float, default=1.0)
    p.add_argument("--log-window-min", type=float, default=60.0)
    p.add_argument("--max-log-errors", type=int, default=1)
    p.add_argument("--format", choices=("text", "json"), default="text")
    p.add_argument("--allow-root", action="store_true",
                   help="permit running as root (default: refuse)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.WARNING,
                        format="%(asctime)s %(levelname)s %(message)s")
    try:
        args = _parse_args(argv)
        # Defence in depth: cron should not run this as root.
        if hasattr(os, "geteuid") and os.geteuid() == 0 and not args.allow_root:
            sys.stderr.write(
                "health_check refuses to run as root; pass --allow-root "
                "if you really mean it.\n")
            return 3
        cfg = load_config()
        results = run_probes(args, cfg)
        overall = overall_status(results)
        out = (render_json(results, overall)
               if args.format == "json"
               else render_text(results, overall))
        print(out)
        return status_to_exit(overall)
    except Exception as exc:  # noqa: BLE001
        log.exception("health_check crashed")
        sys.stderr.write(f"health_check internal error: {exc}\n")
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
