# V5 — Autonomous paper-trading orchestrator (Bybit linear USDT perps)

LLM-routed paper trader. Two heavy-reasoning passes pick + qualify the
trade; a deterministic Python/Rust execution layer sizes, places intents,
fills against a paper broker, and audits everything to disk.

```
Bybit REST/WS  →  13-layer feature engine  →  Prompt A (super grok)  →  watchlist
                                            ↓
                                     Prompt B (grok-4.1-fast)  →  intent  →  ActivationWatcher
                                                                                    ↓
                                                                             paper broker
                                                                                    ↓
                                                            Prompt C (grok-4.1-fast)  →  review
```

- Models: watchlist = `x-ai/grok-4.20-multi-agent` ("super grok"); deep + review = `x-ai/grok-4.1-fast`.
- Daily AI budget cap (default **$3 USD/day**), enforced per call before HTTP.
- 220 tests passing offline (`BYBIT_OFFLINE=1`), 20 skipped (live-only).

> Full architecture: [`docs/00_VISION.md`](docs/00_VISION.md) →
> [`docs/07_DATA_FLOW_DEEPDIVE.md`](docs/07_DATA_FLOW_DEEPDIVE.md).

---

## Table of contents

1. [Prerequisites](#1-prerequisites)
2. [Local install (Windows / dev)](#2-local-install-windows--dev)
3. [Configure secrets](#3-configure-secrets)
4. [Smoke tests](#4-smoke-tests)
5. [Run the loop locally](#5-run-the-loop-locally)
6. [Operator commands (STOP / resume / health)](#6-operator-commands-stop--resume--health)
7. [Production VPS deployment (Linux)](#7-production-vps-deployment-linux)
8. [Dashboard access](#8-dashboard-access)
9. [Backups + log rotation](#9-backups--log-rotation)
10. [Pre-production checklist](#10-pre-production-checklist)
11. [Troubleshooting](#11-troubleshooting)

---

## 1. Prerequisites

| Component  | Version                              | Notes                                           |
|------------|--------------------------------------|-------------------------------------------------|
| Python     | 3.12.x                               | `py -3.12` launcher on Windows                  |
| Rust       | stable-msvc (Win) / stable (Linux)   | for the `des_core` PyO3 kernel; optional        |
| MSVC BT    | 2022 "Desktop development with C++"  | Windows only, only if building Rust kernel      |
| Git        | any                                  |                                                 |
| OpenRouter | account + API key                    | https://openrouter.ai/keys                      |

Python fallbacks exist for every Rust kernel, so `setup.ps1` (no `-Dev`)
works without Rust. The Rust extension is required for production
performance.

---

## 2. Local install (Windows / dev)

```powershell
# from C:\Users\User\Documents\Double-Edged-Savior\DEV\V5
.\setup.ps1 -Dev
```

That script:

1. Verifies Python 3.12 + creates `.venv` if missing.
2. Installs `requirements-dev.txt` (or `requirements.txt` without `-Dev`).
3. `pip install -e . --no-deps`.
4. Builds `des_core` via `maturin develop --release` (Rust kernels).

Activate the venv:

```powershell
.\.venv\Scripts\Activate.ps1
```

> **Always use `V5\.venv`** — never `V1\.venv`. Several recent terminals
> activated the V1 venv by mistake; close them and start a new
> PowerShell inside `V5/`.

---

## 3. Configure secrets

```powershell
copy .env.example .env
notepad .env
```

Set:

```ini
# REQUIRED — get from https://openrouter.ai/keys
OPENROUTER_API_KEY=sk-or-v1-REPLACE_ME

# OPTIONAL — already defaulted in code (Phase 14.5)
MODEL_WATCHLIST=x-ai/grok-4.20-multi-agent
MODEL_DEEP=x-ai/grok-4.1-fast
MODEL_REVIEW=x-ai/grok-4.1-fast

# OPTIONAL — attribution headers (recommended by OpenRouter)
OPENROUTER_REFERER=https://your-host.example
OPENROUTER_TITLE=v5-paper-orchestrator

# OPTIONAL — runtime knobs
DATA_ROOT=                   # absolute or relative-to-V5; default: V5/data
LOG_LEVEL=INFO
BYBIT_OFFLINE=               # set to 1 to skip every live REST/WS call (tests)
AI_DRY_RUN=                  # set to 1 to use the offline mock router
OPENROUTER_LIVE=             # set to 1 to enable live AI tests (costs $)
```

> ⚠️ **Never commit `.env`** — it's gitignored. **Never paste your API
> key in chat / issues / screenshots.** If you do, rotate it immediately
> at https://openrouter.ai/keys.

Quick check:

```powershell
.\.venv\Scripts\python.exe -c "from core.config import load_config; c = load_config(); print('deep =', c.model_deep, '/ key set =', bool(c.openrouter_api_key))"
```

---

## 4. Smoke tests

```powershell
# Offline test sweep (220 tests, ~6 seconds)
$env:BYBIT_OFFLINE = "1"
.\.venv\Scripts\python.exe -m pytest --timeout=60 -q

# Single subset
.\.venv\Scripts\python.exe -m pytest --timeout=60 -q tests\test_exec_loop.py

# Live-data tests (reads Bybit REST; AI tests cost $$)
Remove-Item env:BYBIT_OFFLINE
$env:OPENROUTER_LIVE = "1"   # only if you really want to spend
.\.venv\Scripts\python.exe -m pytest --timeout=120 -q -m "live"
```

Expected: `220 passed, 20 skipped` offline.

Quick LLM connectivity probe (uses your key + budget):

```powershell
.\.venv\Scripts\python.exe scripts\inspect_prompts.py --probe deep --symbol BTCUSDT
```

---

## 5. Run the loop locally

### 5.1 One-shot scanner pass (no orders)

```powershell
.\.venv\Scripts\python.exe scripts\scan_once.py
```

Writes `data/runs/<run_id>/{universe,snapshot,watchlist}.{parquet,json}`.

### 5.2 One-shot watchlist (Prompt A)

```powershell
.\.venv\Scripts\python.exe scripts\watchlist_once.py
```

### 5.3 Full live (paper) exec loop

```powershell
# Default: $10 000 paper equity, 30-min scanner cadence
.\.venv\Scripts\python.exe scripts\run_exec.py

# Custom equity + faster cadence (testing)
.\.venv\Scripts\python.exe scripts\run_exec.py --equity 5000 --scanner-interval 600

# Force the offline mock router (no AI cost)
.\.venv\Scripts\python.exe scripts\run_exec.py --dry-run

# Resume an existing run (replays fills.jsonl into running counters)
.\.venv\Scripts\python.exe scripts\run_exec.py --run-id 20260425_120000_ab12cd
```

### 5.4 Read-only API (dashboard backend)

In a second terminal:

```powershell
.\.venv\Scripts\python.exe scripts\run_api.py --host 127.0.0.1 --port 8765
```

Open `http://127.0.0.1:8765/` for the dashboard.

---

## 6. Operator commands (STOP / resume / health)

### Halt without killing the process

```powershell
# Refuses new intents on next on_trigger; run_exec exits cleanly
"halt $(Get-Date -Format o)" | Out-File -Encoding utf8 data\STOP

# To resume, delete the file before relaunching
Remove-Item data\STOP
```

The STOP-file watcher polls every 10s (`--stop-poll-sec`), then a final
`portfolio.json` is written before exit.

### Health probe

```powershell
.\.venv\Scripts\python.exe scripts\health_check.py --format text
.\.venv\Scripts\python.exe scripts\health_check.py --format json
```

Exit codes: `0=PASS`, `1=WARN`, `2=FAIL`, `3=internal crash`.

Probes: `services / snapshot_age (<45m) / portfolio_age (<90m) / api /
budget (<95% cap) / disk (>1GB) / log_errors`.

### Inspect a run

```powershell
# Latest run id
Get-ChildItem data\runs | Sort-Object LastWriteTime | Select-Object -Last 1

# Tail any audit file
Get-Content data\runs\<rid>\fills.jsonl    -Tail 20
Get-Content data\runs\<rid>\intents.jsonl  -Tail 20
Get-Content data\runs\<rid>\reviews.jsonl  -Tail 20
Get-Content data\runs\<rid>\prompts.jsonl  -Tail 20

# Today's AI spend
Get-Content data\runs\<rid>\budget.json
```

---

## 7. Production VPS deployment (Linux)

Full step-by-step in [`deploy/deploy.md`](deploy/deploy.md). TL;DR:

```bash
# --- on the VPS, as root ---
adduser --system --group --home /opt/v5 v5
apt-get install -y python3.12 python3.12-venv git curl logrotate

# --- as user `v5` ---
sudo -iu v5
cd ~ && git clone <your-fork> v5 && cd v5/V5

python3.12 -m venv .venv
. .venv/bin/activate
pip install --upgrade pip wheel setuptools
pip install -r requirements.txt
pip install -e . --no-deps

# Rust kernel (optional but recommended)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
. "$HOME/.cargo/env"
pip install maturin
( cd rust_core && maturin develop --release )

# Secrets
cp .env.example .env
chmod 600 .env
$EDITOR .env                     # paste OPENROUTER_API_KEY

# Verify
BYBIT_OFFLINE=1 python -m pytest --timeout=60 -q
```

Install systemd units + log rotation **as root**:

```bash
install -m644 deploy/systemd/v5-trader.service  /etc/systemd/system/
install -m644 deploy/systemd/v5-api.service     /etc/systemd/system/
install -m644 deploy/systemd/v5-health.service  /etc/systemd/system/
install -m644 deploy/systemd/v5-health.timer    /etc/systemd/system/
install -m644 deploy/logrotate.v5               /etc/logrotate.d/v5

systemctl daemon-reload
systemctl enable --now v5-trader.service v5-api.service v5-health.timer
systemctl status  v5-trader.service v5-api.service v5-health.timer
```

The units are hardened: `NoNewPrivileges=yes`, `ProtectSystem=strict`,
`ProtectHome=yes`, `MemoryDenyWriteExecute=yes`, restricted system call
filter, read-write only on `/opt/v5/V5/data` and `/opt/v5/V5/logs`.

Verify alive:

```bash
journalctl -u v5-trader.service -f
journalctl -u v5-api.service -f
sudo -iu v5 V5/.venv/bin/python V5/scripts/health_check.py --format text
```

---

## 8. Dashboard access

The API binds **loopback only**. From your laptop:

```bash
ssh -L 8765:localhost:8765 v5-vps
# then open http://127.0.0.1:8765/
```

Do **not** expose 8765 to the public internet. If you need a remote
view, put it behind nginx + Basic Auth or a Tailscale node — see
[`deploy/deploy.md`](deploy/deploy.md) §11.

---

## 9. Backups + log rotation

```bash
# logrotate is auto-invoked daily; manual sanity check:
sudo logrotate -d /etc/logrotate.d/v5

# Nightly rsync of audit data (set up as the v5 user via crontab -e):
0 3 * * *  rsync -aH --delete \
              /opt/v5/V5/data/runs/ backup@bak.example:/srv/v5-runs/
```

`*.jsonl` audit files are **NOT** rotated (append-only, source of truth).
Only `*.log` files rotate hourly.

---

## 10. Pre-production checklist

Run through every box before pointing real money at this:

- [ ] `OPENROUTER_API_KEY` in `.env`, `chmod 600 .env` on Linux.
- [ ] Any key ever pasted into chat / issues / screenshots **rotated** at openrouter.ai/keys.
- [ ] `pytest --timeout=60` reports `220 passed, 20 skipped` offline.
- [ ] `scripts/health_check.py --format text` returns PASS.
- [ ] `data/STOP` is **absent** (no stale halt sentinel).
- [ ] At least one full intent lifecycle observed in `data/runs/<rid>/fills.jsonl`.
- [ ] `replay_from_fills` agrees with `portfolio.json` to 1e-6
      (tested by `tests/test_exec_loop.py::test_replay_matches_running_state`).
- [ ] AI budget configured: `ai_budget_usd_per_day` (default $3) and
      `budget.json` rolls over at UTC midnight.
- [ ] systemd units enabled and surviving a reboot test.
- [ ] Backup target reachable (`rsync -n` dry run).
- [ ] `journalctl -u v5-trader -p err --since today` is empty.
- [ ] Dashboard reachable only via SSH `-L` tunnel (no public port).
- [ ] **7-day Definition of Done**: all three units up, daily cost under
      budget, 3+ intent lifecycles, 7-day fills replay matches portfolio
      to the cent, dashboard cold load < 200 ms.

---

## 11. Troubleshooting

| Symptom                                   | Likely cause                                      | Fix                                                   |
|-------------------------------------------|---------------------------------------------------|-------------------------------------------------------|
| `ModuleNotFoundError: des_core`           | Rust extension not built                          | `cd rust_core && maturin develop --release`           |
| `pytest` import errors                    | wrong venv (V1 instead of V5)                     | `Deactivate; .\.venv\Scripts\Activate.ps1` from V5    |
| `OPENROUTER_API_KEY` empty                | `.env` missing or in wrong folder                 | Must be at `V5/.env`                                  |
| Budget exhausted, no AI calls             | Today's spend ≥ `ai_budget_usd_per_day`           | Wait for UTC-day rollover or raise cap in `.env`      |
| `STOP file detected` in logs              | `data/STOP` exists                                | `rm data/STOP` and restart                            |
| `\r\r\n` in JSONL files (Windows)         | A writer used `os.linesep`                        | Bug — open issue; all writes must use literal `"\n"`  |
| `snapshot_age` WARN > 30 min              | scanner stuck or rate-limited                     | `journalctl -u v5-trader -n 200`; check Bybit status  |
| Dashboard 404 on `/run/<rid>/...`         | run_id contains `..` or absolute path             | Path-traversal hardening — use the `/runs` listing    |
| `replay_from_fills` mismatch              | A non-`_CLOSING_KIND` fill bumped `loser_streak`  | See Phase 12 fix in `loops/exec.py::_record_fill`     |

---

## License + attribution

Internal project. No license granted.

Models routed via [OpenRouter](https://openrouter.ai/). Market data from
[Bybit v5 REST + WS](https://bybit-exchange.github.io/docs/v5/intro).
Charts via [TradingView lightweight-charts](https://github.com/tradingview/lightweight-charts) 4.2.3.
