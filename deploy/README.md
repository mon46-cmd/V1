# V5 — VPS deployment

Two paths, pick one:

| Approach              | When to use                              | Auto-restart? |
|-----------------------|------------------------------------------|---------------|
| **systemd installer** | Real VPS with systemd available          | Yes           |
| **nohup runner**      | Containers, WSL, locked-down VMs         | No            |

The systemd path uses **system-level units** with `User=` directive,
not `--user` units. This avoids the cgroup-delegation and namespace
problems that plague user-mode systemd on stock cloud images.

---

## Prerequisites

Working V5 clone with:

```bash
cd ~/V1            # or wherever you cloned
./setup.sh --dev   # builds .venv + des_core
cp .env.example .env
chmod 600 .env
nano .env          # set OPENROUTER_API_KEY etc.
BYBIT_OFFLINE=1 .venv/bin/python -m pytest -q --timeout=60
# expect: 221 passed, 20 skipped
```

---

## Path A — systemd (recommended)

One command. Run from the V5 clone root:

```bash
sudo deploy/install.sh
```

What it does:

1. Detects the invoking user (`$SUDO_USER`) and that user's homedir.
2. Detects `V5_HOME` from where this script lives.
3. Verifies `.venv/bin/python`, the launcher scripts, and `.env`
   exist before touching anything.
4. Renders `deploy/systemd/*.tmpl` into `/etc/systemd/system/` with
   the absolute paths baked in.
5. `systemctl daemon-reload`, `enable --now` for trader, api, and
   the health timer.

Verify:

```bash
systemctl status v5-trader.service v5-api.service --no-pager -l
journalctl -u v5-trader.service -f                # live tail
curl -s http://127.0.0.1:8765/health | jq .       # API on loopback
```

Stop / start / restart:

```bash
sudo systemctl stop    v5-trader v5-api v5-health.timer
sudo systemctl start   v5-trader v5-api v5-health.timer
sudo systemctl restart v5-trader v5-api
```

Remove completely (leaves `data/` and clone intact):

```bash
sudo deploy/uninstall.sh
```

### Reach the dashboard from your laptop

The API binds to `127.0.0.1:8765` only — never publish it. Tunnel:

```bash
ssh -N -L 8765:127.0.0.1:8765 user@<vps-ip>
# then open http://127.0.0.1:8765 in a browser
```

---

## Path B — nohup runner (no systemd)

For WSL, Docker images without an init, or hostile policies:

```bash
deploy/run_nohup.sh start    # launches trader + api in background
deploy/run_nohup.sh status
deploy/run_nohup.sh tail     # follow data/logs/trader.out
deploy/run_nohup.sh stop
```

PIDs live in `data/run/*.pid`, logs in `data/logs/*.out`. No
auto-restart on crash. Add a cron `@reboot` line if you need
persistence:

```cron
@reboot bash -lc 'cd ~/V1 && deploy/run_nohup.sh start'
```

---

## Log rotation

The systemd journal handles its own rotation. Structured JSON logs
the orchestrator writes under `data/logs/` need an extra step:

```bash
# System-wide (recommended; needs sudo):
sed "s|__V5_HOME__|$HOME/V1|g" deploy/logrotate.v5 \
  | sudo tee /etc/logrotate.d/v5 > /dev/null
sudo logrotate -d /etc/logrotate.d/v5    # dry-run first
```

JSONL audit files (`triggers.jsonl`, `intents.jsonl`, `fills.jsonl`,
`reviews.jsonl`, `prompts.jsonl`) are **intentionally not rotated** —
they are the canonical history. A 7-day run is < 50 MB.

---

## Geo-block warning (Bybit + CloudFront)

Bybit's public REST is fronted by CloudFront and blocks several
regions. If `curl https://api.bybit.com/v5/market/time` returns 403,
your VPS is in a blocked country. Provision in:

- `europe-west4` (Netherlands), `europe-west3` (Frankfurt)
- `asia-northeast1` (Tokyo) — best Bybit RTT
- `asia-east1` (Taiwan)

The trader **will not start** without REST access. Run with
`BYBIT_OFFLINE=1` in the `.env` for synthetic-universe smoke testing.

---

## Threat model (one paragraph)

The VPS holds one secret of value: `OPENROUTER_API_KEY` in
`<V5_HOME>/.env`. The dashboard binds to loopback only and is reached
via `ssh -L`. No inbound port is opened beyond SSH. The trader and
API run as your unprivileged user with `NoNewPrivileges=yes`. The
API is read-only. Logs are pruned hourly. If the key leaks, the
OpenRouter daily budget cap (`AI_BUDGET_USD_PER_DAY`, default $3)
is the blast radius.
