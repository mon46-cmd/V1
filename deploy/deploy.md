# V5 Deployment

The recommended deployment entrypoint is [deploy/README.md](README.md) and,
on a real VPS, the single command is:

```bash
sudo deploy/install.sh
```

Shipped artefacts in this folder:

- `install.sh`: renders absolute-path system units into `/etc/systemd/system/`
- `uninstall.sh`: removes those installed units
- `run_nohup.sh`: fallback for hosts without a usable systemd setup
- `logrotate.v5`: logrotate config template with `__V5_HOME__` placeholder
- `systemd/*.service`: reference units for a default `~/v5` install
- `systemd/*.tmpl`: templates consumed by `install.sh`

If your server cannot run the systemd installer, use:

```bash
deploy/run_nohup.sh start
```# V5 - Debian VPS deployment guide

This is the canonical recipe to bring up a fresh Debian 12 (bookworm)
or Ubuntu 24.04 LTS VPS running the V5 paper-orchestrator under an
unprivileged ``v5`` account with ``--user`` systemd. **Paper trading
only**: the orchestrator never places real orders.

> Estimated time: 20-30 minutes on a fresh 2 vCPU / 4 GB / 40 GB box.
> Tested on Hetzner CX22 (Debian 12) and DigitalOcean s-2vcpu-4gb
> (Ubuntu 24.04).

## Threat model in one paragraph

The VPS holds a single secret of value: ``OPENROUTER_API_KEY`` in
``~/v5/.env``. The dashboard binds to **loopback only** and is reached
via ``ssh -L``. No inbound port is opened beyond SSH. All services run
as a non-root ``v5`` account with ``NoNewPrivileges`` and a tight
``SystemCallFilter``. The API is **read-only** and never mutates the
data tree (``ReadOnlyPaths`` enforced). Logs are pruned hourly. If the
key leaks, the OpenRouter daily budget cap (``AI_BUDGET_USD_PER_DAY``,
default $3) is the blast radius.

## 0. Prerequisites on your laptop

You need:

- SSH key for the VPS (``ssh-keygen -t ed25519`` if you don't have one).
- Your OpenRouter API key (``sk-or-...``).
- Git access to the V5 repo.

## 1. SSH key + admin user (run as root)

```bash
# Provision the VPS (one-shot bootstrap).
adduser --disabled-password --gecos "" v5
usermod -aG sudo v5
mkdir -p /home/v5/.ssh
cp ~/.ssh/authorized_keys /home/v5/.ssh/
chown -R v5:v5 /home/v5/.ssh && chmod 700 /home/v5/.ssh
chmod 600 /home/v5/.ssh/authorized_keys

# Disable root SSH + password auth.
sed -i 's/^#\?PermitRootLogin.*/PermitRootLogin no/' /etc/ssh/sshd_config
sed -i 's/^#\?PasswordAuthentication.*/PasswordAuthentication no/' /etc/ssh/sshd_config
systemctl restart ssh

# Firewall: allow SSH only.
apt update && apt install -y ufw
ufw default deny incoming
ufw default allow outgoing
ufw allow 22/tcp
ufw --force enable

# unattended security upgrades + ntp.
apt install -y unattended-upgrades systemd-timesyncd
dpkg-reconfigure -f noninteractive unattended-upgrades
timedatectl set-ntp true
```

Reconnect as ``v5`` for the rest of the guide.

## 2. System dependencies (as v5, with sudo)

```bash
sudo apt update
sudo apt install -y \
    build-essential pkg-config libssl-dev curl git \
    python3.12 python3.12-venv python3.12-dev \
    logrotate jq

# Allow user services to keep running after logout.
sudo loginctl enable-linger v5
```

If your distro does not ship ``python3.12``, use deadsnakes
(Ubuntu) or the official ``python.org`` source build. The orchestrator
**requires** 3.12 - 3.11 will fail at import time.

## 3. Rust toolchain (only needed if rebuilding ``des_core``)

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
source $HOME/.cargo/env
rustc --version
```

## 4. Clone, venv, install

```bash
cd $HOME
git clone <YOUR-REPO-URL> v5-source
mv v5-source/V5 v5     # the orchestrator lives at $HOME/v5
cd v5

# Build venv + install runtime + dev (so maturin is available).
./setup.sh --dev

# Smoke test offline.
BYBIT_OFFLINE=1 ./.venv/bin/python -m pytest -q --timeout=120
```

The ``--dev`` flag also runs ``maturin develop --release`` and builds
``des_core`` into the venv. If you do not need the Rust kernels (the
Python fallback is used automatically), drop ``--dev``.

## 5. Configure secrets

```bash
cp .env.example .env
chmod 600 .env
nano .env
```

At minimum set:

```
OPENROUTER_API_KEY=sk-or-...
DATA_ROOT=/home/v5/v5/data
LOG_LEVEL=INFO
```

Leave ``BYBIT_OFFLINE`` blank in production. Set ``AI_DRY_RUN=1`` for
the first 24h of paper-trading dry-runs to avoid burning the API key
while you observe.

## 6. systemd --user units

Copy and enable the units:

```bash
mkdir -p $HOME/.config/systemd/user
cp deploy/systemd/v5-trader.service \
   deploy/systemd/v5-api.service \
   deploy/systemd/v5-health.service \
   deploy/systemd/v5-health.timer \
   $HOME/.config/systemd/user/

systemctl --user daemon-reload
systemctl --user enable --now v5-trader.service
systemctl --user enable --now v5-api.service
systemctl --user enable --now v5-health.timer
```

Verify:

```bash
systemctl --user status v5-trader v5-api v5-health.timer
journalctl --user -u v5-trader -n 100 --no-pager
journalctl --user -u v5-api -n 50 --no-pager
```

The dashboard is now live on ``127.0.0.1:8765`` **inside** the VPS.

## 7. Reach the dashboard from your laptop

The API binds to loopback only. Forward it over SSH:

```bash
# On your laptop:
ssh -N -L 8765:127.0.0.1:8765 v5@<VPS-IP>
```

Then open ``http://127.0.0.1:8765`` in a browser. **Never** publish
the dashboard on a public port - it has no auth, no rate limiting,
and exposes prompt audit lines.

## 8. Log rotation

The systemd journal handles its own rotation. The structured JSON
logs the orchestrator writes under ``$HOME/v5/data/logs/`` need an
extra step:

```bash
# Edit the path inside deploy/logrotate.v5 to match your $HOME, then:
crontab -e
# Add:
*/60 * * * * /usr/sbin/logrotate -s $HOME/.cache/v5-logrotate.state $HOME/v5/deploy/logrotate.v5
```

JSONL audit files (``triggers.jsonl``, ``intents.jsonl``,
``fills.jsonl``, ``reviews.jsonl``, ``prompts.jsonl``) are
**intentionally not rotated** - they are the canonical history and a
single 7-day run weighs < 50 MB.

## 9. Health check + alerting

The timer at ``v5-health.timer`` runs the probe every 5 minutes (see
``scripts/health_check.py``). Tail it:

```bash
journalctl --user -u v5-health -n 50 --no-pager
```

To get notified on FAIL, drop in a hook script:

```bash
cat > $HOME/.local/bin/v5-health-alert <<'SH'
#!/bin/bash
out=$($HOME/v5/.venv/bin/python $HOME/v5/scripts/health_check.py \
    --services v5-trader v5-api \
    --api-url http://127.0.0.1:8765 \
    --format json)
status=$?
if [ "$status" -ne 0 ]; then
    # Replace with your alerting transport (curl Slack/Telegram/email).
    echo "$out" | mail -s "v5 health $status" you@example.com
fi
SH
chmod +x $HOME/.local/bin/v5-health-alert
```

then point cron at it instead of the systemd timer if you want
e-mail / webhook alerting.

## 10. Routine ops

| Task | Command |
|---|---|
| restart trader | ``systemctl --user restart v5-trader`` |
| tail trader logs | ``journalctl --user -u v5-trader -f`` |
| pull update + restart | ``cd $HOME/v5 && git pull && ./setup.sh && systemctl --user restart v5-trader v5-api`` |
| run health probe ad hoc | ``./.venv/bin/python scripts/health_check.py --services v5-trader v5-api --api-url http://127.0.0.1:8765`` |
| see today's budget | ``cat $HOME/v5/data/runs/$(ls -t $HOME/v5/data/runs/ \| head -1)/budget.json`` |
| flatten everything | ``systemctl --user stop v5-trader`` (broker writes ``portfolio.json`` on shutdown via systemd ``TimeoutStopSec``) |
| disable for maintenance | ``systemctl --user disable --now v5-trader v5-api v5-health.timer`` |
| disk usage of data tree | ``du -sh $HOME/v5/data/*`` |
| replay last 7d fills | ``./.venv/bin/python -c "from portfolio.state import read_fills, replay_from_fills; ..."`` |

## 11. Backups

The only state worth backing up is ``$HOME/v5/data/runs/``. Daily
rsync is enough:

```bash
crontab -e
# Add (replace destination as needed):
30 3 * * * rsync -az --delete $HOME/v5/data/runs/ backup@bak:/srv/v5-runs/
```

Models, caches and Rust artefacts are reproducible from the repo and
do not need backup.

## 12. Updating

```bash
cd $HOME/v5
git fetch && git pull
./setup.sh --dev          # picks up new deps + rebuilds des_core if needed
./.venv/bin/python -m pytest -q --timeout=120     # offline regression
systemctl --user restart v5-trader v5-api
```

If the new release bumps ``FeatureConfig.version``, the cache is
invalidated automatically (the version is part of every cache key).
``budget.json`` and ``portfolio.json`` are forward-compatible by
design.

## 13. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| ``ModuleNotFoundError: des_core`` | Rust extension not built | ``./setup.sh --dev`` |
| ``OPENROUTER_API_KEY`` empty in audit | ``EnvironmentFile`` not loaded | ``ls -la $HOME/v5/.env``; reload unit (``systemctl --user daemon-reload``) |
| trader restarts in a loop | check ``journalctl --user -u v5-trader -n 200`` and the data tree for ``budget.json`` corruption | restore from ``budget.json.bak`` |
| dashboard 503 / blank | ``v5-api`` not running OR ``cfg.run_root`` empty | run ``v5-trader`` first; new run dirs appear within 30 min |
| snapshot probe FAIL | scanner stuck or rate-limited by Bybit | ``journalctl --user -u v5-trader -f`` and look for 429 / clock-drift warnings |
| clock drift > 2s | systemd-timesyncd disabled | ``timedatectl set-ntp true && systemctl restart systemd-timesyncd`` |

## 14. Decommissioning

```bash
systemctl --user disable --now v5-trader v5-api v5-health.timer
loginctl disable-linger v5
sudo userdel -r v5            # nukes $HOME/v5 and all data
```

If you only want to pause:

```bash
systemctl --user stop v5-trader v5-api v5-health.timer
```

State is persisted; restart resumes at the next 30-minute scanner
boundary.
