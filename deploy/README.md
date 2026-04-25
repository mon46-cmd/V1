# V5 Deploy Quickstart

This guide is written as literal copy-paste blocks.

All examples below assume the repo is cloned into `~/V1`, which matches the
current production workflow.

## 1. Base packages

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y build-essential git curl wget vim unzip pkg-config libssl-dev jq cmake autoconf automake
```

## 2. Rust toolchain

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
source $HOME/.cargo/env
```

## 3. `uv`

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env
export PATH="$HOME/.local/bin:$PATH"
source ~/.bashrc
```

## 4. Python 3.12.13

```bash
uv python install 3.12.13
sudo ln -sf $(uv python find 3.12.13) /usr/local/bin/python3.12
```

## 5. Clone and install the repo

```bash
git clone https://github.com/mon46-cmd/V1
sudo chown -R $USER:$USER ~/V1
cd ~/V1
chmod +x setup.sh
./setup.sh --dev
```

## 6. Create `.env`

```bash
cp .env.example .env
chmod 600 .env
nano .env
```

Minimum fields to set before starting services:

```dotenv
OPENROUTER_API_KEY=sk-or-...
BYBIT_API_KEY=...
BYBIT_API_SECRET=...
```

Leave `BYBIT_OFFLINE` blank for a live VPS. Set it to `1` only for smoke tests.

## 7. Validate the install offline

```bash
cd ~/V1
source .venv/bin/activate
BYBIT_OFFLINE=1 pytest --timeout=60 -q
```

Expected result:

```text
221 passed, 20 skipped
```

## 8. Check Bybit reachability from the VPS

```bash
curl -s -o /dev/null -w "%{http_code}\n" https://api.bybit.com/v5/market/time
```

Interpretation:

- `200`: region is usable, continue.
- `403`: the VPS region is blocked by Bybit/CloudFront. Stop here and move the VPS.

Known good regions:

- `asia-northeast1` (Tokyo)
- `asia-east1` (Taiwan)
- `europe-west4` (Netherlands)
- `europe-west3` (Frankfurt)

## 9. Install the services

The recommended path is the system-level installer. It renders absolute-path
units into `/etc/systemd/system/` and runs them as your normal user.

```bash
cd ~/V1
chmod +x deploy/install.sh deploy/uninstall.sh deploy/run_nohup.sh
sudo deploy/install.sh
```

## 10. Verify the services

```bash
systemctl status v5-trader.service v5-api.service --no-pager -l
journalctl -u v5-trader.service -n 50 --no-pager
curl -s http://127.0.0.1:8765/api/health | jq .
```

The health probe has three outcomes by design:

- exit `0`: PASS
- exit `1`: WARN
- exit `2`: FAIL

The shipped systemd health unit treats `1` as successful so a non-fatal warn
does not mark `v5-health.service` as failed.

## 11. Day-to-day operations

### Restart the trader and API

```bash
sudo systemctl restart v5-trader v5-api
```

### Stop everything

```bash
sudo systemctl stop v5-trader v5-api v5-health.timer
```

### Start everything again

```bash
sudo systemctl start v5-trader v5-api v5-health.timer
```

### Live tail the trader logs

```bash
journalctl -u v5-trader.service -f
```

### Remove the installed services

```bash
cd ~/V1
sudo deploy/uninstall.sh
```

## 12. Reach the API from your laptop

The API binds to loopback only. Tunnel it over SSH instead of opening a public port.

```bash
ssh -N -L 8765:127.0.0.1:8765 user@<vps-ip>
```

Then open `http://127.0.0.1:8765` on your laptop.

## 13. Log rotation

```bash
cd ~/V1
sed "s|__V5_HOME__|$HOME/V1|g" deploy/logrotate.v5 | sudo tee /etc/logrotate.d/v5 > /dev/null
sudo logrotate -d /etc/logrotate.d/v5
```

## 14. Fallback when systemd is unavailable

Use this only on hosts where normal systemd service management is not usable.

```bash
cd ~/V1
chmod +x deploy/run_nohup.sh
deploy/run_nohup.sh start
```

Check status and logs:

```bash
cd ~/V1
deploy/run_nohup.sh status
deploy/run_nohup.sh tail
```

Stop it:

```bash
cd ~/V1
deploy/run_nohup.sh stop
```

## 15. What not to do

- Do not use `systemctl --user` with the shipped units on cloud images.
- Do not publish port `8765` directly to the internet.
- Do not proceed if the Bybit reachability check returns `403`.
