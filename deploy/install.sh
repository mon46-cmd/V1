#!/usr/bin/env bash
# V5 systemd installer (system-level units, runs services as $USER).
#
# Usage (from inside the V5 clone, on the target server):
#     sudo deploy/install.sh
#
# That single command:
#   1. detects the install path ($PWD), the run-as user (the invoking
#      user, NOT root), and homedir
#   2. expands the .tmpl files in deploy/systemd/ with those values
#   3. drops the result into /etc/systemd/system/
#   4. daemon-reloads, enables, and starts the units
#
# To remove later: sudo deploy/uninstall.sh

set -euo pipefail

if [[ $EUID -ne 0 ]]; then
    echo "[install] re-running with sudo..." >&2
    exec sudo --preserve-env=USER,HOME "$0" "$@"
fi

# Identify the invoking (non-root) user.
RUN_USER="${SUDO_USER:-${USER:-}}"
if [[ -z "$RUN_USER" || "$RUN_USER" == "root" ]]; then
    echo "[install] ERROR: could not detect a non-root user to run V5 as." >&2
    echo "          Run this script with 'sudo' from your normal user shell." >&2
    exit 1
fi

RUN_GROUP="$(id -gn "$RUN_USER")"
HOME_DIR="$(getent passwd "$RUN_USER" | cut -d: -f6)"

# Locate the V5 clone: the deploy/ folder that contains this script.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
V5_HOME="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "[install] V5_HOME=$V5_HOME"
echo "[install] User=$RUN_USER  Group=$RUN_GROUP  Home=$HOME_DIR"

# Sanity: required files must exist before we touch systemd.
required=(
    "$V5_HOME/.venv/bin/python"
    "$V5_HOME/scripts/run_exec.py"
    "$V5_HOME/scripts/run_api.py"
    "$V5_HOME/scripts/health_check.py"
    "$V5_HOME/.env"
)
missing=0
for f in "${required[@]}"; do
    if [[ ! -e "$f" ]]; then
        echo "[install] MISSING: $f" >&2
        missing=1
    fi
done
if [[ $missing -ne 0 ]]; then
    echo "[install] ERROR: fix the missing files above and re-run." >&2
    echo "          (Run ./setup.sh --dev and 'cp .env.example .env' first.)" >&2
    exit 1
fi

# Make sure $RUN_USER can read .env (systemd reads it as that user).
if ! sudo -u "$RUN_USER" test -r "$V5_HOME/.env"; then
    echo "[install] WARN: $RUN_USER cannot read $V5_HOME/.env"
    echo "          Fixing permissions..."
    chown "$RUN_USER:$RUN_GROUP" "$V5_HOME/.env"
    chmod 0600 "$V5_HOME/.env"
fi

# Make sure data/ exists and is writable.
mkdir -p "$V5_HOME/data/logs" "$V5_HOME/data/runs"
chown -R "$RUN_USER:$RUN_GROUP" "$V5_HOME/data"

# Render the templates.
TARGET=/etc/systemd/system
units=(v5-trader.service v5-api.service v5-health.service v5-health.timer)
for unit in "${units[@]}"; do
    tmpl="$SCRIPT_DIR/systemd/${unit}.tmpl"
    out="$TARGET/$unit"
    if [[ ! -f "$tmpl" ]]; then
        echo "[install] ERROR: template missing: $tmpl" >&2
        exit 1
    fi
    sed -e "s|__V5_HOME__|$V5_HOME|g" \
        -e "s|__USER__|$RUN_USER|g" \
        -e "s|__GROUP__|$RUN_GROUP|g" \
        "$tmpl" > "$out"
    chmod 0644 "$out"
    echo "[install] wrote $out"
done

# Reload + enable.
systemctl daemon-reload
systemctl enable --now v5-trader.service
systemctl enable --now v5-api.service
systemctl enable --now v5-health.timer

echo
echo "[install] DONE."
echo "[install] Verify:"
echo "          systemctl status v5-trader.service v5-api.service --no-pager -l"
echo "          journalctl -u v5-trader.service -f"
echo "[install] Dashboard (loopback): curl -s http://127.0.0.1:8765/health | jq ."
echo "[install] To stop:  sudo systemctl stop v5-trader v5-api v5-health.timer"
echo "[install] To remove: sudo $V5_HOME/deploy/uninstall.sh"
