#!/usr/bin/env bash
# Removes the V5 systemd units installed by deploy/install.sh.
# Leaves data/ and the clone alone.
set -euo pipefail

if [[ $EUID -ne 0 ]]; then
    exec sudo "$0" "$@"
fi

units=(v5-trader.service v5-api.service v5-health.service v5-health.timer)

systemctl disable --now "${units[@]}" 2>/dev/null || true

for u in "${units[@]}"; do
    rm -f "/etc/systemd/system/$u"
done

systemctl daemon-reload
systemctl reset-failed 2>/dev/null || true

echo "[uninstall] V5 systemd units removed."
echo "[uninstall] Data tree and clone untouched."
