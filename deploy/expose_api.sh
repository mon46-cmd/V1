#!/usr/bin/env bash
set -euo pipefail

V5_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$V5_HOME"

HOST="${1:-0.0.0.0}"
PORT="${2:-8765}"

if [[ ! -f .env ]]; then
    echo "[expose_api] .env missing. Run: cp .env.example .env" >&2
    exit 1
fi

upsert() {
    local key="$1"
    local value="$2"
    if grep -q "^${key}=" .env; then
        sed -i "s|^${key}=.*|${key}=${value}|" .env
    else
        printf "\n%s=%s\n" "$key" "$value" >> .env
    fi
}

upsert API_HOST "$HOST"
upsert API_PORT "$PORT"

echo "[expose_api] wrote API_HOST=$HOST API_PORT=$PORT to $V5_HOME/.env"
echo "[expose_api] restart with: sudo systemctl restart v5-api.service"
echo "[expose_api] if UFW is enabled: sudo ufw allow ${PORT}/tcp"
echo "[expose_api] also open TCP ${PORT} in your cloud firewall/security group"
echo "[expose_api] then browse: http://<VPS-IP>:${PORT}/"