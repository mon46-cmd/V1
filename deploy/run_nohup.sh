#!/usr/bin/env bash
# Fallback runner for environments without systemd (containers, WSL,
# locked-down VMs). Starts the trader + API in the background under
# nohup. Auto-restart is NOT provided - prefer install.sh when systemd
# is available.
#
# Usage:
#     deploy/run_nohup.sh start
#     deploy/run_nohup.sh stop
#     deploy/run_nohup.sh status
#     deploy/run_nohup.sh tail        # follow trader.out
#
# PIDs land in data/run/*.pid, logs in data/logs/*.out.
set -euo pipefail

V5_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$V5_HOME"

PYBIN="$V5_HOME/.venv/bin/python"
LOG_DIR="$V5_HOME/data/logs"
PID_DIR="$V5_HOME/data/run"
mkdir -p "$LOG_DIR" "$PID_DIR"

trader_pidf="$PID_DIR/trader.pid"
api_pidf="$PID_DIR/api.pid"

is_alive() {
    local pidf="$1"
    [[ -f "$pidf" ]] || return 1
    local pid
    pid="$(cat "$pidf" 2>/dev/null || true)"
    [[ -n "$pid" ]] || return 1
    kill -0 "$pid" 2>/dev/null
}

start_one() {
    local name="$1" script="$2" pidf="$3"
    local extra=("${@:4}")
    if is_alive "$pidf"; then
        echo "[$name] already running (pid $(cat "$pidf"))"
        return 0
    fi
    if [[ ! -x "$PYBIN" ]]; then
        echo "[$name] FATAL: $PYBIN missing. Run ./setup.sh --dev first." >&2
        exit 1
    fi
    if [[ ! -f "$V5_HOME/.env" ]]; then
        echo "[$name] FATAL: $V5_HOME/.env missing. cp .env.example .env" >&2
        exit 1
    fi
    # Export .env and exec under nohup.
    set -a; source "$V5_HOME/.env"; set +a
    nohup "$PYBIN" "$script" "${extra[@]}" \
        > "$LOG_DIR/${name}.out" 2>&1 &
    echo $! > "$pidf"
    disown || true
    sleep 1
    if is_alive "$pidf"; then
        echo "[$name] started pid $(cat "$pidf")  log=$LOG_DIR/${name}.out"
    else
        echo "[$name] FAILED to start. Tail:"
        tail -20 "$LOG_DIR/${name}.out" >&2 || true
        rm -f "$pidf"
        exit 1
    fi
}

stop_one() {
    local name="$1" pidf="$2"
    if ! is_alive "$pidf"; then
        echo "[$name] not running"
        rm -f "$pidf"
        return 0
    fi
    local pid
    pid="$(cat "$pidf")"
    echo "[$name] stopping pid $pid ..."
    kill "$pid" 2>/dev/null || true
    for _ in 1 2 3 4 5 6 7 8 9 10; do
        is_alive "$pidf" || break
        sleep 1
    done
    if is_alive "$pidf"; then
        echo "[$name] still alive, sending SIGKILL"
        kill -9 "$pid" 2>/dev/null || true
    fi
    rm -f "$pidf"
    echo "[$name] stopped"
}

cmd="${1:-status}"
case "$cmd" in
    start)
        start_one trader "$V5_HOME/scripts/run_exec.py" "$trader_pidf"
        start_one api    "$V5_HOME/scripts/run_api.py"  "$api_pidf" \
            --host 127.0.0.1 --port 8765
        ;;
    stop)
        stop_one trader "$trader_pidf"
        stop_one api    "$api_pidf"
        ;;
    restart)
        "$0" stop
        sleep 1
        "$0" start
        ;;
    status)
        for pair in "trader:$trader_pidf" "api:$api_pidf"; do
            name="${pair%%:*}"; pidf="${pair##*:}"
            if is_alive "$pidf"; then
                echo "[$name] running pid $(cat "$pidf")"
            else
                echo "[$name] stopped"
            fi
        done
        ;;
    tail)
        tail -F "$LOG_DIR/trader.out"
        ;;
    *)
        echo "usage: $0 {start|stop|restart|status|tail}" >&2
        exit 2
        ;;
esac
