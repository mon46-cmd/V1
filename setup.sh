#!/usr/bin/env bash
# Phase 0 bootstrap for Linux (Debian/Ubuntu target).
#
# Usage:
#   ./setup.sh          # runtime install
#   ./setup.sh --dev    # runtime + dev deps + build rust extension
#
# Prereqs:
#   - python3.12 on PATH
#   - rustup + cargo (required for --dev)

set -euo pipefail
cd "$(dirname "$0")"

log() { printf '[setup] %s\n' "$*"; }
fail() { printf '[setup] %s\n' "$*" >&2; exit 1; }

DEV=0
for arg in "$@"; do
    case "$arg" in
        --dev|-Dev) DEV=1 ;;
    esac
done

command -v python3.12 >/dev/null 2>&1 || fail "python3.12 not found on PATH."
log "Python 3.12 found."

if [ ! -d .venv ]; then
    log "Creating .venv with python3.12 ..."
    python3.12 -m venv .venv
fi

PY=".venv/bin/python"
"$PY" -m pip install --upgrade pip wheel setuptools >/dev/null

if [ "$DEV" = "1" ]; then
    log "Installing runtime + dev dependencies ..."
    "$PY" -m pip install -r requirements-dev.txt
else
    log "Installing runtime dependencies ..."
    "$PY" -m pip install -r requirements.txt
fi

log "Installing V5 package editable ..."
"$PY" -m pip install -e . --no-deps

if [ "$DEV" = "1" ]; then
    command -v cargo >/dev/null 2>&1 || fail "cargo not found. Install rustup and re-run with --dev."
    log "Building des_core with maturin (release) ..."
    (cd rust_core && "$(pwd)/../$PY" -m maturin develop --release)
fi

log "Done. Activate the venv with: source .venv/bin/activate"
