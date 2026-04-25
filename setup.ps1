# Phase 0 bootstrap for Windows (PowerShell 5.1+).
#
# Usage:
#   ./setup.ps1          # runtime install
#   ./setup.ps1 -Dev     # runtime + dev deps + build rust extension
#
# Prereqs (checked below):
#   - Python 3.12 via the `py -3.12` launcher
#   - Rust toolchain (rustup + cargo) with the msvc default
#   - MSVC "Desktop development with C++" workload (Build Tools 2022)

[CmdletBinding()]
param(
    [switch]$Dev
)

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

function Fail($msg) {
    Write-Host "[setup] $msg" -ForegroundColor Red
    exit 1
}

function Info($msg) {
    Write-Host "[setup] $msg" -ForegroundColor Cyan
}

# --- Python 3.12 check ---
try {
    $pyver = & py -3.12 -c "import sys; print('%d.%d' % sys.version_info[:2])"
} catch {
    Fail "Python 3.12 launcher not found. Install Python 3.12 from python.org and retry."
}
if ($pyver.Trim() -ne "3.12") {
    Fail "py -3.12 reports version $pyver, expected 3.12."
}
Info "Python 3.12 found."

# --- venv ---
if (-not (Test-Path .venv)) {
    Info "Creating .venv with py -3.12 ..."
    & py -3.12 -m venv .venv
}
$python = Join-Path $PSScriptRoot ".venv/Scripts/python.exe"
if (-not (Test-Path $python)) { Fail "venv python not found at $python" }

# --- upgrade pip and install deps ---
& $python -m pip install --upgrade pip wheel setuptools | Out-Null
if ($Dev) {
    Info "Installing runtime + dev dependencies ..."
    & $python -m pip install -r requirements-dev.txt
} else {
    Info "Installing runtime dependencies ..."
    & $python -m pip install -r requirements.txt
}

# --- install the package editable ---
Info "Installing V5 package editable ..."
& $python -m pip install -e . --no-deps

# --- Rust extension (dev only) ---
if ($Dev) {
    $cargo = Get-Command cargo -ErrorAction SilentlyContinue
    if (-not $cargo) {
        Fail "cargo not found on PATH. Install rustup (https://rustup.rs) and re-run with -Dev."
    }
    $rustup = Get-Command rustup -ErrorAction SilentlyContinue
    if ($rustup) {
        Info "Pinning rust toolchain to stable-msvc ..."
        & rustup default stable-msvc | Out-Null
    }
    Info "Building des_core with maturin (release) ..."
    Push-Location rust_core
    try {
        & $python -m maturin develop --release
    } finally {
        Pop-Location
    }
}

Info "Done. Activate the venv with: .\\.venv\\Scripts\\Activate.ps1"
