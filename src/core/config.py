"""Central configuration.

Values come from environment variables (optionally via a .env file)
with conservative defaults. Nothing here talks to the network; it is
just a typed, frozen container.

The single public entry point is `load_config()`. Call it once per
process and pass the result around explicitly.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    # dotenv is optional; env vars still work without it.
    pass


# V5/ folder (this file is at V5/src/core/config.py).
REPO_ROOT = Path(__file__).resolve().parents[2]


def _env_str(key: str, default: str = "") -> str:
    return os.getenv(key, default) or default


def _env_bool(key: str, default: bool = False) -> bool:
    v = os.getenv(key, "")
    if not v:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def _env_path(key: str, default: Path) -> Path:
    v = os.getenv(key, "")
    if not v:
        return default
    p = Path(v)
    if not p.is_absolute():
        p = (REPO_ROOT / p).resolve()
    return p


@dataclass(frozen=True)
class Config:
    # --- Paths ---
    repo_root: Path
    data_root: Path
    cache_root: Path
    feature_root: Path
    run_root: Path
    log_root: Path

    # --- Exchange ---
    category: str = "linear"
    quote_currency: str = "USDT"
    bybit_rest_base: str = "https://api.bybit.com"
    bybit_archive_base: str = "https://public.bybit.com/trading"

    # --- HTTP ---
    http_timeout_sec: float = 20.0
    http_rate_delay_sec: float = 0.10  # ~10 req/s
    http_max_retries: int = 5
    http_backoff_base_sec: float = 0.5

    # --- Universe filter ---
    universe_size: int = 30
    min_turnover_usd_24h: float = 20_000_000.0
    max_spread_bps: float = 10.0
    min_listing_age_days: int = 30
    min_price_usd: float = 0.0001
    exclude_symbols: tuple[str, ...] = field(default_factory=lambda: (
        "USDCUSDT", "USDEUSDT", "FDUSDUSDT", "TUSDUSDT", "DAIUSDT",
    ))
    exclude_substrings: tuple[str, ...] = field(default_factory=lambda: ("-",))

    # --- Timeframes ---
    watch_interval: str = "15"
    higher_tfs: tuple[str, ...] = ("60", "240")

    # --- Trigger + cooldown ---
    trigger_flags: tuple[str, ...] = field(default_factory=lambda: (
        "flag_volume_climax", "flag_sweep_up", "flag_sweep_dn",
    ))
    prompt_cooldown_candles: int = 3
    cooldown_bypass_atr_mult: float = 0.8
    cooldown_bypass_floor_pct: float = 0.01

    # --- AI ---
    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    # Watchlist (Prompt A) -- "regular super grok" -- heaviest reasoning, picks pairs.
    model_watchlist: str = "x-ai/grok-4.20-multi-agent"
    # Deep (Prompt B) + Review (Prompt C) -- grok-4.1-fast: cheap, fast, deterministic.
    model_deep: str = "x-ai/grok-4.1-fast"
    model_review: str = "x-ai/grok-4.1-fast"
    openrouter_referer: str = ""
    openrouter_title: str = "v5-paper-orchestrator"
    ai_timeout_sec: float = 90.0
    ai_budget_usd_per_day: float = 3.0

    # --- Paper broker ---
    paper_equity_usd: float = 10_000.0
    per_trade_risk_pct: float = 0.01
    max_concurrent_positions: int = 3
    taker_fee_bps: float = 6.0
    slippage_bps: float = 2.0
    tp1_scale_out_pct: float = 0.5

    # --- Runtime flags ---
    bybit_offline: bool = False
    ai_dry_run: bool = False
    openrouter_live: bool = False
    log_level: str = "INFO"

    # --- Feature engine ---
    feature_version: str = "v0.0.1"
    snapshot_concurrency: int = 6


def load_config() -> Config:
    """Build the frozen Config and ensure the data tree exists."""
    data_root = _env_path("DATA_ROOT", REPO_ROOT / "data")
    cache_root = data_root / "cache"
    feature_root = data_root / "features"
    run_root = data_root / "runs"
    log_root = data_root / "logs"

    cfg = Config(
        repo_root=REPO_ROOT,
        data_root=data_root,
        cache_root=cache_root,
        feature_root=feature_root,
        run_root=run_root,
        log_root=log_root,
        openrouter_api_key=_env_str("OPENROUTER_API_KEY"),
        model_watchlist=_env_str("MODEL_WATCHLIST", "x-ai/grok-4.20-multi-agent"),
        model_deep=_env_str("MODEL_DEEP", "x-ai/grok-4.1-fast"),
        model_review=_env_str("MODEL_REVIEW", "x-ai/grok-4.1-fast"),
        openrouter_referer=_env_str("OPENROUTER_REFERER"),
        openrouter_title=_env_str("OPENROUTER_TITLE", "v5-paper-orchestrator"),
        bybit_offline=_env_bool("BYBIT_OFFLINE"),
        ai_dry_run=_env_bool("AI_DRY_RUN"),
        openrouter_live=_env_bool("OPENROUTER_LIVE"),
        log_level=_env_str("LOG_LEVEL", "INFO").upper(),
    )
    from .paths import ensure_dirs
    ensure_dirs(cfg)
    return cfg
