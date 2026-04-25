"""Portfolio state: cached projection of the fill stream.

`portfolio.json` is the canonical cached projection. The fill log
(`fills.jsonl`) is the source of truth — :func:`replay_from_fills`
can rebuild the state from scratch and any cache mismatch is logged
and the cache is rewritten.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from core.time import now_utc, to_utc
from portfolio.broker import Fill

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------
@dataclass
class PortfolioState:
    as_of: pd.Timestamp
    equity_usd: float
    cash_usd: float
    realized_pnl_usd: float
    fees_paid_usd: float
    open_positions: list[dict] = field(default_factory=list)
    closed_positions_24h: int = 0
    watchlist: list[str] = field(default_factory=list)
    risk_multiplier: float = 1.0
    loser_streak: int = 0

    def to_record(self) -> dict:
        d = asdict(self)
        d["as_of"] = self.as_of.isoformat()
        return d


# ---------------------------------------------------------------------
def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True),
                   encoding="utf-8")
    if path.exists():
        bak = path.with_suffix(path.suffix + ".bak")
        try:
            shutil.copy2(path, bak)
        except OSError:  # noqa: PERF203 - best-effort
            pass
    os.replace(tmp, path)


def save_state(state: PortfolioState, path: Path) -> None:
    """Atomic JSON write with a one-version `.bak` sidecar."""
    _atomic_write_json(Path(path), state.to_record())


def load_state(path: Path) -> PortfolioState | None:
    p = Path(path)
    if not p.exists():
        return None
    raw = json.loads(p.read_text(encoding="utf-8"))
    return PortfolioState(
        as_of=to_utc(raw["as_of"]),
        equity_usd=float(raw["equity_usd"]),
        cash_usd=float(raw["cash_usd"]),
        realized_pnl_usd=float(raw["realized_pnl_usd"]),
        fees_paid_usd=float(raw["fees_paid_usd"]),
        open_positions=list(raw.get("open_positions") or []),
        closed_positions_24h=int(raw.get("closed_positions_24h") or 0),
        watchlist=list(raw.get("watchlist") or []),
        risk_multiplier=float(raw.get("risk_multiplier") or 1.0),
        loser_streak=int(raw.get("loser_streak") or 0),
    )


# ---------------------------------------------------------------------
def append_fill(fills_path: Path, fill: Fill) -> None:
    """Append a single fill record to ``fills.jsonl`` (UTF-8, one line)."""
    p = Path(fills_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(fill.to_record(), sort_keys=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def read_fills(fills_path: Path) -> list[dict]:
    p = Path(fills_path)
    if not p.exists():
        return []
    out: list[dict] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            log.warning("skipping malformed fills.jsonl line: %r", line[:80])
    return out


# ---------------------------------------------------------------------
def replay_from_fills(
    fills: Iterable[dict],
    *,
    starting_equity_usd: float,
    now: pd.Timestamp | None = None,
) -> PortfolioState:
    """Reconstruct ``PortfolioState`` purely from a fill stream.

    Open positions are detected by tracking ``position_id`` running
    qty: opened on ``entry``, decreased by tp1/tp2/stop/time/manual.
    """
    cash = float(starting_equity_usd)
    realized = 0.0
    fees = 0.0
    losers = 0
    last_close_ts: pd.Timestamp | None = None

    # Per-position running state (qty + entry meta).
    open_meta: dict[str, dict] = {}
    closed_24h_anchor: list[pd.Timestamp] = []

    for raw in fills:
        kind = raw["kind"]
        pid = raw["position_id"]
        qty = float(raw["qty"])
        price = float(raw["price"])
        fee = float(raw.get("fee_usd") or 0.0)
        pnl = float(raw.get("pnl_usd") or 0.0)
        side = raw["side"]
        ts = to_utc(raw["ts"])

        cash -= fee
        fees += fee

        if kind == "entry":
            # Cash impact of the entry: notional moves from cash into the
            # position on the long side (and is borrowed on shorts);
            # paper-trading models margin via realized pnl + cash so we
            # only pay the fee here. Notional exposure is tracked via
            # ``open_positions``.
            open_meta[pid] = {
                "position_id": pid,
                "intent_id": raw.get("intent_id", ""),
                "symbol": raw["symbol"],
                "side": side,
                "entry_price": price,
                "initial_qty": qty,
                "remaining_qty": qty,
                "opened_at": ts.isoformat(),
            }
            continue

        cash += pnl
        realized += pnl

        meta = open_meta.get(pid)
        if meta is not None:
            meta["remaining_qty"] = max(0.0, meta["remaining_qty"] - qty)
            if meta["remaining_qty"] <= 0:
                # Position fully closed. The ``kind != "tp1"`` filter is
                # implicit: with the default 50% scale-out a tp1 fill
                # never drives ``remaining_qty`` to zero, and at 100%
                # scale-out we DO want the close to count.
                if last_close_ts is None or ts > last_close_ts:
                    last_close_ts = ts
                if pnl < 0:
                    losers += 1
                elif pnl > 0:
                    losers = 0
                closed_24h_anchor.append(ts)
                open_meta.pop(pid, None)

    as_of = now or now_utc()
    cutoff = as_of - pd.Timedelta(hours=24)
    closed_24h = sum(1 for ts in closed_24h_anchor if ts >= cutoff)

    # Equity = cash + unrealized. We don't have last marks here (that
    # belongs to the broker), so unrealized = 0 in pure replay.
    equity = cash
    return PortfolioState(
        as_of=as_of,
        equity_usd=equity,
        cash_usd=cash,
        realized_pnl_usd=realized,
        fees_paid_usd=fees,
        open_positions=list(open_meta.values()),
        closed_positions_24h=closed_24h,
        risk_multiplier=0.5 if losers >= 2 else 1.0,
        loser_streak=losers,
    )


__all__ = [
    "PortfolioState",
    "append_fill",
    "load_state",
    "read_fills",
    "replay_from_fills",
    "save_state",
]
