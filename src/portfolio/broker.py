"""Paper broker: positions, fills, intrabar fill rules.

The broker is bar-driven for paper-trading: feed it 15m OHLC bars
through :meth:`Broker.on_bar` and it will simulate worst-case
intrabar SL-before-TP, TP1 scale-out + break-even, TP2 (or implied
runner), and the time stop. Each call returns the list of
:class:`Fill` events emitted for that bar.

The broker is deliberately a single, single-threaded mutator: every
state change goes through ``open_from_intent``, ``on_bar``, or
``close_position`` and produces an append-only stream of fills. The
fill log is the source of truth; the in-memory state is a cached
projection that can always be rebuilt by replaying the log.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

import pandas as pd

from core.ids import ulid
from portfolio.intents import Intent, IntentStatus

log = logging.getLogger(__name__)


FillKind = Literal[
    "entry", "tp1", "tp2", "stop", "time", "manual", "funding",
]
Side = Literal["long", "short"]


# ---------------------------------------------------------------------
@dataclass(frozen=True)
class Bar:
    """OHLCV bar, primary timeframe (15m)."""

    ts: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    symbol: str = ""


@dataclass(frozen=True)
class Fill:
    """Append-only fill event."""

    fill_id: str
    ts: pd.Timestamp
    symbol: str
    side: Side
    kind: FillKind
    price: float
    qty: float           # absolute, base units
    fee_usd: float
    pnl_usd: float       # realized; 0 on entry/funding
    intent_id: str
    position_id: str

    def to_record(self) -> dict:
        return {
            "fill_id": self.fill_id,
            "ts": self.ts.isoformat(),
            "symbol": self.symbol,
            "side": self.side,
            "kind": self.kind,
            "price": self.price,
            "qty": self.qty,
            "fee_usd": self.fee_usd,
            "pnl_usd": self.pnl_usd,
            "intent_id": self.intent_id,
            "position_id": self.position_id,
        }


@dataclass
class Position:
    """Open position book entry."""

    position_id: str
    intent_id: str
    symbol: str
    side: Side
    entry_price: float
    initial_qty: float
    remaining_qty: float
    stop_loss: float            # mutated to BE after TP1
    take_profit_1: float
    take_profit_2: float
    time_horizon_bars: int
    bars_held: int = 0
    tp1_filled: bool = False
    be_armed: bool = False      # SL has been moved to entry
    opened_at: pd.Timestamp | None = None

    def is_long(self) -> bool:
        return self.side == "long"


# ---------------------------------------------------------------------
@dataclass
class BrokerConfig:
    """Tunables. Defaults match ``docs/04_EXECUTION.md`` section 3."""

    taker_fee_bps: float = 6.0
    entry_slippage_bps: float = 2.0   # touch + close_*; 0 for breakout
    tp1_scale_out_pct: float = 0.5
    move_sl_to_be_after_tp1: bool = True


def _bps(x: float) -> float:
    return x * 1e-4


def _implied_tp2(entry: float, tp1: float) -> float:
    """Symmetric runner target: tp2 = tp1 + (tp1 - entry)."""
    return tp1 + (tp1 - entry)


def _entry_slippage_bps(intent: Intent, cfg: BrokerConfig) -> float:
    if intent.activation_kind == "breakout":
        return 0.0
    return cfg.entry_slippage_bps


# ---------------------------------------------------------------------
@dataclass
class Broker:
    """Bar-driven paper broker.

    Holds open positions in memory; emits :class:`Fill` events. Equity
    bookkeeping (cash, realized pnl, fees) is the responsibility of
    :mod:`portfolio.state` which projects the fill stream.
    """

    cfg: BrokerConfig = field(default_factory=BrokerConfig)
    positions: dict[str, Position] = field(default_factory=dict)

    # ---- public API --------------------------------------------------
    def open_from_intent(
        self,
        intent: Intent,
        *,
        fill_price: float,
        fill_ts: pd.Timestamp,
        qty: float | None = None,
    ) -> tuple[Position, Fill]:
        """Open a new position from an activated intent.

        Returns ``(position, entry_fill)``. The fill is annotated with
        an entry slippage adjustment (worse for the trader). ``qty``
        defaults to ``intent.qty``.
        """
        if intent.status != IntentStatus.ACTIVATED:
            raise ValueError(f"intent {intent.intent_id} not activated")
        q = float(qty if qty is not None else intent.qty)
        if q <= 0:
            raise ValueError("qty must be > 0")

        slip = _bps(_entry_slippage_bps(intent, self.cfg))
        # Worse fill direction: long pays up, short receives less.
        if intent.side == "long":
            adj_price = fill_price * (1.0 + slip)
        else:
            adj_price = fill_price * (1.0 - slip)

        tp1 = float(intent.take_profit_1)
        tp2 = (float(intent.take_profit_2)
               if intent.take_profit_2 is not None
               else _implied_tp2(adj_price, tp1))

        position_id = ulid()
        pos = Position(
            position_id=position_id,
            intent_id=intent.intent_id,
            symbol=intent.symbol,
            side=intent.side,  # type: ignore[arg-type]
            entry_price=adj_price,
            initial_qty=q,
            remaining_qty=q,
            stop_loss=float(intent.stop_loss),
            take_profit_1=tp1,
            take_profit_2=tp2,
            time_horizon_bars=int(intent.time_horizon_bars),
            opened_at=fill_ts,
        )
        self.positions[position_id] = pos

        fee = _bps(self.cfg.taker_fee_bps) * adj_price * q
        fill = Fill(
            fill_id=ulid(),
            ts=fill_ts,
            symbol=intent.symbol,
            side=pos.side,
            kind="entry",
            price=adj_price,
            qty=q,
            fee_usd=fee,
            pnl_usd=0.0,
            intent_id=intent.intent_id,
            position_id=position_id,
        )
        return pos, fill

    def close_position(
        self,
        position_id: str,
        *,
        price: float,
        ts: pd.Timestamp,
        kind: FillKind = "manual",
    ) -> Fill | None:
        """Close the remaining qty at ``price``. Idempotent."""
        pos = self.positions.get(position_id)
        if pos is None or pos.remaining_qty <= 0:
            return None
        fill = self._exit_fill(pos, price=price, qty=pos.remaining_qty,
                               kind=kind, ts=ts)
        pos.remaining_qty = 0.0
        del self.positions[position_id]
        return fill

    def close_all(self, *, price_map: dict[str, float], ts: pd.Timestamp,
                  kind: FillKind = "manual") -> list[Fill]:
        """Close every open position at the given per-symbol price."""
        fills: list[Fill] = []
        for pid in list(self.positions.keys()):
            pos = self.positions[pid]
            px = price_map.get(pos.symbol)
            if px is None:
                continue
            f = self.close_position(pid, price=px, ts=ts, kind=kind)
            if f is not None:
                fills.append(f)
        return fills

    def on_bar(self, bar: Bar) -> list[Fill]:
        """Process one 15m bar. Returns the fills it generated.

        Ordering rules (worst-case for the trader):

        1. Open against gap: if open is past SL, exit at open.
        2. Open with gap: if open past TP1, fill TP1 at open; then
           re-check the same bar's range for TP2 / new BE stop.
        3. Within-bar both-touched: SL wins.
        4. After TP1, SL moves to ``entry`` (BE) if configured.
        5. Time stop: when ``bars_held >= time_horizon_bars`` after the
           bar's other rules, close at the bar's close.
        """
        fills: list[Fill] = []
        for pid in list(self.positions.keys()):
            pos = self.positions.get(pid)
            if pos is None or pos.symbol != bar.symbol:
                continue
            fills.extend(self._process_bar_for_position(pos, bar))
        return fills

    # ---- internals ---------------------------------------------------
    def _process_bar_for_position(self, pos: Position, bar: Bar) -> list[Fill]:
        out: list[Fill] = []

        # 1 + 2: gap evaluation against the bar's open.
        gap_fills = self._evaluate_gap(pos, bar)
        out.extend(gap_fills)
        if pos.remaining_qty <= 0:
            self.positions.pop(pos.position_id, None)
            pos.bars_held += 1
            return out

        # 3 + 4: within-bar evaluation.
        intra = self._evaluate_intrabar(pos, bar)
        out.extend(intra)
        if pos.remaining_qty <= 0:
            self.positions.pop(pos.position_id, None)
            pos.bars_held += 1
            return out

        # 5: time stop. ``bars_held`` increments once per processed bar.
        pos.bars_held += 1
        if pos.bars_held >= pos.time_horizon_bars:
            out.append(self._exit_fill(pos, price=bar.close,
                                       qty=pos.remaining_qty,
                                       kind="time", ts=bar.ts))
            pos.remaining_qty = 0.0
            self.positions.pop(pos.position_id, None)
        return out

    # ---- gap handling -------------------------------------------------
    def _evaluate_gap(self, pos: Position, bar: Bar) -> list[Fill]:
        out: list[Fill] = []
        if pos.is_long():
            # Gap through SL.
            if bar.open <= pos.stop_loss:
                out.append(self._exit_fill(pos, price=bar.open,
                                           qty=pos.remaining_qty,
                                           kind="stop", ts=bar.ts))
                pos.remaining_qty = 0.0
                return out
            # Gap through TP1 (and possibly TP2).
            if not pos.tp1_filled and bar.open >= pos.take_profit_1:
                out.extend(self._fill_tp1(pos, price=bar.open, ts=bar.ts))
                if pos.remaining_qty > 0 and bar.open >= pos.take_profit_2:
                    out.append(self._exit_fill(pos, price=bar.open,
                                               qty=pos.remaining_qty,
                                               kind="tp2", ts=bar.ts))
                    pos.remaining_qty = 0.0
            return out

        # short
        if bar.open >= pos.stop_loss:
            out.append(self._exit_fill(pos, price=bar.open,
                                       qty=pos.remaining_qty,
                                       kind="stop", ts=bar.ts))
            pos.remaining_qty = 0.0
            return out
        if not pos.tp1_filled and bar.open <= pos.take_profit_1:
            out.extend(self._fill_tp1(pos, price=bar.open, ts=bar.ts))
            if pos.remaining_qty > 0 and bar.open <= pos.take_profit_2:
                out.append(self._exit_fill(pos, price=bar.open,
                                           qty=pos.remaining_qty,
                                           kind="tp2", ts=bar.ts))
                pos.remaining_qty = 0.0
        return out

    # ---- intrabar handling -------------------------------------------
    def _evaluate_intrabar(self, pos: Position, bar: Bar) -> list[Fill]:
        out: list[Fill] = []
        if pos.is_long():
            sl_hit = bar.low <= pos.stop_loss
            tp1_reachable = (not pos.tp1_filled) and bar.high >= pos.take_profit_1
            tp2_reachable = pos.tp1_filled and bar.high >= pos.take_profit_2

            # Worst case: if SL is touched in the same bar, SL fires first.
            if sl_hit:
                out.append(self._exit_fill(pos, price=pos.stop_loss,
                                           qty=pos.remaining_qty,
                                           kind="stop", ts=bar.ts))
                pos.remaining_qty = 0.0
                return out

            if tp1_reachable:
                out.extend(self._fill_tp1(pos, price=pos.take_profit_1,
                                          ts=bar.ts))
                # After TP1, BE arms; if TP2 also touched, BE wins by spec.
                if pos.remaining_qty > 0 and bar.high >= pos.take_profit_2:
                    if pos.be_armed and bar.low <= pos.stop_loss:
                        # BE before TP2 (worst case).
                        out.append(self._exit_fill(pos, price=pos.stop_loss,
                                                   qty=pos.remaining_qty,
                                                   kind="stop", ts=bar.ts))
                        pos.remaining_qty = 0.0
                    else:
                        out.append(self._exit_fill(pos, price=pos.take_profit_2,
                                                   qty=pos.remaining_qty,
                                                   kind="tp2", ts=bar.ts))
                        pos.remaining_qty = 0.0
                return out

            if tp2_reachable:
                # Already past TP1 from a prior bar; BE may already be sl.
                if bar.low <= pos.stop_loss:
                    out.append(self._exit_fill(pos, price=pos.stop_loss,
                                               qty=pos.remaining_qty,
                                               kind="stop", ts=bar.ts))
                    pos.remaining_qty = 0.0
                else:
                    out.append(self._exit_fill(pos, price=pos.take_profit_2,
                                               qty=pos.remaining_qty,
                                               kind="tp2", ts=bar.ts))
                    pos.remaining_qty = 0.0
            return out

        # short symmetric
        sl_hit = bar.high >= pos.stop_loss
        tp1_reachable = (not pos.tp1_filled) and bar.low <= pos.take_profit_1
        tp2_reachable = pos.tp1_filled and bar.low <= pos.take_profit_2

        if sl_hit:
            out.append(self._exit_fill(pos, price=pos.stop_loss,
                                       qty=pos.remaining_qty,
                                       kind="stop", ts=bar.ts))
            pos.remaining_qty = 0.0
            return out

        if tp1_reachable:
            out.extend(self._fill_tp1(pos, price=pos.take_profit_1, ts=bar.ts))
            if pos.remaining_qty > 0 and bar.low <= pos.take_profit_2:
                if pos.be_armed and bar.high >= pos.stop_loss:
                    out.append(self._exit_fill(pos, price=pos.stop_loss,
                                               qty=pos.remaining_qty,
                                               kind="stop", ts=bar.ts))
                    pos.remaining_qty = 0.0
                else:
                    out.append(self._exit_fill(pos, price=pos.take_profit_2,
                                               qty=pos.remaining_qty,
                                               kind="tp2", ts=bar.ts))
                    pos.remaining_qty = 0.0
            return out

        if tp2_reachable:
            if bar.high >= pos.stop_loss:
                out.append(self._exit_fill(pos, price=pos.stop_loss,
                                           qty=pos.remaining_qty,
                                           kind="stop", ts=bar.ts))
                pos.remaining_qty = 0.0
            else:
                out.append(self._exit_fill(pos, price=pos.take_profit_2,
                                           qty=pos.remaining_qty,
                                           kind="tp2", ts=bar.ts))
                pos.remaining_qty = 0.0
        return out

    # ---- fill builders -----------------------------------------------
    def _fill_tp1(self, pos: Position, *, price: float,
                  ts: pd.Timestamp) -> list[Fill]:
        scale = max(0.0, min(1.0, self.cfg.tp1_scale_out_pct))
        tp1_qty = pos.initial_qty * scale
        if tp1_qty > pos.remaining_qty:
            tp1_qty = pos.remaining_qty
        if tp1_qty <= 0:
            return []
        fill = self._exit_fill(pos, price=price, qty=tp1_qty,
                               kind="tp1", ts=ts)
        pos.remaining_qty -= tp1_qty
        pos.tp1_filled = True
        if self.cfg.move_sl_to_be_after_tp1:
            pos.stop_loss = pos.entry_price
            pos.be_armed = True
        return [fill]

    def _exit_fill(self, pos: Position, *, price: float, qty: float,
                   kind: FillKind, ts: pd.Timestamp) -> Fill:
        fee = _bps(self.cfg.taker_fee_bps) * price * qty
        if pos.is_long():
            pnl = (price - pos.entry_price) * qty
        else:
            pnl = (pos.entry_price - price) * qty
        return Fill(
            fill_id=ulid(),
            ts=ts,
            symbol=pos.symbol,
            side=pos.side,
            kind=kind,
            price=price,
            qty=qty,
            fee_usd=fee,
            pnl_usd=pnl,
            intent_id=pos.intent_id,
            position_id=pos.position_id,
        )


__all__ = ["Bar", "Broker", "BrokerConfig", "Fill", "FillKind", "Position"]
