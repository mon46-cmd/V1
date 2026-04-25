"""Intent + atomic confirmation (Phase 9).

Between trigger-bar close and the LLM reply, a hostile spike can stop
us out at a naive market entry. The defence is to treat the AI output
as an ``Intent`` and validate it at tick resolution before opening a
position. This module owns the watcher; the broker (Phase 10) consumes
its activation events.

Public surface:

- ``Intent`` -- frozen dataclass mirroring ``DeepSignal`` plus
  bookkeeping (id / status / kill_reason / qty).
- ``IntentEvent`` -- ``activated`` / ``killed`` / ``expired`` event
  emitted by the watcher.
- ``IntentQueue`` -- in-memory queue + ``intents.jsonl`` audit writer.
- ``ActivationWatcher`` -- per-symbol activation/invalidation logic.
  Synchronous ``process_tick`` / ``process_book`` / ``process_clock``
  methods are pure-ish (return events, mutate intent status) and are
  what tests exercise. ``run(feed)`` wraps them in an async loop.

Intent lifecycle (per ``docs/04_EXECUTION.md`` section 2):

    armed --(activation rule met)----> activated  -> emit ENTRY fill
        \\                            ^
         \\---(SL hit before fill)----- killed("sl_before_entry")
         \\---(now >= expires_at)----- expired
"""
from __future__ import annotations

import json
import logging
import threading
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Literal, Protocol

import pandas as pd

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Tick + book primitives (pure, decoupled from downloader.ws)
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class Tick:
    """A trade print. ``ts`` is UTC; ``side`` is the aggressor side."""

    ts: pd.Timestamp
    price: float
    size: float = 0.0
    side: Literal["Buy", "Sell"] | None = None
    symbol: str = ""


@dataclass(frozen=True)
class BookTop:
    """Top-of-book snapshot. Sizes in base units, not USD."""

    ts: pd.Timestamp
    bid: float
    bid_size: float
    ask: float
    ask_size: float
    symbol: str = ""

    @property
    def mid(self) -> float:
        return 0.5 * (self.bid + self.ask)


# ---------------------------------------------------------------------
# Intent + status
# ---------------------------------------------------------------------
ActivationKind = Literal["touch", "close_above", "close_below", "breakout"]
Side = Literal["long", "short"]


class IntentStatus(str, Enum):
    PENDING = "pending"   # created, not yet armed (rare; we arm immediately)
    ARMED = "armed"       # waiting for activation
    ACTIVATED = "activated"
    EXPIRED = "expired"
    KILLED = "killed"
    FILLED = "filled"     # after broker confirms entry fill


@dataclass(frozen=True)
class Intent:
    """An immutable structured trade plan.

    Status transitions are modeled by ``replace(intent, status=...)``;
    the queue stores the latest version keyed by ``intent_id``.
    """

    intent_id: str
    created_at: pd.Timestamp
    expires_at: pd.Timestamp
    symbol: str
    side: Side
    entry: float
    entry_trigger: float | None
    activation_kind: ActivationKind
    stop_loss: float
    take_profit_1: float
    take_profit_2: float | None
    time_horizon_bars: int
    qty: float
    trigger_flag: str
    prompt_version: str
    status: IntentStatus = IntentStatus.ARMED
    kill_reason: str = ""
    activated_price: float | None = None
    activated_at: pd.Timestamp | None = None

    # ---- helpers ------------------------------------------------------
    def is_terminal(self) -> bool:
        return self.status in (
            IntentStatus.ACTIVATED, IntentStatus.EXPIRED,
            IntentStatus.KILLED, IntentStatus.FILLED,
        )

    def to_record(self) -> dict:
        d = asdict(self)
        d["status"] = self.status.value
        for k in ("created_at", "expires_at", "activated_at"):
            v = d.get(k)
            d[k] = v.isoformat() if isinstance(v, pd.Timestamp) else v
        return d


# ---------------------------------------------------------------------
# Events emitted by the watcher
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class IntentEvent:
    kind: Literal["activated", "killed", "expired"]
    intent: Intent  # the post-transition snapshot
    reason: str = ""

    def to_record(self) -> dict:
        return {
            "kind": self.kind,
            "reason": self.reason,
            "intent": self.intent.to_record(),
        }


# ---------------------------------------------------------------------
# Intent queue + audit writer
# ---------------------------------------------------------------------
@dataclass
class IntentQueue:
    """In-memory queue with append-only ``intents.jsonl`` audit.

    Thread-safe within a single process. Every ``submit`` and every
    ``update`` writes one record. Exposes ``active()`` (non-terminal)
    and ``by_symbol(sym)`` for the watcher to bind per-symbol state.
    """

    audit_path: Path | None = None
    _intents: dict[str, Intent] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def submit(self, intent: Intent) -> Intent:
        with self._lock:
            self._intents[intent.intent_id] = intent
        self._write({"event": "submit", "intent": intent.to_record()})
        return intent

    def update(self, intent: Intent, *, reason: str = "") -> Intent:
        """Replace an intent in-place; logs the new status."""
        with self._lock:
            self._intents[intent.intent_id] = intent
        self._write({
            "event": "update",
            "status": intent.status.value,
            "reason": reason,
            "intent": intent.to_record(),
        })
        return intent

    def get(self, intent_id: str) -> Intent | None:
        return self._intents.get(intent_id)

    def active(self) -> list[Intent]:
        return [i for i in self._intents.values() if not i.is_terminal()]

    def by_symbol(self, symbol: str, *, only_active: bool = True) -> list[Intent]:
        out = [i for i in self._intents.values() if i.symbol == symbol]
        if only_active:
            out = [i for i in out if not i.is_terminal()]
        return out

    def all(self) -> list[Intent]:
        return list(self._intents.values())

    def _write(self, record: dict) -> None:
        if self.audit_path is None:
            return
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, default=str, sort_keys=True)
        # Append-only writes are atomic for small lines (< PIPE_BUF) on
        # POSIX; on Windows we rely on a single ``write`` per call. Use
        # ``"\n"`` (text mode translates to the platform line ending);
        # ``os.linesep`` here would double-translate to ``\r\r\n``.
        with self.audit_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


# ---------------------------------------------------------------------
# Activation watcher
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class WatcherConfig:
    """Tunables for the activation watcher."""

    # Breakout: top-of-book size in BASE units required at the level.
    breakout_min_book_size: float = 0.0
    # Breakout: in USD; the watcher takes the max of the two thresholds.
    breakout_min_book_usd: float = 5_000.0
    # Length of a synthetic close-confirmation candle (close_above/below).
    close_candle_seconds: float = 1.0


class TickFeed(Protocol):
    """Minimal async iterator yielding ``Tick`` or ``BookTop``."""

    def __aiter__(self): ...
    async def __anext__(self) -> Tick | BookTop: ...


@dataclass
class _CloseAccumulator:
    """Per-intent OHLC builder for close_above / close_below."""

    open_ts: pd.Timestamp | None = None
    o: float = 0.0
    h: float = 0.0
    l: float = 0.0  # noqa: E741
    c: float = 0.0

    def add(self, tick: Tick, candle_seconds: float) -> tuple[bool, float | None]:
        """Feed a tick; returns ``(closed, close_price)``."""
        if self.open_ts is None:
            self.open_ts = tick.ts
            self.o = self.h = self.l = self.c = tick.price
            return False, None

        elapsed = (tick.ts - self.open_ts).total_seconds()
        # Update high/low/close with the new tick first.
        self.h = max(self.h, tick.price)
        self.l = min(self.l, tick.price)
        self.c = tick.price
        if elapsed >= candle_seconds:
            close_px = self.c
            # Reset the bucket to start AT this tick.
            self.open_ts = tick.ts
            self.o = self.h = self.l = self.c = tick.price
            return True, close_px
        return False, None


@dataclass
class ActivationWatcher:
    """Drives every armed intent to a terminal status.

    The watcher is *synchronous* at the core: ``process_tick``,
    ``process_book``, and ``process_clock`` mutate intent state and
    return any events generated. The async ``run(feed)`` is a thin loop
    that dispatches each item from a feed and yields events.

    Intentionally minimal: no broker side-effects, no fills produced.
    The exec loop (Phase 11) is responsible for forwarding ``activated``
    events to the broker.
    """

    queue: IntentQueue
    cfg: WatcherConfig = field(default_factory=WatcherConfig)
    _closers: dict[str, _CloseAccumulator] = field(default_factory=dict)

    # ---- per-event entry points --------------------------------------
    def process_tick(self, tick: Tick) -> list[IntentEvent]:
        events: list[IntentEvent] = []
        for intent in list(self.queue.by_symbol(tick.symbol, only_active=True)):
            if intent.status != IntentStatus.ARMED:
                continue
            ev = self._evaluate_tick(intent, tick)
            if ev is not None:
                events.append(ev)
        return events

    def process_book(self, book: BookTop) -> list[IntentEvent]:
        events: list[IntentEvent] = []
        for intent in list(self.queue.by_symbol(book.symbol, only_active=True)):
            if intent.status != IntentStatus.ARMED:
                continue
            if intent.activation_kind != "breakout":
                continue
            ev = self._evaluate_breakout(intent, book)
            if ev is not None:
                events.append(ev)
        return events

    def process_clock(self, now: pd.Timestamp) -> list[IntentEvent]:
        events: list[IntentEvent] = []
        if now.tzinfo is None:
            now = now.tz_localize("UTC")
        for intent in list(self.queue.active()):
            if intent.status != IntentStatus.ARMED:
                continue
            if now >= intent.expires_at:
                killed = replace(
                    intent, status=IntentStatus.EXPIRED,
                    kill_reason="expired",
                )
                self.queue.update(killed, reason="expired")
                self._closers.pop(intent.intent_id, None)
                events.append(IntentEvent(
                    kind="expired", intent=killed, reason="expired",
                ))
        return events

    # ---- internals ---------------------------------------------------
    def _evaluate_tick(self, intent: Intent, tick: Tick) -> IntentEvent | None:
        # 1. Hard invalidation: SL traded through before activation.
        if self._sl_hit(intent, tick.price):
            killed = replace(
                intent, status=IntentStatus.KILLED,
                kill_reason="sl_before_entry",
            )
            self.queue.update(killed, reason="sl_before_entry")
            self._closers.pop(intent.intent_id, None)
            return IntentEvent(
                kind="killed", intent=killed, reason="sl_before_entry",
            )

        # 2. Activation rule.
        kind = intent.activation_kind
        trig = intent.entry_trigger if intent.entry_trigger is not None else intent.entry
        if kind == "touch":
            if self._touch_hit(intent, tick.price, trig):
                return self._activate(intent, fill_price=trig, ts=tick.ts)
            return None
        if kind in ("close_above", "close_below"):
            return self._evaluate_close(intent, tick, trig)
        # breakout activations come from the book channel; tick channel
        # only checks for SL invalidation above.
        return None

    def _evaluate_close(
        self, intent: Intent, tick: Tick, trigger: float,
    ) -> IntentEvent | None:
        accu = self._closers.setdefault(intent.intent_id, _CloseAccumulator())
        closed, close_px = accu.add(tick, self.cfg.close_candle_seconds)
        if not closed or close_px is None:
            return None
        kind = intent.activation_kind
        if kind == "close_above" and close_px > trigger:
            return self._activate(intent, fill_price=close_px, ts=tick.ts)
        if kind == "close_below" and close_px < trigger:
            return self._activate(intent, fill_price=close_px, ts=tick.ts)
        return None

    def _evaluate_breakout(
        self, intent: Intent, book: BookTop,
    ) -> IntentEvent | None:
        trig = intent.entry_trigger if intent.entry_trigger is not None else intent.entry
        # Long breakout: ask price > trigger AND ask_size depth sufficient.
        if intent.side == "long":
            if not (book.ask > 0 and book.ask > trig):
                return None
            depth_ok = self._depth_ok(book.ask, book.ask_size)
            if not depth_ok:
                return None
            return self._activate(intent, fill_price=book.ask, ts=book.ts)
        # Short breakout: bid < trigger AND bid_size depth sufficient.
        if not (book.bid > 0 and book.bid < trig):
            return None
        if not self._depth_ok(book.bid, book.bid_size):
            return None
        return self._activate(intent, fill_price=book.bid, ts=book.ts)

    def _depth_ok(self, price: float, size: float) -> bool:
        usd = price * size
        return (
            size >= self.cfg.breakout_min_book_size
            and usd >= self.cfg.breakout_min_book_usd
        )

    @staticmethod
    def _touch_hit(intent: Intent, price: float, trigger: float) -> bool:
        if intent.side == "long":
            return price <= trigger
        return price >= trigger

    @staticmethod
    def _sl_hit(intent: Intent, price: float) -> bool:
        if intent.side == "long":
            return price <= intent.stop_loss
        return price >= intent.stop_loss

    def _activate(
        self, intent: Intent, *, fill_price: float, ts: pd.Timestamp,
    ) -> IntentEvent:
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        activated = replace(
            intent,
            status=IntentStatus.ACTIVATED,
            activated_price=float(fill_price),
            activated_at=ts,
        )
        self.queue.update(activated, reason="activated")
        self._closers.pop(intent.intent_id, None)
        return IntentEvent(kind="activated", intent=activated, reason="")

    # ---- async driver ------------------------------------------------
    async def run(
        self,
        feed: TickFeed,
        *,
        emit: "Callable[[IntentEvent], Awaitable[None]] | None" = None,  # noqa: F821
    ) -> None:
        """Consume an async feed of ``Tick`` / ``BookTop`` items.

        Stops when the feed terminates. If ``emit`` is provided, every
        event is awaited through it (e.g. to forward to the broker).
        """
        async for item in feed:
            if isinstance(item, Tick):
                evs: Iterable[IntentEvent] = self.process_tick(item)
            elif isinstance(item, BookTop):
                evs = self.process_book(item)
            else:
                continue
            if emit is None:
                continue
            for ev in evs:
                await emit(ev)


# ---------------------------------------------------------------------
# Helpers to build an Intent from a DeepSignal
# ---------------------------------------------------------------------
def intent_from_signal(
    *,
    signal,  # ai.schemas.DeepSignal -- avoid hard import to keep portfolio independent
    qty: float,
    trigger_flag: str,
    now: pd.Timestamp,
    activation_window_sec: float = 180.0,
    intent_id: str | None = None,
) -> Intent:
    """Construct an ``Intent`` from a non-flat ``DeepSignal``.

    Raises ``ValueError`` if the signal lacks entry/sl/tp1 or is flat.
    The signal's ``check_consistency`` should be clean before calling.
    """
    if getattr(signal, "action", "flat") == "flat":
        raise ValueError("cannot build intent from flat signal")
    entry = signal.entry
    sl = signal.stop_loss
    tp1 = signal.take_profit_1
    if entry is None or sl is None or tp1 is None:
        raise ValueError("signal missing entry/sl/tp1")
    from core.ids import short_id  # local import: keep cycle-free

    if now.tzinfo is None:
        now = now.tz_localize("UTC")
    return Intent(
        intent_id=intent_id or short_id(),
        created_at=now,
        expires_at=now + pd.Timedelta(seconds=activation_window_sec),
        symbol=signal.symbol,
        side=signal.action,  # "long" / "short"
        entry=float(entry),
        entry_trigger=(float(signal.entry_trigger)
                       if signal.entry_trigger is not None else None),
        activation_kind=(signal.activation_kind or "touch"),
        stop_loss=float(sl),
        take_profit_1=float(tp1),
        take_profit_2=(float(signal.take_profit_2)
                       if signal.take_profit_2 is not None else None),
        time_horizon_bars=int(signal.time_horizon_bars or 16),
        qty=float(qty),
        trigger_flag=trigger_flag,
        prompt_version=signal.prompt_version,
    )


__all__ = [
    "ActivationKind",
    "ActivationWatcher",
    "BookTop",
    "Intent",
    "IntentEvent",
    "IntentQueue",
    "IntentStatus",
    "Side",
    "Tick",
    "TickFeed",
    "WatcherConfig",
    "intent_from_signal",
]
