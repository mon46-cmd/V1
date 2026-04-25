"""Phase 11 - exec loop wiring.

Single-process orchestrator that joins:

  scanner.fired_trigger
      -> ai.chat_deep
      -> intent_from_signal + risk.size_intent
      -> IntentQueue.submit
      -> ActivationWatcher (consumes ticks/books)
      -> Broker.open_from_intent on `activated`
      -> Broker.on_bar drives SL / TP / time-stop fills
      -> append-only fills.jsonl + atomic portfolio.json

The class :class:`ExecLoop` exposes the three callbacks the rest of
the system needs:

- :meth:`on_trigger` -- async; the scanner's ``deep_callback``.
- :meth:`emit_event` -- async; the watcher's ``emit`` coroutine.
- :meth:`on_bar` -- sync; called per closed 15m bar.

Equity / cash bookkeeping is tracked incrementally and persisted to
``portfolio.json`` after every fill. The fill stream remains the
source of truth -- :func:`portfolio.state.replay_from_fills` will
rebuild the same equity to 1e-6 (Phase 10 invariant).
"""
from __future__ import annotations

import json
import logging
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from ai import AIClient, DeepSignal, ReviewResponse
from core.config import Config
from core.ids import run_id as _new_run_id
from core.paths import run_dir
from core.time import now_utc, to_utc
from features.config import FeatureConfig
from portfolio.broker import Bar, Broker, BrokerConfig, Fill
from portfolio.intents import (
    ActivationWatcher,
    Intent,
    IntentEvent,
    IntentQueue,
    IntentStatus,
    WatcherConfig,
    intent_from_signal,
)
from portfolio.risk import (
    InstrumentSpec,
    RiskCaps,
    SizingDecision,
    circuit_breaker_multiplier,
    size_intent,
)
from portfolio.state import (
    PortfolioState,
    append_fill,
    save_state,
)

from .scanner import Scanner
from .triggers import TriggerDecision

log = logging.getLogger(__name__)


# Fill kinds that close (the remaining qty of) a position. ``tp1`` is
# excluded because it is always a partial scale-out at the default
# 50%; ``entry`` is excluded for obvious reasons.
_CLOSING_KINDS = frozenset({"stop", "tp2", "time", "manual", "funding"})


# ---------------------------------------------------------------------
@dataclass
class ExecConfig:
    """Tunables for the exec loop."""

    starting_equity_usd: float = 10_000.0
    activation_window_sec: float = 180.0
    save_state_every_fill: bool = True
    # Default instrument clamp; in production the per-symbol spec
    # should be looked up from instrument-info, but Phase 11 keeps it
    # uniform until that wiring lands.
    instrument: InstrumentSpec = field(default_factory=InstrumentSpec)


# ---------------------------------------------------------------------
@dataclass
class ReviewConfig:
    """Phase 12 - hooks that fire ``ai.chat_review`` on open positions."""

    # Drawdown threshold: fraction of the SL distance the mark has
    # travelled adversely from the entry. 0.6 means "60% of the way
    # to the stop".
    drawdown_pct_of_sl_distance: float = 0.6

    # Funding-approach window: we proactively review if the next
    # funding settlement is closer than this many seconds away.
    funding_window_sec: float = 30 * 60.0

    # Minimum gap between two review calls on the SAME position. The
    # build plan asserts "at most one review call per hour".
    min_review_interval_sec: float = 3600.0

    # Hook switches.
    enable_drawdown_hook: bool = True
    enable_tp1_hook: bool = True
    enable_regime_flip_hook: bool = True
    enable_funding_hook: bool = True


# Fill kinds the review apply-layer cares about.
_REVIEW_ACTIONS = frozenset({"hold", "tighten_stop", "scale_out", "stop", "flip"})


# ---------------------------------------------------------------------
# Whitelist of context blocks the deep payload accepts. Anything else
# is silently ignored so callers can pass a fat snapshot bundle dict
# without leaking irrelevant keys to the LLM.
_DEEP_CONTEXT_KEYS: tuple[str, ...] = (
    "mtf",          # {"1h": [...bars...], "4h": [...bars...]}
    "deriv",        # {"funding_rate": ..., "oi_usd": ..., "basis_bps": ...}
    "flow",         # {"cvd_z": ..., "taker_imbalance": ..., ...}
    "book",         # {"imbalance_top10": ..., "spread_bps": ...}
    "vp",           # {"poc": ..., "vah": ..., "val": ...}
    "levels",       # {"sr_levels": [...], "hh_ll": ...}
    "peer",         # {"cluster_id": ..., "cluster_leader": ..., "rank": ...}
    "regime",       # {"label": "trend|range|chop", "score": ...}
    "news",         # [{...recent headlines...}]
)


def _build_deep_payload(
    symbol: str,
    bar: dict,
    decision: TriggerDecision,
    *,
    context: dict | None = None,
) -> dict:
    """Prompt-B payload from the trigger bar plus optional context.

    The minimal core (trigger block + snapshot row) is preserved for
    backwards-compat; any keys from ``context`` that match the
    ``_DEEP_CONTEXT_KEYS`` whitelist are merged in. Callers should
    populate context from the same FeatureBundle that produced the
    snapshot (multi-TF bars, derivatives, peer cluster, etc.) so the
    LLM sees the full picture instead of a single bar.
    """
    mark = float(bar.get("close", decision.close or 0.0))
    payload: dict = {
        "as_of": (bar.get("timestamp").isoformat()
                  if isinstance(bar.get("timestamp"), pd.Timestamp)
                  else str(bar.get("timestamp", ""))),
        "symbol": symbol,
        "trigger": {
            "flag": decision.flag,
            "decision": decision.decision,
            "mark_price": mark,
            "atr_pct": decision.atr_pct,
            "move_pct": decision.move_pct,
            "bars_elapsed": decision.bars_elapsed,
        },
        "snapshot_row": {k: v for k, v in bar.items()
                         if not isinstance(v, pd.Timestamp)},
    }
    if context:
        for k in _DEEP_CONTEXT_KEYS:
            if k in context and context[k] is not None:
                payload[k] = context[k]
    return payload


# ---------------------------------------------------------------------
@dataclass
class ExecLoop:
    """Wire scanner triggers -> deep LLM -> intent -> broker."""

    cfg: Config
    feature_cfg: FeatureConfig
    ai: AIClient
    queue: IntentQueue
    watcher: ActivationWatcher
    broker: Broker
    exec_cfg: ExecConfig = field(default_factory=ExecConfig)
    review_cfg: ReviewConfig = field(default_factory=ReviewConfig)
    run_id: str | None = None

    # Running portfolio counters (the cached projection).
    _cash_usd: float = 0.0
    _realized_pnl_usd: float = 0.0
    _fees_paid_usd: float = 0.0
    _loser_streak: int = 0
    _watchlist: list[str] = field(default_factory=list)
    _last_marks: dict[str, float] = field(default_factory=dict)
    _closed_24h: deque[pd.Timestamp] = field(default_factory=deque)

    # Review-hook state.
    _last_review_at: dict[str, pd.Timestamp] = field(default_factory=dict)
    _tp1_seen: set[str] = field(default_factory=set)
    _pending_reviews: list[tuple[str, str, float, pd.Timestamp]] = field(default_factory=list)
    _last_regime: str | None = None
    _review_count: int = 0

    # Resolved on first use.
    _fills_path: Path | None = None
    _portfolio_path: Path | None = None
    _intents_path: Path | None = None
    _reviews_path: Path | None = None

    # ---- factory ------------------------------------------------------
    @classmethod
    def build(
        cls,
        *,
        cfg: Config,
        feature_cfg: FeatureConfig,
        ai: AIClient,
        run_id: str | None = None,
        exec_cfg: ExecConfig | None = None,
        review_cfg: ReviewConfig | None = None,
        broker_cfg: BrokerConfig | None = None,
        watcher_cfg: WatcherConfig | None = None,
    ) -> "ExecLoop":
        rid = run_id or _new_run_id()
        d = run_dir(cfg, rid)
        d.mkdir(parents=True, exist_ok=True)
        intents_path = d / "intents.jsonl"
        queue = IntentQueue(audit_path=intents_path)
        watcher = ActivationWatcher(queue, cfg=watcher_cfg or WatcherConfig())
        broker = Broker(cfg=broker_cfg or BrokerConfig())
        loop = cls(
            cfg=cfg,
            feature_cfg=feature_cfg,
            ai=ai,
            queue=queue,
            watcher=watcher,
            broker=broker,
            exec_cfg=exec_cfg or ExecConfig(),
            review_cfg=review_cfg or ReviewConfig(),
            run_id=rid,
        )
        loop._fills_path = d / "fills.jsonl"
        loop._portfolio_path = d / "portfolio.json"
        loop._intents_path = intents_path
        loop._reviews_path = d / "reviews.jsonl"
        loop._cash_usd = float(loop.exec_cfg.starting_equity_usd)
        # Startup recovery: if a previous fills.jsonl exists at this
        # run_id, replay it to restore cash / realized / fees /
        # loser_streak / closed_24h. Open broker positions are NOT
        # rehydrated (the fill stream lacks SL/TP metadata to
        # reconstruct them); operators should flatten before restart.
        if loop._fills_path.exists():
            loop._replay_fills_into_self()
        loop._save_state(now=now_utc())
        return loop

    def _replay_fills_into_self(self) -> None:
        """Hydrate running counters from ``fills.jsonl`` (idempotent)."""
        from portfolio.state import read_fills, replay_from_fills
        if self._fills_path is None or not self._fills_path.exists():
            return
        fills = read_fills(self._fills_path)
        if not fills:
            return
        replay = replay_from_fills(
            fills, starting_equity_usd=self.exec_cfg.starting_equity_usd,
            now=now_utc(),
        )
        self._cash_usd = float(replay.cash_usd)
        self._realized_pnl_usd = float(replay.realized_pnl_usd)
        self._fees_paid_usd = float(replay.fees_paid_usd)
        self._loser_streak = int(replay.loser_streak)
        # Re-seed the 24h closed-position deque from the replay anchor.
        self._closed_24h.clear()
        cutoff = now_utc() - pd.Timedelta(hours=24)
        for raw in fills:
            kind = raw.get("kind")
            if kind not in _CLOSING_KINDS:
                continue
            ts = to_utc(raw["ts"])
            if ts >= cutoff:
                self._closed_24h.append(ts)
        log.info(
            "replay restored cash=%.2f realized=%.2f fees=%.2f "
            "loser_streak=%d closed_24h=%d from %d fills",
            self._cash_usd, self._realized_pnl_usd, self._fees_paid_usd,
            self._loser_streak, len(self._closed_24h), len(fills),
        )

    def stop_file_path(self) -> Path | None:
        """Return the configured STOP-file path (data_root/STOP)."""
        if self.cfg is None or getattr(self.cfg, "data_root", None) is None:
            return None
        return Path(self.cfg.data_root) / "STOP"

    def stop_requested(self) -> bool:
        """True iff a sentinel STOP file exists. Operators can drop
        ``data/STOP`` to refuse new intents and request a graceful
        shutdown without killing the process."""
        sp = self.stop_file_path()
        return bool(sp and sp.exists())

    # ---- scanner integration ------------------------------------------
    def make_scanner(self) -> Scanner:
        """Return a Scanner pre-wired to this loop's ``on_trigger``."""
        return Scanner(
            cfg=self.cfg,
            feature_cfg=self.feature_cfg,
            ai=self.ai,
            deep_callback=self.on_trigger,
            run_id=self.run_id,
        )

    # ---- public callbacks --------------------------------------------
    async def on_trigger(
        self,
        symbol: str,
        bar: dict,
        decision: TriggerDecision,
        *,
        context: dict | None = None,
    ) -> Intent | None:
        """Scanner deep-callback: deep prompt -> intent -> queue.

        ``context`` is an optional dict of MTF / derivatives / peer /
        flow / book / vp blocks (see ``_DEEP_CONTEXT_KEYS``) merged
        into the Prompt-B payload. Returns the queued :class:`Intent`
        or ``None`` if the deep signal was flat / inconsistent /
        sizing rejected / the STOP-file is set.
        """
        if self.stop_requested():
            log.warning("on_trigger refused: STOP file present (%s)",
                        self.stop_file_path())
            return None
        payload = _build_deep_payload(symbol, bar, decision, context=context)
        try:
            signal: DeepSignal = await self.ai.chat_deep(symbol, payload)
        except Exception:  # noqa: BLE001
            log.exception("chat_deep failed symbol=%s", symbol)
            return None

        if signal.action == "flat":
            log.info("deep flat symbol=%s", symbol)
            return None

        # Risk sizing.
        mark = float(payload["trigger"].get("mark_price") or 0.0)
        provisional = intent_from_signal(
            signal=signal,
            qty=1.0,  # placeholder; resized below
            trigger_flag=decision.flag or "",
            now=now_utc(),
            activation_window_sec=self.exec_cfg.activation_window_sec,
        )
        equity = self._equity_usd_estimate(mark_map={symbol: mark})
        symbol_exposure = self._symbol_exposure(symbol)
        agg_exposure = self._aggregate_exposure()
        rmult = circuit_breaker_multiplier(self._loser_streak)

        decision_size: SizingDecision = size_intent(
            intent=provisional,
            equity_usd=equity,
            open_positions=len(self.broker.positions),
            symbol_exposure_usd=symbol_exposure,
            aggregate_exposure_usd=agg_exposure,
            instrument=self.exec_cfg.instrument,
            caps=RiskCaps.from_config(self.cfg),
            risk_multiplier=rmult,
        )
        if not decision_size.accepted:
            log.info("intent rejected symbol=%s reason=%s",
                     symbol, decision_size.reason)
            return None

        sized = intent_from_signal(
            signal=signal,
            qty=decision_size.qty,
            trigger_flag=decision.flag or "",
            now=now_utc(),
            activation_window_sec=self.exec_cfg.activation_window_sec,
        )
        self.queue.submit(sized)
        log.info("intent queued symbol=%s id=%s qty=%.6f",
                 symbol, sized.intent_id, sized.qty)
        return sized

    async def emit_event(self, event: IntentEvent) -> None:
        """ActivationWatcher emit-callback: open broker pos on activation."""
        intent = event.intent
        if event.kind != "activated":
            log.info("intent %s symbol=%s -> %s reason=%s",
                     intent.intent_id, intent.symbol, event.kind, event.reason)
            return
        if intent.activated_price is None or intent.activated_at is None:
            log.warning("activation event missing fill price symbol=%s",
                        intent.symbol)
            return
        try:
            _, fill = self.broker.open_from_intent(
                intent,
                fill_price=float(intent.activated_price),
                fill_ts=intent.activated_at,
            )
        except Exception:  # noqa: BLE001
            log.exception("broker.open_from_intent failed symbol=%s",
                          intent.symbol)
            return
        self._record_fill(fill, mark_price=float(intent.activated_price))

    def on_bar(self, bar: Bar) -> list[Fill]:
        """Drive the broker for one closed 15m bar.

        After fills are recorded, surviving open positions on this
        symbol are checked for the drawdown hook. TP1 fills enqueue a
        review automatically inside ``_record_fill``.
        """
        fills = self.broker.on_bar(bar)
        self._last_marks[bar.symbol] = float(bar.close)
        for f in fills:
            self._record_fill(f, mark_price=float(bar.close))
        # Drawdown hook: only positions still open on this symbol.
        if self.review_cfg.enable_drawdown_hook:
            for pos in list(self.broker.positions.values()):
                if pos.symbol != bar.symbol:
                    continue
                self._check_drawdown(pos, mark=float(bar.close), now=bar.ts)
        return fills

    async def on_bar_async(self, bar: Bar) -> list[Fill]:
        """Convenience: ``on_bar`` followed by ``run_pending_reviews``."""
        fills = self.on_bar(bar)
        await self.run_pending_reviews()
        return fills

    def close_all(self, *, ts: pd.Timestamp,
                  price_map: dict[str, float] | None = None,
                  kind: str = "manual") -> list[Fill]:
        """Flatten every open position at the given (or last seen) marks."""
        marks = dict(self._last_marks)
        if price_map:
            marks.update(price_map)
        fills = self.broker.close_all(price_map=marks, ts=ts, kind=kind)  # type: ignore[arg-type]
        for f in fills:
            self._record_fill(f, mark_price=marks.get(f.symbol, f.price))
        return fills

    # ---- state projection ---------------------------------------------
    def snapshot_state(self, *, now: pd.Timestamp | None = None) -> PortfolioState:
        ts = now or now_utc()
        unrealized = self._unrealized_pnl_usd()
        equity = self._cash_usd + unrealized
        return PortfolioState(
            as_of=ts,
            equity_usd=float(equity),
            cash_usd=float(self._cash_usd),
            realized_pnl_usd=float(self._realized_pnl_usd),
            fees_paid_usd=float(self._fees_paid_usd),
            open_positions=[self._position_dict(p)
                            for p in self.broker.positions.values()],
            closed_positions_24h=self._closed_in_24h(ts),
            watchlist=list(self._watchlist),
            risk_multiplier=circuit_breaker_multiplier(self._loser_streak),
            loser_streak=int(self._loser_streak),
        )

    def set_watchlist(self, symbols: Iterable[str]) -> int:
        """Replace the active watchlist; kill any pending intent on a
        symbol that is no longer watchlisted. Returns the number of
        intents killed.
        """
        from dataclasses import replace as _replace
        new_list = list(symbols)
        new_set = set(new_list)
        self._watchlist = new_list
        killed = 0
        for intent in list(self.queue.active()):
            # Only drop pre-activation intents; activated/filled ones
            # already have a live position the broker manages.
            if intent.symbol in new_set:
                continue
            if intent.status not in (IntentStatus.ARMED, IntentStatus.PENDING):
                continue
            dropped = _replace(
                intent, status=IntentStatus.KILLED,
                kill_reason="watchlist_dropped",
            )
            self.queue.update(dropped, reason="watchlist_dropped")
            killed += 1
        if killed:
            log.info("watchlist rotation killed %d pending intents", killed)
        return killed

    # ---- review hooks (Phase 12) --------------------------------------
    async def notify_regime_flip(
        self,
        new_regime: str,
        *,
        mark_map: dict[str, float] | None = None,
        now: pd.Timestamp | None = None,
    ) -> list[ReviewResponse]:
        """Queue a review on every open position when the regime changes.

        The first call simply records the regime; later calls only
        fire reviews when the regime actually flips.
        """
        ts = now or now_utc()
        prior = self._last_regime
        self._last_regime = new_regime
        if (prior is None or prior == new_regime
                or not self.review_cfg.enable_regime_flip_hook):
            return []
        marks = dict(self._last_marks)
        if mark_map:
            marks.update({k: float(v) for k, v in mark_map.items()})
        for pos in list(self.broker.positions.values()):
            mark = marks.get(pos.symbol, pos.entry_price)
            self._queue_review(pos.position_id, "regime_flip",
                               mark=float(mark), now=ts,
                               extra={"prior_regime": prior,
                                      "new_regime": new_regime})
        return await self.run_pending_reviews()

    async def on_funding_window(
        self,
        *,
        symbol: str,
        next_funding_at: pd.Timestamp,
        mark: float | None = None,
        now: pd.Timestamp | None = None,
        funding_rate: float | None = None,
    ) -> list[ReviewResponse]:
        """Fire a review on every open position in ``symbol`` when the
        next funding settlement is inside ``funding_window_sec``.

        Auto-stop short-circuit: when ``funding_rate`` is supplied AND
        we are inside the imminent-window (``< 300s``) AND the
        funding sign is adverse for the position (longs pay positive,
        shorts pay negative) AND unrealized PnL is < 0, the position
        is closed directly via ``broker.close_position(kind="funding")``
        instead of consulting the review LLM.
        """
        if not self.review_cfg.enable_funding_hook:
            return []
        ts = now or now_utc()
        delta_sec = (next_funding_at - ts).total_seconds()
        if delta_sec <= 0 or delta_sec > self.review_cfg.funding_window_sec:
            return []
        m = float(mark) if mark is not None else self._last_marks.get(symbol)
        if m is None:
            return []

        # --- imminent + adverse + losing => hard auto-stop ---------------
        IMMINENT_SEC = 300.0
        if funding_rate is not None and delta_sec < IMMINENT_SEC:
            fr = float(funding_rate)
            for pos in list(self.broker.positions.values()):
                if pos.symbol != symbol:
                    continue
                # Long pays when fr > 0; short pays when fr < 0.
                adverse = (pos.is_long() and fr > 0) or \
                          ((not pos.is_long()) and fr < 0)
                unrealized = ((m - pos.entry_price) * pos.remaining_qty
                              if pos.is_long() else
                              (pos.entry_price - m) * pos.remaining_qty)
                if adverse and unrealized < 0:
                    fill = self.broker.close_position(
                        pos.position_id, price=m, ts=ts, kind="funding")
                    if fill is not None:
                        self._record_fill(fill, mark_price=m)
                    self._audit_review(
                        pid=pos.position_id, reason="funding_autostop",
                        ts=ts, action="stop",
                        note=(f"auto-stop: fr={fr:+.5f} "
                              f"sec={delta_sec:.0f} unr={unrealized:+.2f}"),
                        extra={"funding_rate": fr,
                               "seconds_to_funding": float(delta_sec),
                               "mark_price": m})
        # --- otherwise queue the LLM review ----------------------------
        for pos in list(self.broker.positions.values()):
            if pos.symbol != symbol:
                continue
            self._queue_review(pos.position_id, "funding_approach",
                               mark=m, now=ts,
                               extra={"next_funding_at": next_funding_at.isoformat(),
                                      "seconds_to_funding": float(delta_sec),
                                      "funding_rate": funding_rate})
        return await self.run_pending_reviews()

    async def run_pending_reviews(self) -> list[ReviewResponse]:
        """Drain the pending-review queue: call ``ai.chat_review`` for
        each, apply the verdict (tighten_stop / scale_out / stop), and
        audit the round-trip to ``reviews.jsonl``.
        """
        if not self._pending_reviews:
            return []
        out: list[ReviewResponse] = []
        pending = self._pending_reviews
        self._pending_reviews = []
        for pid, reason, mark, ts in pending:
            pos = self.broker.positions.get(pid)
            if pos is None:
                # Position closed before the review fired; record skip.
                self._audit_review(pid=pid, reason=reason, ts=ts,
                                   action="skipped",
                                   note="position no longer open")
                continue
            payload = self._review_payload(pos, reason=reason,
                                           mark=mark, ts=ts)
            try:
                resp = await self.ai.chat_review(pos.symbol, payload)
            except Exception:  # noqa: BLE001
                log.exception("chat_review failed pid=%s reason=%s",
                              pid, reason)
                self._audit_review(pid=pid, reason=reason, ts=ts,
                                   action="error", note="chat_review raised")
                continue
            self._review_count += 1
            self._apply_review(resp, pos, mark=mark, ts=ts, reason=reason)
            out.append(resp)
        return out

    def _check_drawdown(self, pos, *, mark: float, now: pd.Timestamp) -> None:
        sl_dist = abs(pos.entry_price - pos.stop_loss)
        if sl_dist <= 0:
            return
        if pos.is_long():
            adverse = max(0.0, pos.entry_price - mark)
        else:
            adverse = max(0.0, mark - pos.entry_price)
        if adverse / sl_dist >= self.review_cfg.drawdown_pct_of_sl_distance:
            self._queue_review(pos.position_id, "drawdown",
                               mark=float(mark), now=now)

    def _queue_review(self, pid: str, reason: str, *,
                      mark: float, now: pd.Timestamp,
                      extra: dict | None = None) -> bool:
        """Queue a review subject to the per-position 1/hour cap.

        Returns True if queued, False if throttled. The throttle
        timestamp is set on queue so multiple hooks firing in the
        same bar coalesce to a single review.
        """
        last = self._last_review_at.get(pid)
        if last is not None:
            elapsed = (now - last).total_seconds()
            if elapsed < self.review_cfg.min_review_interval_sec:
                self._audit_review(pid=pid, reason=reason, ts=now,
                                   action="throttled",
                                   note=f"last review {elapsed:.0f}s ago",
                                   extra=extra)
                return False
        self._last_review_at[pid] = now
        self._pending_reviews.append((pid, reason, float(mark), now))
        return True

    def _review_payload(self, pos, *, reason: str,
                        mark: float, ts: pd.Timestamp) -> dict:
        if pos.is_long():
            unrealized = (mark - pos.entry_price) * pos.remaining_qty
        else:
            unrealized = (pos.entry_price - mark) * pos.remaining_qty
        return {
            "as_of": ts.isoformat(),
            "trigger_reason": reason,
            "position": {
                "position_id": pos.position_id,
                "symbol": pos.symbol,
                "side": pos.side,
                "entry": pos.entry_price,
                "stop_loss": pos.stop_loss,
                "tp1": pos.take_profit_1,
                "tp2": pos.take_profit_2,
                "mark_price": float(mark),
                "remaining_qty": pos.remaining_qty,
                "tp1_filled": pos.tp1_filled,
                "be_armed": pos.be_armed,
                "bars_held": pos.bars_held,
                "time_horizon_bars": pos.time_horizon_bars,
                "unrealized_pnl_usd": float(unrealized),
            },
        }

    def _apply_review(self, resp: ReviewResponse, pos, *,
                      mark: float, ts: pd.Timestamp, reason: str) -> None:
        action = resp.action if resp.action in _REVIEW_ACTIONS else "hold"
        applied = action
        note = ""
        if action == "tighten_stop":
            new_sl = float(resp.new_stop_loss) if resp.new_stop_loss is not None else None
            if new_sl is None:
                applied, note = "hold", "tighten_stop missing new_stop_loss"
            elif pos.is_long():
                # Long: must be strictly above the current SL and at or
                # below the entry. Reject otherwise.
                if not (pos.stop_loss < new_sl <= pos.entry_price):
                    applied, note = "hold", (
                        f"reject tighten_stop: new={new_sl} not in "
                        f"({pos.stop_loss}, {pos.entry_price}]")
                else:
                    pos.stop_loss = new_sl
            else:
                if not (pos.entry_price <= new_sl < pos.stop_loss):
                    applied, note = "hold", (
                        f"reject tighten_stop: new={new_sl} not in "
                        f"[{pos.entry_price}, {pos.stop_loss})")
                else:
                    pos.stop_loss = new_sl
        elif action == "scale_out":
            # Synthetic TP1 at mark on the still-untouched leg. Mark
            # the TP1 hook as fired BEFORE invoking _fill_tp1 so the
            # tp1 fill threading back through _record_fill does NOT
            # queue a duplicate review (which would only get
            # throttled, but pollutes reviews.jsonl).
            if not pos.tp1_filled:
                self._tp1_seen.add(pos.position_id)
                fills = self.broker._fill_tp1(pos, price=float(mark), ts=ts)  # type: ignore[attr-defined]
                for f in fills:
                    self._record_fill(f, mark_price=float(mark))
            else:
                applied, note = "hold", "scale_out skipped: tp1 already filled"
        elif action in {"stop", "flip"}:
            fill = self.broker.close_position(pos.position_id, price=float(mark),
                                              ts=ts, kind="manual")
            if fill is not None:
                self._record_fill(fill, mark_price=float(mark))
            if action == "flip":
                note = "flip-as-stop: reversal not implemented in Phase 12"

        self._audit_review(pid=pos.position_id, reason=reason, ts=ts,
                           action=applied, note=note,
                           extra={"raw_action": resp.action,
                                  "new_stop_loss": resp.new_stop_loss,
                                  "confidence": resp.confidence,
                                  "rationale": resp.rationale,
                                  "mark_price": float(mark)})

    def _audit_review(self, *, pid: str, reason: str, ts: pd.Timestamp,
                      action: str, note: str = "",
                      extra: dict | None = None) -> None:
        if self._reviews_path is None:
            return
        rec = {
            "ts": ts.isoformat(),
            "position_id": pid,
            "reason": reason,
            "action": action,
            "note": note,
        }
        if extra:
            rec.update(extra)
        self._reviews_path.parent.mkdir(parents=True, exist_ok=True)
        # Plain "\n" in text-mode UTF-8: ``os.linesep`` would
        # double-translate to ``\r\r\n`` on Windows.
        with self._reviews_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, default=str) + "\n")

    # ---- internals ----------------------------------------------------
    def _record_fill(self, fill: Fill, *, mark_price: float) -> None:
        if self._fills_path is not None:
            append_fill(self._fills_path, fill)
        self._cash_usd -= fill.fee_usd
        self._fees_paid_usd += fill.fee_usd
        if fill.kind != "entry":
            self._cash_usd += fill.pnl_usd
            self._realized_pnl_usd += fill.pnl_usd
            # loser_streak: count only the closing leg of a position so
            # the projection matches ``portfolio.state.replay_from_fills``
            # exactly. ``broker.positions`` membership is unreliable
            # here because the broker may have popped the position
            # earlier in this same fill batch (TP1+stop in one bar).
            if fill.kind in _CLOSING_KINDS:
                if fill.pnl_usd < 0:
                    self._loser_streak += 1
                elif fill.pnl_usd > 0:
                    self._loser_streak = 0
                self._closed_24h.append(fill.ts)
        # TP1 hook: queue a review the first time we see TP1 on a
        # position. The position is still open (50% scaled out by
        # default) so the review can act on the remaining qty.
        if (fill.kind == "tp1"
                and fill.position_id not in self._tp1_seen
                and self.review_cfg.enable_tp1_hook):
            self._tp1_seen.add(fill.position_id)
            self._queue_review(fill.position_id, "tp1",
                               mark=float(mark_price), now=fill.ts)
        if mark_price > 0:
            self._last_marks[fill.symbol] = float(mark_price)
        if self.exec_cfg.save_state_every_fill:
            self._save_state(now=fill.ts)

    def _save_state(self, *, now: pd.Timestamp | None = None) -> None:
        if self._portfolio_path is None:
            return
        save_state(self.snapshot_state(now=now or now_utc()), self._portfolio_path)

    def _equity_usd_estimate(self, *, mark_map: dict[str, float]) -> float:
        marks = dict(self._last_marks)
        marks.update({k: float(v) for k, v in mark_map.items() if v > 0})
        unrealized = self._unrealized_pnl_usd(mark_map=marks)
        return float(self._cash_usd + unrealized)

    def _unrealized_pnl_usd(self, *,
                            mark_map: dict[str, float] | None = None) -> float:
        marks = mark_map if mark_map is not None else self._last_marks
        total = 0.0
        for pos in self.broker.positions.values():
            mark = marks.get(pos.symbol)
            if mark is None or mark <= 0:
                continue
            if pos.is_long():
                total += (mark - pos.entry_price) * pos.remaining_qty
            else:
                total += (pos.entry_price - mark) * pos.remaining_qty
        return float(total)

    def _symbol_exposure(self, symbol: str) -> float:
        return float(sum(
            pos.entry_price * pos.remaining_qty
            for pos in self.broker.positions.values()
            if pos.symbol == symbol
        ))

    def _aggregate_exposure(self) -> float:
        return float(sum(
            pos.entry_price * pos.remaining_qty
            for pos in self.broker.positions.values()
        ))

    def _closed_in_24h(self, now: pd.Timestamp) -> int:
        cutoff = now - pd.Timedelta(hours=24)
        # Trim the deque opportunistically.
        while self._closed_24h and self._closed_24h[0] < cutoff:
            self._closed_24h.popleft()
        return len(self._closed_24h)

    @staticmethod
    def _position_dict(pos) -> dict:
        return {
            "position_id": pos.position_id,
            "intent_id": pos.intent_id,
            "symbol": pos.symbol,
            "side": pos.side,
            "entry_price": pos.entry_price,
            "initial_qty": pos.initial_qty,
            "remaining_qty": pos.remaining_qty,
            "stop_loss": pos.stop_loss,
            "take_profit_1": pos.take_profit_1,
            "take_profit_2": pos.take_profit_2,
            "tp1_filled": pos.tp1_filled,
            "be_armed": pos.be_armed,
            "bars_held": pos.bars_held,
            "opened_at": (pos.opened_at.isoformat()
                          if pos.opened_at is not None else None),
        }


__all__ = ["ExecConfig", "ExecLoop", "ReviewConfig"]
