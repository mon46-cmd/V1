"""Long-running loops: scanner and exec. Phase 8+ and 11+."""
from .cooldowns import CooldownEntry, CooldownStore
from .exec import ExecConfig, ExecLoop, ReviewConfig
from .scanner import DeepCallback, Scanner, ScannerResult
from .triggers import (
    DEC_BYPASS_MOVE,
    DEC_COOLDOWN_ACTIVE,
    DEC_DUP_BAR,
    DEC_FRESH,
    DEC_NO_BAR,
    DEC_NO_FLAG,
    DEC_POST_COOLDOWN,
    POSITIVE_DECISIONS,
    CooldownState,
    TriggerDecision,
    detect_trigger,
)

__all__ = [
    "CooldownEntry",
    "CooldownState",
    "CooldownStore",
    "DeepCallback",
    "ExecConfig",
    "ExecLoop",
    "ReviewConfig",
    "Scanner",
    "ScannerResult",
    "TriggerDecision",
    "detect_trigger",
    "DEC_FRESH",
    "DEC_POST_COOLDOWN",
    "DEC_BYPASS_MOVE",
    "DEC_COOLDOWN_ACTIVE",
    "DEC_DUP_BAR",
    "DEC_NO_FLAG",
    "DEC_NO_BAR",
    "POSITIVE_DECISIONS",
]
