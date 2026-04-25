"""Bar-level Tier A feature engine (Phase 4).

Public surface used by the snapshot builder and the dashboard:

- ``FeatureConfig`` - versioned knobs (change version to invalidate caches).
- ``REGISTRY`` / ``FeatureEntry`` - declarative feature metadata.
- ``SymbolBundle`` / ``compute`` - run the pipeline for one symbol.
- ``check_no_lookahead`` / ``profile_frame`` - validators.
- ``SNAPSHOT_COLUMNS`` / ``DASHBOARD_COLUMNS`` / ``profile_columns`` -
  profile column contracts.
"""
from features.align import attach_asof, attach_mtf
from features.config import FeatureConfig, feature_cache_key
from features.context import compute_context
from features.flags import FLAG_COLUMNS, compute_flags
from features.layer1_vol import compute_layer1
from features.layer3_regime import compute_layer3
from features.layer4_tech import compute_layer4
from features.layer5_derivs import compute_layer5
from features.layer8_vp import compute_layer8
from features.layer10_xasset import compute_layer10
from features.layer11_calendar import compute_layer11
from features.layer12_composite import compute_layer12
from features.peer import CLUSTER_COLUMNS, PEER_COLUMNS, RANK_COLUMNS, PeerReport, compute_peer
from features.pipeline import SymbolBundle, compute
from features.registry import (
    DASHBOARD_COLUMNS,
    MTF_ATTACH,
    PROFILES,
    REGISTRY,
    SNAPSHOT_COLUMNS,
    FeatureEntry,
    Registry,
    profile_columns,
)
from features.snapshot import (
    SnapshotReport,
    build_snapshot,
    build_snapshot_for_symbol,
    fetch_symbol_bundle,
    save_snapshot,
)
from features.validators import (
    FeatureQualityReport,
    LookaheadReport,
    check_no_lookahead,
    profile_frame,
    registry_warmup,
)

__all__ = [
    "FeatureConfig",
    "feature_cache_key",
    "REGISTRY",
    "Registry",
    "FeatureEntry",
    "MTF_ATTACH",
    "SNAPSHOT_COLUMNS",
    "DASHBOARD_COLUMNS",
    "PROFILES",
    "profile_columns",
    "SymbolBundle",
    "compute",
    "compute_layer1",
    "compute_layer3",
    "compute_layer4",
    "compute_layer5",
    "compute_layer8",
    "compute_layer10",
    "compute_layer11",
    "compute_layer12",
    "compute_context",
    "compute_flags",
    "FLAG_COLUMNS",
    "compute_peer",
    "PeerReport",
    "RANK_COLUMNS",
    "CLUSTER_COLUMNS",
    "PEER_COLUMNS",
    "fetch_symbol_bundle",
    "build_snapshot",
    "build_snapshot_for_symbol",
    "save_snapshot",
    "SnapshotReport",
    "attach_asof",
    "attach_mtf",
    "check_no_lookahead",
    "profile_frame",
    "registry_warmup",
    "LookaheadReport",
    "FeatureQualityReport",
]
