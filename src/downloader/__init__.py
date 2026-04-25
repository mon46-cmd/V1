"""Exchange I/O: REST, archive, cache, validators (Phase 1).

Public surface:

- ``HttpClient`` - shared aiohttp session with token-bucket rate limit.
- ``RestClient`` - normalized Bybit v5 public endpoints.
- ``ParquetCache`` - atomic parquet reader/writer with merge-append.
- ``ArchiveClient`` - daily csv.gz tick archive fetcher.
- ``validate_ohlcv`` / ``validate_grid`` / ``validate_ticks`` - schema +
  gap checks returning report dataclasses.
"""
from downloader.archive import ArchiveClient
from downloader.cache import ParquetCache
from downloader.http import HttpClient
from downloader.orderbook import BookGap, BookStats, OrderBookL2
from downloader.rest import RestClient
from downloader.tick_pipeline import TickPipeline, TickPipelineStats
from downloader.universe import (
    UNIVERSE_COLUMNS,
    UniverseReport,
    build_universe,
    filter_universe,
    load_universe,
    save_universe,
)
from downloader.validators import (
    GridReport,
    TickReport,
    oi_freq_for,
    validate_grid,
    validate_ohlcv,
    validate_ticks,
)
from downloader.ws import WsClient

__all__ = [
    "HttpClient",
    "RestClient",
    "ParquetCache",
    "ArchiveClient",
    "WsClient",
    "OrderBookL2",
    "BookStats",
    "BookGap",
    "TickPipeline",
    "TickPipelineStats",
    "UNIVERSE_COLUMNS",
    "UniverseReport",
    "build_universe",
    "filter_universe",
    "load_universe",
    "save_universe",
    "GridReport",
    "TickReport",
    "validate_ohlcv",
    "validate_grid",
    "validate_ticks",
    "oi_freq_for",
]
