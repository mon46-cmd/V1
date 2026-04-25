"""LIVE tests for `downloader.rest.RestClient`.

Hit the real Bybit public REST endpoints. Skipped entirely when
``BYBIT_OFFLINE=1`` or when the host is unreachable. Keep the number
of calls small to respect the public-endpoint budget.
"""
from __future__ import annotations

import os
import socket

import pandas as pd
import pytest


pytestmark = [
    pytest.mark.skipif(
        os.getenv("BYBIT_OFFLINE", "").lower() in ("1", "true", "yes"),
        reason="BYBIT_OFFLINE is set",
    ),
]


def _has_internet(host: str = "api.bybit.com", port: int = 443, timeout: float = 3.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


if not _has_internet():  # pragma: no cover
    pytestmark.append(pytest.mark.skip(reason="api.bybit.com unreachable"))


async def _with_clients(fn):
    from core.config import load_config
    from downloader.http import HttpClient
    from downloader.rest import RestClient

    cfg = load_config()
    async with HttpClient(cfg) as http:
        rest = RestClient(http, cfg)
        return await fn(rest)


async def test_tickers_returns_usdt_perps(tmp_data_root) -> None:  # noqa: ARG001
    async def _do(rest):
        data = await rest.tickers()
        assert isinstance(data, list)
        assert len(data) > 50
        btc = next(t for t in data if t["symbol"] == "BTCUSDT")
        assert btc["price"] > 0
        assert btc["turnover_24h"] > 0
        return data

    await _with_clients(_do)


async def test_instruments_has_btcusdt(tmp_data_root) -> None:  # noqa: ARG001
    async def _do(rest):
        df = await rest.instruments()
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 50
        assert (df["symbol"] == "BTCUSDT").any()

    await _with_clients(_do)


async def test_klines_monotonic_utc_float(tmp_data_root) -> None:  # noqa: ARG001
    async def _do(rest):
        end = pd.Timestamp.utcnow().tz_localize(None).tz_localize("UTC")
        start = end - pd.Timedelta(hours=6)
        df = await rest.klines("BTCUSDT", "15", start, end)
        assert not df.empty
        assert str(df["timestamp"].dt.tz) == "UTC"
        assert df["timestamp"].is_monotonic_increasing
        assert df["timestamp"].is_unique
        for c in ("open", "high", "low", "close", "volume", "turnover"):
            assert df[c].dtype == "float64"
        # High >= Low, both > 0.
        assert (df["high"] >= df["low"]).all()
        assert (df["low"] > 0).all()

    await _with_clients(_do)


async def test_mark_and_premium_klines(tmp_data_root) -> None:  # noqa: ARG001
    async def _do(rest):
        end = pd.Timestamp.utcnow().tz_localize(None).tz_localize("UTC")
        start = end - pd.Timedelta(hours=6)
        mk = await rest.mark_klines("BTCUSDT", "15", start, end)
        pk = await rest.premium_klines("BTCUSDT", "15", start, end)
        for df in (mk, pk):
            assert not df.empty
            assert str(df["timestamp"].dt.tz) == "UTC"
            for c in ("open", "high", "low", "close"):
                assert df[c].dtype == "float64"

    await _with_clients(_do)


async def test_funding_history(tmp_data_root) -> None:  # noqa: ARG001
    async def _do(rest):
        end = pd.Timestamp.utcnow().tz_localize(None).tz_localize("UTC")
        start = end - pd.Timedelta(days=10)
        df = await rest.funding("BTCUSDT", start, end)
        assert not df.empty
        assert str(df["timestamp"].dt.tz) == "UTC"
        assert df["timestamp"].is_monotonic_increasing
        assert df["funding_rate"].dtype == "float64"

    await _with_clients(_do)


async def test_open_interest(tmp_data_root) -> None:  # noqa: ARG001
    async def _do(rest):
        end = pd.Timestamp.utcnow().tz_localize(None).tz_localize("UTC")
        start = end - pd.Timedelta(days=2)
        df = await rest.open_interest("BTCUSDT", "1h", start, end)
        assert not df.empty
        assert str(df["timestamp"].dt.tz) == "UTC"
        assert df["open_interest"].dtype == "float64"
        assert (df["open_interest"] > 0).all()

    await _with_clients(_do)


async def test_long_short_ratio(tmp_data_root) -> None:  # noqa: ARG001
    async def _do(rest):
        df = await rest.long_short_ratio("BTCUSDT", "1h", limit=50)
        assert not df.empty
        assert str(df["timestamp"].dt.tz) == "UTC"
        for c in ("buy_ratio", "sell_ratio"):
            assert df[c].dtype == "float64"

    await _with_clients(_do)


async def test_orderbook_snapshot(tmp_data_root) -> None:  # noqa: ARG001
    async def _do(rest):
        ob = await rest.orderbook("BTCUSDT", depth=25)
        assert ob["symbol"] == "BTCUSDT"
        assert ob["bids"] and ob["asks"]
        best_bid = ob["bids"][0][0]
        best_ask = ob["asks"][0][0]
        assert best_ask > best_bid > 0

    await _with_clients(_do)


async def test_recent_trades(tmp_data_root) -> None:  # noqa: ARG001
    async def _do(rest):
        df = await rest.recent_trades("BTCUSDT", limit=50)
        assert not df.empty
        assert str(df["timestamp"].dt.tz) == "UTC"
        assert df["price"].dtype == "float64"
        assert df["size"].dtype == "float64"
        assert set(df["side"].unique()).issubset({"Buy", "Sell"})

    await _with_clients(_do)


async def test_validators_pass_on_live_klines(tmp_data_root) -> None:  # noqa: ARG001
    from downloader.validators import validate_ohlcv

    async def _do(rest):
        end = pd.Timestamp.utcnow().tz_localize(None).tz_localize("UTC")
        start = end - pd.Timedelta(hours=6)
        df = await rest.klines("BTCUSDT", "15", start, end)
        report = validate_ohlcv(df, "BTCUSDT", "15")
        assert report.status in ("PASS", "WARN"), report.as_dict()

    await _with_clients(_do)
