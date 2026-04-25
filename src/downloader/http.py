"""Async HTTP client with token-bucket rate limit and jittered retry.

One shared `HttpClient` is opened by the scanner/exec loop via
``async with`` and handed to REST/archive clients. It enforces a global
minimum delay between calls (default ~10 req/s) and retries 429/5xx
with exponential backoff + jitter.

Bybit v5 public endpoints wrap responses in ``{retCode, retMsg, result}``.
Use ``envelope=True`` (the default) to unwrap and surface retCode != 0
as ``BybitApiError``.
"""
from __future__ import annotations

import asyncio
import logging
import random
from types import TracebackType
from typing import Any

import aiohttp

from core.config import Config
from downloader.errors import BybitApiError, HttpError

log = logging.getLogger(__name__)


class _RateLimiter:
    """Ensures at least `min_delay` seconds between successive sends."""

    def __init__(self, min_delay_sec: float) -> None:
        self._min_delay = max(0.0, min_delay_sec)
        self._last = 0.0
        self._lock = asyncio.Lock()

    async def wait(self) -> None:
        async with self._lock:
            loop = asyncio.get_event_loop()
            sleep_for = self._min_delay - (loop.time() - self._last)
            if sleep_for > 0:
                await asyncio.sleep(sleep_for)
            self._last = loop.time()


class HttpClient:
    """Async JSON/bytes client scoped by ``async with``.

    Construct once per process. Pass the instance to ``RestClient`` and
    ``ArchiveClient`` so they share the rate limit and connection pool.
    """

    def __init__(
        self,
        cfg: Config,
        *,
        base_url: str | None = None,
        user_agent: str = "v5-orchestrator/0.0.1",
    ) -> None:
        self._base_url = (base_url or cfg.bybit_rest_base).rstrip("/")
        self._timeout = aiohttp.ClientTimeout(total=cfg.http_timeout_sec)
        self._max_retries = cfg.http_max_retries
        self._backoff_base = cfg.http_backoff_base_sec
        self._rate = _RateLimiter(cfg.http_rate_delay_sec)
        self._user_agent = user_agent
        self._session: aiohttp.ClientSession | None = None

    @property
    def base_url(self) -> str:
        return self._base_url

    async def __aenter__(self) -> "HttpClient":
        self._session = aiohttp.ClientSession(
            timeout=self._timeout,
            headers={"User-Agent": self._user_agent},
        )
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None

    async def get_json(
        self,
        path: str,
        params: dict[str, Any] | None = None,
        *,
        envelope: bool = True,
    ) -> Any:
        """GET a JSON path relative to the base URL.

        With ``envelope=True`` (default) the Bybit ``{retCode, retMsg,
        result}`` wrapper is validated and the ``result`` dict is
        returned. Otherwise the parsed JSON is returned as-is.
        """
        if self._session is None:
            raise RuntimeError("HttpClient used outside 'async with' block")
        url = f"{self._base_url}{path}"
        last_exc: Exception | None = None

        for attempt in range(self._max_retries + 1):
            await self._rate.wait()
            try:
                async with self._session.get(url, params=params) as resp:
                    status = resp.status
                    if status in (429, 500, 502, 503, 504):
                        last_exc = HttpError(status, url, await resp.text())
                        if attempt < self._max_retries:
                            await self._backoff(attempt)
                            continue
                        raise last_exc
                    if status != 200:
                        raise HttpError(status, url, await resp.text())
                    data = await resp.json()
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                last_exc = exc
                if attempt < self._max_retries:
                    log.debug("transport error on %s: %s (retry %d)", path, exc, attempt + 1)
                    await self._backoff(attempt)
                    continue
                raise HttpError(0, url, str(exc)) from exc

            if not envelope:
                return data
            ret_code = int(data.get("retCode", -1))
            if ret_code != 0:
                raise BybitApiError(ret_code, str(data.get("retMsg", "")), url)
            return data.get("result") or {}

        raise HttpError(0, url, f"retries exhausted: {last_exc!r}")

    async def get_bytes(self, url: str) -> bytes:
        """Fetch a raw binary URL (e.g. the public archive CSV.gz).

        Takes an absolute URL (the archive is on a different host).
        """
        if self._session is None:
            raise RuntimeError("HttpClient used outside 'async with' block")
        last_exc: Exception | None = None
        for attempt in range(self._max_retries + 1):
            await self._rate.wait()
            try:
                async with self._session.get(url) as resp:
                    if resp.status == 404:
                        raise HttpError(404, url, "not found")
                    if resp.status in (429, 500, 502, 503, 504):
                        last_exc = HttpError(resp.status, url, "")
                        if attempt < self._max_retries:
                            await self._backoff(attempt)
                            continue
                        raise last_exc
                    if resp.status != 200:
                        raise HttpError(resp.status, url, await resp.text())
                    return await resp.read()
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                last_exc = exc
                if attempt < self._max_retries:
                    await self._backoff(attempt)
                    continue
                raise HttpError(0, url, str(exc)) from exc
        raise HttpError(0, url, f"retries exhausted: {last_exc!r}")

    async def _backoff(self, attempt: int) -> None:
        delay = self._backoff_base * (2 ** attempt)
        await asyncio.sleep(random.uniform(0.0, delay))
