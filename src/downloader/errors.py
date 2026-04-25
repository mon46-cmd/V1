"""Downloader error taxonomy.

All network and cache failures raised by `src/downloader/*` should be one
of these; callers can decide whether to retry, log, or propagate.
"""
from __future__ import annotations


class DownloaderError(Exception):
    """Base class for every exception raised by the downloader."""


class HttpError(DownloaderError):
    """Raised when a transport call fails (non-2xx, timeout, conn reset)."""

    def __init__(self, status: int, url: str, body: str) -> None:
        super().__init__(f"HTTP {status} {url}: {body[:200]}")
        self.status = status
        self.url = url
        self.body = body


class BybitApiError(DownloaderError):
    """Raised when Bybit returns retCode != 0 in a JSON envelope."""

    def __init__(self, ret_code: int, ret_msg: str, url: str) -> None:
        super().__init__(f"Bybit ret={ret_code} msg={ret_msg!r} url={url}")
        self.ret_code = ret_code
        self.ret_msg = ret_msg
        self.url = url


class CacheError(DownloaderError):
    """Raised on unrecoverable cache read/write problems."""
