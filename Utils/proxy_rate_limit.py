"""Utility functions for Webshare proxy rotation and per-platform
rate limit enforcement.

This module is the spec-compliant proxy helper referenced by audit A3.
It is fail-soft: missing proxy config never crashes the pipeline.
"""

from __future__ import annotations

import itertools
import logging
import os
import threading
import time
from typing import Optional

logger = logging.getLogger(__name__)

__all__ = [
    "get_next_proxy",
    "rate_limit_wait",
    "get_proxy_dict",
    "reset_proxy_cycle",
    "ProxyRateLimiter",
]


def _load_proxies() -> list[str]:
    """Load Webshare proxy URLs from environment.

    Reads:
      - WEBSHARE_PROXY_LIST (comma-separated list of proxy URLs)
      - WEBSHARE_PROXY_URL  (single primary proxy URL)

    Fallback rules:
      - If WEBSHARE_PROXY_LIST is empty, fall back to [WEBSHARE_PROXY_URL]
      - If both are missing/empty, log a warning and return []

    Returns:
        List of proxy URL strings in the format accepted by requests/httpx.
    """
    proxy_list_str = (os.getenv("WEBSHARE_PROXY_LIST") or "").strip()
    proxies: list[str] = [p.strip() for p in proxy_list_str.split(",") if p.strip()]

    if not proxies:
        primary = (os.getenv("WEBSHARE_PROXY_URL") or "").strip()
        if primary:
            proxies = [primary]
        else:
            logger.warning(
                "proxy_rate_limit: missing WEBSHARE_PROXY_LIST and WEBSHARE_PROXY_URL — running without proxy"
            )
            proxies = []

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for p in proxies:
        if p not in seen:
            unique.append(p)
            seen.add(p)
    return unique


_proxy_list: list[str] = _load_proxies()
_proxy_lock = threading.Lock()
_proxy_cycle = itertools.cycle(_proxy_list) if _proxy_list else None
_proxy_index: int = 0


def get_next_proxy() -> Optional[str]:
    """Retrieve the next proxy in a thread-safe round-robin fashion."""
    global _proxy_index
    with _proxy_lock:
        if _proxy_cycle is None:
            logger.debug("No proxies configured — running without proxy")
            return None
        _proxy_index += 1
        return next(_proxy_cycle)


def get_proxy_dict(platform: str = "") -> Optional[dict[str, str]]:
    """Return a requests/httpx-compatible proxy dict, or None if disabled.

    Args:
        platform: Optional platform label for logging context.

    Returns:
        Dict with "http" and "https" keys, or None.
    """
    try:
        proxy_url = get_next_proxy()
        if proxy_url is None:
            return None
        return {"http": proxy_url, "https": proxy_url}
    except Exception as e:  # noqa: BLE001
        if platform:
            logger.warning("get_proxy_dict(%s) failed — running without proxy: %s", platform, e)
        else:
            logger.warning("get_proxy_dict failed — running without proxy: %s", e)
        return None


def rate_limit_wait(platform: str, seconds: float) -> None:
    """Sleep for the specified number of seconds to respect rate limitations."""
    logger.debug("rate_limit_wait: %.1fs | platform=%s", seconds, platform)
    time.sleep(seconds)
    if seconds > 10:
        logger.warning("Long rate limit wait: %.1fs for %s", seconds, platform)


def reset_proxy_cycle() -> None:
    """Reset proxy iterator for testing purposes."""
    global _proxy_cycle, _proxy_index
    with _proxy_lock:
        if _proxy_list:
            _proxy_cycle = itertools.cycle(_proxy_list)
            _proxy_index = 0
        logger.debug("proxy_cycle reset")


class ProxyRateLimiter:
    """OOP wrapper around module-level proxy rotation state."""

    def get_next_proxy(self) -> Optional[str]:
        """Return the next proxy URL string in round-robin rotation."""
        return get_next_proxy()

    def get_proxy_dict(self) -> Optional[dict[str, str]]:
        """Return the next proxy as a requests/httpx-compatible dict."""
        return get_proxy_dict()

    def reset_cycle(self) -> None:
        """Reset proxy rotation back to index 0."""
        reset_proxy_cycle()

    def rotate(self) -> None:
        """Advance to the next proxy (for use after a proxy failure)."""
        get_next_proxy()

