"""Utility functions for Webshare proxy rotation and per-platform
rate limit enforcement. Called by scraper_tools.py and apply_tools.py."""

import os
import time
import logging
import itertools
import threading
from typing import Optional

logger = logging.getLogger(__name__)

__all__ = ["get_next_proxy", "rate_limit_wait", "get_proxy_dict", "reset_proxy_cycle", "ProxyRateLimiter"]


def _load_proxies() -> list[str]:
    """Load all Webshare proxy URLs from environment variables.

    Reads WEBSHARE_PROXY_1_1 through WEBSHARE_PROXY_1_10
    for individual proxy slots, then WEBSHARE_PROXY_LIST_1
    as a comma-separated fallback. Deduplicates the result.

    Returns:
        List of proxy URL strings in http://user:pass@host:port
        format. Empty list if no proxies configured.
    """
    proxies: list[str] = []

    # Load individual proxy slots 1-10
    for i in range(1, 11):
        val = os.getenv(f"WEBSHARE_PROXY_1_{i}", "").strip()
        if val:
            proxies.append(val)

    # Comma-separated list as fallback/supplement
    list_val = os.getenv("WEBSHARE_PROXY_LIST_1", "").strip()
    if list_val:
        for p in list_val.split(","):
            p = p.strip()
            if p and p not in proxies:
                proxies.append(p)

    if not proxies:
        logger.warning(
            "proxy_ratelimit: no WEBSHARE_PROXY_1_* keys set "
            "— running without proxy"
        )
    return proxies


_proxy_list: list[str] = _load_proxies()
_proxy_lock = threading.Lock()
_proxy_cycle = itertools.cycle(_proxy_list) if _proxy_list else None
_proxy_index: int = 0


def get_next_proxy() -> Optional[str]:
    """Retrieve the next proxy in a thread-safe round-robin fashion."""
    global _proxy_index
    _proxy_lock.acquire()
    try:
        if _proxy_cycle is None:
            logger.debug("No proxies configured — running without proxy")
            return None
        _proxy_index += 1
        return next(_proxy_cycle)
    finally:
        _proxy_lock.release()


def get_proxy_dict() -> Optional[dict[str, str]]:
    """Retrieve Playwright- and Requests-compatible dictionary for the next proxy."""
    proxy_url = get_next_proxy()
    if proxy_url is None:
        return None
        
    return {
        "server": proxy_url,
        "http": proxy_url,
        "https": proxy_url
    }


def rate_limit_wait(platform: str, seconds: float) -> None:
    """Sleep for the specified number of seconds to respect platform rate limitations."""
    logger.debug(f"rate_limit_wait: {seconds:.1f}s | platform={platform}")
    time.sleep(seconds)
    if seconds > 10:
        logger.warning(f"Long rate limit wait: {seconds:.1f}s for {platform}")


def reset_proxy_cycle() -> None:
    """Reset proxy iterator for testing purposes."""
    global _proxy_cycle, _proxy_index
    _proxy_lock.acquire()
    try:
        if _proxy_list:
            _proxy_cycle = itertools.cycle(_proxy_list)
            _proxy_index = 0
        logger.debug("proxy_cycle reset")
    finally:
        _proxy_lock.release()


class ProxyRateLimiter:
    """OOP wrapper around module-level proxy rotation state.

    Thin class interface for code that prefers OOP-style access to
    proxy rotation. All methods delegate to the module-level functions
    and shared state (_proxy_list, _proxy_lock, _proxy_cycle).
    """

    def get_next_proxy(self) -> Optional[str]:
        """Return the next proxy URL string in round-robin rotation.

        Returns:
            Proxy URL string or None if no proxies configured.
        """
        return get_next_proxy()

    def get_proxy_dict(self) -> Optional[dict[str, str]]:
        """Return the next proxy as a Playwright- and Requests-compatible dict.

        Returns:
            Dict with 'server', 'http', and 'https' keys, or None.
        """
        return get_proxy_dict()

    def reset_cycle(self) -> None:
        """Reset proxy rotation back to index 0."""
        reset_proxy_cycle()

    def rotate(self) -> None:
        """Advance to the next proxy (for use after a proxy failure)."""
        get_next_proxy()


# Thread-safe singleton creation lock
_default_limiter_lock: threading.Lock = threading.Lock()
_default_limiter: Optional["ProxyRateLimiter"] = None


def _get_default_limiter() -> ProxyRateLimiter:
    """Get or create the module-level default ProxyRateLimiter.

    Thread-safe: uses a dedicated lock to prevent double-initialisation
    under concurrent import.

    Returns:
        Singleton ProxyRateLimiter instance.
    """
    global _default_limiter
    with _default_limiter_lock:
        if _default_limiter is None:
            _default_limiter = ProxyRateLimiter()
    return _default_limiter
