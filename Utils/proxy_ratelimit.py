"""Utility functions for Webshare proxy rotation and per-platform
rate limit enforcement. Called by scraper_tools.py and apply_tools.py."""

import os
import time
import logging
import itertools
import threading
from typing import Optional

logger = logging.getLogger(__name__)

__all__ = ["get_next_proxy", "rate_limit_wait", "get_proxy_dict", "reset_proxy_cycle"]

_proxy_list: list[str] = [p.strip() for p in os.getenv("WEBSHARE_PROXY_LIST", "").split(",") if p.strip()]
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
