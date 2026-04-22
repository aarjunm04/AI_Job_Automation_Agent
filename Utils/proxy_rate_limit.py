"""utils/proxy_rate_limit.py

Thread-safe proxy pool manager for all 20 Webshare proxies.
Provides round-robin rotation, dead-proxy tracking with auto-recovery,
per-proxy rate limiting, and Playwright-compatible proxy dicts.

All consumers (scraper_tools, apply_tools, scraper_service) import
from this module. No WEBSHARE_PROXY_LIST anywhere in the codebase.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from threading import Lock, Event
from typing import Dict, List, Optional, Tuple

import agentops
from agentops.sdk.decorators import operation

__all__ = [
    "ProxyPool",
    "get_next_proxy",
    "get_proxy_dict",
    "get_playwright_proxy",
    "mark_proxy_dead",
    "mark_proxy_success",
    "reset_cycle",
    "get_proxy_stats",
    "is_proxy_pool_healthy",
    "health_check_proxies",
]

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("LOG_LEVEL", "INFO"))

# ---------------------------------------------------------------------------
# CONFIG — all from env, no hardcoding
# ---------------------------------------------------------------------------

PROXY_DEAD_TIMEOUT_SECONDS: int = int(os.getenv("PROXY_DEAD_TIMEOUT_SECONDS", "300"))
PROXY_MAX_RETRIES: int = int(os.getenv("PROXY_MAX_RETRIES", "2"))
PROXY_REQUEST_TIMEOUT_SECONDS: int = int(os.getenv("PROXY_REQUEST_TIMEOUT_SECONDS", "15"))
PROXY_ROTATION_METHOD: str = os.getenv("PROXY_ROTATION_METHOD", "round_robin")
PROXY_FAILOVER_ENABLED: bool = os.getenv("PROXY_FAILOVER_ENABLED", "true").lower() == "true"
# Rate limit: max requests per proxy per minute window
PROXY_RATE_LIMIT_RPM: int = int(os.getenv("PROXY_RATE_LIMIT_RPM", "20"))

# ---------------------------------------------------------------------------
# PROXY REGISTRY — 20 individual env vars, 2 accounts × 10 proxies
# ---------------------------------------------------------------------------

_PROXY_ENV_KEYS: List[str] = [
    # Account 1 — 10 proxies
    "WEBSHARE_PROXY_1_1",
    "WEBSHARE_PROXY_1_2",
    "WEBSHARE_PROXY_1_3",
    "WEBSHARE_PROXY_1_4",
    "WEBSHARE_PROXY_1_5",
    "WEBSHARE_PROXY_1_6",
    "WEBSHARE_PROXY_1_7",
    "WEBSHARE_PROXY_1_8",
    "WEBSHARE_PROXY_1_9",
    "WEBSHARE_PROXY_1_10",
    # Account 2 — 10 proxies
    "WEBSHARE_PROXY_2_1",
    "WEBSHARE_PROXY_2_2",
    "WEBSHARE_PROXY_2_3",
    "WEBSHARE_PROXY_2_4",
    "WEBSHARE_PROXY_2_5",
    "WEBSHARE_PROXY_2_6",
    "WEBSHARE_PROXY_2_7",
    "WEBSHARE_PROXY_2_8",
    "WEBSHARE_PROXY_2_9",
    "WEBSHARE_PROXY_2_10",
]

# ---------------------------------------------------------------------------
# DATA STRUCTURES
# ---------------------------------------------------------------------------

@dataclass
class ProxyEntry:
    """Single proxy with state tracking.

    Attributes:
        key: The env var key this proxy was loaded from.
        url: Full proxy URL — http://username:password@ip:port
        is_dead: Whether this proxy is currently marked dead.
        dead_since: Epoch timestamp when proxy was marked dead.
        request_count: Total requests routed through this proxy.
        success_count: Total successful requests.
        fail_count: Total failed requests.
        last_used: Epoch timestamp of last use.
        rpm_window_start: Start of current rate-limit window (epoch).
        rpm_window_count: Request count in current 60s window.
    """

    key: str
    url: str
    is_dead: bool = False
    dead_since: float = 0.0
    request_count: int = 0
    success_count: int = 0
    fail_count: int = 0
    last_used: float = 0.0
    rpm_window_start: float = field(default_factory=time.time)
    rpm_window_count: int = 0

    @property
    def parsed(self) -> Dict[str, str]:
        """Parse proxy URL into components.

        Returns:
            Dict with keys: scheme, username, password, host, port, url
        """
        # Format: http://username:password@ip:port
        try:
            scheme = self.url.split("://")[0]
            rest = self.url.split("://")[1]
            creds, hostport = rest.rsplit("@", 1)
            username, password = creds.split(":", 1)
            host, port = hostport.rsplit(":", 1)
            return {
                "scheme": scheme,
                "username": username,
                "password": password,
                "host": host,
                "port": port,
                "url": self.url,
            }
        except Exception as exc:
            logger.error("Failed to parse proxy URL for %s: %s", self.key, exc)
            return {"url": self.url, "scheme": "http", "username": "",
                    "password": "", "host": "", "port": ""}

    @property
    def is_rate_limited(self) -> bool:
        """Check if proxy has exceeded RPM rate limit.

        Returns:
            True if rate limit exceeded in current window.
        """
        now = time.time()
        if now - self.rpm_window_start >= 60.0:
            # Window expired — reset
            self.rpm_window_start = now
            self.rpm_window_count = 0
            return False
        return self.rpm_window_count >= PROXY_RATE_LIMIT_RPM

    def record_use(self) -> None:
        """Increment usage counters for this proxy."""
        now = time.time()
        self.last_used = now
        self.request_count += 1
        if now - self.rpm_window_start >= 60.0:
            self.rpm_window_start = now
            self.rpm_window_count = 1
        else:
            self.rpm_window_count += 1

    def maybe_recover(self) -> bool:
        """Auto-recover proxy if dead timeout has elapsed.

        Returns:
            True if proxy was recovered.
        """
        if self.is_dead and (time.time() - self.dead_since) >= PROXY_DEAD_TIMEOUT_SECONDS:
            self.is_dead = False
            self.dead_since = 0.0
            logger.info("Proxy %s auto-recovered after timeout", self.key)
            return True
        return False


# ---------------------------------------------------------------------------
# PROXY POOL — singleton
# ---------------------------------------------------------------------------

class ProxyPool:
    """Thread-safe round-robin proxy pool for all 20 Webshare proxies.

    Loads proxies from individual env vars at init time.
    Provides rotation, dead tracking, rate limit enforcement,
    and Playwright/requests-compatible proxy dicts.

    Usage::

        pool = ProxyPool.get_instance()
        proxy = pool.next()            # ProxyEntry or None
        d = pool.as_requests_dict()    # {"http": ..., "https": ...}
        pool.mark_dead("WEBSHARE_PROXY_1_3")
        pool.mark_success("WEBSHARE_PROXY_1_3")
    """

    _instance: Optional["ProxyPool"] = None
    _lock: Lock = Lock()

    def __init__(self) -> None:
        """Load all proxies from env and initialise pool state."""
        self._proxies: List[ProxyEntry] = []
        self._index: int = 0
        self._rotation_lock: Lock = Lock()
        self._load_proxies()
        logger.info(
            "ProxyPool initialised — %d/%d proxies loaded",
            len(self._proxies),
            len(_PROXY_ENV_KEYS),
        )

    @classmethod
    def get_instance(cls) -> "ProxyPool":
        """Return the singleton ProxyPool instance (thread-safe).

        Returns:
            The singleton ProxyPool.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Destroy the singleton (for testing only).

        Warning:
            Never call this in production code.
        """
        with cls._lock:
            cls._instance = None

    # ------------------------------------------------------------------
    # Private: loading
    # ------------------------------------------------------------------

    def _load_proxies(self) -> None:
        """Load all 20 proxy env vars into the pool.

        Missing or empty env vars are skipped with a warning.
        """
        loaded = 0
        missing = 0
        for key in _PROXY_ENV_KEYS:
            url = os.getenv(key, "").strip()
            if not url:
                logger.warning("ProxyPool: %s is not set — skipping", key)
                missing += 1
                continue
            if not url.startswith("http://") and not url.startswith("https://"):
                logger.warning(
                    "ProxyPool: %s has invalid URL format '%s' — skipping", key, url[:40]
                )
                missing += 1
                continue
            self._proxies.append(ProxyEntry(key=key, url=url))
            loaded += 1

        if loaded == 0:
            logger.critical(
                "ProxyPool: NO proxies loaded — all WEBSHARE_PROXY_* keys missing. "
                "Scraping will run without proxy protection."
            )
        elif missing > 0:
            logger.warning("ProxyPool: %d proxy keys missing from env", missing)

    # ------------------------------------------------------------------
    # Public: rotation
    # ------------------------------------------------------------------

    def next(self) -> Optional[ProxyEntry]:
        """Return the next available proxy using round-robin.

        Skips dead proxies (unless auto-recovery timeout has elapsed)
        and rate-limited proxies. Returns None only if every proxy
        is dead or rate-limited.

        Returns:
            ProxyEntry if a proxy is available, None otherwise.
        """
        if not self._proxies:
            return None

        with self._rotation_lock:
            total = len(self._proxies)
            # Attempt up to full cycle to find a live proxy
            for _ in range(total):
                entry = self._proxies[self._index % total]
                self._index = (self._index + 1) % total

                # Auto-recover if timeout elapsed
                entry.maybe_recover()

                if entry.is_dead:
                    continue
                if entry.is_rate_limited:
                    logger.debug("Proxy %s is rate-limited — skipping", entry.key)
                    continue

                entry.record_use()
                logger.debug(
                    "ProxyPool: dispatching %s (req#%d)",
                    entry.key,
                    entry.request_count,
                )
                return entry

            # All proxies dead or rate-limited
            logger.error(
                "ProxyPool: all %d proxies are dead or rate-limited", total
            )
            return None

    def mark_dead(self, key: str) -> None:
        """Mark a proxy as dead by its env key.

        Args:
            key: The env var key, e.g. ``"WEBSHARE_PROXY_1_3"``.
        """
        with self._rotation_lock:
            for entry in self._proxies:
                if entry.key == key:
                    if not entry.is_dead:
                        entry.is_dead = True
                        entry.dead_since = time.time()
                        entry.fail_count += 1
                        logger.warning(
                            "ProxyPool: %s marked dead (fail_count=%d)",
                            key,
                            entry.fail_count,
                        )
                    return
        logger.warning("ProxyPool.mark_dead: key %s not found in pool", key)

    def mark_success(self, key: str) -> None:
        """Record a successful request for a proxy.

        Args:
            key: The env var key, e.g. ``"WEBSHARE_PROXY_1_3"``.
        """
        with self._rotation_lock:
            for entry in self._proxies:
                if entry.key == key:
                    entry.success_count += 1
                    return

    # ------------------------------------------------------------------
    # Public: proxy format helpers
    # ------------------------------------------------------------------

    def as_requests_dict(self, entry: Optional[ProxyEntry] = None) -> Dict[str, str]:
        """Return a requests-library-compatible proxy dict.

        Args:
            entry: Specific ProxyEntry to use, or None to auto-pick next.

        Returns:
            Dict ``{"http": url, "https": url}`` or empty dict if no proxy.
        """
        target = entry or self.next()
        if target is None:
            return {}
        return {"http": target.url, "https": target.url}

    def as_playwright_dict(self, entry: Optional[ProxyEntry] = None) -> Optional[Dict[str, str]]:
        """Return a Playwright-compatible proxy dict.

        Args:
            entry: Specific ProxyEntry to use, or None to auto-pick next.

        Returns:
            Dict ``{"server": url, "username": u, "password": p}``
            or None if no proxy available.
        """
        target = entry or self.next()
        if target is None:
            return None
        parsed = target.parsed
        return {
            "server": f"{parsed['scheme']}://{parsed['host']}:{parsed['port']}",
            "username": parsed["username"],
            "password": parsed["password"],
        }

    def as_httpx_dict(self, entry: Optional[ProxyEntry] = None) -> Optional[str]:
        """Return an httpx-compatible proxy URL string.

        Args:
            entry: Specific ProxyEntry to use, or None to auto-pick next.

        Returns:
            Proxy URL string or None if no proxy available.
        """
        target = entry or self.next()
        if target is None:
            return None
        return target.url

    # ------------------------------------------------------------------
    # Public: stats and health
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, object]:
        """Return a summary of the current pool state.

        Returns:
            Dict with total, alive, dead, rate_limited, per-proxy detail.
        """
        with self._rotation_lock:
            alive = 0
            dead = 0
            rate_limited = 0
            per_proxy = []
            for entry in self._proxies:
                entry.maybe_recover()
                status = "dead" if entry.is_dead else (
                    "rate_limited" if entry.is_rate_limited else "alive"
                )
                if status == "alive":
                    alive += 1
                elif status == "dead":
                    dead += 1
                else:
                    rate_limited += 1

                per_proxy.append({
                    "key": entry.key,
                    "status": status,
                    "request_count": entry.request_count,
                    "success_count": entry.success_count,
                    "fail_count": entry.fail_count,
                    "rpm_window_count": entry.rpm_window_count,
                })

            return {
                "total": len(self._proxies),
                "alive": alive,
                "dead": dead,
                "rate_limited": rate_limited,
                "proxies": per_proxy,
            }

    def is_healthy(self) -> bool:
        """Return True if at least one proxy is alive and not rate-limited.

        Returns:
            Boolean health status.
        """
        s = self.stats()
        return int(s["alive"]) > 0

    def reset_all(self) -> None:
        """Force-recover all dead proxies and reset rate limit windows.

        Use for testing or manual recovery only.
        """
        with self._rotation_lock:
            for entry in self._proxies:
                entry.is_dead = False
                entry.dead_since = 0.0
                entry.rpm_window_start = time.time()
                entry.rpm_window_count = 0
            self._index = 0
        logger.info("ProxyPool: all proxies force-reset")


# ---------------------------------------------------------------------------
# MODULE-LEVEL CONVENIENCE FUNCTIONS
# (used by scraper_tools, apply_tools, scraper_service)
# ---------------------------------------------------------------------------

@operation
def get_next_proxy() -> Optional[ProxyEntry]:
    """Get the next available proxy from the singleton pool.

    Returns:
        ProxyEntry if available, None if all proxies dead/rate-limited.
    """
    return ProxyPool.get_instance().next()


@operation
def get_proxy_dict() -> Dict[str, str]:
    """Get next proxy as a requests-compatible dict.

    Returns:
        ``{"http": url, "https": url}`` or empty dict.
    """
    return ProxyPool.get_instance().as_requests_dict()


@operation
def get_playwright_proxy() -> Optional[Dict[str, str]]:
    """Get next proxy as a Playwright-compatible dict.

    Returns:
        ``{"server": ..., "username": ..., "password": ...}`` or None.
    """
    return ProxyPool.get_instance().as_playwright_dict()


@operation
def get_httpx_proxy() -> Optional[str]:
    """Get next proxy as an httpx URL string.

    Returns:
        Proxy URL string or None.
    """
    return ProxyPool.get_instance().as_httpx_dict()


@operation
def mark_proxy_dead(key: str) -> None:
    """Mark a proxy dead by its env key.

    Args:
        key: Env var key, e.g. ``"WEBSHARE_PROXY_1_3"``.
    """
    ProxyPool.get_instance().mark_dead(key)


@operation
def mark_proxy_success(key: str) -> None:
    """Record a successful request for a proxy.

    Args:
        key: Env var key, e.g. ``"WEBSHARE_PROXY_1_3"``.
    """
    ProxyPool.get_instance().mark_success(key)


@operation
def reset_cycle() -> None:
    """Force-reset all dead proxies and rate-limit windows.

    Use only in test environments or manual recovery scenarios.
    """
    ProxyPool.get_instance().reset_all()


@operation
def get_proxy_stats() -> Dict[str, object]:
    """Return current pool statistics dict.

    Returns:
        Stats dict with total/alive/dead/rate_limited and per-proxy detail.
    """
    return ProxyPool.get_instance().stats()


@operation
def is_proxy_pool_healthy() -> bool:
    """Return True if at least one proxy is alive and not rate-limited.

    Returns:
        Boolean pool health status.
    """
    return ProxyPool.get_instance().is_healthy()


# ---------------------------------------------------------------------------
# PROXY HEALTH CHECK — filter dead proxies at boot
# ---------------------------------------------------------------------------

async def health_check_proxies(
    proxies: list[str],
    test_url: str = "https://httpbin.org/ip",
    timeout_ms: int = 8000,
) -> list[str]:
    """Check each proxy with a lightweight HEAD request.
    Returns only the proxies that respond within timeout_ms.

    Args:
        proxies: List of proxy strings in host:port:user:pass format.
        test_url: URL to test connectivity against.
        timeout_ms: Timeout in milliseconds per proxy.

    Returns:
        Filtered list of live proxies only.
    """
    from playwright.async_api import async_playwright
    live: list[str] = []
    async with async_playwright() as pw:
        for proxy_str in proxies:
            try:
                parts = proxy_str.split(":")
                proxy_cfg = {
                    "server": f"http://{parts[0]}:{parts[1]}",
                    "username": parts[2] if len(parts) > 2 else "",
                    "password": parts[3] if len(parts) > 3 else "",
                }
                browser = await pw.chromium.launch(proxy=proxy_cfg)
                page = await browser.new_page()
                await page.goto(test_url, timeout=timeout_ms)
                await browser.close()
                live.append(proxy_str)
                logger.debug("Proxy live: %s", parts[0])
            except Exception as e:
                logger.warning(
                    "Proxy dead (culled): %s — %s", proxy_str[:20], e
                )
    logger.info(
        "Proxy health check: %d/%d live", len(live), len(proxies)
    )
    return live
