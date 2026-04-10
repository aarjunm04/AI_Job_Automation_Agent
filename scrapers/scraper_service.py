"""
scrapers/scraper_service.py

ENTERPRISE-GRADE PLAYWRIGHT SCRAPER STACK
=========================================

Playwright infrastructure + site-specific scrapers for the AI Job Automation
Agent.

Responsibilities:
├── Load Webshare static proxies from ~/java.env
├── Round-robin proxy rotation + 1 GB bandwidth accounting per proxy
├── Single shared Chromium browser with stealth hardening
├── Site scrapers (Phase 1 + safety-net):
│     Wellfound, WeWorkRemotely, YC/WorkAtAStartup
│     Turing, Crossover, Arc.dev, Nodesk, Toptal
├── Production retry logic + failure isolation
└── Raw job extraction (no normalization/dedupe)

Legacy classes kept (not exported):
    RemoteOKAPIScraper, SimplyHiredScraper, StackOverflowScraper, HiringCafeScraper

Usage by ScraperEngine:
    scraper = WellfoundScraper(jobs_limit=20)
    raw_jobs = await scraper.run(playwright_manager)
"""

from __future__ import annotations

import os
import time
import json
import atexit
import asyncio
import logging
import random
import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

# External dependencies
from dotenv import load_dotenv
import requests as _http_requests
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field
from playwright.async_api import async_playwright, Browser, BrowserContext

from config.config_loader import config_loader

# =================================================================================
# LOGGING
# =================================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("playwright_scrapers.log"),
    ],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ENVIRONMENT LOADING
# ---------------------------------------------------------------------------
_env_path = os.getenv("DOTENV_PATH", os.path.expanduser("~/java.env"))
if os.path.exists(_env_path):
    load_dotenv(dotenv_path=_env_path, override=False)


# =================================================================================
# STEALTH HELPERS (module-level — shared by all scrapers)
# =================================================================================


async def human_delay(min_ms: int = 800, max_ms: int = 2400) -> None:
    """Wait a random human-like delay between min_ms and max_ms milliseconds."""
    await asyncio.sleep(random.randint(min_ms, max_ms) / 1000)


async def human_type(page: Any, selector: str, text: str) -> None:
    """Type text into an input field character-by-character to mimic human input.

    Args:
        page: Playwright Page object.
        selector: CSS selector of the input element to type into.
        text: The string to type.
    """
    await page.click(selector)
    for char in text:
        await page.type(selector, char, delay=random.randint(40, 120))


# =================================================================================
# PROXY INFRASTRUCTURE
# =================================================================================


@dataclass
class ProxyNode:
    """Individual proxy with usage tracking."""

    server: str
    username: str
    password: str
    used_mb: int = 0
    failures: int = 0
    cooldown_until: float = 0.0

    @property
    def available(self) -> bool:
        return (
            self.used_mb < 1024  # 1 GB limit per proxy
            and self.failures < 5  # 5-failure threshold
            and time.time() > self.cooldown_until
        )

    def mark_failure(self) -> None:
        """Record a proxy failure and apply cooldown at threshold."""
        self.failures += 1
        if self.failures >= 5:
            self.cooldown_until = time.time() + 300  # 5-minute cooldown

    def record_bandwidth(self, kb: int) -> None:
        """Track bandwidth usage in MB (minimum 1 MB per call)."""
        self.used_mb += max(1, kb // 1024)


class ProxyManager:
    """Manages static Webshare proxies loaded from ~/java.env."""

    def __init__(self) -> None:

        self.proxies: List[ProxyNode] = []
        self._proxy_index: int = 0
        self._lock = asyncio.Lock()
        self._load_static_proxies()
        logger.info("Loaded %d static proxies", len(self.proxies))

    def _load_static_proxies(self) -> None:
        """
        Parse proxy URLs from env vars.

        Supported var names:
            WEBSHARE_PROXY_2_1 .. WEBSHARE_PROXY_2_10
            PROXY_1 .. PROXY_10

        Expected URL format: http://username:password@ip:port
        Parse failures are logged as warnings and never raise.
        """
        env_keys: List[str] = [f"WEBSHARE_PROXY_2_{i}" for i in range(1, 11)]
        env_keys.extend([f"PROXY_{i}" for i in range(1, 11)])

        for key in env_keys:
            raw = os.getenv(key)
            if not raw:
                continue
            try:
                # Normalize protocol
                raw = raw.replace("https://", "http://")
                raw_no_proto = raw.replace("http://", "")
                auth_part, server_part = raw_no_proto.split("@", 1)
                username, password = auth_part.split(":", 1)
                node = ProxyNode(
                    server=f"http://{server_part}",
                    username=username,
                    password=password,
                )
                self.proxies.append(node)
                logger.info("Loaded proxy from %s", key)
            except Exception as e:
                if isinstance(e, (KeyboardInterrupt, SystemExit)):
                    raise
                logger.warning("Failed to parse proxy %s: %s", key, e)

        if not self.proxies:
            logger.warning("No valid proxies found. Running without proxies.")

    def select_proxy(self) -> Optional[ProxyNode]:
        """Round-robin selection with fallback to least-failed proxy."""
        if not self.proxies:
            return None

        for _ in range(len(self.proxies)):
            node = self.proxies[self._proxy_index]
            self._proxy_index = (self._proxy_index + 1) % len(self.proxies)
            if node.available:
                return node

        # All proxies degraded — return the least-failed one
        return min(self.proxies, key=lambda p: (p.failures, p.used_mb))

    def rotate_proxy(self) -> Optional[ProxyNode]:
        """Force-advance the round-robin index and return the next available proxy."""
        if not self.proxies:
            return None
        self._proxy_index = (self._proxy_index + 1) % len(self.proxies)
        return self.select_proxy()

    @staticmethod
    def _verify_proxy(proxy_node: ProxyNode, timeout: float = 5.0) -> bool:
        """Verify proxy health with a lightweight HTTP request."""
        try:
            resp = _http_requests.get(
                "https://api.ipify.org?format=json",
                proxies={
                    "http": f"http://{proxy_node.username}:{proxy_node.password}@"
                            f"{proxy_node.server.replace('http://', '')}",
                    "https": f"http://{proxy_node.username}:{proxy_node.password}@"
                             f"{proxy_node.server.replace('http://', '')}",
                },
                timeout=timeout,
            )
            return resp.status_code == 200
        except Exception:
            return False


# =================================================================================
# SHARED BROWSER INFRASTRUCTURE
# =================================================================================


class PlaywrightManager:
    """Centralized browser + proxy lifecycle manager (lazily initialized)."""

    def __init__(self) -> None:
        self.proxy_manager = ProxyManager()
        self._browser: Optional[Browser] = None
        self._lock = asyncio.Lock()
        self._initialized = False
        self._stealth_user_agent = (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )

    async def initialize(self) -> None:
        """Lazy init: launch Chromium with stealth args."""
        async with self._lock:
            if self._initialized:
                return

            playwright = await async_playwright().start()
            self._browser = await playwright.chromium.launch(
                headless=True,
                args=[
                    "--disable-blink-features=AutomationControlled",
                    "--no-sandbox",
                    "--disable-dev-shm-usage",
                    "--disable-web-security",
                    "--disable-features=VizDisplayCompositor",
                ],
            )
            self._initialized = True
            logger.info("Playwright browser + stealth initialized")

    async def new_context(self, require_proxy: bool = False) -> BrowserContext:
        """Create a new browser context with proxy rotation + stealth init script."""
        await self.initialize()

        proxy_node = self.proxy_manager.select_proxy()

        # Verify proxy health — try up to 3 proxies before falling back to direct
        if proxy_node and not self.proxy_manager._verify_proxy(proxy_node):
            logger.warning("Proxy %s failed health check — rotating", proxy_node.server)
            proxy_node.mark_failure()
            for _ in range(2):
                proxy_node = self.proxy_manager.rotate_proxy()
                if proxy_node and self.proxy_manager._verify_proxy(proxy_node):
                    break
                if proxy_node:
                    proxy_node.mark_failure()
                    proxy_node = None
            if not proxy_node:
                if require_proxy:
                    raise RuntimeError("All proxies failed health check — require_proxy set, aborting")
                logger.warning("All proxy candidates failed health check — using direct")

        proxy_config = (
            {
                "server": proxy_node.server,
                "username": proxy_node.username,
                "password": proxy_node.password,
            }
            if proxy_node
            else None
        )

        context = await self._browser.new_context(
            proxy=proxy_config,
            user_agent=self._stealth_user_agent,
            viewport={"width": 1366, "height": 768},
            locale="en-US",
            timezone_id="America/New_York",
            permissions=["geolocation"],
        )

        # Stealth init script: hide webdriver, fake plugins, set languages
        await context.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', {
                get: () => undefined,
            });
            Object.defineProperty(navigator, 'plugins', {
                get: () => [1, 2, 3, 4, 5],
            });
            Object.defineProperty(navigator, 'languages', {
                get: () => ['en-US', 'en'],
            });
        """)

        # Attach proxy node for bandwidth accounting
        if proxy_node:
            context._proxy_node = proxy_node  # type: ignore[attr-defined]

        return context

    async def report_success(self, context: BrowserContext, estimated_kb: int) -> None:
        """Record successful scrape bandwidth usage."""
        proxy_node = getattr(context, "_proxy_node", None)
        if proxy_node:
            proxy_node.record_bandwidth(estimated_kb)

    async def report_failure(self, context: BrowserContext) -> None:
        """Mark associated proxy as failed."""
        proxy_node = getattr(context, "_proxy_node", None)
        if proxy_node:
            proxy_node.mark_failure()

    async def shutdown(self) -> None:
        """Clean shutdown — close browser if open."""
        if self._browser:
            await self._browser.close()
            logger.info("Playwright browser closed")


# =================================================================================
# BASE SCRAPER
# =================================================================================


class BasePlaywrightScraper:
    """Shared logic: retries, scrolling, extraction, bandwidth accounting."""

    # Class-level defaults — subclasses MUST set at minimum name, start_url,
    # job_card_selector, title_selector, and link_selector.
    name: str = "base"
    start_url: str = ""
    job_card_selector: str = ""
    title_selector: str = ""
    company_selector: Optional[str] = None
    location_selector: Optional[str] = None
    link_selector: Optional[str] = None
    scroll_times: int = 3
    max_retries: int = 2
    require_proxy: bool = False

    def __init__(self, jobs_limit: int = 20) -> None:
        self.jobs_limit = jobs_limit

    async def run(self, manager: PlaywrightManager) -> List[Dict[str, Any]]:
        """Entry point: run with exponential-backoff retries."""
        for attempt in range(self.max_retries + 1):
            try:
                return await self._scrape_once(manager)
            except Exception as e:
                logger.warning("%s attempt %d failed: %s", self.name, attempt + 1, e)
                if attempt == self.max_retries:
                    logger.error("%s all retries exhausted", self.name)
                    return []
                await asyncio.sleep(2**attempt)
        return []

    async def _scrape_once(self, manager: PlaywrightManager) -> List[Dict[str, Any]]:
        """Single scrape pass: navigate → scroll → extract cards."""
        results: List[Dict[str, Any]] = []
        context: Optional[BrowserContext] = None

        try:
            context = await manager.new_context(require_proxy=self.require_proxy)
            page = await context.new_page()
            page.set_default_navigation_timeout(15000)
            page.set_default_timeout(10000)

            await page.goto(self.start_url, timeout=15000, wait_until="domcontentloaded")
            logger.debug("%s: navigated to %s", self.name, self.start_url)

            for _ in range(self.scroll_times):
                await page.evaluate("window.scrollBy(0, document.body.scrollHeight)")
                await page.wait_for_timeout(1000 + random.randint(0, 500))

            cards = await page.query_selector_all(self.job_card_selector)
            logger.info("%s: found %d job cards", self.name, len(cards))

            for card in cards[: self.jobs_limit]:
                job = await self._extract_job(card)
                if job:
                    job["source"] = self.name
                    results.append(job)

            estimated_kb = len(results) * 150
            await manager.report_success(context, estimated_kb)

            logger.info("%s: extracted %d jobs", self.name, len(results))
            return results

        except Exception as e:
            logger.error("%s: scrape failed — %s", self.name, e)
            if context:
                await manager.report_failure(context)
            raise
        finally:
            if context:
                try:
                    await context.close()
                except Exception:
                    pass

    async def _extract_job(self, card) -> Optional[Dict[str, Any]]:
        """Extract a raw job dict from a card element."""
        try:
            title = await self._safe_text(card, self.title_selector)
            company = await self._safe_text(card, self.company_selector)
            location = await self._safe_text(card, self.location_selector)
            link = await self._safe_attr(card, self.link_selector, "href")

            if not title or not link:
                return None

            # Make relative links absolute
            if link.startswith("/"):
                from urllib.parse import urlparse
                parsed = urlparse(self.start_url)
                link = f"{parsed.scheme}://{parsed.netloc}{link}"

            return {
                "title": title,
                "company": company,
                "location": location or "Remote",
                "job_url": link,
                "description": "",
            }
        except Exception:
            return None

    async def _safe_text(self, root, selector: Optional[str]) -> str:
        """Safely extract inner text; returns empty string on any failure."""
        if not selector:
            return ""
        try:
            el = await root.query_selector(selector)
            return (await el.inner_text()).strip() if el else ""
        except Exception:
            return ""

    async def _safe_attr(self, root, selector: Optional[str], attr: str) -> str:
        """Safely extract attribute value; returns empty string on any failure."""
        if not selector:
            return ""
        try:
            el = await root.query_selector(selector)
            return (await el.get_attribute(attr)) or "" if el else ""
        except Exception:
            return ""


# =================================================================================
# SITE-SPECIFIC SCRAPERS — PHASE 1 + SAFETY-NET
# =================================================================================


class WellfoundScraper(BasePlaywrightScraper):
    """
    Wellfound (formerly AngelList Talent) startup jobs scraper.

    Browser recon (2026-03-21) confirmed that https://wellfound.com/jobs presents
    a search landing page and company hero section — no job listings are accessible
    without a logged-in session. The session cookies required (including Cloudflare
    clearance) cannot be reliably automated without violating ToS or risking bans.

    This scraper gracefully skips with a log warning when no login session is
    available. If WELLFOUND_SESSION_COOKIE is set in the environment, the cookie
    is injected and the scraper attempts to access the authenticated job feed
    at https://wellfound.com/role/r/software-engineer (public remote listings).
    """

    name = "wellfound"
    start_url = "https://wellfound.com/role/r/software-engineer"
    # Verified selectors from authenticated job feed DOM (Next.js rendered)
    job_card_selector = "div[class*='styles_component'] a[href*='/jobs/'][href*='-at-']"
    title_selector = "span[class*='JobOpening_name']"
    company_selector = "h2[class*='StartupJobStandout_name']"
    location_selector = "span[class*='StartupJobStandout_location']"
    link_selector = "a[href*='/jobs/'][href*='-at-']"
    scroll_times = 5
    max_retries = 2
    _USER_AGENT: str = os.getenv(
        "PLAYWRIGHT_USER_AGENT",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36",
    )

    async def scrape(self, queries: list, max_jobs: int) -> list:
        """Public entry point — gracefully skips if no session cookie is configured.

        Args:
            queries: List of search query strings (unused; feed is pre-filtered).
            max_jobs: Maximum number of job dicts to return.

        Returns:
            List of normalised job dicts or empty list on skip/failure.
        """
        session_cookie = os.getenv("WELLFOUND_SESSION_COOKIE", "")
        if not session_cookie:
            logger.warning(
                "WellfoundScraper: WELLFOUND_SESSION_COOKIE not set — "
                "Wellfound requires a logged-in session to access job listings. "
                "Skipping Wellfound scrape. Set WELLFOUND_SESSION_COOKIE in "
                "~/java.env to enable this platform."
            )
            return []
        self.jobs_limit = max_jobs
        return await self.run(GLOBAL_PLAYWRIGHT_MANAGER)

    def _get_next_page_url(self, current_url: str, page: int) -> Optional[str]:
        """Wellfound uses server-side pagination via ?page= param.

        Args:
            current_url: The URL of the current page.
            page: The next page number (1-indexed).

        Returns:
            URL of the next page or None if on page 1 (no next for first pass).
        """
        if page <= 1:
            return None
        base = self.start_url.split("?")[0]
        return f"{base}?page={page}"

    async def _scrape_once(self, manager: PlaywrightManager) -> List[Dict[str, Any]]:
        """Navigate with injected session cookie and extract job cards."""
        results: List[Dict[str, Any]] = []
        context: Optional[BrowserContext] = None

        try:
            context = await manager.new_context()

            # Inject session cookie — required for any job listings to appear
            session_cookie = os.getenv("WELLFOUND_SESSION_COOKIE", "")
            if session_cookie:
                await context.add_cookies(
                    [
                        {
                            "name": "_session",
                            "value": session_cookie,
                            "domain": "wellfound.com",
                            "path": "/",
                            "httpOnly": True,
                            "secure": True,
                        }
                    ]
                )
                logger.info("WellfoundScraper: injected session cookie")

            page = await context.new_page()
            timeout_ms = int(os.getenv("PLAYWRIGHT_TIMEOUT_MS", "60000"))
            page.set_default_navigation_timeout(timeout_ms)
            page.set_default_timeout(15000)

            await page.goto(
                self.start_url,
                timeout=timeout_ms,
                wait_until="domcontentloaded",
            )
            await human_delay()

            # Check for login wall — if redirected to /login or /users/sign_in
            current = page.url
            if "/login" in current or "/sign_in" in current or "/sign_up" in current:
                logger.warning(
                    "WellfoundScraper: redirected to login wall (%s). "
                    "Session cookie may be expired. Skipping.",
                    current,
                )
                return []

            try:
                await page.wait_for_selector(self.job_card_selector, timeout=15000)
            except Exception:
                logger.warning(
                    "WellfoundScraper: no job cards found at %s — "
                    "session may be invalid or page structure changed.",
                    self.start_url,
                )
                return []

            for _ in range(self.scroll_times):
                await page.evaluate("window.scrollBy(0, document.body.scrollHeight)")
                await asyncio.sleep(1.0 + random.random() * 0.5)

            cards = await page.query_selector_all(self.job_card_selector)
            logger.info("%s: found %d job cards", self.name, len(cards))

            scraped_at = datetime.datetime.utcnow().isoformat() + "Z"
            for card in cards[: self.jobs_limit]:
                job = await self._extract_job(card)
                if job:
                    job["scraped_at"] = scraped_at
                    results.append(job)

            await manager.report_success(context, len(results) * 200)
            logger.info("%s: extracted %d jobs", self.name, len(results))
            return results

        except Exception as e:
            logger.error("%s: scrape failed — %s", self.name, e)
            if context:
                await manager.report_failure(context)
            raise
        finally:
            if context:
                try:
                    await context.close()
                except Exception:
                    pass

    async def _extract_job(self, card) -> Optional[Dict[str, Any]]:
        """Extract normalised job dict from a Wellfound job card element.

        Args:
            card: Playwright ElementHandle for a single job card.

        Returns:
            Normalised job dict or None if title or url is missing.
        """
        try:
            raw_html: Optional[str] = None
            try:
                outer = await card.evaluate("el => el.outerHTML")
                raw_html = str(outer)[:500] if outer else None
            except Exception:
                pass

            title = await self._safe_text(card, self.title_selector)
            company = await self._safe_text(card, self.company_selector)
            location = await self._safe_text(card, self.location_selector)
            link = await self._safe_attr(card, self.link_selector, "href")

            if not title or not link:
                logger.debug(
                    "WellfoundScraper._extract_job: returning None — title=%r link=%r",
                    title,
                    link,
                )
                return None

            if link.startswith("/"):
                link = f"https://wellfound.com{link}"

            return {
                "title": title.strip(),
                "company": company.strip() if company else "",
                "location": location.strip() if location else None,
                "url": link,
                "source": self.name,
                "source_platform": "wellfound",
                "scraped_at": datetime.datetime.utcnow().isoformat() + "Z",
                "raw_html_snippet": raw_html,
            }
        except Exception as exc:
            logger.debug("WellfoundScraper._extract_job: exception — %s", exc)
            return None


class YCStartupScraper(BasePlaywrightScraper):
    """
    YC / Work At A Startup jobs scraper.

    Browser recon (2026-03-21) confirmed:
    - 30 jobs are publicly visible without login at /jobs
    - DOM is Tailwind-rendered (no stable class hashes — uses utility classes)
    - Card container: div.mb-2.rounded-md (border + bg utility classes also present)
    - Title link: div.job-name > a
    - Company: div.company-details > a > span.font-bold
    - Location: div.job-details span (2nd span — first is employment type)
    - No __NEXT_DATA__ on the public guest view
    - Pagination: "Sign up to see more" gates results beyond 30

    Produces up to 30 jobs per run without login. Login support may be added
    in a future phase by injecting WORKATASTARTUP_SESSION_COOKIE.
    """

    name = "yc_startups"
    start_url = "https://www.workatastartup.com/jobs"
    # Verified selectors from live DOM inspection (2026-03-21, Tailwind classes)
    job_card_selector = "div.mb-2.rounded-md"
    title_selector = "div.job-name a"
    company_selector = "div.company-details a span.font-bold"
    location_selector = "div.job-details span:nth-of-type(2)"
    link_selector = "div.job-name a"
    scroll_times = 3
    max_retries = 3
    _USER_AGENT: str = os.getenv(
        "PLAYWRIGHT_USER_AGENT",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36",
    )

    async def scrape(self, queries: list, max_jobs: int) -> list:
        """Public entry point — iterates and collects up to max_jobs.

        Args:
            queries: Search query strings (site uses its own pre-filtered feed;
                     queries are noted in logs but not passed as URL params
                     since the public view does not support search without login).
            max_jobs: Maximum number of normalised job dicts to return.

        Returns:
            List of normalised job dicts. Never raises — returns whatever
            was collected before any failure.
        """
        self.jobs_limit = max_jobs
        results: list = []
        for attempt in range(self.max_retries):
            try:
                results = await self._scrape_once(GLOBAL_PLAYWRIGHT_MANAGER)
                break
            except Exception as exc:
                wait = 2 ** attempt
                logger.warning(
                    "YCStartupScraper.scrape: attempt %d/%d failed: %s — retrying in %ds",
                    attempt + 1,
                    self.max_retries,
                    exc,
                    wait,
                )
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(wait)
                else:
                    logger.error("YCStartupScraper.scrape: all retries exhausted")
        return results

    def _get_next_page_url(self, current_url: str, page: int) -> Optional[str]:
        """WorkAtAStartup limits guest access to 30 jobs on a single page.

        Args:
            current_url: The current page URL.
            page: Requested next page number.

        Returns:
            None — no multi-page navigation available without login.
        """
        return None

    async def _scrape_once(self, manager: PlaywrightManager) -> List[Dict[str, Any]]:
        """Single scrape pass: navigate, scroll, extract cards via verified CSS selectors."""
        results: List[Dict[str, Any]] = []
        context: Optional[BrowserContext] = None

        try:
            context = await manager.new_context()
            page = await context.new_page()
            timeout_ms = int(os.getenv("PLAYWRIGHT_TIMEOUT_MS", "60000"))
            page.set_default_navigation_timeout(timeout_ms)
            page.set_default_timeout(15000)

            await page.goto(
                self.start_url,
                timeout=timeout_ms,
                wait_until="domcontentloaded",
            )
            await human_delay()

            try:
                await page.wait_for_selector(self.job_card_selector, timeout=15000)
            except Exception:
                logger.warning(
                    "YCStartupScraper: job card selector %r not found — "
                    "page may have changed structure",
                    self.job_card_selector,
                )
                return []

            for _ in range(self.scroll_times):
                await page.evaluate("window.scrollBy(0, document.body.scrollHeight)")
                await asyncio.sleep(1.0 + random.random() * 0.5)

            cards = await page.query_selector_all(self.job_card_selector)
            logger.info("%s: found %d job cards via CSS selector", self.name, len(cards))

            scraped_at = datetime.datetime.utcnow().isoformat() + "Z"
            for card in cards[: self.jobs_limit]:
                job = await self._extract_job(card)
                if job:
                    job["scraped_at"] = scraped_at
                    results.append(job)

            await manager.report_success(context, len(results) * 150)
            logger.info("%s: extracted %d jobs", self.name, len(results))
            return results

        except Exception as e:
            logger.error("%s: scrape failed — %s", self.name, e)
            if context:
                await manager.report_failure(context)
            raise
        finally:
            if context:
                try:
                    await context.close()
                except Exception:
                    pass

    async def _extract_job(self, card) -> Optional[Dict[str, Any]]:
        """Extract normalised job dict from a single YC startup job card element.

        Args:
            card: Playwright ElementHandle for a single job card.

        Returns:
            Normalised job dict, or None if title or url is missing.
        """
        try:
            raw_html: Optional[str] = None
            try:
                outer = await card.evaluate("el => el.outerHTML")
                raw_html = str(outer)[:500] if outer else None
            except Exception:
                pass

            title = await self._safe_text(card, self.title_selector)
            company = await self._safe_text(card, self.company_selector)
            location = await self._safe_text(card, self.location_selector)
            link = await self._safe_attr(card, self.link_selector, "href")

            if not title or not link:
                logger.debug(
                    "YCStartupScraper._extract_job: returning None — title=%r link=%r",
                    title,
                    link,
                )
                return None

            if link.startswith("/"):
                link = "https://www.workatastartup.com" + link

            return {
                "title": title.strip(),
                "company": company.strip() if company else "",
                "location": location.strip() if location else None,
                "url": link,
                "source": self.name,
                "source_platform": "yc",
                "scraped_at": datetime.datetime.utcnow().isoformat() + "Z",
                "raw_html_snippet": raw_html,
            }
        except Exception as exc:
            logger.debug("YCStartupScraper._extract_job: exception — %s", exc)
            return None


class TuringScraper(BasePlaywrightScraper):
    """
    Turing.com remote developer jobs scraper.

    Turing lists remote engineering roles on their public job board.

    TODO: Turing's job board may require authentication or render jobs via
    a client-side API. If the CSS selectors below stop working, inspect the
    Network tab for a JSON API call and consider switching to an HTTP scraper.
    """

    name = "turing"
    start_url = "https://www.turing.com/remote-developer-jobs"
    # TODO: verify against current turing.com DOM
    job_card_selector = "div.job-card, div[class*='JobCard'], article.job"
    title_selector = "h3.job-title, h3 a, a[class*='job-title']"
    company_selector = "span.company-name, span[class*='company']"
    location_selector = "span.job-location, span[class*='location']"
    link_selector = "a.job-link, h3 a, a[class*='job']"
    scroll_times = 3
    max_retries = 2


class CrossoverScraper(BasePlaywrightScraper):
    """
    Crossover.com remote high-paying jobs scraper.

    Crossover lists curated remote jobs; their React-rendered job board is
    publicly accessible without authentication.

    TODO: verify selectors against current crossover.com DOM after launch.
    """

    name = "crossover"
    start_url = "https://www.crossover.com/jobs"
    # TODO: verify against current crossover.com DOM
    job_card_selector = (
        "div.job-card, div[class*='JobCard'], div[class*='job-listing']"
    )
    title_selector = "h3, h2 a, a[class*='title']"
    company_selector = "span[class*='company'], p[class*='company']"
    location_selector = "span[class*='location'], span[class*='remote']"
    link_selector = "a[class*='job'], a[class*='title'], h3 a"
    scroll_times = 3
    max_retries = 2


class ArcDevScraper(BasePlaywrightScraper):
    """
    Arc.dev remote jobs scraper.

    Browser recon (2026-03-21) confirmed:
    - All job listings are publicly accessible without login
    - Card container: div.job-card (confirmed via live outerHTML inspection)
    - Title: a.job-title (anchor with relative href like /remote-jobs/j/...)
    - Company: div.company-name
    - Location: div.required-countries (may be absent for some Arc-exclusive jobs)
    - __NEXT_DATA__ present — used as primary extraction strategy
    - No traditional pagination on the landing page (curated latest list)

    Primary strategy: __NEXT_DATA__ JSON extraction (stable, fast).
    Fallback: CSS selector extraction from live DOM.
    """

    name = "arc"
    start_url = "https://arc.dev/remote-jobs"
    # Verified selectors from live DOM inspection (2026-03-21)
    job_card_selector = "div.job-card"
    title_selector = "a.job-title"
    company_selector = "div.company-name"
    location_selector = "div.required-countries"
    link_selector = "a.job-title"
    scroll_times = 4
    max_retries = 3
    _USER_AGENT: str = os.getenv(
        "PLAYWRIGHT_USER_AGENT",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36",
    )

    async def scrape(self, queries: list, max_jobs: int) -> list:
        """Public entry point — collects up to max_jobs from Arc.dev remote jobs.

        Args:
            queries: Search query strings (noted in logs; Arc landing shows
                     curated listings, full search requires parameterised URLs).
            max_jobs: Maximum number of normalised job dicts to return.

        Returns:
            List of normalised job dicts. Never raises.
        """
        self.jobs_limit = max_jobs
        results: list = []
        for attempt in range(self.max_retries):
            try:
                results = await self._scrape_once(GLOBAL_PLAYWRIGHT_MANAGER)
                break
            except Exception as exc:
                wait = 2 ** attempt
                logger.warning(
                    "ArcDevScraper.scrape: attempt %d/%d failed: %s — retrying in %ds",
                    attempt + 1,
                    self.max_retries,
                    exc,
                    wait,
                )
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(wait)
                else:
                    logger.error("ArcDevScraper.scrape: all retries exhausted")
        return results

    def _get_next_page_url(self, current_url: str, page: int) -> Optional[str]:
        """Arc.dev uses infinite scroll — no URL-pattern pagination.

        Args:
            current_url: The current page URL.
            page: Requested next page number.

        Returns:
            None — pagination is scroll-based; additional results loaded
            via scroll in _scrape_once.
        """
        return None

    async def _scrape_once(self, manager: PlaywrightManager) -> List[Dict[str, Any]]:
        """Navigate, attempt __NEXT_DATA__ extraction, fall back to CSS selectors."""
        results: List[Dict[str, Any]] = []
        context: Optional[BrowserContext] = None

        try:
            context = await manager.new_context()
            page = await context.new_page()
            timeout_ms = int(os.getenv("PLAYWRIGHT_TIMEOUT_MS", "60000"))
            page.set_default_navigation_timeout(timeout_ms)
            page.set_default_timeout(15000)

            await page.goto(
                self.start_url,
                timeout=timeout_ms,
                wait_until="domcontentloaded",
            )
            await human_delay()

            # ---- Strategy 1: __NEXT_DATA__ JSON extraction ----------------
            try:
                next_data_el = await page.query_selector("script#__NEXT_DATA__")
                if next_data_el:
                    raw_json = await next_data_el.inner_text()
                    parsed = json.loads(raw_json)
                    jobs_from_json = self._extract_from_next_data(parsed)
                    if jobs_from_json:
                        logger.info(
                            "%s: extracted %d jobs via __NEXT_DATA__",
                            self.name,
                            len(jobs_from_json),
                        )
                        await manager.report_success(context, len(jobs_from_json) * 200)
                        return jobs_from_json[: self.jobs_limit]
            except Exception as next_err:
                logger.debug(
                    "%s: __NEXT_DATA__ extraction failed (%s) — falling back to CSS",
                    self.name,
                    next_err,
                )

            # ---- Strategy 2: CSS selector fallback ------------------------
            try:
                await page.wait_for_selector(self.job_card_selector, timeout=15000)
            except Exception:
                logger.warning(
                    "ArcDevScraper: job card selector %r not found after 15s",
                    self.job_card_selector,
                )
                return []

            for _ in range(self.scroll_times):
                await page.evaluate("window.scrollBy(0, document.body.scrollHeight)")
                await asyncio.sleep(1.2 + random.random() * 0.6)

            cards = await page.query_selector_all(self.job_card_selector)
            logger.info("%s: found %d job cards via CSS", self.name, len(cards))

            scraped_at = datetime.datetime.utcnow().isoformat() + "Z"
            for card in cards[: self.jobs_limit]:
                job = await self._extract_job(card)
                if job:
                    job["scraped_at"] = scraped_at
                    results.append(job)

            await manager.report_success(context, len(results) * 200)
            logger.info("%s: extracted %d jobs (CSS fallback)", self.name, len(results))
            return results

        except Exception as e:
            logger.error("%s: scrape failed — %s", self.name, e)
            if context:
                await manager.report_failure(context)
            raise
        finally:
            if context:
                try:
                    await context.close()
                except Exception:
                    pass

    def _extract_from_next_data(
        self, data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Walk the Arc.dev Next.js page props to extract job listings.

        Args:
            data: Parsed JSON from script#__NEXT_DATA__.

        Returns:
            List of normalised job dicts extracted from the JSON tree.
        """
        results: List[Dict[str, Any]] = []
        scraped_at = datetime.datetime.utcnow().isoformat() + "Z"
        try:
            page_props = data.get("props", {}).get("pageProps", {})
            # Arc.dev stores jobs under various keys depending on page type
            jobs_raw: List[Any] = (
                page_props.get("jobs")
                or page_props.get("jobListings")
                or page_props.get("remoteJobs")
                or page_props.get("listings")
                or []
            )
            for j in jobs_raw:
                if not isinstance(j, dict):
                    continue
                title = str(
                    j.get("title") or j.get("name") or j.get("role") or ""
                ).strip()
                company = str(
                    (j.get("company") or {}).get("name")
                    or j.get("companyName")
                    or j.get("company")
                    or ""
                ).strip()
                location = str(
                    j.get("location")
                    or j.get("requiredCountries")
                    or j.get("remote")
                    or "Remote"
                ).strip()
                slug = str(
                    j.get("slug") or j.get("randomKey") or j.get("id") or ""
                ).strip()
                url = str(j.get("url") or j.get("job_url") or j.get("link") or "").strip()

                if not url and slug:
                    url = f"https://arc.dev/remote-jobs/j/{slug}"

                if not title or not url:
                    continue
                if url.startswith("/"):
                    url = f"https://arc.dev{url}"

                results.append(
                    {
                        "title": title,
                        "company": company,
                        "location": location if location != "true" else "Remote",
                        "url": url,
                        "source": self.name,
                        "source_platform": "arcdev",
                        "scraped_at": scraped_at,
                        "raw_html_snippet": None,
                    }
                )
        except Exception as exc:
            logger.debug("%s: error walking __NEXT_DATA__: %s", self.name, exc)
        return results

    async def _extract_job(self, card) -> Optional[Dict[str, Any]]:
        """Extract normalised job dict from a single Arc.dev job card element.

        Args:
            card: Playwright ElementHandle for a single job card.

        Returns:
            Normalised job dict, or None if title or url is missing.
        """
        try:
            raw_html: Optional[str] = None
            try:
                outer = await card.evaluate("el => el.outerHTML")
                raw_html = str(outer)[:500] if outer else None
            except Exception:
                pass

            title = await self._safe_text(card, self.title_selector)
            company = await self._safe_text(card, self.company_selector)
            location = await self._safe_text(card, self.location_selector)
            link = await self._safe_attr(card, self.link_selector, "href")

            if not title or not link:
                logger.debug(
                    "ArcDevScraper._extract_job: returning None — title=%r link=%r",
                    title,
                    link,
                )
                return None

            if link.startswith("/"):
                link = f"https://arc.dev{link}"

            return {
                "title": title.strip(),
                "company": company.strip() if company else "",
                "location": location.strip() if location else None,
                "url": link,
                "source": self.name,
                "source_platform": "arcdev",
                "scraped_at": datetime.datetime.utcnow().isoformat() + "Z",
                "raw_html_snippet": raw_html,
            }
        except Exception as exc:
            logger.debug("ArcDevScraper._extract_job: exception — %s", exc)
            return None


class NodeskScraper(BasePlaywrightScraper):
    """
    Nodesk.co remote jobs scraper (safety-net source).

    Nodesk is a minimalist remote-jobs aggregator with server-rendered HTML,
    making it highly reliable as a fallback source.

    TODO: verify selectors against current nodesk.co/remote-jobs DOM.
    """

    name = "nodesk"
    start_url = "https://nodesk.co/remote-jobs/"
    # TODO: verify against current nodesk.co DOM
    job_card_selector = "article.job, li.job, div.job-listing, article"
    title_selector = "h2 a, h3 a, a.job-title"
    company_selector = "span.company, p.company, span[class*='company']"
    location_selector = "span.location, span[class*='location']"
    link_selector = "h2 a, h3 a, a.job-title"
    scroll_times = 2
    max_retries = 2
    require_proxy = True


class ToptalScraper(BasePlaywrightScraper):
    """
    Toptal remote freelance / full-time jobs scraper (safety-net source).

    Toptal's public job postings page lists roles available to Toptal network
    members as well as some public-facing openings.

    TODO: verify selectors against current toptal.com/freelance-jobs DOM.
    """

    name = "toptal"
    start_url = "https://www.toptal.com/freelance-jobs"
    # TODO: verify against current toptal.com DOM; the site uses server-side
    # rendering so CSS selectors should be stable across releases.
    job_card_selector = "div.job-card, li.job-item, div[class*='JobCard']"
    title_selector = "h3 a, a.job-title, a[class*='title']"
    company_selector = "span.company, span[class*='company']"
    location_selector = "span.location, span[class*='location']"
    link_selector = "h3 a, a.job-title, a[class*='job']"
    scroll_times = 2
    max_retries = 2
    require_proxy = True


# ---- Legacy scrapers (kept for reference; NOT exported) ----------------------

class RemoteOKPlaywrightScraper(BasePlaywrightScraper):
    """Legacy Playwright scraper for RemoteOK. Replaced by RemoteOKAPIScraper."""

    name = "remoteok_pw"
    start_url = "https://remoteok.com/"
    job_card_selector = "tr.job"
    title_selector = "h2"
    company_selector = "h3"
    location_selector = "div.location"
    link_selector = "a.preventLink"
    scroll_times = 3


class SimplyHiredScraper(BasePlaywrightScraper):
    """Legacy scraper — not part of current architecture."""

    name = "simplyhired"
    start_url = "https://www.simplyhired.com/search?q=remote"
    job_card_selector = "div[data-testid='job-card']"
    title_selector = "h2 a"
    company_selector = ".company-name"
    location_selector = ".job-location"
    link_selector = "h2 a"
    scroll_times = 3


class StackOverflowScraper(BasePlaywrightScraper):
    """Legacy scraper — Stack Overflow Jobs was shut down in 2022."""

    name = "stackoverflow"
    start_url = "https://stackoverflow.com/jobs"
    job_card_selector = ".job-link-wrapper"
    title_selector = "a.job-link"
    company_selector = ".job-company"
    location_selector = ".job-location"
    link_selector = "a.job-link"
    scroll_times = 3


class HiringCafeScraper(BasePlaywrightScraper):
    """Legacy scraper — not part of current architecture."""

    name = "hiring_cafe"
    start_url = "https://hiring.cafe/"
    job_card_selector = ".job-listing"
    title_selector = "h3 a"
    company_selector = ".company"
    location_selector = ".location"
    link_selector = "h3 a"
    scroll_times = 2


# =================================================================================
# API-BASED SCRAPERS
# =================================================================================


class BaseAPIScraper:
    """Base class for lightweight REST API scrapers."""

    name: str = "base_api"

    def __init__(self, jobs_limit: int = 50):
        self.jobs_limit = jobs_limit

    async def run(self, manager: Optional[Any] = None) -> List[Dict[str, Any]]:
        """Async entry point — runs the synchronous _run_sync in a threadpool."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._run_sync)

    def _run_sync(self) -> List[Dict[str, Any]]:
        """Synchronous implementation to be overridden by subclasses."""
        raise NotImplementedError


class WeWorkRemotelyScraper(BaseAPIScraper):
    """
    WeWorkRemotely RSS feed scraper.
    Uses official RSS feeds to fetch latest remote jobs.
    """

    name = "weworkremotely"
    rss_feeds = [
        "https://weworkremotely.com/remote-jobs.rss",
        "https://weworkremotely.com/categories/remote-programming-jobs.rss",
        "https://weworkremotely.com/categories/remote-devops-sysadmin-jobs.rss",
    ]

    def _run_sync(self) -> List[Dict[str, Any]]:
        import xml.etree.ElementTree as ET

        results: List[Dict[str, Any]] = []
        for feed_url in self.rss_feeds:
            try:
                response = _http_requests.get(feed_url, timeout=20)
                response.raise_for_status()

                root = ET.fromstring(response.content)
                items = root.findall(".//item")

                for item in items:
                    title = item.findtext("title")
                    link = item.findtext("link")
                    description = item.findtext("description")
                    pub_date = item.findtext("pubDate")

                    # WWR RSS often includes company in title "Company: Title"
                    company = ""
                    if title and ":" in title:
                        company_part, title_part = title.split(":", 1)
                        company = company_part.strip()
                        title = title_part.strip()

                    if not title or not link:
                        continue

                    results.append(
                        {
                            "title": title,
                            "company": company,
                            "location": "Remote",
                            "job_url": link,
                            "description": description or "",
                            "source": "weworkremotely",
                            "platform": "weworkremotely",
                            "posted_date": pub_date,
                            "scraped_at": datetime.datetime.utcnow().isoformat() + "Z",
                        }
                    )
                    if len(results) >= self.jobs_limit:
                        break

                if len(results) >= self.jobs_limit:
                    break
            except Exception as e:
                logger.error("WWR RSS feed %s failed: %s", feed_url, e)

        return results


class RemoteOKAPIScraper(BaseAPIScraper):
    """RemoteOK Public API scraper."""

    name = "remoteok"
    endpoint = "https://remoteok.com/api"

    def __init__(self, jobs_per_site: int = 50, **kwargs) -> None:
        super().__init__(**kwargs)
        self.jobs_per_site = jobs_per_site

    def _run_sync(self) -> List[Dict[str, Any]]:
        try:
            # RemoteOK requires a User-Agent
            headers = {"User-Agent": "job-automation-agent/1.1 (github.com/aarjunm04)"}
            response = _http_requests.get(self.endpoint, headers=headers, timeout=20)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            logger.error("RemoteOK API failed: %s", e)
            return []

        results: List[Dict[str, Any]] = []
        for item in data:
            if not isinstance(item, dict) or (not item.get("position") and not item.get("title")):
                continue
            
            title = str(item.get("position") or item.get("title") or "").strip()
            # Normalize to unified schema
            results.append({
                "title": title,
                "company": str(item.get("company") or "").strip(),
                "location": str(item.get("location") or "Remote").strip(),
                "job_url": str(item.get("url") or "").strip(),
                "description": str(item.get("description") or "").strip(),
                "source": "remoteok",
                "platform": "remoteok",
                "posted_date": item.get("date"),
                "job_type": item.get("job_type"),
                "salary": item.get("salary"),
                "tags": item.get("tags") or [],
                "scraped_at": datetime.datetime.utcnow().isoformat() + "Z",
            })
            if len(results) >= self.jobs_per_site:
                break
        return results


class HimalayasScraper(BaseAPIScraper):
    """
    Himalayas.app Remote Jobs API.
    Uses offset-based pagination and requires no proxy/key.
    """

    name = "himalayas"
    endpoint = "https://himalayas.app/jobs/api"

    def __init__(self, jobs_per_site: int = 50, **kwargs) -> None:
        super().__init__(**kwargs)
        self.jobs_per_site = jobs_per_site

    def _run_sync(self) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        limit = 50
        offset = 0
        
        # Paginate to reach jobs_limit
        page = 1
        while len(results) < self.jobs_per_site:
            try:
                params = {"limit": limit, "offset": offset}
                logger.info("Himalayas request: %s params=%s", self.endpoint, params)
                # NO PROXY as per Batch 11 requirements
                response = _http_requests.get(
                    self.endpoint, 
                    params=params, 
                    headers={"Accept": "application/json"},
                    timeout=20
                )
                response.raise_for_status()
                data = response.json()
                
                jobs = data.get("jobs", [])
                total = data.get("totalCount", 0)
                logger.debug("Himalayas page=%d raw_count=%d total=%d", 
                         page, len(jobs), total)
                
                if not jobs:
                    break
                
                for j in jobs:
                    title = j.get("title")
                    # Himalayas uses 'applicationLink'
                    url = j.get("applicationLink")
                    if not title or not url:
                        continue
                        
                    results.append({
                        "title": title,
                        "company": j.get("company", {}).get("name") if isinstance(j.get("company"), dict) else j.get("company"),
                        "location": j.get("location") or "Remote",
                        "job_url": url,
                        "description": j.get("description") or "",
                        "source": "himalayas",
                        "platform": "himalayas",
                        "posted_date": j.get("createdAt") or j.get("postedAt"),
                        "job_type": j.get("jobType"),
                        "salary": j.get("salary"),
                        "tags": j.get("skills") or [],
                        "scraped_at": datetime.datetime.utcnow().isoformat() + "Z",
                    })
                    if len(results) >= self.jobs_per_site:
                        break
                
                offset += limit
                page += 1
                if len(jobs) < limit:
                    break
            except Exception as e:
                logger.error("Himalayas API failed at offset %d: %s", offset, e)
                break
        return results


class ArbeitnowScraper(BaseAPIScraper):
    """Arbeitnow.com Remote Jobs API."""

    name = "arbeitnow"
    endpoint = "https://www.arbeitnow.com/api/job-board-api"

    def _run_sync(self) -> List[Dict[str, Any]]:
        try:
            response = _http_requests.get(self.endpoint, timeout=20)
            response.raise_for_status()
            data = response.json()
            jobs = data.get("data", [])
        except Exception as e:
            logger.error("Arbeitnow API failed: %s", e)
            return []

        results: List[Dict[str, Any]] = []
        for j in jobs:
            title = j.get("title")
            url = j.get("url")
            if not title or not url:
                continue
            
            results.append({
                "title": title,
                "company": j.get("company_name"),
                "location": j.get("location") or "Remote",
                "job_url": url,
                "description": j.get("description") or "",
                "source": "arbeitnow",
                "platform": "arbeitnow",
                "tags": j.get("tags") or [],
                "scraped_at": datetime.datetime.utcnow().isoformat() + "Z",
            })
            if len(results) >= self.jobs_limit:
                break
        return results


class JobicyScraper(BaseAPIScraper):
    """Jobicy.com Remote Jobs API."""

    name = "jobicy"
    endpoint = "https://jobicy.com/api/v2/remote-jobs"

    def _run_sync(self) -> List[Dict[str, Any]]:
        try:
            # Jobicy takes 'count' and 'geo' params
            params = {"count": self.jobs_limit}
            response = _http_requests.get(self.endpoint, params=params, timeout=20)
            response.raise_for_status()
            data = response.json()
            jobs = data.get("jobs", [])
        except Exception as e:
            logger.error("Jobicy API failed: %s", e)
            return []

        results: List[Dict[str, Any]] = []
        for j in jobs:
            title = j.get("jobTitle")
            url = j.get("url")
            if not title or not url:
                continue
            
            results.append({
                "title": title,
                "company": j.get("companyName"),
                "location": j.get("jobGeo") or "Remote",
                "job_url": url,
                "description": j.get("jobDescription") or "",
                "source": "jobicy",
                "platform": "jobicy",
                "posted_date": j.get("pubDate"),
                "job_type": j.get("jobType"),
                "salary": j.get("annualSalaryMin"), # Jobicy provides salary info
                "scraped_at": datetime.datetime.utcnow().isoformat() + "Z",
            })
            if len(results) >= self.jobs_limit:
                break
        return results


class RemotiveScraper(BaseAPIScraper):
    """Remotive.io Remote Jobs API (Public)."""

    name = "remotive"
    # Fallback to .com domain (was .io which is dead)
    endpoint = (
        config_loader.get_platform("remotive").get("jobs_url")
        if config_loader else "https://remotive.com/api/remote-jobs"
    )

    def _run_sync(self) -> List[Dict[str, Any]]:
        try:
            # NO PROXY as per Batch 11 requirements
            response = _http_requests.get(self.endpoint, timeout=20)
            response.raise_for_status()
            data = response.json()
            jobs = data.get("jobs", [])
        except Exception as e:
            logger.error("Remotive API failed: %s", e)
            return []

        results: List[Dict[str, Any]] = []
        for j in jobs[:self.jobs_limit]:
            title = j.get("title")
            url = j.get("url")
            if not title or not url:
                continue
            
            results.append({
                "title": title,
                "company": j.get("company_name"),
                "location": j.get("candidate_required_location") or "Remote",
                "job_url": url,
                "description": j.get("description") or "",
                "source": "remotive",
                "platform": "remotive",
                "posted_date": j.get("publication_date"),
                "job_type": j.get("job_type"),
                "salary": j.get("salary"),
                "tags": j.get("tags") or [],
                "scraped_at": datetime.datetime.utcnow().isoformat() + "Z",
            })
        return results


# =================================================================================
# GLOBAL MANAGER INSTANCE
# =================================================================================

GLOBAL_PLAYWRIGHT_MANAGER = PlaywrightManager()


def _shutdown_playwright() -> None:
    """
    atexit handler: close the shared Chromium browser.

    Handles the case where there is no running event loop (common in atexit
    callbacks) by creating a temporary loop. Any exception is caught so that
    the shutdown hook itself never crashes the process.
    """
    try:
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                raise RuntimeError("event loop is closed")
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        try:
            loop.run_until_complete(GLOBAL_PLAYWRIGHT_MANAGER.shutdown())
        except Exception as e:
            logger.warning("Error during Playwright shutdown: %s", e)
    except Exception as e:
        logger.warning("_shutdown_playwright: unexpected error: %s", e)


atexit.register(_shutdown_playwright)


# =================================================================================
# PUBLIC API
# =================================================================================

__all__ = [
    "app",
    "PlaywrightManager",
    "GLOBAL_PLAYWRIGHT_MANAGER",
    "WellfoundScraper",
    "WeWorkRemotelyScraper",
    "YCStartupScraper",
    "TuringScraper",
    "CrossoverScraper",
    "ArcDevScraper",
    "NodeskScraper",
    "ToptalScraper",
    "RemoteOKAPIScraper",
    "HimalayasScraper",
    "ArbeitnowScraper",
    "JobicyScraper",
    "RemotiveScraper",
]


# =================================================================================
# FASTAPI HTTP SERVICE
# =================================================================================

app = FastAPI(title="Playwright Scraper Service", version="1.0.0")

_SCRAPER_API_KEY: str = os.getenv("SCRAPER_SERVICE_API_KEY", "")


def _verify_scraper_api_key(
    x_api_key: str = Header(..., alias="X-API-Key"),
) -> str:
    """Verify X-API-Key header for the scraper service.

    Args:
        x_api_key: Value of the ``X-API-Key`` request header.

    Returns:
        The validated key string.

    Raises:
        HTTPException: 401 if the key is missing or does not match.
    """
    if _SCRAPER_API_KEY and x_api_key == _SCRAPER_API_KEY:
        return x_api_key
    if not _SCRAPER_API_KEY:
        logger.warning("SCRAPER_SERVICE_API_KEY not set — auth disabled")
        return x_api_key
    raise HTTPException(status_code=401, detail="Invalid API key")


class ScrapeRequest(BaseModel):
    """Request body for POST /scrape.

    Attributes:
        run_batch_id: UUID identifying the batch run.
        search_queries: List of search query strings.
        platforms: List of platform names to scrape.
        max_jobs: Maximum number of jobs to return.
    """

    run_batch_id: str = Field(..., description="UUID of the run batch")
    search_queries: List[str] = Field(default_factory=list, description="Search queries")
    platforms: List[str] = Field(default_factory=list, description="Platforms to scrape")
    max_jobs: int = Field(default=150, ge=1, le=1000, description="Max jobs to return")


@app.post("/scrape")
async def scrape_jobs(
    request: ScrapeRequest,
    api_key: str = Header(..., alias="X-API-Key"),
) -> Dict[str, Any]:
    """Trigger a scrape batch and return results.

    Instantiates a :class:`~scrapers.scraper_engine.ScraperEngine` with
    ``min_jobs_target`` set from the request payload and runs the full
    scrape orchestration pipeline.

    Args:
        request: Scrape parameters including batch ID and limits.
        api_key: ``X-API-Key`` header value.

    Returns:
        Dict with ``run_batch_id``, ``jobs_found``, ``jobs`` list,
        ``platforms_scraped``, and ``duration_seconds``.
    """
    _verify_scraper_api_key(api_key)
    start = time.time()
    try:
        from scrapers.scraper_engine import ScraperEngine

        engine = ScraperEngine(min_jobs_target=request.max_jobs)
        jobs, metrics = await engine.run()
        duration = round(time.time() - start, 2)
        logger.info(
            "POST /scrape batch=%s jobs=%d duration=%.2fs",
            request.run_batch_id,
            len(jobs),
            duration,
        )
        return {
            "run_batch_id": request.run_batch_id,
            "jobs_found": len(jobs),
            "jobs": jobs,
            "platforms_scraped": list({j.get("source", "unknown") for j in jobs}),
            "duration_seconds": duration,
        }
    except Exception as exc:
        logger.error("POST /scrape failed: %s", exc)
        return {
            "run_batch_id": request.run_batch_id,
            "jobs_found": 0,
            "jobs": [],
            "platforms_scraped": [],
            "duration_seconds": round(time.time() - start, 2),
            "error": str(exc),
        }


@app.get("/health")
async def health() -> Dict[str, Any]:
    """Health check — verify Playwright can launch a browser.

    Launches and immediately closes a headless Chromium instance to
    confirm the browser binary is functional.

    Returns:
        Dict with ``status``, ``playwright`` health, and
        ``proxy_pool_size``.
    """
    result: Dict[str, Any] = {
        "status": "ok",
        "playwright": "ok",
        "proxy_pool_size": len(GLOBAL_PLAYWRIGHT_MANAGER.proxy_manager.proxies),
    }
    try:
        pw = await async_playwright().start()
        browser = await pw.chromium.launch(headless=True, args=["--no-sandbox"])
        await browser.close()
        await pw.stop()
    except Exception as exc:
        logger.warning("Health: Playwright check failed: %s", exc)
        result["playwright"] = "error"
        result["status"] = "error"
    return result
