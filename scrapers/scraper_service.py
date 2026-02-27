"""
scrapers/scraper_service.py

ENTERPRISE-GRADE PLAYWRIGHT SCRAPER STACK
=========================================

Playwright infrastructure + site-specific scrapers for the AI Job Automation
Agent.

Responsibilities:
├── Load Webshare static proxies from ~/narad.env
├── Round-robin proxy rotation + 1 GB bandwidth accounting per proxy
├── Single shared Chromium browser with stealth hardening
├── Site scrapers (Phase 1 + safety-net):
│     Wellfound, WeWorkRemotely, YC/WorkAtAStartup
│     Turing, Crossover, Arc.dev, Nodesk, Toptal
├── Production retry logic + failure isolation
└── Raw job extraction (no normalization/dedupe)

Legacy classes kept (not exported):
    RemoteOKScraper, SimplyHiredScraper, StackOverflowScraper, HiringCafeScraper

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
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path

# External dependencies
from dotenv import load_dotenv
from playwright.async_api import async_playwright, Browser, BrowserContext

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
LOG = logging.getLogger("playwright_scrapers")
LOG.setLevel(logging.INFO)

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
    """Manages static Webshare proxies loaded from ~/narad.env."""

    def __init__(self) -> None:
        load_dotenv(Path.home() / "narad.env")
        self.proxies: List[ProxyNode] = []
        self._proxy_index: int = 0
        self._lock = asyncio.Lock()
        self._load_static_proxies()
        LOG.info("Loaded %d static proxies", len(self.proxies))

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
                LOG.info("Loaded proxy from %s", key)
            except Exception as e:
                LOG.warning("Failed to parse proxy %s: %s", key, e)

        if not self.proxies:
            LOG.warning("No valid proxies found. Running without proxies.")

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
            LOG.info("Playwright browser + stealth initialized")

    async def new_context(self) -> BrowserContext:
        """Create a new browser context with proxy rotation + stealth init script."""
        await self.initialize()

        proxy_node = self.proxy_manager.select_proxy()
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
            LOG.info("Playwright browser closed")


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

    def __init__(self, jobs_limit: int = 20) -> None:
        self.jobs_limit = jobs_limit

    async def run(self, manager: PlaywrightManager) -> List[Dict[str, Any]]:
        """Entry point: run with exponential-backoff retries."""
        for attempt in range(self.max_retries + 1):
            try:
                return await self._scrape_once(manager)
            except Exception as e:
                LOG.warning("%s attempt %d failed: %s", self.name, attempt + 1, e)
                if attempt == self.max_retries:
                    LOG.error("%s all retries exhausted", self.name)
                    return []
                await asyncio.sleep(2**attempt)
        return []

    async def _scrape_once(self, manager: PlaywrightManager) -> List[Dict[str, Any]]:
        """Single scrape pass: navigate → scroll → extract cards."""
        results: List[Dict[str, Any]] = []
        context: Optional[BrowserContext] = None

        try:
            context = await manager.new_context()
            page = await context.new_page()
            page.set_default_navigation_timeout(45000)
            page.set_default_timeout(30000)

            await page.goto(self.start_url, wait_until="networkidle")
            LOG.debug("%s: navigated to %s", self.name, self.start_url)

            for _ in range(self.scroll_times):
                await page.evaluate("window.scrollBy(0, document.body.scrollHeight)")
                await page.wait_for_timeout(1000 + random.randint(0, 500))

            cards = await page.query_selector_all(self.job_card_selector)
            LOG.info("%s: found %d job cards", self.name, len(cards))

            for card in cards[: self.jobs_limit]:
                job = await self._extract_job(card)
                if job:
                    job["source"] = self.name
                    results.append(job)

            estimated_kb = len(results) * 150
            await manager.report_success(context, estimated_kb)

            LOG.info("%s: extracted %d jobs", self.name, len(results))
            return results

        except Exception as e:
            LOG.error("%s: scrape failed — %s", self.name, e)
            if context:
                await manager.report_failure(context)
            raise
        finally:
            if context:
                await context.close()

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

    Wellfound is protected by Cloudflare. If WELLFOUND_CF_CLEARANCE env var
    is set, the cf_clearance cookie is injected before navigation to bypass
    the JS challenge.
    """

    name = "wellfound"
    start_url = "https://wellfound.com/jobs"
    # TODO: re-verify selectors against current Wellfound DOM on breakage
    job_card_selector = "div[data-test='JobListing']"
    title_selector = "a[data-test='jobTitle']"
    company_selector = "a[data-test='companyName']"
    location_selector = "span[data-test='jobLocation']"
    link_selector = "a[data-test='jobTitle']"
    scroll_times = 4
    max_retries = 2

    async def _scrape_once(self, manager: PlaywrightManager) -> List[Dict[str, Any]]:
        """Override to inject optional Cloudflare clearance cookie."""
        results: List[Dict[str, Any]] = []
        context: Optional[BrowserContext] = None

        try:
            context = await manager.new_context()

            # --- Optional Cloudflare bypass -----------------------------------
            cf_clearance = os.getenv("WELLFOUND_CF_CLEARANCE")
            if cf_clearance:
                try:
                    await context.add_cookies(
                        [
                            {
                                "name": "cf_clearance",
                                "value": cf_clearance,
                                "domain": "wellfound.com",
                                "path": "/",
                                "httpOnly": True,
                                "secure": True,
                            }
                        ]
                    )
                    LOG.info("WellfoundScraper: injected cf_clearance cookie")
                except Exception as cookie_err:
                    LOG.warning(
                        "WellfoundScraper: failed to inject cf_clearance cookie: %s",
                        cookie_err,
                    )
            # ------------------------------------------------------------------

            page = await context.new_page()
            page.set_default_navigation_timeout(45000)
            page.set_default_timeout(30000)

            await page.goto(self.start_url, wait_until="networkidle")

            for _ in range(self.scroll_times):
                await page.evaluate("window.scrollBy(0, document.body.scrollHeight)")
                await page.wait_for_timeout(1000 + random.randint(0, 500))

            cards = await page.query_selector_all(self.job_card_selector)
            LOG.info("%s: found %d job cards", self.name, len(cards))

            for card in cards[: self.jobs_limit]:
                job = await self._extract_job(card)
                if job:
                    job["source"] = self.name
                    results.append(job)

            await manager.report_success(context, len(results) * 150)
            LOG.info("%s: extracted %d jobs", self.name, len(results))
            return results

        except Exception as e:
            LOG.error("%s: scrape failed — %s", self.name, e)
            if context:
                await manager.report_failure(context)
            raise
        finally:
            if context:
                await context.close()


class WeWorkRemotelyScraper(BasePlaywrightScraper):
    """WeWorkRemotely remote job listings."""

    name = "weworkremotely"
    start_url = "https://weworkremotely.com/remote-jobs"
    # TODO: re-verify selectors on DOM changes; WWR uses server-rendered HTML
    job_card_selector = "section.jobs article li"
    title_selector = "span.title"
    company_selector = "span.company"
    location_selector = None  # Always remote
    link_selector = "a"
    scroll_times = 2
    max_retries = 2


class YCStartupScraper(BasePlaywrightScraper):
    """
    YC / Work At A Startup jobs scraper.

    Primary strategy: locate the __NEXT_DATA__ JSON embedded in the page and
    extract job listings from it (more reliable than CSS selectors on a
    Next.js-rendered app that may add class hashes on rebuild).

    Fallback: standard CSS-selector extraction via BasePlaywrightScraper.
    """

    name = "yc_startups"
    start_url = "https://www.workatastartup.com/jobs"
    # CSS fallback selectors (used if __NEXT_DATA__ path fails)
    # TODO: re-verify these against current workatastartup.com DOM
    job_card_selector = "div.job-card, div[class*='JobCard'], li.job"
    title_selector = "a.job-title, h3 a, a[class*='title']"
    company_selector = "span.company-name, a[class*='company']"
    location_selector = "span.job-location, span[class*='location']"
    link_selector = "a.job-title, h3 a, a[class*='title']"
    scroll_times = 2
    max_retries = 2

    async def _scrape_once(self, manager: PlaywrightManager) -> List[Dict[str, Any]]:
        """Try __NEXT_DATA__ extraction first; fall back to CSS selectors."""
        results: List[Dict[str, Any]] = []
        context: Optional[BrowserContext] = None

        try:
            context = await manager.new_context()
            page = await context.new_page()
            page.set_default_navigation_timeout(45000)
            page.set_default_timeout(30000)

            await page.goto(self.start_url, wait_until="networkidle")

            for _ in range(self.scroll_times):
                await page.evaluate("window.scrollBy(0, document.body.scrollHeight)")
                await page.wait_for_timeout(1000 + random.randint(0, 500))

            # ---- Strategy 1: __NEXT_DATA__ JSON extraction ----------------
            try:
                next_data_el = await page.query_selector("script#__NEXT_DATA__")
                if next_data_el:
                    raw_json = await next_data_el.inner_text()
                    parsed = json.loads(raw_json)
                    jobs_from_json = self._extract_from_next_data(parsed)
                    if jobs_from_json:
                        LOG.info(
                            "%s: extracted %d jobs via __NEXT_DATA__",
                            self.name,
                            len(jobs_from_json),
                        )
                        await manager.report_success(context, len(jobs_from_json) * 150)
                        return jobs_from_json[: self.jobs_limit]
            except Exception as next_err:
                LOG.debug(
                    "%s: __NEXT_DATA__ extraction failed (%s), falling back to CSS",
                    self.name,
                    next_err,
                )

            # ---- Strategy 2: CSS selector fallback ------------------------
            cards = await page.query_selector_all(self.job_card_selector)
            LOG.info(
                "%s: found %d job cards via CSS selector", self.name, len(cards)
            )

            for card in cards[: self.jobs_limit]:
                job = await self._extract_job(card)
                if job:
                    job["source"] = self.name
                    results.append(job)

            await manager.report_success(context, len(results) * 150)
            LOG.info("%s: extracted %d jobs (CSS fallback)", self.name, len(results))
            return results

        except Exception as e:
            LOG.error("%s: scrape failed — %s", self.name, e)
            if context:
                await manager.report_failure(context)
            raise
        finally:
            if context:
                await context.close()

    def _extract_from_next_data(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Walk the Next.js page props tree looking for a jobs/listings array.

        TODO: Inspect the actual __NEXT_DATA__ structure at runtime and
        adjust the key path below to match the current page props shape.
        """
        results: List[Dict[str, Any]] = []
        try:
            # Common Next.js shape: data["props"]["pageProps"]["jobs"]
            page_props = data.get("props", {}).get("pageProps", {})
            jobs_raw: List[Any] = (
                page_props.get("jobs")
                or page_props.get("listings")
                or page_props.get("positions")
                or []
            )
            for j in jobs_raw:
                if not isinstance(j, dict):
                    continue
                title = str(
                    j.get("title") or j.get("role") or j.get("name") or ""
                ).strip()
                company = str(
                    j.get("company")
                    or (j.get("startup") or {}).get("name")
                    or ""
                ).strip()
                location = str(
                    j.get("location") or j.get("remote_ok") or "Remote"
                ).strip()
                url = str(
                    j.get("url")
                    or j.get("job_url")
                    or j.get("link")
                    or ""
                ).strip()
                description = str(j.get("description") or "").strip()

                if not title or not url:
                    continue
                if url.startswith("/"):
                    url = f"https://www.workatastartup.com{url}"
                results.append(
                    {
                        "title": title,
                        "company": company,
                        "location": location,
                        "job_url": url,
                        "description": description,
                        "source": self.name,
                    }
                )
        except Exception as e:
            LOG.debug("%s: error walking __NEXT_DATA__: %s", self.name, e)
        return results


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

    Arc.dev (formerly CodementorX) runs a curated remote-first job board for
    engineers. The board is publicly accessible.

    TODO: verify selectors against current arc.dev/remote-jobs DOM.
    """

    name = "arc"
    start_url = "https://arc.dev/remote-jobs"
    # TODO: verify against current arc.dev DOM
    job_card_selector = (
        "div.job-card, li.job, div[class*='JobCard'], div[data-testid='job-card']"
    )
    title_selector = "h3 a, a.job-title, a[class*='title']"
    company_selector = "span.company, span[class*='company'], p[class*='company']"
    location_selector = "span.location, span[class*='location']"
    link_selector = "a.job-title, h3 a, a[class*='job']"
    scroll_times = 3
    max_retries = 2


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


# ---- Legacy scrapers (kept for reference; NOT exported) ----------------------

class RemoteOKScraper(BasePlaywrightScraper):
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
            LOG.warning("Error during Playwright shutdown: %s", e)
    except Exception as e:
        LOG.warning("_shutdown_playwright: unexpected error: %s", e)


atexit.register(_shutdown_playwright)


# =================================================================================
# PUBLIC API
# =================================================================================

__all__ = [
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
]
