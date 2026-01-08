"""
playwright/scraper_service.py

ENTERPRISE-GRADE PLAYWRIGHT SCRAPER STACK
=========================================
COMPLETE Playwright infrastructure + 7 site-specific scrapers.

Responsibilities:
├── Load 10 Webshare static proxies from narad.env
├── Round-robin proxy rotation + 1GB bandwidth accounting
├── Single shared Chromium browser with stealth hardening
├── 7 site-specific scrapers: Wellfound, WeWorkRemotely, RemoteOK,
│   SimplyHired, StackOverflow, YC Startups, Hiring Cafe
├── Production retry logic + failure isolation
└── Raw job extraction (no normalization/dedupe)

Usage by ScraperEngine:
    scraper = WellfoundScraper(jobs_limit=20)
    raw_jobs = await scraper.run(playwright_manager)
"""

from __future__ import annotations

import os
import time
import json
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
# ENTERPRISE LOGGING
# =================================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("playwright_scrapers.log")
    ]
)
LOG = logging.getLogger("playwright_scrapers")
LOG.setLevel(logging.INFO)

# =================================================================================
# PROXY INFRASTRUCTURE (10 STATIC PROXIES FROM narad.env)
# =================================================================================

@dataclass
class ProxyNode:
    """Individual proxy with usage tracking"""
    server: str
    username: str
    password: str
    used_mb: int = 0
    failures: int = 0
    cooldown_until: float = 0.0

    @property
    def available(self) -> bool:
        return (
            self.used_mb < 1024 and  # 1GB limit per proxy
            self.failures < 5 and     # 5 failure threshold
            time.time() > self.cooldown_until
        )

    def mark_failure(self):
        """Record proxy failure"""
        self.failures += 1
        if self.failures >= 5:
            self.cooldown_until = time.time() + 300  # 5min cooldown

    def record_bandwidth(self, kb: int):
        """Track bandwidth usage"""
        self.used_mb += max(1, kb // 1024)

class ProxyManager:
    """Manages 10 static Webshare proxies from narad.env."""

    def __init__(self):
        load_dotenv("narad.env")
        self.proxies: List[ProxyNode] = []
        self._proxy_index: int = 0
        self._lock = asyncio.Lock()
        self._load_static_proxies()
        LOG.info("Loaded %d static Webshare proxies", len(self.proxies))

    def _load_static_proxies(self):
        """
        Load Webshare static proxies from narad.env.

        Expected format, as in your narad.env:
            WEBSHARE_PROXY_2_1=http://username:password@ip:port
            ...
            WEBSHARE_PROXY_2_10=http://username:password@ip:port
        """
        env_keys: List[str] = [f"WEBSHARE_PROXY_2_{i}" for i in range(1, 11)]
        # Optional: also allow PROXY_1..PROXY_10
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
            LOG.warning("No valid proxies found in narad.env. Running without proxies.")

    def select_proxy(self) -> Optional[ProxyNode]:
        """Round-robin selection of an available proxy, with fallback."""
        if not self.proxies:
            return None

        # Primary: round-robin over available proxies
        for _ in range(len(self.proxies)):
            node = self.proxies[self._proxy_index]
            self._proxy_index = (self._proxy_index + 1) % len(self.proxies)
            if node.available:
                return node

        # Fallback: least-failed, then least-used
        return min(self.proxies, key=lambda p: (p.failures, p.used_mb))

# =================================================================================
# SHARED BROWSER INFRASTRUCTURE
# =================================================================================

class PlaywrightManager:
    """Centralized browser + proxy lifecycle manager"""
    
    def __init__(self):
        self.proxy_manager = ProxyManager()
        self._browser: Optional[Browser] = None
        self._lock = asyncio.Lock()
        self._initialized = False

    async def initialize(self):
        """Lazy init: launch browser + load proxies"""
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
                    "--disable-features=VizDisplayCompositor"
                ]
            )
            
            # Stealth hardening
            self._stealth_user_agent = (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
            
            self._initialized = True
            LOG.info("✅ Playwright browser + stealth initialized")

    async def new_context(self) -> BrowserContext:
        """Create new context with proxy rotation + stealth"""
        await self.initialize()
        
        proxy_node = self.proxy_manager.select_proxy()
        proxy_config = {
            "server": proxy_node.server,
            "username": proxy_node.username,
            "password": proxy_node.password
        } if proxy_node else None

        context = await self._browser.new_context(
            proxy=proxy_config,
            user_agent=self._stealth_user_agent,
            viewport={"width": 1366, "height": 768},
            locale="en-US",
            timezone_id="America/New_York",
            permissions=["geolocation"]
        )
        
        # Ultimate stealth script
        await context.add_init_script("""
            // Remove webdriver property
            Object.defineProperty(navigator, 'webdriver', {
                get: () => undefined,
            });
            
            // Override plugins
            Object.defineProperty(navigator, 'plugins', {
                get: () => [1, 2, 3, 4, 5],
            });
            
            // Mock languages
            Object.defineProperty(navigator, 'languages', {
                get: () => ['en-US', 'en'],
            });
        """)
        
        # Attach proxy for accounting
        if proxy_node:
            context._proxy_node = proxy_node
            
        return context

    async def report_success(self, context: BrowserContext, estimated_kb: int):
        """Track successful scrape bandwidth"""
        proxy_node = getattr(context, '_proxy_node', None)
        if proxy_node:
            proxy_node.record_bandwidth(estimated_kb)

    async def report_failure(self, context: BrowserContext):
        """Mark proxy as failed"""
        proxy_node = getattr(context, '_proxy_node', None)
        if proxy_node:
            proxy_node.mark_failure()

    async def shutdown(self):
        """Clean shutdown"""
        if self._browser:
            await self._browser.close()
            LOG.info("Playwright browser closed")

# =================================================================================
# BASE SCRAPER (SHARED LOGIC)
# =================================================================================

class BasePlaywrightScraper:
    """Shared logic for all site scrapers"""
    
    # Must be overridden by subclasses
    name: str = "base"
    start_url: str = ""
    job_card_selector: str = ""
    title_selector: str = ""
    company_selector: Optional[str] = None
    location_selector: Optional[str] = None
    link_selector: Optional[str] = None
    scroll_times: int = 3
    max_retries: int = 2

    def __init__(self, jobs_limit: int):
        self.jobs_limit = jobs_limit

    async def run(self, manager: PlaywrightManager) -> List[Dict[str, Any]]:
        """Main entry point called by ScraperEngine"""
        for attempt in range(self.max_retries + 1):
            try:
                return await self._scrape_once(manager)
            except Exception as e:
                LOG.warning(f"{self.name} attempt {attempt + 1} failed: {e}")
                if attempt == self.max_retries:
                    LOG.error(f"{self.name} all retries exhausted")
                    return []
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
        
        return []

    async def _scrape_once(self, manager: PlaywrightManager) -> List[Dict[str, Any]]:
        """Single scrape attempt"""
        results: List[Dict[str, Any]] = []
        context: Optional[BrowserContext] = None
        
        try:
            context = await manager.new_context()
            page = await context.new_page()
            page.set_default_navigation_timeout(45000)
            page.set_default_timeout(30000)

            # Navigate + wait for content
            await page.goto(self.start_url, wait_until="networkidle")
            LOG.debug(f"{self.name}: navigated to {self.start_url}")

            # Scroll to load dynamic content
            for i in range(self.scroll_times):
                await page.evaluate("window.scrollBy(0, document.body.scrollHeight)")
                await page.wait_for_timeout(1000 + random.randint(0, 500))

            # Extract job cards
            cards = await page.query_selector_all(self.job_card_selector)
            LOG.info(f"{self.name}: found {len(cards)} job cards")

            for card in cards[:self.jobs_limit]:
                job = await self._extract_job(card)
                if job:
                    job["source"] = self.name
                    results.append(job)

            # Report success + estimated bandwidth
            estimated_kb = len(results) * 150  # ~150KB per job card
            await manager.report_success(context, estimated_kb)
            
            LOG.info(f"✅ {self.name}: extracted {len(results)} jobs")
            return results

        except Exception as e:
            LOG.error(f"{self.name}: scrape failed - {e}")
            if context:
                await manager.report_failure(context)
            raise
        finally:
            if context:
                await context.close()

    async def _extract_job(self, card) -> Optional[Dict[str, Any]]:
        """Extract raw job data from card"""
        try:
            title = await self._safe_text(card, self.title_selector)
            company = await self._safe_text(card, self.company_selector)
            location = await self._safe_text(card, self.location_selector)
            link = await self._safe_attr(card, self.link_selector, "href")

            if not title or not link:
                return None

            # Make link absolute
            if link.startswith("/"):
                page_url = getattr(self, 'start_url', '')
                link = page_url.rsplit('/', 1)[0] + link

            return {
                "title": title,
                "company": company,
                "location": location or "Remote",
                "job_url": link,
                "description": ""
            }
        except Exception:
            return None

    async def _safe_text(self, root, selector: Optional[str]) -> str:
        """Safely extract text with fallback"""
        if not selector:
            return ""
        try:
            el = await root.query_selector(selector)
            return (await el.inner_text()).strip() if el else ""
        except:
            return ""

    async def _safe_attr(self, root, selector: Optional[str], attr: str) -> str:
        """Safely extract attribute with fallback"""
        if not selector:
            return ""
        try:
            el = await root.query_selector(selector)
            return (await el.get_attribute(attr)) or "" if el else ""
        except:
            return ""

# =================================================================================
# SITE-SPECIFIC SCRAPERS (7 SITES)
# =================================================================================

class WellfoundScraper(BasePlaywrightScraper):
    name = "wellfound"
    start_url = "https://wellfound.com/jobs"
    job_card_selector = "div[data-test='JobListing']"
    title_selector = "a[data-test='jobTitle']"
    company_selector = "a[data-test='companyName']"
    location_selector = "span[data-test='jobLocation']"
    link_selector = "a[data-test='jobTitle']"
    scroll_times = 4

class WeWorkRemotelyScraper(BasePlaywrightScraper):
    name = "weworkremotely"
    start_url = "https://weworkremotely.com/remote-jobs"
    job_card_selector = "section.jobs article li"
    title_selector = "span.title"
    company_selector = "span.company"
    location_selector = None  # Always remote
    link_selector = "a"
    scroll_times = 2

class RemoteOKScraper(BasePlaywrightScraper):
    name = "remoteok"
    start_url = "https://remoteok.com/"
    job_card_selector = "tr.job"
    title_selector = "h2"
    company_selector = "h3"
    location_selector = "div.location"
    link_selector = "a.preventLink"
    scroll_times = 3

class SimplyHiredScraper(BasePlaywrightScraper):
    name = "simplyhired"
    start_url = "https://www.simplyhired.com/search?q=remote"
    job_card_selector = "div[data-testid='job-card']"
    title_selector = "h2 a"
    company_selector = ".company-name"
    location_selector = ".job-location"
    link_selector = "h2 a"
    scroll_times = 3

class StackOverflowScraper(BasePlaywrightScraper):
    name = "stackoverflow"
    start_url = "https://stackoverflow.com/jobs"
    job_card_selector = ".job-link-wrapper"
    title_selector = "a.job-link"
    company_selector = ".job-company"
    location_selector = ".job-location"
    link_selector = "a.job-link"
    scroll_times = 3

class YCStartupScraper(BasePlaywrightScraper):
    name = "yc_startups"
    start_url = "https://www.ycombinator.com/jobs"
    job_card_selector = ".job-card"
    title_selector = ".job-title a"
    company_selector = ".company-name"
    location_selector = ".job-location"
    link_selector = ".job-title a"
    scroll_times = 2

class HiringCafeScraper(BasePlaywrightScraper):
    name = "hiring_cafe"
    start_url = "https://hiring.cafe/"
    job_card_selector = ".job-listing"
    title_selector = "h3 a"
    company_selector = ".company"
    location_selector = ".location"
    link_selector = "h3 a"
    scroll_times = 2

# =================================================================================
# GLOBAL MANAGER INSTANCE (for ScraperEngine)
# =================================================================================

GLOBAL_PLAYWRIGHT_MANAGER = PlaywrightManager()

# Auto-shutdown on exit
import atexit
import asyncio

def _shutdown_playwright():
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        # No running loop, create a new one just to close browser
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    loop.run_until_complete(GLOBAL_PLAYWRIGHT_MANAGER.shutdown())

atexit.register(_shutdown_playwright)


__all__ = [
    "PlaywrightManager", "GLOBAL_PLAYWRIGHT_MANAGER",
    "WellfoundScraper", "WeWorkRemotelyScraper", "RemoteOKScraper",
    "SimplyHiredScraper", "StackOverflowScraper", "YCStartupScraper", "HiringCafeScraper"
]
