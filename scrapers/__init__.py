"""
scrapers/__init__.py

AI JOB AUTOMATION AGENT — SCRAPERS PACKAGE
==========================================

Purpose:
    Scrape and normalize job listings from multiple platforms into a unified
    schema, returning them as plain Python dicts for AI agents to consume.

    This package does NOT perform database writes and does NOT make auto-apply
    vs. manual-review routing decisions. Those responsibilities belong
    exclusively to downstream agents.

Public API (what agents should import):

    from scrapers import ScraperEngine

    # Synchronous usage
    engine = ScraperEngine()
    jobs, metrics = engine.run_sync()

    # Async usage
    engine = ScraperEngine()
    jobs, metrics = await engine.run()

    # Ingest the full payload (jobs + metadata)
    payload = engine.get_ingestion_payload()
    # payload == {
    #     "jobs": [...],           # list of normalized job dicts
    #     "metadata": {
    #         "total_jobs": int,
    #         "scraped_at": str,   # ISO-8601 UTC
    #         "metrics": {...},
    #     },
    # }

Advanced (direct access to browser/adapter layer):

    from scrapers import PlaywrightManager, GLOBAL_PLAYWRIGHT_MANAGER
    from scrapers import JobSpyAdapter
"""

from scrapers.scraper_engine import ScraperEngine
from scrapers.scraper_service import PlaywrightManager, GLOBAL_PLAYWRIGHT_MANAGER
from scrapers.jobspy_adapter import JobSpyAdapter

__all__ = [
    # Primary engine — the main entry point for all agents
    "ScraperEngine",
    # Playwright browser lifecycle manager (advanced / direct use)
    "PlaywrightManager",
    "GLOBAL_PLAYWRIGHT_MANAGER",
    # JobSpy adapter (LinkedIn, Indeed, ZipRecruiter, Glassdoor)
    "JobSpyAdapter",
]
