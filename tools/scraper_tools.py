"""
Scraper tools for AI Job Application Agent.

This module wraps the existing scraper_engine.py and scraper_service.py into
CrewAI tool functions consumed by the Scraper Agent. All scraping logic is
delegated to the underlying engines - this file only handles tool interfaces,
Postgres persistence, and error handling.
"""

import os
import json
import logging
import time
import asyncio
from typing import Optional, List, Dict, Any

from crewai.tools import tool
import agentops
import psycopg2
import psycopg2.extras

from scrapers.scraper_engine import (
    ScraperEngine,
    RemoteOKAPIScraper,
    HimalayasAPIScraper,
    SerpAPIGoogleJobsScraper,
    ResourceManager,
)
from scrapers.scraper_service import (
    GLOBAL_PLAYWRIGHT_MANAGER,
    WellfoundScraper,
    WeWorkRemotelyScraper,
    YCStartupScraper,
    TuringScraper,
    CrossoverScraper,
    ArcDevScraper,
    NodeskScraper,
    ToptalScraper,
)
from tools.postgres_tools import upsert_job_post, log_event, get_platform_config

# Module-level logger
logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("LOG_LEVEL", "INFO"))

# Module-level lazy-initialized engines
_scraper_engine: Optional[ScraperEngine] = None
_serpapi_key_index: int = 0

# Database URL for direct queries
DB_URL = (
    os.getenv("LOCAL_POSTGRES_URL")
    if os.getenv("ACTIVE_DB", "local") == "local"
    else os.getenv("SUPABASE_URL")
)

__all__ = [
    "run_jobspy_scrape",
    "run_rest_api_scrape",
    "run_playwright_scrape",
    "run_serpapi_scrape",
    "run_safety_net_scrape",
    "normalise_and_dedup",
    "get_scrape_summary",
]


def _get_engine() -> ScraperEngine:
    """
    Get or initialize the ScraperEngine singleton.

    Returns:
        ScraperEngine: Initialized scraper engine instance.
    """
    global _scraper_engine
    if _scraper_engine is None:
        min_jobs_target = int(os.getenv("JOBS_PER_RUN_MINIMUM", "100"))
        _scraper_engine = ScraperEngine(min_jobs_target=min_jobs_target)
        logger.info("ScraperEngine initialized")
    return _scraper_engine


def _with_retry(func, max_retries: int = 3):
    """
    Retry decorator for scraper functions.

    Args:
        func: Function to wrap with retry logic.
        max_retries: Maximum number of retry attempts.

    Returns:
        Wrapped function with retry logic.
    """

    def wrapper(*args, **kwargs):
        last_exception = None
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                if attempt < max_retries - 1:
                    sleep_time = 2**attempt
                    logger.warning(
                        f"{func.__name__} failed (attempt {attempt + 1}/{max_retries}): {e}. "
                        f"Retrying in {sleep_time}s..."
                    )
                    time.sleep(sleep_time)
                else:
                    logger.error(
                        f"{func.__name__} failed after {max_retries} attempts: {e}"
                    )
        raise last_exception

    return wrapper


@tool
@agentops.track_tool
def run_jobspy_scrape(
    run_batch_id: str,
    search_query: str,
    location: str = "Remote",
    results_wanted: int = 50,
) -> str:
    """
    Scrape LinkedIn and Indeed via JobSpy library.

    Args:
        run_batch_id: UUID of the run batch.
        search_query: Job search query string.
        location: Location filter (default: "Remote").
        results_wanted: Number of results to fetch per platform.

    Returns:
        JSON string with scraping results and statistics.
    """
    try:
        engine = _get_engine()

        # JobSpy is integrated into ScraperEngine - run full engine
        # and filter for jobspy results
        jobs, metrics = asyncio.run(engine.run())

        # Filter for jobspy sources (linkedin, indeed)
        jobspy_jobs = [
            j for j in jobs if j.get("source") in ["linkedin", "indeed"]
        ]

        jobs_upserted = 0
        errors = []

        for job in jobspy_jobs:
            try:
                result = upsert_job_post(
                    run_batch_id=run_batch_id,
                    source_platform=job.get("source", "jobspy"),
                    title=job.get("title", ""),
                    company=job.get("company", ""),
                    url=job.get("job_url", ""),
                    location=job.get("location", location),
                    posted_at=job.get("posted_date", ""),
                )
                result_data = json.loads(result)
                if "error" not in result_data:
                    jobs_upserted += 1
                else:
                    errors.append(result_data["error"])
            except Exception as e:
                logger.error(f"Failed to upsert job: {e}")
                errors.append(str(e))

        logger.info(
            f"JobSpy scrape completed: {len(jobspy_jobs)} found, {jobs_upserted} upserted"
        )

        return json.dumps(
            {
                "platform": "jobspy",
                "jobs_found": len(jobspy_jobs),
                "jobs_upserted": jobs_upserted,
                "run_batch_id": run_batch_id,
                "errors": errors[:10],  # Limit error list
            }
        )

    except Exception as e:
        logger.error(f"JobSpy scrape failed: {e}")
        log_event(
            run_batch_id=run_batch_id,
            level="ERROR",
            event_type="jobspy_scrape_failed",
            message=f"JobSpy scrape failed: {str(e)}",
        )
        return json.dumps(
            {
                "platform": "jobspy",
                "jobs_found": 0,
                "jobs_upserted": 0,
                "run_batch_id": run_batch_id,
                "errors": [str(e)],
            }
        )


@tool
@agentops.track_tool
def run_rest_api_scrape(
    run_batch_id: str, platforms: str = "remoteok,himalayas"
) -> str:
    """
    Scrape REST API platforms (RemoteOK, Himalayas).

    Args:
        run_batch_id: UUID of the run batch.
        platforms: Comma-separated list of platforms to scrape.

    Returns:
        JSON string with scraping results and statistics.
    """
    platform_list = [p.strip().lower() for p in platforms.split(",")]
    jobs_found = 0
    jobs_upserted = 0
    errors = []
    platforms_attempted = []

    for platform_name in platform_list:
        try:
            # Get platform config for rate limiting
            config_result = get_platform_config(platform_name)
            config = json.loads(config_result)

            if "error" in config:
                logger.warning(f"No config found for {platform_name}, using defaults")

            platforms_attempted.append(platform_name)

            # Initialize appropriate scraper
            scraper = None
            if platform_name == "remoteok":
                scraper = RemoteOKAPIScraper(jobs_per_site=50)
            elif platform_name == "himalayas":
                scraper = HimalayasAPIScraper(jobs_per_site=50)
            else:
                errors.append(f"Unknown platform: {platform_name}")
                continue

            if scraper:

                @_with_retry
                def scrape_sync():
                    return asyncio.run(scraper.run())

                raw_jobs = scrape_sync()

                # Normalize and upsert each job
                for job in raw_jobs:
                    try:
                        result = upsert_job_post(
                            run_batch_id=run_batch_id,
                            source_platform=platform_name,
                            title=job.get("title", ""),
                            company=job.get("company", ""),
                            url=job.get("job_url", ""),
                            location=job.get("location", "Remote"),
                            posted_at=job.get("posted_date", ""),
                        )
                        result_data = json.loads(result)
                        if "error" not in result_data:
                            jobs_upserted += 1
                        else:
                            errors.append(result_data["error"])
                    except Exception as e:
                        logger.error(f"Failed to upsert job: {e}")
                        errors.append(str(e))

                jobs_found += len(raw_jobs)
                logger.info(
                    f"{platform_name} scrape completed: {len(raw_jobs)} jobs found"
                )

                # Respect rate limits
                max_per_run = config.get("max_per_run", 50)
                if jobs_found >= max_per_run:
                    logger.info(
                        f"Reached max_per_run limit ({max_per_run}) for {platform_name}"
                    )
                    break

        except Exception as e:
            logger.error(f"REST API scrape failed for {platform_name}: {e}")
            log_event(
                run_batch_id=run_batch_id,
                level="ERROR",
                event_type="rest_api_scrape_failed",
                message=f"{platform_name} scrape failed: {str(e)}",
            )
            errors.append(f"{platform_name}: {str(e)}")

    return json.dumps(
        {
            "platforms_attempted": platforms_attempted,
            "jobs_found": jobs_found,
            "jobs_upserted": jobs_upserted,
            "errors": errors[:10],
        }
    )


@tool
@agentops.track_tool
def run_playwright_scrape(
    run_batch_id: str, platform: str, max_jobs: int = 30
) -> str:
    """
    Scrape a single platform using Playwright browser automation.

    Args:
        run_batch_id: UUID of the run batch.
        platform: Platform name (wellfound, weworkremotely, ycombinator, arc, turing, crossover).
        max_jobs: Maximum number of jobs to scrape.

    Returns:
        JSON string with scraping results and statistics.
    """
    try:
        # Get proxy from environment
        proxy_list_str = os.getenv("WEBSHARE_PROXY_LIST", "")
        proxies = [p.strip() for p in proxy_list_str.split(",") if p.strip()]
        proxy_used = proxies[0] if proxies else "none"

        # Map platform name to scraper class
        scraper_map = {
            "wellfound": WellfoundScraper,
            "weworkremotely": WeWorkRemotelyScraper,
            "ycombinator": YCStartupScraper,
            "yc": YCStartupScraper,
            "arc": ArcDevScraper,
            "turing": TuringScraper,
            "crossover": CrossoverScraper,
            "nodesk": NodeskScraper,
            "toptal": ToptalScraper,
        }

        scraper_class = scraper_map.get(platform.lower())
        if not scraper_class:
            return json.dumps(
                {
                    "platform": platform,
                    "jobs_found": 0,
                    "jobs_upserted": 0,
                    "proxy_used": proxy_used,
                    "errors": [f"Unknown platform: {platform}"],
                }
            )

        scraper = scraper_class(jobs_limit=max_jobs)

        @_with_retry
        def scrape_with_playwright():
            return asyncio.run(scraper.run(GLOBAL_PLAYWRIGHT_MANAGER))

        raw_jobs = scrape_with_playwright()

        jobs_upserted = 0
        errors = []
        blocked = False

        for job in raw_jobs:
            try:
                result = upsert_job_post(
                    run_batch_id=run_batch_id,
                    source_platform=platform,
                    title=job.get("title", ""),
                    company=job.get("company", ""),
                    url=job.get("job_url", ""),
                    location=job.get("location", "Remote"),
                    posted_at=job.get("posted_date", ""),
                )
                result_data = json.loads(result)
                if "error" not in result_data:
                    jobs_upserted += 1
                else:
                    errors.append(result_data["error"])
            except Exception as e:
                logger.error(f"Failed to upsert job: {e}")
                errors.append(str(e))

        logger.info(
            f"Playwright scrape completed for {platform}: {len(raw_jobs)} jobs found, {jobs_upserted} upserted"
        )

        return json.dumps(
            {
                "platform": platform,
                "jobs_found": len(raw_jobs),
                "jobs_upserted": jobs_upserted,
                "proxy_used": proxy_used,
                "errors": errors[:10],
                "blocked": blocked,
            }
        )

    except Exception as e:
        error_msg = str(e).lower()
        blocked = "captcha" in error_msg or "blocked" in error_msg or "403" in error_msg

        if blocked:
            log_event(
                run_batch_id=run_batch_id,
                level="WARNING",
                event_type="playwright_blocked",
                message=f"{platform} blocked or CAPTCHA detected: {str(e)}",
            )
        else:
            log_event(
                run_batch_id=run_batch_id,
                level="ERROR",
                event_type="playwright_scrape_failed",
                message=f"{platform} scrape failed: {str(e)}",
            )

        logger.error(f"Playwright scrape failed for {platform}: {e}")

        return json.dumps(
            {
                "platform": platform,
                "jobs_found": 0,
                "jobs_upserted": 0,
                "proxy_used": "none",
                "errors": [str(e)],
                "blocked": blocked,
            }
        )


@tool
@agentops.track_tool
def run_serpapi_scrape(
    run_batch_id: str,
    search_query: str,
    location: str = "Remote",
    results_wanted: int = 25,
) -> str:
    """
    Scrape Google Jobs via SerpAPI with key rotation.

    Args:
        run_batch_id: UUID of the run batch.
        search_query: Job search query string.
        location: Location filter (default: "Remote").
        results_wanted: Number of results to fetch.

    Returns:
        JSON string with scraping results and statistics.
    """
    global _serpapi_key_index

    try:
        # Collect available SerpAPI keys
        serpapi_keys = []
        for i in range(1, 5):
            key = os.getenv(f"SERPAPI_API_KEY_{i}")
            if key:
                serpapi_keys.append((f"SERPAPI_API_KEY_{i}", key))

        if not serpapi_keys:
            return json.dumps(
                {
                    "jobs_found": 0,
                    "jobs_upserted": 0,
                    "api_key_used": "none",
                    "errors": ["No SerpAPI keys configured"],
                }
            )

        # Round-robin key selection
        key_name, key_value = serpapi_keys[_serpapi_key_index % len(serpapi_keys)]
        _serpapi_key_index += 1

        # Initialize ResourceManager and scraper
        resource_manager = ResourceManager(monthly_quota=250)
        scraper = SerpAPIGoogleJobsScraper(
            jobs_per_site=results_wanted,
            resource_manager=resource_manager,
            query=search_query,
        )

        @_with_retry
        def scrape_sync():
            return asyncio.run(scraper.run())

        raw_jobs = scrape_sync()

        jobs_upserted = 0
        errors = []

        for job in raw_jobs:
            try:
                result = upsert_job_post(
                    run_batch_id=run_batch_id,
                    source_platform="google_jobs",
                    title=job.get("title", ""),
                    company=job.get("company", ""),
                    url=job.get("job_url", ""),
                    location=job.get("location", location),
                    posted_at=job.get("posted_date", ""),
                )
                result_data = json.loads(result)
                if "error" not in result_data:
                    jobs_upserted += 1
                else:
                    errors.append(result_data["error"])
            except Exception as e:
                logger.error(f"Failed to upsert job: {e}")
                errors.append(str(e))

        logger.info(
            f"SerpAPI scrape completed: {len(raw_jobs)} jobs found, {jobs_upserted} upserted"
        )

        # Extract key index for reporting
        key_index = key_name.split("_")[-1]

        return json.dumps(
            {
                "jobs_found": len(raw_jobs),
                "jobs_upserted": jobs_upserted,
                "api_key_used": f"key_{key_index}",
                "errors": errors[:10],
            }
        )

    except Exception as e:
        error_msg = str(e).lower()

        # Check for quota errors and retry with next key
        if "429" in error_msg or "quota" in error_msg:
            log_event(
                run_batch_id=run_batch_id,
                level="WARNING",
                event_type="serpapi_quota_exceeded",
                message=f"SerpAPI quota exceeded, rotating to next key: {str(e)}",
            )

            # Try next key once
            if len(serpapi_keys) > 1:
                try:
                    key_name, key_value = serpapi_keys[
                        _serpapi_key_index % len(serpapi_keys)
                    ]
                    _serpapi_key_index += 1

                    resource_manager = ResourceManager(monthly_quota=250)
                    scraper = SerpAPIGoogleJobsScraper(
                        jobs_per_site=results_wanted,
                        resource_manager=resource_manager,
                        query=search_query,
                    )

                    raw_jobs = asyncio.run(scraper.run())

                    jobs_upserted = 0
                    for job in raw_jobs:
                        result = upsert_job_post(
                            run_batch_id=run_batch_id,
                            source_platform="google_jobs",
                            title=job.get("title", ""),
                            company=job.get("company", ""),
                            url=job.get("job_url", ""),
                            location=job.get("location", location),
                            posted_at=job.get("posted_date", ""),
                        )
                        result_data = json.loads(result)
                        if "error" not in result_data:
                            jobs_upserted += 1

                    key_index = key_name.split("_")[-1]
                    return json.dumps(
                        {
                            "jobs_found": len(raw_jobs),
                            "jobs_upserted": jobs_upserted,
                            "api_key_used": f"key_{key_index}",
                            "errors": [],
                        }
                    )
                except Exception as retry_error:
                    logger.error(f"SerpAPI retry also failed: {retry_error}")

        log_event(
            run_batch_id=run_batch_id,
            level="ERROR",
            event_type="serpapi_scrape_failed",
            message=f"SerpAPI scrape failed: {str(e)}",
        )

        logger.error(f"SerpAPI scrape failed: {e}")

        return json.dumps(
            {
                "jobs_found": 0,
                "jobs_upserted": 0,
                "api_key_used": "none",
                "errors": [str(e)],
            }
        )


@tool
@agentops.track_tool
def run_safety_net_scrape(run_batch_id: str, current_job_count: int) -> str:
    """
    Run safety-net scrapers (Nodesk, Toptal) if minimum job count not met.

    Args:
        run_batch_id: UUID of the run batch.
        current_job_count: Current number of jobs collected.

    Returns:
        JSON string with safety net results.
    """
    try:
        minimum = int(os.getenv("JOBS_PER_RUN_MINIMUM", "100"))

        if current_job_count >= minimum:
            return json.dumps(
                {
                    "safety_net_triggered": False,
                    "reason": "minimum_already_met",
                    "current_count": current_job_count,
                    "minimum": minimum,
                }
            )

        log_event(
            run_batch_id=run_batch_id,
            level="INFO",
            event_type="safety_net_triggered",
            message=f"Only {current_job_count} jobs found, activating Nodesk + Toptal",
        )

        logger.info(
            f"Safety net triggered: {current_job_count}/{minimum} jobs. Running Nodesk and Toptal."
        )

        additional_jobs_found = 0
        additional_jobs_upserted = 0
        platforms = ["nodesk", "toptal"]

        for platform in platforms:
            try:
                result = run_playwright_scrape(
                    run_batch_id=run_batch_id, platform=platform, max_jobs=30
                )
                result_data = json.loads(result)

                additional_jobs_found += result_data.get("jobs_found", 0)
                additional_jobs_upserted += result_data.get("jobs_upserted", 0)

                logger.info(
                    f"Safety net {platform}: {result_data.get('jobs_found', 0)} jobs found"
                )

            except Exception as e:
                logger.error(f"Safety net scrape failed for {platform}: {e}")
                log_event(
                    run_batch_id=run_batch_id,
                    level="ERROR",
                    event_type="safety_net_scrape_failed",
                    message=f"{platform} safety net scrape failed: {str(e)}",
                )

        return json.dumps(
            {
                "safety_net_triggered": True,
                "additional_jobs_found": additional_jobs_found,
                "additional_jobs_upserted": additional_jobs_upserted,
                "platforms": platforms,
                "current_count": current_job_count,
                "minimum": minimum,
            }
        )

    except Exception as e:
        logger.error(f"Safety net scrape failed: {e}")
        return json.dumps(
            {
                "safety_net_triggered": False,
                "reason": "error",
                "error": str(e),
            }
        )


@tool
@agentops.track_tool
def normalise_and_dedup(run_batch_id: str) -> str:
    """
    Normalize and deduplicate jobs for the current run batch.

    Args:
        run_batch_id: UUID of the run batch.

    Returns:
        JSON string with deduplication results.
    """
    if not DB_URL:
        return json.dumps(
            {
                "duplicates_removed": 0,
                "jobs_remaining": 0,
                "run_batch_id": run_batch_id,
                "error": "DB_URL not configured",
            }
        )

    conn = None
    try:
        conn = psycopg2.connect(DB_URL)
        conn.autocommit = False
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        # Find duplicate URLs within this run batch
        cursor.execute(
            """
            SELECT url, COUNT(*) as count, ARRAY_AGG(id ORDER BY created_at DESC) as ids
            FROM job_posts
            WHERE run_batch_id = %s
            GROUP BY url
            HAVING COUNT(*) > 1
            """,
            (run_batch_id,),
        )

        duplicates = cursor.fetchall()
        duplicates_removed = 0

        for dup in duplicates:
            # Keep the most recent (first in array), delete the rest
            ids_to_delete = dup["ids"][1:]
            if ids_to_delete:
                cursor.execute(
                    """
                    DELETE FROM job_posts
                    WHERE id = ANY(%s)
                    """,
                    (ids_to_delete,),
                )
                duplicates_removed += len(ids_to_delete)

        # Get final count
        cursor.execute(
            """
            SELECT COUNT(*) as count
            FROM job_posts
            WHERE run_batch_id = %s
            """,
            (run_batch_id,),
        )

        result = cursor.fetchone()
        jobs_remaining = result["count"] if result else 0

        conn.commit()

        logger.info(
            f"Deduplication completed: {duplicates_removed} duplicates removed, {jobs_remaining} jobs remaining"
        )

        return json.dumps(
            {
                "duplicates_removed": duplicates_removed,
                "jobs_remaining": jobs_remaining,
                "run_batch_id": run_batch_id,
            }
        )

    except Exception as e:
        if conn:
            conn.rollback()
        logger.error(f"Deduplication failed: {e}")
        return json.dumps(
            {
                "duplicates_removed": 0,
                "jobs_remaining": 0,
                "run_batch_id": run_batch_id,
                "error": str(e),
            }
        )
    finally:
        if conn:
            conn.close()


@tool
@agentops.track_tool
def get_scrape_summary(run_batch_id: str) -> str:
    """
    Get summary statistics for the current scrape run.

    Args:
        run_batch_id: UUID of the run batch.

    Returns:
        JSON string with scrape summary and platform breakdown.
    """
    if not DB_URL:
        return json.dumps(
            {
                "run_batch_id": run_batch_id,
                "total_jobs": 0,
                "by_platform": {},
                "minimum_met": False,
                "error": "DB_URL not configured",
            }
        )

    conn = None
    try:
        conn = psycopg2.connect(DB_URL)
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        # Get platform breakdown
        cursor.execute(
            """
            SELECT source_platform, COUNT(*) as count
            FROM job_posts
            WHERE run_batch_id = %s
            GROUP BY source_platform
            ORDER BY count DESC
            """,
            (run_batch_id,),
        )

        platform_results = cursor.fetchall()
        by_platform = {row["source_platform"]: row["count"] for row in platform_results}

        total_jobs = sum(by_platform.values())
        minimum = int(os.getenv("JOBS_PER_RUN_MINIMUM", "100"))
        minimum_met = total_jobs >= minimum

        logger.info(
            f"Scrape summary: {total_jobs} total jobs across {len(by_platform)} platforms"
        )

        return json.dumps(
            {
                "run_batch_id": run_batch_id,
                "total_jobs": total_jobs,
                "by_platform": by_platform,
                "minimum_met": minimum_met,
                "minimum_target": minimum,
            }
        )

    except Exception as e:
        logger.error(f"Failed to get scrape summary: {e}")
        return json.dumps(
            {
                "run_batch_id": run_batch_id,
                "total_jobs": 0,
                "by_platform": {},
                "minimum_met": False,
                "error": str(e),
            }
        )
    finally:
        if conn:
            conn.close()
