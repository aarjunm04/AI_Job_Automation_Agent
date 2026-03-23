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
import concurrent.futures
from typing import Optional, List, Dict, Any

from crewai.tools import tool
import agentops
from agentops.sdk.decorators import agent, operation
import psycopg2
import psycopg2.extras

from scrapers.jobspy_adapter import JobSpyAdapter
from scrapers.scraper_engine import (
    ScraperEngine,
    RemoteOKAPIScraper,
    HimalayasAPIScraper,
)
from tools.serpapi_tool import search_google_jobs
from scrapers.scraper_service import (
    GLOBAL_PLAYWRIGHT_MANAGER,
    WellfoundScraper,
    WeWorkRemotelyScraper,
    NodeskScraper,
)
from tools.postgres_tools import upsert_job_post, log_event, get_platform_config
from utils.db_utils import get_db_conn

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
@operation
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
        # Run JobSpy ONLY (avoid triggering other scrapers).
        allowed_countries = []
        try:
            allowed_countries = _get_engine().filter_engine.allowed_countries
        except Exception:
            allowed_countries = []

        hours_old = 72
        adapter = JobSpyAdapter(
            jobs_per_site=int(results_wanted),
            concurrency=4,
            hours_old=hours_old,
            allowed_countries=allowed_countries,
        )
        raw_jobs = asyncio.run(adapter.run())

        jobs_upserted = 0
        errors = []

        for job in raw_jobs:
            try:
                result = upsert_job_post.run(
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
            f"JobSpy scrape completed: {len(raw_jobs)} found, {jobs_upserted} upserted"
        )

        return json.dumps(
            {
                "platform": "jobspy",
                "jobs_found": len(raw_jobs),
                "jobs_upserted": jobs_upserted,
                "run_batch_id": run_batch_id,
                "errors": errors[:10],  # Limit error list
            }
        )

    except Exception as e:
        logger.error(f"JobSpy scrape failed: {e}")
        log_event.run(
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
@operation
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
            config_result = get_platform_config.run(platform_name)
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
                        result = upsert_job_post.run(
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
            log_event.run(
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
@operation
@agentops.track_tool
def run_playwright_scrape(
    run_batch_id: str, platform: str, max_jobs: int = 30
) -> str:
    """
    Scrape a single platform using Playwright browser automation.

    Args:
        run_batch_id: UUID of the run batch.
        platform: Platform name (wellfound, weworkremotely, nodesk).
        max_jobs: Maximum number of jobs to scrape.

    Returns:
        JSON string with scraping results and statistics.
    """
    try:
        playwright_timeout_ms = 30000
        playwright_timeout_s = max(1.0, playwright_timeout_ms / 1000.0)

        # Get proxy from environment
        proxy_list_str = os.getenv("WEBSHARE_PROXY_LIST", "")
        proxies = [p.strip() for p in proxy_list_str.split(",") if p.strip()]
        proxy_used = proxies[0] if proxies else "none"

        # Map platform name to scraper class
        scraper_map = {
            "wellfound": WellfoundScraper,
            "weworkremotely": WeWorkRemotelyScraper,
            "nodesk": NodeskScraper,
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

        def _do_scrape():
            return asyncio.run(scraper.run(GLOBAL_PLAYWRIGHT_MANAGER))

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            future = ex.submit(_do_scrape)
            try:
                raw_jobs = future.result(timeout=playwright_timeout_s)
            except concurrent.futures.TimeoutError:
                logger.warning(
                    f"{platform}: scrape exceeded {playwright_timeout_ms}ms wall timeout — returning []"
                )
                return json.dumps(
                    {
                        "platform": platform,
                        "jobs_found": 0,
                        "jobs_upserted": 0,
                        "proxy_used": proxy_used,
                        "errors": [f"{platform} scrape exceeded {playwright_timeout_ms}ms wall timeout"],
                        "blocked": False,
                    }
                )

        jobs_upserted = 0
        errors = []
        blocked = False

        for job in raw_jobs:
            try:
                result = upsert_job_post.run(
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
            log_event.run(
                run_batch_id=run_batch_id,
                level="WARNING",
                event_type="playwright_blocked",
                message=f"{platform} blocked or CAPTCHA detected: {str(e)}",
            )
        else:
            log_event.run(
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
@operation
@agentops.track_tool
def run_serpapi_scrape(
    run_batch_id: str,
    query: str,
    location: str = "Remote",
    results_wanted: int = 25,
    **kwargs,
) -> str:
    """
    Scrape Google Jobs via SerpAPI with key rotation.

    Args:
        run_batch_id: UUID of the run batch.
        query: Job search query string.
        location: Location filter (default: "Remote").
        results_wanted: Number of results to fetch.

    Returns:
        JSON string with scraping results and statistics.
    """
    try:
        # Delegate to serpapi_tool — handles key rotation, AgentOps tracking,
        # credit tracking, and fail-soft error handling internally.
        result_str = search_google_jobs(
            query=query,
            location=location,
            num_results=results_wanted,
        )
        raw_jobs: list = []
        try:
            parsed = json.loads(result_str)
            if isinstance(parsed, list):
                raw_jobs = parsed
            elif isinstance(parsed, dict) and "error" in parsed:
                logger.warning(
                    "SerpAPI scrape returned error: %s", parsed["error"]
                )
                log_event.run(
                    run_batch_id=run_batch_id,
                    level="WARNING",
                    event_type="serpapi_scrape_failed",
                    message=f"SerpAPI error: {parsed['error']}",
                )
                return json.dumps(
                    {
                        "jobs_found": 0,
                        "jobs_upserted": 0,
                        "api_key_used": "none",
                        "errors": [parsed["error"]],
                    }
                )
        except (json.JSONDecodeError, TypeError) as parse_err:
            logger.error("SerpAPI result parse failed: %s", parse_err)
            raw_jobs = []

        jobs_upserted = 0
        errors = []

        for job in raw_jobs:
            try:
                result = upsert_job_post.run(
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
            except Exception as upsert_err:
                logger.error("Failed to upsert job: %s", upsert_err)
                errors.append(str(upsert_err))

        logger.info(
            "SerpAPI scrape completed: %d jobs found, %d upserted",
            len(raw_jobs),
            jobs_upserted,
        )

        return json.dumps(
            {
                "jobs_found": len(raw_jobs),
                "jobs_upserted": jobs_upserted,
                "api_key_used": "serpapi_tool",
                "errors": errors[:10],
            }
        )

    except Exception as e:
        log_event.run(
            run_batch_id=run_batch_id,
            level="ERROR",
            event_type="serpapi_scrape_failed",
            message=f"SerpAPI scrape failed: {str(e)}",
        )
        logger.error("SerpAPI scrape failed: %s", e)
        return json.dumps(
            {
                "jobs_found": 0,
                "jobs_upserted": 0,
                "api_key_used": "none",
                "errors": [str(e)],
            }
        )


@tool
@operation
@agentops.track_tool
def run_safety_net_scrape(run_batch_id: str, current_job_count: int) -> str:
    """
    Run safety-net scrapers (Nodesk) if minimum job count not met.

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

        log_event.run(
            run_batch_id=run_batch_id,
            level="INFO",
            event_type="safety_net_triggered",
            message=f"Only {current_job_count} jobs found, activating Nodesk",
        )

        logger.info(
            f"Safety net triggered: {current_job_count}/{minimum} jobs. Running Nodesk."
        )

        additional_jobs_found = 0
        additional_jobs_upserted = 0
        platforms = ["nodesk"]

        for platform in platforms:
            try:
                result = run_playwright_scrape.run(
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
                log_event.run(
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
@operation
@agentops.track_tool
def normalise_and_dedup(run_batch_id: str) -> str:
    """
    Normalize and deduplicate jobs for the current run batch.

    Args:
        run_batch_id: UUID of the run batch.

    Returns:
        JSON string with deduplication results.
    """
    conn = None
    try:
        conn = get_db_conn()
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        # Find duplicate URLs within this run batch
        cursor.execute(
            """
            SELECT url, COUNT(*) as count, ARRAY_AGG(id ORDER BY created_at DESC) as ids
            FROM jobs
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
                    DELETE FROM jobs
                    WHERE id = ANY(%s)
                    """,
                    (ids_to_delete,),
                )
                duplicates_removed += len(ids_to_delete)

        # Get final count
        cursor.execute(
            """
            SELECT COUNT(*) as count
            FROM jobs
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
@operation
@agentops.track_tool
def get_scrape_summary(run_batch_id: str) -> str:
    """
    Get summary statistics for the current scrape run.

    Args:
        run_batch_id: UUID of the run batch.

    Returns:
        JSON string with scrape summary and platform breakdown.
    """
    last_exc: Optional[Exception] = None
    for attempt in range(1, 4):
        conn = None
        try:
            conn = get_db_conn()
            if not conn:
                return json.dumps({
                    "run_batch_id": run_batch_id,
                    "total_jobs": 0,
                    "by_platform": {},
                    "minimum_met": False,
                    "error": "DB connection failed",
                })
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

            # Get platform breakdown
            cursor.execute(
                """
                SELECT source_platform, COUNT(*) as count
                FROM jobs
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
            last_exc = e
            if attempt < 3:
                time.sleep(2 ** attempt)
                logger.warning(
                    f"get_scrape_summary attempt {attempt}/3 failed: {e} — retrying"
                )
            else:
                logger.error(f"Failed to get scrape summary: {e}")
        finally:
            if conn:
                conn.close()

    return json.dumps(
        {
            "run_batch_id": run_batch_id,
            "total_jobs": 0,
            "by_platform": {},
            "minimum_met": False,
            "error": str(last_exc),
        }
    )
