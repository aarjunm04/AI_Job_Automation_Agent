from agentops.sdk.decorators import operation
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
import nest_asyncio
nest_asyncio.apply()
import concurrent.futures
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)

from crewai.tools import tool
import agentops
from agentops.sdk.decorators import agent, operation
import psycopg2
import psycopg2.extras

from scrapers.jobspy_adapter import JobSpyAdapter
from scrapers.scraper_engine import (
    ScraperEngine,
)
from tools.serpapi_tool import search_google_jobs
from scrapers.scraper_service import (
    GLOBAL_PLAYWRIGHT_MANAGER,
    WellfoundScraper,
    WeWorkRemotelyScraper,
    NodeskScraper,
    RemoteOKAPIScraper,
    HimalayasScraper,
)
from tools.postgres_tools import upsert_job_post, _upsert_job_post, log_event, _log_event, get_platform_config, _get_platform_config
from utils.db_utils import get_db_conn
from config.config_loader import config_loader



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
        min_jobs_target = config_loader.get_run_config().get("jobs_per_run_target", 100)
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
def run_jobspy_scrape(pipeline_run_id: str) -> str:
    """Run JobSpy across all configured search queries.

    Reads search_queries from config/user_profile.json via config_loader.
    Creates one JobSpyAdapter per query, runs them sequentially, and
    aggregates all results. Upserts every discovered job into Postgres.
    Fail-soft: one bad query does not abort the rest.

    Args:
        pipeline_run_id: UUID string for this pipeline run, used for dedup/logging.

    Returns:
        JSON string with aggregate stats dict (jobs_found, jobs_upserted, etc.).
    """
    prefs: dict = config_loader.get_job_preferences()
    queries: list[str] = prefs.get("search_queries", ["AI Engineer"])
    run_cfg: dict = config_loader.get_run_config()
    hours_old: int = int(run_cfg.get("hours_old", 72))
    results_per_query: int = int(run_cfg.get("results_per_query", 25))

    all_jobs: list[dict] = []
    for query in queries:
        try:
            # FIX 2: JobSpyAdapter missing cfg arg and dead concurrency kwarg
            from config.config_loader import ConfigLoader
            _cfg = ConfigLoader()
            adapter = JobSpyAdapter(
                cfg=_cfg,
                jobs_per_site=results_per_query,
                hours_old=hours_old,
            )
            # Override the search_term on the adapter for this specific query
            adapter.search_term = query
            jobs = asyncio.run(adapter.run())
            all_jobs.extend(jobs if isinstance(jobs, list) else [])
            logger.info(
                "run_jobspy_scrape: query='%s' returned %d jobs", query, len(jobs)
            )
        except Exception as exc:
            logger.error(
                "run_jobspy_scrape: query='%s' failed — %s", query, exc
            )
            continue  # fail-soft — one bad query does not abort the run

    logger.info(
        "run_jobspy_scrape: total aggregated jobs=%d across %d queries",
        len(all_jobs), len(queries)
    )

    # Upsert all aggregated jobs into Postgres
    jobs_upserted = 0
    errors: list[str] = []
    for job in all_jobs:
        try:
            result = _upsert_job_post(
                pipeline_run_id=pipeline_run_id,
                source_platform=job.get("source", "jobspy"),
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
        except Exception as upsert_exc:
            logger.error("run_jobspy_scrape: upsert failed — %s", upsert_exc)
            errors.append(str(upsert_exc))

    return json.dumps(
        {
            "platform": "jobspy",
            "jobs_found": len(all_jobs),
            "jobs_upserted": jobs_upserted,
            "queries_run": len(queries),
            "pipeline_run_id": pipeline_run_id,
            "errors": errors[:10],
        }
    )


@tool
def run_rest_api_scrape(
    pipeline_run_id: str, platforms: str = "remoteok,himalayas"
) -> str:
    """
    Scrape REST API platforms (RemoteOK, Himalayas).

    Args:
        pipeline_run_id: UUID of the run batch.
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
            config_result = _get_platform_config(platform_name)
            config = json.loads(config_result)

            if "error" in config:
                logger.warning(f"No config found for {platform_name}, using defaults")

            platforms_attempted.append(platform_name)

            # Initialize appropriate scraper
            scraper = None
            if platform_name == "remoteok":
                # FIX 1: Wrong kwarg on RemoteOKAPIScraper
                scraper = RemoteOKAPIScraper(jobs_limit=50)
            elif platform_name == "himalayas":
                # FIX 1: Wrong kwarg on HimalayasScraper
                scraper = HimalayasScraper(jobs_limit=50)
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
                        result = _upsert_job_post(
                            pipeline_run_id=pipeline_run_id,
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
            _log_event(
                pipeline_run_id=pipeline_run_id,
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
def run_playwright_scrape(
    pipeline_run_id: str, platform: str, max_jobs: int = 30
) -> str:
    """
    Scrape a single platform using Playwright browser automation.

    Args:
        pipeline_run_id: UUID of the run batch.
        platform: Platform name (wellfound, weworkremotely, nodesk).
        max_jobs: Maximum number of jobs to scrape.

    Returns:
        JSON string with scraping results and statistics.
    """
    # FIX 3: proxy_list_str undefined in run_playwright_scrape
    _proxy_parts: list[str] = []
    for _i in range(1, 11):
        for _pool in (1, 2):
            _val = os.getenv(f"WEBSHARE_PROXY_{_pool}_{_i}", "").strip()
            if _val:
                _proxy_parts.append(_val)
    proxy_list_str: str = ",".join(_proxy_parts)

    try:
        playwright_timeout_ms = 30000
        playwright_timeout_s = max(1.0, playwright_timeout_ms / 1000.0)

        # Get proxy from environment
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
                result = _upsert_job_post(
                    pipeline_run_id=pipeline_run_id,
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
            _log_event(
                pipeline_run_id=pipeline_run_id,
                level="WARNING",
                event_type="playwright_blocked",
                message=f"{platform} blocked or CAPTCHA detected: {str(e)}",
            )
        else:
            _log_event(
                pipeline_run_id=pipeline_run_id,
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
def run_serpapi_scrape(
    pipeline_run_id: str,
    query: str = "",
    location: str = "Remote",
    results_wanted: int = 25,
    **kwargs,
) -> str:
    """
    Scrape Google Jobs via SerpAPI with key rotation.
    Reads search_queries from config/user_profile.json.
    If query arg is non-empty it is used as a single override;
    otherwise all configured search_queries are run in sequence.
    """
    prefs: dict = config_loader.get_job_preferences()
    configured_queries: list[str] = prefs.get("search_queries", ["AI Engineer"])
    queries_to_run: list[str] = [query] if query.strip() else configured_queries

    all_raw_jobs: list[dict] = []
    all_errors: list[str] = []

    for q in queries_to_run:
        try:
            result_str = search_google_jobs(
                query=q,
                location=location,
                num_results=results_wanted,
            )
            
            if not result_str:
                logger.warning("run_serpapi_scrape: query='%s' returned no results", q)
                continue

            try:
                result_data = json.loads(result_str)
                jobs = result_data.get("jobs", [])
                all_raw_jobs.extend(jobs)
                logger.info(
                    "run_serpapi_scrape: query='%s' returned %d jobs", q, len(jobs)
                )
            except json.JSONDecodeError:
                logger.error("run_serpapi_scrape: query='%s' failed to parse JSON: %s", q, result_str[:200])
                all_errors.append(f"JSON parse error for query '{q}'")
                continue

        except Exception as exc:
            logger.error("run_serpapi_scrape: query='%s' failed — %s", q, exc)
            all_errors.append(str(exc))
            continue  # fail-soft

    # Upsert all aggregated jobs into Postgres
    jobs_upserted = 0
    for job in all_raw_jobs:
        try:
            result = _upsert_job_post(
                pipeline_run_id=pipeline_run_id,
                source_platform="serpapi_google_jobs",
                title=job.get("title", ""),
                company=job.get("company_name", ""),
                url=job.get("related_links", [{}])[0].get("link", ""),
                location=job.get("location", "Remote"),
                posted_at=job.get("detected_extensions", {}).get("posted_at", ""),
                raw_job_data=job,
            )
            result_data = json.loads(result)
            if "error" not in result_data:
                jobs_upserted += 1
            else:
                all_errors.append(result_data["error"])
        except Exception as upsert_exc:
            logger.error("run_serpapi_scrape: upsert failed — %s", upsert_exc)
            all_errors.append(str(upsert_exc))

    return json.dumps(
        {
            "platform": "serpapi",
            "jobs_found": len(all_raw_jobs),
            "jobs_upserted": jobs_upserted,
            "queries_run": len(queries_to_run),
            "pipeline_run_id": pipeline_run_id,
            "errors": all_errors[:10],
        }
    )


@tool
def run_safety_net_scrape(pipeline_run_id: str, current_job_count: int) -> str:
    """
    Run safety-net scrapers (Nodesk) if minimum job count not met.

    Args:
        pipeline_run_id: UUID of the run batch.
        current_job_count: Current number of jobs collected.

    Returns:
        JSON string with safety net results.
    """
    try:
        minimum = config_loader.get_run_config().get("jobs_per_run_target", 100)

        if current_job_count >= minimum:
            return json.dumps(
                {
                    "safety_net_triggered": False,
                    "reason": "minimum_already_met",
                    "current_count": current_job_count,
                    "minimum": minimum,
                }
            )

        _log_event(
            pipeline_run_id=pipeline_run_id,
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
                result = _run_playwright_scrape(
                    pipeline_run_id=pipeline_run_id, platform=platform, max_jobs=30
                )
                result_data = json.loads(result)

                additional_jobs_found += result_data.get("jobs_found", 0)
                additional_jobs_upserted += result_data.get("jobs_upserted", 0)

                logger.info(
                    f"Safety net {platform}: {result_data.get('jobs_found', 0)} jobs found"
                )

            except Exception as e:
                logger.error(f"Safety net scrape failed for {platform}: {e}")
                _log_event(
                    pipeline_run_id=pipeline_run_id,
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
def normalise_and_dedup(pipeline_run_id: str) -> str:
    """
    Normalize and deduplicate jobs for the current run batch.

    Args:
        pipeline_run_id: UUID of the run batch.

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
            WHERE pipeline_run_id = %s
            GROUP BY url
            HAVING COUNT(*) > 1
            """,
            (pipeline_run_id,),
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
            WHERE pipeline_run_id = %s
            """,
            (pipeline_run_id,),
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
                "pipeline_run_id": pipeline_run_id,
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
                "pipeline_run_id": pipeline_run_id,
                "error": str(e),
            }
        )
    finally:
        if conn:
            conn.close()


@tool
def get_scrape_summary(pipeline_run_id: str) -> str:
    """
    Get summary statistics for the current scrape run.

    Args:
        pipeline_run_id: UUID of the run batch.

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
                    "pipeline_run_id": pipeline_run_id,
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
                WHERE pipeline_run_id = %s
                GROUP BY source_platform
                ORDER BY count DESC
                """,
                (pipeline_run_id,),
            )

            platform_results = cursor.fetchall()
            by_platform = {row["source_platform"]: row["count"] for row in platform_results}

            total_jobs = sum(by_platform.values())
            minimum = config_loader.get_run_config().get("jobs_per_run_target", 100)
            minimum_met = total_jobs >= minimum

            logger.info(
                f"Scrape summary: {total_jobs} total jobs across {len(by_platform)} platforms"
            )

            return json.dumps(
                {
                    "pipeline_run_id": pipeline_run_id,
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
            "pipeline_run_id": pipeline_run_id,
            "total_jobs": 0,
            "by_platform": {},
            "minimum_met": False,
            "error": str(last_exc),
        }
    )


# ═══════════════════════════════════════════════════════════════════════════════
# .func ALIASES — raw function access (bypasses CrewAI Tool Pydantic wrapper)
# Use these underscore aliases when calling tools DIRECTLY from agent code.
# NEVER call the @tool version directly from an agent — always use _alias.
# ═══════════════════════════════════════════════════════════════════════════════
_run_jobspy_scrape     = run_jobspy_scrape.func     if hasattr(run_jobspy_scrape,     "func") else run_jobspy_scrape
_run_rest_api_scrape   = run_rest_api_scrape.func   if hasattr(run_rest_api_scrape,   "func") else run_rest_api_scrape
_run_playwright_scrape = run_playwright_scrape.func if hasattr(run_playwright_scrape, "func") else run_playwright_scrape
_run_serpapi_scrape    = run_serpapi_scrape.func    if hasattr(run_serpapi_scrape,    "func") else run_serpapi_scrape
_run_safety_net_scrape = run_safety_net_scrape.func if hasattr(run_safety_net_scrape, "func") else run_safety_net_scrape
_normalise_and_dedup   = normalise_and_dedup.func   if hasattr(normalise_and_dedup,   "func") else normalise_and_dedup
_get_scrape_summary    = get_scrape_summary.func    if hasattr(get_scrape_summary,    "func") else get_scrape_summary

# ═══════════════════════════════════════════════════════════════════════════════
# RUNTIME GUARDS — Ensure aliases are callable at module load
# ═══════════════════════════════════════════════════════════════════════════════
for _name, _alias in [
    ("run_jobspy_scrape", _run_jobspy_scrape),
    ("run_rest_api_scrape", _run_rest_api_scrape),
    ("run_playwright_scrape", _run_playwright_scrape),
    ("run_serpapi_scrape", _run_serpapi_scrape),
    ("run_safety_net_scrape", _run_safety_net_scrape),
    ("normalise_and_dedup", _normalise_and_dedup),
    ("get_scrape_summary", _get_scrape_summary),
]:
    assert callable(_alias), f"CRITICAL: Tool alias {_name} is not callable. Check @tool decoration."

