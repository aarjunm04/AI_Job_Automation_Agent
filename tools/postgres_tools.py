"""
Postgres data access layer for AI Job Application Agent.

This module provides CrewAI tool functions for all database operations across
the entire pipeline. All agents use these tools for reading and writing to the
PostgreSQL database (local Docker or Supabase).

All tools handle connection management, retry logic, and fail-soft error handling.
"""

import os
import json
import logging
import time
from typing import Any, Callable, Optional
from functools import wraps

import psycopg2
import psycopg2.extras
from psycopg2.extensions import connection as PgConnection
from crewai.tools import tool
import agentops
from agentops.sdk.decorators import agent, operation

# Module-level logger
logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("LOG_LEVEL", "INFO"))

# Database URL selection based on ACTIVE_DB environment variable
DB_URL = (
    os.getenv("LOCAL_POSTGRES_URL")
    if os.getenv("ACTIVE_DB", "local") == "local"
    else os.getenv("SUPABASE_URL")
)

__all__ = [
    "upsert_job_post",
    "save_job_score",
    "create_application",
    "update_application_status",
    "create_run_batch",
    "update_run_batch_stats",
    "log_event",
    "get_queued_jobs",
    "get_platform_config",
    "get_run_stats",
]


def _get_conn() -> PgConnection:
    """
    Opens and returns a psycopg2 connection to the database.

    Returns:
        psycopg2.extensions.connection: Database connection with autocommit=False.

    Raises:
        RuntimeError: If DB_URL is None or connection fails.
    """
    if not DB_URL:
        raise RuntimeError(
            "Database URL not configured. Set LOCAL_POSTGRES_URL or SUPABASE_URL in narad.env"
        )

    try:
        conn = psycopg2.connect(DB_URL)
        conn.autocommit = False
        return conn
    except Exception as e:
        logger.critical(f"Failed to connect to database: {e}")
        raise RuntimeError(f"Database connection failed: {e}")


def _with_retry(max_retries: int = 3) -> Callable:
    """
    Decorator that wraps database calls with retry logic.

    Args:
        max_retries: Maximum number of retry attempts (default: 3).

    Returns:
        Decorated function with retry logic.
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
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
                        logger.critical(
                            f"{func.__name__} failed after {max_retries} attempts: {e}"
                        )

            raise last_exception

        return wrapper

    return decorator


@tool
@operation
def upsert_job_post(
    run_batch_id: str,
    source_platform: str,
    title: str,
    company: str,
    url: str,
    location: str = "",
    posted_at: str = "",
) -> str:
    """
    Insert or update a job post in the database.

    Args:
        run_batch_id: UUID of the run batch.
        source_platform: Platform where job was discovered.
        title: Job title.
        company: Company name.
        url: Job posting URL (unique constraint).
        location: Job location (optional).
        posted_at: ISO8601 timestamp when job was posted (optional).

    Returns:
        JSON string with job_post_id and action (inserted or updated).
    """

    @_with_retry(max_retries=3)
    def _execute() -> str:
        conn = None
        try:
            conn = _get_conn()
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

            posted_at_value = posted_at if posted_at else None

            cursor.execute(
                """
                INSERT INTO job_posts (run_batch_id, source_platform, title, company, url, location, posted_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (url) DO UPDATE SET
                    title = EXCLUDED.title,
                    company = EXCLUDED.company,
                    location = EXCLUDED.location,
                    source_platform = EXCLUDED.source_platform,
                    run_batch_id = EXCLUDED.run_batch_id
                RETURNING id, (xmax = 0) AS inserted
                """,
                (
                    run_batch_id,
                    source_platform,
                    title,
                    company,
                    url,
                    location,
                    posted_at_value,
                ),
            )

            result = cursor.fetchone()
            conn.commit()

            action = "inserted" if result["inserted"] else "updated"
            logger.info(
                f"Job post {action}: {title} at {company} (id: {result['id']})"
            )

            return json.dumps({"job_post_id": str(result["id"]), "action": action})

        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Failed to upsert job post: {e}")
            return json.dumps(
                {"error": "upsert_job_post_failed", "detail": str(e)}
            )
        finally:
            if conn:
                conn.close()

    try:
        return _execute()
    except Exception as e:
        return json.dumps({"error": "upsert_job_post_failed", "detail": str(e)})


@tool
@operation
def save_job_score(
    job_post_id: str,
    resume_id: str,
    fit_score: float,
    eligibility_pass: bool,
    reasons_json: dict,
) -> str:
    """
    Save or update a job score for a job post and resume combination.

    Args:
        job_post_id: UUID of the job post.
        resume_id: UUID of the resume used for scoring.
        fit_score: Fit score (0.0 to 1.0).
        eligibility_pass: Whether job passed eligibility filter.
        reasons_json: Dictionary with scoring reasoning.

    Returns:
        JSON string with job_score_id.
    """

    @_with_retry(max_retries=3)
    def _execute() -> str:
        conn = None
        try:
            conn = _get_conn()
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

            cursor.execute(
                """
                INSERT INTO job_scores (job_post_id, resume_id, fit_score, eligibility_pass, reasons_json)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id
                """,
                (job_post_id, resume_id, fit_score, eligibility_pass, json.dumps(reasons_json)),
            )

            result = cursor.fetchone()
            conn.commit()

            logger.info(
                f"Job score saved: job_post_id={job_post_id}, fit_score={fit_score}, "
                f"eligibility={eligibility_pass}"
            )

            return json.dumps({"job_score_id": str(result["id"])})

        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Failed to save job score: {e}")
            return json.dumps({"error": "save_job_score_failed", "detail": str(e)})
        finally:
            if conn:
                conn.close()

    try:
        return _execute()
    except Exception as e:
        return json.dumps({"error": "save_job_score_failed", "detail": str(e)})


@tool
@operation
def create_application(
    job_post_id: str,
    resume_id: str,
    user_id: str,
    mode: str,
    status: str,
    platform: str,
    error_code: str = "",
) -> str:
    """
    Create an application record and optionally queue it for manual review.

    If status is 'manual_queued', also creates a queued_jobs entry with priority
    derived from the fit_score.

    Args:
        job_post_id: UUID of the job post.
        resume_id: UUID of the resume used.
        user_id: UUID of the user.
        mode: Application mode ('auto' or 'manual').
        status: Application status ('applied', 'failed', or 'manual_queued').
        platform: Platform name.
        error_code: Error code if status is 'failed' (optional).

    Returns:
        JSON string with application_id and queued_job_id (if queued).
    """

    @_with_retry(max_retries=3)
    def _execute() -> str:
        conn = None
        try:
            conn = _get_conn()
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

            error_code_value = error_code if error_code else None

            cursor.execute(
                """
                INSERT INTO applications (job_post_id, resume_id, user_id, mode, status, platform, error_code)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (job_post_id, resume_id, user_id, mode, status, platform, error_code_value),
            )

            application_result = cursor.fetchone()
            application_id = str(application_result["id"])
            queued_job_id = None

            if status == "manual_queued":
                cursor.execute(
                    """
                    SELECT fit_score FROM job_scores WHERE job_post_id = %s
                    ORDER BY scored_at DESC LIMIT 1
                    """,
                    (job_post_id,),
                )
                score_result = cursor.fetchone()

                if score_result:
                    fit_score = score_result["fit_score"]
                    if fit_score >= 0.75:
                        priority = 1
                    elif fit_score >= 0.50:
                        priority = 3
                    else:
                        priority = 5
                else:
                    priority = 5

                cursor.execute(
                    """
                    INSERT INTO queued_jobs (application_id, job_post_id, priority)
                    VALUES (%s, %s, %s)
                    RETURNING id
                    """,
                    (application_id, job_post_id, priority),
                )

                queued_result = cursor.fetchone()
                queued_job_id = str(queued_result["id"])

            conn.commit()

            logger.info(
                f"Application created: id={application_id}, status={status}, "
                f"queued={queued_job_id is not None}"
            )

            return json.dumps(
                {"application_id": application_id, "queued_job_id": queued_job_id}
            )

        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Failed to create application: {e}")
            return json.dumps(
                {"error": "create_application_failed", "detail": str(e)}
            )
        finally:
            if conn:
                conn.close()

    try:
        return _execute()
    except Exception as e:
        return json.dumps({"error": "create_application_failed", "detail": str(e)})


@tool
@operation
def update_application_status(
    application_id: str, status: str, error_code: str = ""
) -> str:
    """
    Update the status of an existing application.

    Args:
        application_id: UUID of the application.
        status: New status ('applied', 'failed', or 'manual_queued').
        error_code: Error code if status is 'failed' (optional).

    Returns:
        JSON string with updated flag and application_id.
    """

    @_with_retry(max_retries=3)
    def _execute() -> str:
        conn = None
        try:
            conn = _get_conn()
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

            error_code_value = error_code if error_code else None

            cursor.execute(
                """
                UPDATE applications
                SET status = %s, error_code = %s
                WHERE id = %s
                RETURNING id
                """,
                (status, error_code_value, application_id),
            )

            result = cursor.fetchone()
            conn.commit()

            updated = result is not None
            logger.info(
                f"Application status updated: id={application_id}, status={status}, updated={updated}"
            )

            return json.dumps({"updated": updated, "application_id": application_id})

        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Failed to update application status: {e}")
            return json.dumps(
                {"error": "update_application_status_failed", "detail": str(e)}
            )
        finally:
            if conn:
                conn.close()

    try:
        return _execute()
    except Exception as e:
        return json.dumps(
            {"error": "update_application_status_failed", "detail": str(e)}
        )


@tool
@operation
def create_run_batch(run_index_in_week: int) -> str:
    """
    Create a new run batch for the current date.

    Args:
        run_index_in_week: Index of this run in the week (1, 2, or 3).

    Returns:
        JSON string with run_batch_id and run_date.
    """

    @_with_retry(max_retries=3)
    def _execute() -> str:
        conn = None
        try:
            conn = _get_conn()
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

            cursor.execute(
                """
                INSERT INTO run_batches (run_index_in_week)
                VALUES (%s)
                RETURNING id, run_date
                """,
                (run_index_in_week,),
            )

            result = cursor.fetchone()
            conn.commit()

            logger.info(
                f"Run batch created: id={result['id']}, date={result['run_date']}, "
                f"index={run_index_in_week}"
            )

            return json.dumps(
                {
                    "run_batch_id": str(result["id"]),
                    "run_date": str(result["run_date"]),
                }
            )

        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Failed to create run batch: {e}")
            return json.dumps(
                {"error": "create_run_batch_failed", "detail": str(e)}
            )
        finally:
            if conn:
                conn.close()

    try:
        return _execute()
    except Exception as e:
        return json.dumps({"error": "create_run_batch_failed", "detail": str(e)})


@tool
@operation
def update_run_batch_stats(
    run_batch_id: str, jobs_discovered: int, jobs_auto_applied: int, jobs_queued: int
) -> str:
    """
    Update run batch statistics and close the batch.

    Args:
        run_batch_id: UUID of the run batch.
        jobs_discovered: Total jobs discovered.
        jobs_auto_applied: Jobs automatically applied to.
        jobs_queued: Jobs queued for manual review.

    Returns:
        JSON string with updated flag.
    """

    @_with_retry(max_retries=3)
    def _execute() -> str:
        conn = None
        try:
            conn = _get_conn()
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

            cursor.execute(
                """
                UPDATE run_batches
                SET jobs_discovered = %s,
                    jobs_auto_applied = %s,
                    jobs_queued = %s,
                    closed_at = NOW()
                WHERE id = %s
                RETURNING id
                """,
                (jobs_discovered, jobs_auto_applied, jobs_queued, run_batch_id),
            )

            result = cursor.fetchone()
            conn.commit()

            updated = result is not None
            logger.info(
                f"Run batch stats updated: id={run_batch_id}, discovered={jobs_discovered}, "
                f"applied={jobs_auto_applied}, queued={jobs_queued}"
            )

            return json.dumps({"updated": updated})

        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Failed to update run batch stats: {e}")
            return json.dumps(
                {"error": "update_run_batch_stats_failed", "detail": str(e)}
            )
        finally:
            if conn:
                conn.close()

    try:
        return _execute()
    except Exception as e:
        return json.dumps(
            {"error": "update_run_batch_stats_failed", "detail": str(e)}
        )


@tool
@operation
def log_event(
    run_batch_id: str,
    level: str,
    event_type: str,
    message: str,
    application_id: str = "",
    job_post_id: str = "",
) -> str:
    """
    Log an event to the logs_events table.

    This function never retries and silently swallows errors to prevent
    logging failures from disrupting the pipeline.

    Args:
        run_batch_id: UUID of the run batch.
        level: Log level ('INFO', 'WARNING', 'ERROR', 'CRITICAL').
        event_type: Type of event.
        message: Log message.
        application_id: UUID of related application (optional).
        job_post_id: UUID of related job post (optional).

    Returns:
        JSON string with logged flag and log_id.
    """
    conn = None
    try:
        conn = _get_conn()
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        app_id_value = application_id if application_id else None
        job_id_value = job_post_id if job_post_id else None

        cursor.execute(
            """
            INSERT INTO logs_events (run_batch_id, level, event_type, message, application_id, job_post_id)
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (run_batch_id, level, event_type, message, app_id_value, job_id_value),
        )

        result = cursor.fetchone()
        conn.commit()

        logger.debug(f"Event logged: {level} - {event_type} - {message}")

        return json.dumps({"logged": True, "log_id": str(result["id"])})

    except Exception as e:
        if conn:
            conn.rollback()
        logger.warning(f"Failed to log event (non-critical): {e}")
        return json.dumps({"logged": False, "log_id": None})
    finally:
        if conn:
            conn.close()


@tool
@operation
def get_queued_jobs(limit: int = 50) -> str:
    """
    Retrieve queued jobs ordered by priority and queue time.

    Args:
        limit: Maximum number of jobs to return (default: 50).

    Returns:
        JSON string with array of queued job objects.
    """

    @_with_retry(max_retries=3)
    def _execute() -> str:
        conn = None
        try:
            conn = _get_conn()
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

            cursor.execute(
                """
                SELECT
                    qj.id AS queued_job_id,
                    qj.job_post_id,
                    qj.application_id,
                    jp.url,
                    jp.title,
                    jp.company,
                    jp.source_platform AS platform,
                    qj.priority,
                    qj.notes,
                    qj.queued_at
                FROM queued_jobs qj
                JOIN job_posts jp ON qj.job_post_id = jp.id
                JOIN applications a ON qj.application_id = a.id
                ORDER BY qj.priority ASC, qj.queued_at ASC
                LIMIT %s
                """,
                (limit,),
            )

            results = cursor.fetchall()

            jobs = []
            for row in results:
                jobs.append(
                    {
                        "queued_job_id": str(row["queued_job_id"]),
                        "job_post_id": str(row["job_post_id"]),
                        "application_id": str(row["application_id"]),
                        "url": row["url"],
                        "title": row["title"],
                        "company": row["company"],
                        "platform": row["platform"],
                        "priority": row["priority"],
                        "notes": row["notes"],
                        "queued_at": str(row["queued_at"]),
                    }
                )

            logger.info(f"Retrieved {len(jobs)} queued jobs")

            return json.dumps(jobs)

        except Exception as e:
            logger.error(f"Failed to get queued jobs: {e}")
            return json.dumps({"error": "get_queued_jobs_failed", "detail": str(e)})
        finally:
            if conn:
                conn.close()

    try:
        return _execute()
    except Exception as e:
        return json.dumps({"error": "get_queued_jobs_failed", "detail": str(e)})


@tool
@operation
def get_platform_config(platform: str) -> str:
    """
    Retrieve platform configuration limits.

    Args:
        platform: Platform name.

    Returns:
        JSON string with platform config (returns defaults if not found).
    """

    @_with_retry(max_retries=3)
    def _execute() -> str:
        conn = None
        try:
            conn = _get_conn()
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

            cursor.execute(
                """
                SELECT platform, max_per_run, max_per_day, max_concurrent_sessions
                FROM config_limits
                WHERE platform = %s
                """,
                (platform,),
            )

            result = cursor.fetchone()

            if result:
                config = {
                    "platform": result["platform"],
                    "max_per_run": result["max_per_run"],
                    "max_per_day": result["max_per_day"],
                    "max_concurrent_sessions": result["max_concurrent_sessions"],
                }
            else:
                config = {
                    "platform": platform,
                    "max_per_run": 50,
                    "max_per_day": 100,
                    "max_concurrent_sessions": 1,
                }

            logger.info(f"Platform config retrieved: {platform}")

            return json.dumps(config)

        except Exception as e:
            logger.error(f"Failed to get platform config: {e}")
            return json.dumps(
                {"error": "get_platform_config_failed", "detail": str(e)}
            )
        finally:
            if conn:
                conn.close()

    try:
        return _execute()
    except Exception as e:
        return json.dumps({"error": "get_platform_config_failed", "detail": str(e)})


@tool
@operation
def get_run_stats(run_batch_id: str) -> str:
    """
    Retrieve run batch statistics including error count.

    Args:
        run_batch_id: UUID of the run batch.

    Returns:
        JSON string with run batch stats and error count.
    """

    @_with_retry(max_retries=3)
    def _execute() -> str:
        conn = None
        try:
            conn = _get_conn()
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

            cursor.execute(
                """
                SELECT id, run_date, run_index_in_week, jobs_discovered,
                       jobs_auto_applied, jobs_queued, started_at, closed_at
                FROM run_batches
                WHERE id = %s
                """,
                (run_batch_id,),
            )

            batch_result = cursor.fetchone()

            if not batch_result:
                return json.dumps(
                    {"error": "run_batch_not_found", "detail": f"No run batch with id {run_batch_id}"}
                )

            cursor.execute(
                """
                SELECT COUNT(*) AS error_count
                FROM logs_events
                WHERE run_batch_id = %s AND level = 'ERROR'
                """,
                (run_batch_id,),
            )

            error_result = cursor.fetchone()

            stats = {
                "id": str(batch_result["id"]),
                "run_date": str(batch_result["run_date"]),
                "run_index_in_week": batch_result["run_index_in_week"],
                "jobs_discovered": batch_result["jobs_discovered"],
                "jobs_auto_applied": batch_result["jobs_auto_applied"],
                "jobs_queued": batch_result["jobs_queued"],
                "started_at": str(batch_result["started_at"]),
                "closed_at": str(batch_result["closed_at"]) if batch_result["closed_at"] else None,
                "error_count": error_result["error_count"],
            }

            logger.info(f"Run stats retrieved: {run_batch_id}")

            return json.dumps(stats)

        except Exception as e:
            logger.error(f"Failed to get run stats: {e}")
            return json.dumps({"error": "get_run_stats_failed", "detail": str(e)})
        finally:
            if conn:
                conn.close()

    try:
        return _execute()
    except Exception as e:
        return json.dumps({"error": "get_run_stats_failed", "detail": str(e)})
