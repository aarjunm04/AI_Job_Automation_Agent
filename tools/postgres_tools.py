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
from typing import Any, Callable, Optional, Union
from functools import wraps
from datetime import datetime
import math
import pandas as pd

import psycopg2
import psycopg2.extras
import agentops
from psycopg2.extensions import connection as PgConnection
from utils.db_utils import get_db_conn
from crewai.tools import tool
from agentops import operation

logger = logging.getLogger(__name__)

 
 
__all__ = [
    "upsert_job_post",
    "save_job_score",
    "create_application",
    "update_application_status",
    "create_run_batch",
    "update_run_batch_stats",
    "log_event",
    "_log_event_fn",
    "get_queued_jobs",
    "get_platform_config",
    "get_run_stats",
    "get_recent_applications",
    "get_pending_manual_queue_db",
    "_upsert_job_post",
    "_log_event",
    "_get_platform_config",
    "_get_run_stats",
]


def _priority_text(score: int) -> str:
    """Map legacy integer priority score to canonical text value."""
    if score <= 3:
        return "high"
    elif score <= 6:
        return "mid"
    else:
        return "low"


def _sanitize_timestamp(value: Any) -> Optional[datetime]:
    """Return None if value is NaN, inf, or not a valid datetime."""
    if pd.isnull(value):  # handles NaT, NaN, None all at once
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, datetime):
        return value
    try:
        # Attempt to convert to a pandas Timestamp and then to a Python datetime
        ts = pd.Timestamp(value)
        if pd.isnull(ts):
            return None
        return ts.to_pydatetime()
    except (ValueError, TypeError):
        logger.warning(f"Could not convert value '{value}' to a valid timestamp, returning None.")
        return None


def _get_conn() -> PgConnection:
    """Get database connection via centralised db_utils."""
    return get_db_conn()


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


def _fetch_user_config() -> dict[str, Any]:
    """Fetch ``user_settings`` and ``platform_settings`` JSONB columns from the users table.

    Executes ``SELECT user_settings, platform_settings FROM users ORDER BY id LIMIT 1``.
    Retries up to 3 times on ``psycopg2.OperationalError`` with exponential back-off
    (``time.sleep(2**attempt)``).  All other exceptions are caught, logged at WARNING, and
    cause an immediate return of ``{}`` — this function never raises.

    Returns:
        Dict with keys ``"user_settings"`` and ``"platform_settings"``, each being a
        Python dict parsed from the corresponding JSONB column.  Returns ``{}`` when the
        table is empty, the DB URL is unset, or all retry attempts are exhausted.
    """
    conn: Optional[PgConnection] = None
    for attempt in range(3):
        try:
            conn = _get_conn()
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cursor.execute(
                "SELECT user_settings, platform_settings FROM users ORDER BY id LIMIT 1"
            )
            row = cursor.fetchone()
            if row is None:
                logger.warning("_fetch_user_config: users table is empty — returning {}")
                return {}
            return {
                "user_settings": row["user_settings"] or {},
                "platform_settings": row["platform_settings"] or {},
            }
        except psycopg2.OperationalError as op_err:
            sleep_time: int = 2 ** attempt
            logger.warning(
                "_fetch_user_config: OperationalError attempt %d/3: %s — retrying in %ds",
                attempt + 1,
                op_err,
                sleep_time,
            )
            time.sleep(sleep_time)
        except Exception as exc:
            logger.warning("_fetch_user_config: unexpected DB error: %s", exc)
            return {}
        finally:
            if conn:
                try:
                    conn.close()
                except Exception:  # noqa: BLE001
                    pass
            conn = None
    logger.warning("_fetch_user_config: all 3 retries exhausted — returning empty dict")
    return {}


@tool
def upsert_job_post(
    pipeline_run_id: str,
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
        pipeline_run_id: UUID of the run batch.
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

            posted_at_value = _sanitize_timestamp(posted_at)
            for attempt in range(3):
                try:
                    cursor.execute(
                        """
                        INSERT INTO jobs (pipeline_run_id, source_platform, title, company, url, location, posted_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (url) DO UPDATE SET
                            title = EXCLUDED.title,
                            company = EXCLUDED.company,
                            location = EXCLUDED.location,
                            source_platform = EXCLUDED.source_platform,
                            pipeline_run_id = EXCLUDED.pipeline_run_id
                        RETURNING id, (xmax = 0) AS inserted
                        """,
                        (
                            pipeline_run_id,
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
                        f"Successfully {action} job post: {result['id']} for {url}"
                    )
                    return json.dumps(
                        {"job_post_id": result["id"], "action": action}
                    )
                except psycopg2.Error as e:
                    logger.error(f"Failed to upsert job post: {e}")
                    conn.rollback()
                    if attempt < 2:
                        time.sleep(2**attempt)
                    else:
                        raise
        finally:
            if conn:
                conn.close()

    return _execute()


@tool
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

            # Inline patch for user_id type mismatch
            try:
                cursor.execute("ALTER TABLE applications DROP CONSTRAINT IF EXISTS applications_user_id_fkey")
            except Exception:
                pass
            cursor.execute("ALTER TABLE applications ALTER COLUMN user_id TYPE TEXT")

            cursor.execute(
                """
                INSERT INTO applications (job_post_id, resume_id, user_id, mode, status, platform, error_code, notion_synced, notion_synced_at, proof_json)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (
                    job_post_id,
                    resume_id,
                    user_id,
                    mode,
                    status,
                    platform,
                    error_code_value,
                    False,
                    None,
                    None,
                ),
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
                    (application_id, job_post_id, _priority_text(int(priority))),
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
def create_run_batch(run_index_in_week: int) -> str:
    """
    Create a new run batch for the current date.

    Args:
        run_index_in_week: Index of this run in the week (1, 2, or 3).

    Returns:
        JSON string with pipeline_run_id and run_date.
    """

    @_with_retry(max_retries=3)
    def _execute() -> str:
        conn = None
        try:
            conn = _get_conn()
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

            cursor.execute(
                """
                INSERT INTO pipeline_runs (run_index_in_week)
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

            # FIX 4: pipeline_runs table never written
            try:
                cursor.execute(
                    """
                    INSERT INTO pipeline_runs (pipeline_run_id, mode, status, dry_run, started_at, updated_at)
                    VALUES (%s, %s, %s, %s, NOW(), NOW())
                    ON CONFLICT DO NOTHING
                    """,
                    (str(result["id"]), "full", "running", False),
                )
                conn.commit()
            except Exception as rb_exc:
                logger.warning("create_run_batch: pipeline_runs insert failed (non-critical): %s", rb_exc)

            return json.dumps(
                {
                    "pipeline_run_id": str(result["id"]),
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
def update_run_batch_stats(
    pipeline_run_id: str, jobs_found: int, jobs_applied: int, jobs_queued: int
) -> str:
    """
    Update run batch statistics and close the batch.

    Args:
        pipeline_run_id: UUID of the run batch.
        jobs_found: Total jobs discovered.
        jobs_applied: Jobs automatically applied to.
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
                UPDATE pipeline_runs
                SET jobs_found = %s,
                    jobs_applied = %s,
                    jobs_queued = %s,
                    completed_at = NOW()
                WHERE id = %s
                RETURNING id
                """,
                (jobs_found, jobs_applied, jobs_queued, pipeline_run_id),
            )

            result = cursor.fetchone()
            conn.commit()

            updated = result is not None
            logger.info(
                f"Run batch stats updated: id={pipeline_run_id}, discovered={jobs_found}, "
                f"applied={jobs_applied}, queued={jobs_queued}"
            )

            # FIX 4: pipeline_runs table never written
            try:
                cursor.execute(
                    """
                    UPDATE pipeline_runs
                    SET jobs_found = %s,
                        jobs_applied = %s,
                        jobs_queued = %s,
                        status = 'completed',
                        completed_at = NOW(),
                        updated_at = NOW()
                    WHERE pipeline_run_id = %s
                    """,
                    (jobs_found, jobs_applied, jobs_queued, pipeline_run_id),
                )
                conn.commit()
            except Exception as rb_exc:
                logger.warning("update_run_batch_stats: pipeline_runs update failed (non-critical): %s", rb_exc)

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


@tool("log_event")
def log_event(
    pipeline_run_id: str,
    event_type: str,
    level: str,
    agent: str,
    message: str,
    job_post_id: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> str:
    """CrewAI tool wrapper for log_event."""
    if job_post_id:
        md = dict(metadata or {})
        md.setdefault("job_post_id", job_post_id)
        metadata = md
    _log_event_fn(
        pipeline_run_id=pipeline_run_id,
        event_type=event_type,
        level=level,
        agent=agent,
        message=message,
        metadata=metadata,
    )
    return "logged"

def _log_event_fn(
    pipeline_run_id: str,
    event_type: str,
    level: str,
    agent: str,
    message: str,
    metadata: dict | None = None,
) -> None:
    """Log an event to the audit_logs table.

    Fail-soft: never raises.
    """
    conn = None
    try:
        conn = _get_conn()
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        metadata_value = json.dumps(metadata) if metadata else None

        cursor.execute(
            """
            INSERT INTO audit_logs (pipeline_run_id, event_type, level, agent, message, metadata)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (pipeline_run_id, event_type, level, agent, message, metadata_value),
        )
        conn.commit()
    except Exception as e:
        if conn:
            try:
                conn.rollback()
            except Exception:  # noqa: BLE001
                pass
        logger.warning(f"Failed to log event (non-critical): {e}")
    finally:
        if conn:
            try:
                conn.close()
            except Exception:  # noqa: BLE001
                pass


@tool
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
                JOIN jobs jp ON qj.job_post_id = jp.id
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
def get_platform_config(platform: str) -> str:
    """Retrieve per-platform rate-limit config from ``users.platform_settings`` JSONB column.

    Reads ``platform_settings->platform_limits->{platform}`` from the first row of the
    ``users`` table.  The lookup is case-insensitive (lowercases the ``platform`` arg).
    Retry logic (max 3 × exponential back-off on ``OperationalError``) is handled by
    the shared ``_fetch_user_config()`` helper.  Fail-soft: returns a safe default dict
    on any DB error or when the platform key is absent.

    Args:
        platform: Platform name (case-insensitive match against ``platform_limits`` keys).

    Returns:
        JSON string with the matching ``platform_limits`` entry, or
        ``{"max_per_run": 20, "rate_limit_per_request_seconds": 3}`` when the platform
        is not found or any DB failure occurs.
    """
    _DEFAULT: dict[str, Any] = {"max_per_run": 20, "rate_limit_per_request_seconds": 3}
    try:
        cfg: dict[str, Any] = _fetch_user_config()
        platform_limits: dict[str, Any] = (
            cfg.get("platform_settings", {}).get("platform_limits", {})
        )
        platform_cfg: Optional[dict[str, Any]] = platform_limits.get(platform.lower())
        if platform_cfg is None:
            logger.info(
                "get_platform_config: platform '%s' not in platform_limits — using defaults",
                platform,
            )
            return json.dumps(_DEFAULT)
        logger.info("get_platform_config: config retrieved for platform '%s'", platform)
        return json.dumps(platform_cfg)
    except Exception as exc:
        logger.warning(
            "get_platform_config: error fetching config for '%s': %s — using defaults",
            platform,
            exc,
        )
        return json.dumps(_DEFAULT)


@tool
def get_run_stats(pipeline_run_id: str) -> str:
    """
    Retrieve run batch statistics including error count.

    Args:
        pipeline_run_id: UUID of the run batch.

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
                SELECT id, run_date, run_index_in_week, jobs_found,
                       jobs_applied, jobs_queued, started_at, completed_at
                FROM pipeline_runs
                WHERE id = %s
                """,
                (pipeline_run_id,),
            )

            batch_result = cursor.fetchone()

            if not batch_result:
                return json.dumps(
                    {"error": "run_batch_not_found", "detail": f"No run batch with id {pipeline_run_id}"}
                )

            cursor.execute(
                """
                SELECT COUNT(*) AS error_count
                FROM audit_logs
                WHERE pipeline_run_id = %s AND level = 'ERROR'
                """,
                (pipeline_run_id,),
            )

            error_result = cursor.fetchone()

            stats = {
                "id": str(batch_result["id"]),
                "run_date": str(batch_result["run_date"]),
                "run_index_in_week": batch_result["run_index_in_week"],
                "jobs_found": batch_result["jobs_found"],
                "jobs_applied": batch_result["jobs_applied"],
                "jobs_queued": batch_result["jobs_queued"],
                "started_at": str(batch_result["started_at"]),
                "completed_at": str(batch_result["completed_at"]) if batch_result["completed_at"] else None,
                "error_count": error_result["error_count"],
            }

            logger.info(f"Run stats retrieved: {pipeline_run_id}")

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

@tool
def get_recent_applications(limit: int = 20) -> str:
    """
    Retrieve most recent application records.
    """
    @_with_retry(max_retries=3)
    def _execute() -> str:
        conn = None
        try:
            conn = _get_conn()
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cursor.execute("SELECT * FROM applications ORDER BY submitted_at DESC LIMIT %s", (limit,))
            return json.dumps([dict(row) for row in cursor.fetchall()], default=str)
        except Exception as e:
            logger.error(f"Failed to get recent applications: {e}")
            return json.dumps({"error": "get_recent_applications_failed", "detail": str(e)})
        finally:
            if conn:
                conn.close()
    try:
        return _execute()
    except Exception as e:
        return json.dumps({"error": "get_recent_applications_failed", "detail": str(e)})


@tool
def get_pending_manual_queue_db(pipeline_run_id: str) -> str:
    """Fetch all manual-queue applications from Postgres for a given run batch.

    Queries ``applications JOIN jobs`` where ``applications.status = 'manual_queued'``
    and ``jobs.pipeline_run_id`` matches. Returns full job metadata ready for Notion sync.

    Args:
        pipeline_run_id: UUID of the run batch to query.

    Returns:
        JSON string with array of application + job objects ordered by fit_score DESC.
    """
    try:
        import psycopg2.extras
        from utils.db_utils import get_db_conn
        conn = get_db_conn()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(
            """
	            SELECT
	                a.id              AS application_id,
	                a.status,
	                a.applied_at,
	                j.id              AS job_id,
	                j.title,
                j.company,
                j.job_url,
                j.source_platform AS platform,
                j.location,
                j.job_type,
                j.salary_range,
                js.fit_score,
                js.best_resume_id
            FROM applications a
            JOIN jobs j           ON j.id = a.job_post_id
            LEFT JOIN job_scores js ON js.job_post_id = j.id
            WHERE a.status = 'manual_queued'
              AND j.pipeline_run_id = %s
            ORDER BY js.fit_score DESC NULLS LAST
            """,
            (pipeline_run_id,),
        )
        rows = [dict(r) for r in cur.fetchall()]
        conn.close()
        logger.info(
            "get_pending_manual_queue_db: found %d rows for batch %s",
            len(rows), pipeline_run_id,
        )
        return json.dumps(rows, default=str)
    except Exception as exc:
        logger.error("get_pending_manual_queue_db: failed: %s", exc)
        return json.dumps([])



# ---------------------------------------------------------------------------
# Private aliases for direct programmatic calls (bypass CrewAI Tool wrapper)
# ---------------------------------------------------------------------------
# ═══════════════════════════════════════════════════════════════════════════════
# .func ALIASES — raw function access (bypasses CrewAI Tool Pydantic wrapper)
# ═══════════════════════════════════════════════════════════════════════════════
_upsert_job_post            = upsert_job_post.func            if hasattr(upsert_job_post,            'func') else upsert_job_post
_save_job_score             = save_job_score.func             if hasattr(save_job_score,             'func') else save_job_score
_create_application         = create_application.func         if hasattr(create_application,         'func') else create_application
_update_application_status  = update_application_status.func  if hasattr(update_application_status,  'func') else update_application_status
_create_run_batch           = create_run_batch.func           if hasattr(create_run_batch,           'func') else create_run_batch
_update_run_batch_stats     = update_run_batch_stats.func     if hasattr(update_run_batch_stats,     'func') else update_run_batch_stats
_log_event                  = log_event.func                  if hasattr(log_event,                  'func') else log_event
_get_queued_jobs            = get_queued_jobs.func            if hasattr(get_queued_jobs,            'func') else get_queued_jobs
_get_platform_config        = get_platform_config.func        if hasattr(get_platform_config,        'func') else get_platform_config
_get_run_stats              = get_run_stats.func              if hasattr(get_run_stats,              'func') else get_run_stats
_get_recent_applications    = get_recent_applications.func    if hasattr(get_recent_applications,    'func') else get_recent_applications
_get_pending_manual_queue   = get_pending_manual_queue_db.func if hasattr(get_pending_manual_queue_db, 'func') else get_pending_manual_queue_db

# ═══════════════════════════════════════════════════════════════════════════════
# RUNTIME GUARDS — Ensure aliases are callable at module load
# ═══════════════════════════════════════════════════════════════════════════════
for _name, _alias in [
    ("upsert_job_post", _upsert_job_post),
    ("save_job_score", _save_job_score),
    ("create_application", _create_application),
    ("update_application_status", _update_application_status),
    ("create_run_batch", _create_run_batch),
    ("update_run_batch_stats", _update_run_batch_stats),
    ("log_event", _log_event),
    ("get_queued_jobs", _get_queued_jobs),
    ("get_platform_config", _get_platform_config),
    ("get_run_stats", _get_run_stats),
    ("get_recent_applications", _get_recent_applications),
    ("get_pending_manual_queue_db", _get_pending_manual_queue),
]:
    assert callable(_alias), f"CRITICAL: Tool alias {_name} is not callable. Check @tool decoration."

