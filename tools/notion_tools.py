from agentops.sdk.decorators import operation
"""
Notion tools for AI Job Application Agent.

This module wraps the NotionClient into CrewAI tools consumed by the Tracker Agent.
All Notion sync operations are non-blocking - failures never stop the pipeline.
"""

import os
import json
import logging
import time
from datetime import datetime, timezone
from typing import Optional

from crewai.tools import tool
import agentops
from agentops.sdk.decorators import agent, operation

from integrations.notion import NotionClient
from tools.postgres_tools import log_event
_log_event = log_event.func if hasattr(log_event, "func") else log_event

# Module-level logger
logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("LOG_LEVEL", "INFO"))

def _retry_call(fn, *args, max_retries: int = 3, **kwargs):
    """Execute fn with exponential backoff retry."""
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            last_exc = exc
            wait = 2.0 ** attempt
            logger.warning(
                "Attempt %d/%d failed: %s — retrying in %.1fs",
                attempt + 1, max_retries, exc, wait,
            )
            time.sleep(wait)
    raise RuntimeError(
        f"All {max_retries} attempts failed. Last error: {last_exc}"
    )

# Module-level lazy-initialized client
_notion_client: Optional[NotionClient] = None

__all__ = [
    "sync_application_to_job_tracker",
    "queue_job_to_applications_db",
    "update_notion_page_status",
    "get_pending_manual_queue",
    "check_notion_connection",
]


def _get_client() -> NotionClient:
    """
    Get or initialize the NotionClient singleton.

    Returns:
        NotionClient: Initialized Notion client instance.
    """
    global _notion_client
    if _notion_client is None:
        _notion_client = NotionClient()
        logger.info("NotionClient initialized")
    return _notion_client


@tool
def sync_application_to_job_tracker(
    application_id: str,
    job_post_id: str,
    run_batch_id: str,
    title: str,
    company: str,
    job_url: str,
    platform: str,
    resume_used: str,
    ctc: str = "",
    notes: str = "",
    location: str = "",
    job_type: str = "full-time",
) -> str:
    """
    Sync an applied job to the Notion Job Tracker database.

    Creates a new page in the Job Tracker DB with stage "Applied" and logs
    the sync event. This is called after successful job applications.

    Args:
        application_id: UUID of the application record.
        job_post_id: UUID of the job post.
        run_batch_id: UUID of the run batch.
        title: Job title.
        company: Company name.
        job_url: Job posting URL.
        platform: Job platform name.
        resume_used: Resume variant used for application.
        ctc: Compensation/salary information (optional).
        notes: Additional notes (optional).
        location: Job location (optional).
        job_type: Job type (default: "full-time").

    Returns:
        JSON string with sync result and Notion page ID.
    """
    try:
        client = _get_client()

        # Get today's date in ISO format
        date_applied = datetime.now(timezone.utc).date().isoformat()

        # Determine applied_via based on context (default to Auto for now)
        # TODO: Pass mode parameter to distinguish Auto vs Manual
        applied_via = "Auto"

        # Create page in Job Tracker DB
        try:
            response = None
            for attempt in range(3):
                try:
                    response = client.create_job_tracker_page(
                        title=title,
                        company=company,
                        job_url=job_url,
                        stage="Applied",
                        date_applied=date_applied,
                        platform=platform,
                        applied_via=applied_via,
                        ctc=ctc,
                        notes=notes,
                        job_type=job_type,
                        location=location,
                        resume_used=resume_used,
                    )
                    break
                except Exception as e:
                    if attempt == 2:
                        logger.error("Notion create_job_tracker_page failed: %s", e)
                        raise
                    time.sleep(2 ** attempt)
            if response is None:
                return None
        except Exception as e:
            logger.warning("Notion sync failed after retries: %s", e)
            return None

        notion_page_id = response.get("id", "")

        # Log success event
        _log_event(
            run_batch_id=run_batch_id,
            level="INFO",
            event_type="notion_synced",
            message=f"Job Tracker page created for {company} — {title}",
            application_id=application_id,
            job_post_id=job_post_id,
        )

        logger.info(
            f"Successfully synced application to Job Tracker: {company} - {title}"
        )

        return json.dumps(
            {"notion_page_id": notion_page_id, "synced": True, "db": "job_tracker"}
        )

    except Exception as e:
        logger.error(f"Failed to sync application to Job Tracker: {e}")

        # Log error event
        _log_event(
            run_batch_id=run_batch_id,
            level="ERROR",
            event_type="notion_sync_failed",
            message=f"Job Tracker sync failed for {company} — {title}: {str(e)}",
            application_id=application_id,
            job_post_id=job_post_id,
        )

        # Return error but don't fail the pipeline
        return json.dumps({"synced": False, "error": str(e), "db": "job_tracker"})


@tool
def queue_job_to_applications_db(
    job_post_id: str,
    run_batch_id: str,
    title: str,
    company: str,
    job_url: str,
    platform: str,
    fit_score: float,
    resume_suggested: str,
    ctc: str = "",
    notes: str = "",
    location: str = "",
    job_type: str = "full-time",
    deadline: str = "",
) -> str:
    """
    Queue a job to the Notion Applications database for manual review.

    Creates a new page in the Applications DB with status "Queued" and priority
    derived from the fit score. This is called for jobs routed to manual queue.

    Args:
        job_post_id: UUID of the job post.
        run_batch_id: UUID of the run batch.
        title: Job title.
        company: Company name.
        job_url: Job posting URL.
        platform: Job platform name.
        fit_score: Fit score (0.0 to 1.0).
        resume_suggested: Suggested resume variant.
        ctc: Compensation/salary information (optional).
        notes: Additional notes (optional).
        location: Job location (optional).
        job_type: Job type (default: "full-time").
        deadline: Application deadline in ISO format (optional).

    Returns:
        JSON string with queue result and Notion page ID.
    """
    try:
        client = _get_client()

        # Derive priority from fit_score
        if fit_score >= 0.75:
            priority = "High"
        elif fit_score >= 0.50:
            priority = "Medium"
        else:
            priority = "Low"

        # Create page in Applications DB
        try:
            response = None
            for attempt in range(3):
                try:
                    response = client.create_applications_page(
                        title=title,
                        company=company,
                        job_url=job_url,
                        deadline=deadline,
                        platform=platform,
                        status="Queued",
                        ctc=ctc,
                        priority=priority,
                        fit_score=fit_score,
                        job_type=job_type,
                        location=location,
                        notes=notes,
                        resume_suggested=resume_suggested,
                    )
                    break
                except Exception as e:
                    if attempt == 2:
                        logger.error("Notion create_applications_page failed: %s", e)
                        raise
                    time.sleep(2 ** attempt)
            if response is None:
                return None
        except Exception as e:
            logger.warning("Notion sync failed after retries: %s", e)
            return None

        notion_page_id = response.get("id", "")

        # Log success event
        _log_event(
            run_batch_id=run_batch_id,
            level="INFO",
            event_type="notion_queued",
            message=f"Applications DB page created for {company} — {title}",
            job_post_id=job_post_id,
        )

        logger.info(
            f"Successfully queued job to Applications DB: {company} - {title} (priority: {priority})"
        )

        return json.dumps(
            {
                "notion_page_id": notion_page_id,
                "queued": True,
                "db": "applications",
                "priority": priority,
            }
        )

    except Exception as e:
        logger.error(f"Failed to queue job to Applications DB: {e}")

        # Log error event
        _log_event(
            run_batch_id=run_batch_id,
            level="ERROR",
            event_type="notion_queue_failed",
            message=f"Applications DB queue failed for {company} — {title}: {str(e)}",
            job_post_id=job_post_id,
        )

        # Return error but don't fail the pipeline
        return json.dumps({"queued": False, "error": str(e), "db": "applications"})


@tool
def update_notion_page_status(
    page_id: str, status: str, run_batch_id: str
) -> str:
    """
    Update the Status property of a Notion page.

    Args:
        page_id: Notion page ID.
        status: New status value.
        run_batch_id: UUID of the run batch.

    Returns:
        JSON string with update result.
    """
    try:
        client = _get_client()

        try:
            response = None
            for attempt in range(3):
                try:
                    response = client.update_page_status(page_id=page_id, status=status)
                    break
                except Exception as e:
                    if attempt == 2:
                        logger.error("Notion update_page_status failed: %s", e)
                        raise
                    time.sleep(2 ** attempt)
            if response is None:
                return None
        except Exception as e:
            logger.warning("Notion sync failed after retries: %s", e)
            return None

        # Log success event
        _log_event(
            run_batch_id=run_batch_id,
            level="INFO",
            event_type="notion_status_updated",
            message=f"Notion page {page_id} status updated to: {status}",
        )

        logger.info(f"Successfully updated Notion page {page_id} status to: {status}")

        return json.dumps(
            {"updated": True, "page_id": page_id, "status": status}
        )

    except Exception as e:
        logger.error(f"Failed to update Notion page status: {e}")

        # Log error event
        _log_event(
            run_batch_id=run_batch_id,
            level="ERROR",
            event_type="notion_status_update_failed",
            message=f"Notion page {page_id} status update failed: {str(e)}",
        )

        return json.dumps(
            {"updated": False, "page_id": page_id, "status": status, "error": str(e)}
        )


@tool
def get_pending_manual_queue(run_batch_id: str) -> str:
    """
    Get all pending jobs from the Notion Applications database.

    Queries the Applications DB for pages with status "Queued" and returns
    a list of pending manual jobs.

    Args:
        run_batch_id: UUID of the run batch.

    Returns:
        JSON string with array of pending job objects.
    """
    try:
        client = _get_client()

        # Query Applications DB for queued jobs
        filter_payload = {
            "property": "Status",
            "select": {"equals": "Queued"},
        }

        pages = client.query_database(
            database_id=os.getenv("NOTION_APPLICATIONS_DB_ID", ""),
            filter_payload=filter_payload,
            page_size=100,
        )

        # Extract relevant fields from each page
        pending_jobs = []
        for page in pages:
            properties = page.get("properties", {})

            # Extract title
            title_prop = properties.get("Job Title", {})
            title_content = title_prop.get("title", [])
            title = title_content[0].get("text", {}).get("content", "") if title_content else ""

            # Extract company
            company_prop = properties.get("Company", {})
            company_content = company_prop.get("rich_text", [])
            company = company_content[0].get("text", {}).get("content", "") if company_content else ""

            # Extract job URL
            job_url = properties.get("Job URL", {}).get("url", "")

            # Extract priority
            priority_prop = properties.get("Priority", {})
            priority = priority_prop.get("select", {}).get("name", "Medium")

            # Extract fit score
            fit_score = properties.get("Fit Score", {}).get("number", 0.0)

            pending_jobs.append(
                {
                    "page_id": page.get("id", ""),
                    "title": title,
                    "company": company,
                    "job_url": job_url,
                    "priority": priority,
                    "fit_score": fit_score,
                }
            )

        logger.info(f"Retrieved {len(pending_jobs)} pending jobs from Applications DB")

        return json.dumps(pending_jobs)

    except Exception as e:
        logger.error(f"Failed to get pending manual queue: {e}")

        # Log error event
        _log_event(
            run_batch_id=run_batch_id,
            level="ERROR",
            event_type="notion_query_failed",
            message=f"Applications DB query failed: {str(e)}",
        )

        return json.dumps([])


@tool
def check_notion_connection(run_batch_id: str) -> str:
    """
    Check Notion API connection health.

    Performs a health check on the Notion API connection and logs the result.

    Args:
        run_batch_id: UUID of the run batch.

    Returns:
        JSON string with connection status and bot information.
    """
    try:
        client = _get_client()

        health = client.health_check()

        # Log event based on connection status
        if health["connected"]:
            _log_event(
                run_batch_id=run_batch_id,
                level="INFO",
                event_type="notion_health_check",
                message=f"Notion connection healthy: {health['bot_name']}",
            )
            logger.info(f"Notion health check passed: {health['bot_name']}")
        else:
            _log_event(
                run_batch_id=run_batch_id,
                level="ERROR",
                event_type="notion_health_check_failed",
                message=f"Notion connection failed: {health['error']}",
            )
            logger.error(f"Notion health check failed: {health['error']}")

        return json.dumps(health)

    except Exception as e:
        logger.error(f"Notion health check exception: {e}")

        # Log error event
        _log_event(
            run_batch_id=run_batch_id,
            level="ERROR",
            event_type="notion_health_check_error",
            message=f"Notion health check error: {str(e)}",
        )

        return json.dumps({"connected": False, "bot_name": None, "error": str(e)})


# ═══════════════════════════════════════════════════════════════════════════════
# .func ALIASES — raw function access (bypasses CrewAI Tool Pydantic wrapper)
# ═══════════════════════════════════════════════════════════════════════════════
_sync_application_to_job_tracker = sync_application_to_job_tracker.func if hasattr(sync_application_to_job_tracker, "func") else sync_application_to_job_tracker
_queue_job_to_applications_db    = queue_job_to_applications_db.func    if hasattr(queue_job_to_applications_db,    "func") else queue_job_to_applications_db
_update_notion_page_status       = update_notion_page_status.func       if hasattr(update_notion_page_status,       "func") else update_notion_page_status
_get_pending_manual_queue        = get_pending_manual_queue.func        if hasattr(get_pending_manual_queue,        "func") else get_pending_manual_queue
_check_notion_connection         = check_notion_connection.func         if hasattr(check_notion_connection,         "func") else check_notion_connection
