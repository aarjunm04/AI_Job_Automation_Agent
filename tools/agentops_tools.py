"""
AgentOps observability tools for AI Job Application Agent.

Provides CrewAI tool functions for recording agent errors and fallback events
to both AgentOps tracing and the Postgres logs_events table.  Every agent that
uses xAI or paid LLM providers should call these tools to maintain a full
audit trail of failures and fallback activations.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Optional

import psycopg2
import agentops
from agentops.sdk.decorators import agent, operation
from crewai.tools import tool

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("LOG_LEVEL", "INFO"))

# Database URL selection
_DB_URL: Optional[str] = (
    os.getenv("LOCAL_POSTGRES_URL")
    if os.getenv("ACTIVE_DB", "local") == "local"
    else os.getenv("SUPABASE_URL")
)

__all__ = [
    "record_agent_error",
    "record_fallback_event",
]


def _log_to_db(
    run_batch_id: str,
    level: str,
    event_type: str,
    message: str,
    job_post_id: Optional[str] = None,
    application_id: Optional[str] = None,
) -> None:
    """
    Write a log entry to the logs_events table in Postgres.

    Silently swallows exceptions so logging failures never interrupt the
    pipeline.

    Args:
        run_batch_id: UUID of the current run batch.
        level: Severity level — INFO | WARNING | ERROR | CRITICAL.
        event_type: Short snake_case label for the event.
        message: Human-readable event description.
        job_post_id: Optional FK to job_posts.id.
        application_id: Optional FK to applications.id.
    """
    if not _DB_URL:
        logger.warning("agentops_tools._log_to_db: DB_URL not configured — skipping DB log")
        return

    conn = None
    try:
        conn = psycopg2.connect(_DB_URL)
        conn.autocommit = False
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO logs_events
                (run_batch_id, level, event_type, message, job_post_id, application_id)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (
                run_batch_id,
                level,
                event_type,
                message,
                job_post_id or None,
                application_id or None,
            ),
        )
        conn.commit()
    except Exception as exc:  # noqa: BLE001
        if conn:
            try:
                conn.rollback()
            except Exception:  # noqa: BLE001
                pass
        logger.warning("agentops_tools._log_to_db: failed to write to Postgres — %s", exc)
    finally:
        if conn:
            try:
                conn.close()
            except Exception:  # noqa: BLE001
                pass


@tool
@operation
def record_agent_error(
    agent_type: str,
    error_message: str,
    run_batch_id: str,
    error_code: str = "",
    job_post_id: str = "",
) -> str:
    """
    Record an agent error to AgentOps and Postgres audit log.

    Call this whenever an agent catches an unexpected exception that it cannot
    recover from.  The error is emitted as an AgentOps ActionEvent for tracing
    and also written to logs_events for permanent audit storage.

    Args:
        agent_type: Class name of the agent that raised the error
            (e.g. ``"AnalyserAgent"``).
        error_message: Full exception message or summary.
        run_batch_id: UUID of the current run batch.
        error_code: Short machine-readable error code (optional).
        job_post_id: UUID of the job post being processed when the error
            occurred (optional).

    Returns:
        JSON string ``{"recorded": bool, "agent_type": str, "error_code": str}``.
    """
    code = error_code.strip() if error_code else "AGENT_ERROR"
    full_message = (
        f"[{agent_type}] {code}: {error_message}"
        if code != "AGENT_ERROR"
        else f"[{agent_type}] {error_message}"
    )

    logger.error("record_agent_error — %s", full_message)

    # Emit to AgentOps as an ActionEvent
    try:
        agentops.record(
            agentops.ActionEvent(
                action_type="agent_error",
                params={
                    "agent_type": agent_type,
                    "error_code": code,
                    "error_message": error_message,
                    "run_batch_id": run_batch_id,
                    "job_post_id": job_post_id or None,
                },
                returns={"status": "error"},
            )
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "record_agent_error: AgentOps ActionEvent failed (non-critical) — %s", exc
        )

    # Persist to Postgres
    _log_to_db(
        run_batch_id=run_batch_id,
        level="ERROR",
        event_type="agent_error",
        message=full_message,
        job_post_id=job_post_id or None,
    )

    return json.dumps(
        {
            "recorded": True,
            "agent_type": agent_type,
            "error_code": code,
        }
    )


@tool
@operation
def record_fallback_event(
    agent_type: str,
    from_provider: str,
    to_provider: str,
    run_batch_id: str,
    fallback_level: int = 1,
    reason: str = "",
) -> str:
    """
    Record an LLM provider fallback activation to AgentOps and Postgres.

    Call this immediately after switching from a failed LLM provider to a
    fallback provider so the event is captured in AgentOps traces and the
    Postgres audit log.

    Args:
        agent_type: Class name of the agent performing the fallback.
        from_provider: Name of the provider that failed
            (e.g. ``"xai/grok-4-fast-reasoning"``).
        to_provider: Name of the fallback provider being activated.
        run_batch_id: UUID of the current run batch.
        fallback_level: 1 for fallback_1, 2 for fallback_2 (default: 1).
        reason: Optional human-readable reason for the fallback.

    Returns:
        JSON string ``{"recorded": bool, "fallback_level": int,
        "from_provider": str, "to_provider": str}``.
    """
    message = (
        f"[{agent_type}] LLM fallback level {fallback_level}: "
        f"{from_provider} → {to_provider}"
    )
    if reason:
        message += f" | reason: {reason}"

    logger.warning("record_fallback_event — %s", message)

    # Emit to AgentOps
    try:
        agentops.record(
            agentops.ActionEvent(
                action_type="llm_fallback",
                params={
                    "agent_type": agent_type,
                    "from_provider": from_provider,
                    "to_provider": to_provider,
                    "fallback_level": fallback_level,
                    "reason": reason,
                    "run_batch_id": run_batch_id,
                },
                returns={"status": "fallback_activated"},
            )
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "record_fallback_event: AgentOps ActionEvent failed (non-critical) — %s", exc
        )

    # Persist to Postgres
    _log_to_db(
        run_batch_id=run_batch_id,
        level="WARNING",
        event_type="llm_fallback",
        message=message,
    )

    return json.dumps(
        {
            "recorded": True,
            "fallback_level": fallback_level,
            "from_provider": from_provider,
            "to_provider": to_provider,
        }
    )
