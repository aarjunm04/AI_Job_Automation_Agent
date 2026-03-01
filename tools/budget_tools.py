"""
Budget enforcement tools for AI Job Application Agent.

This module enforces the $10/month hard cap and the $0.38/run xAI cap.
The Master Agent calls these tools before and after every Apply Agent invocation
to ensure budget compliance.

Module-level state tracks costs per run and is reset at the start of each run.
"""

import os
import json
import logging
import time

import psycopg2
import psycopg2.extras
from crewai.tools import tool
import agentops
from agentops.sdk.decorators import agent, operation

# Module-level logger
logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("LOG_LEVEL", "INFO"))

# Budget caps from environment
XAI_COST_CAP_PER_RUN = float(os.getenv("XAI_COST_CAP_PER_RUN", "0.38"))
TOTAL_MONTHLY_BUDGET = float(os.getenv("TOTAL_MONTHLY_BUDGET", "10.00"))

# Database URL selection
DB_URL = (
    os.getenv("LOCAL_POSTGRES_URL")
    if os.getenv("ACTIVE_DB", "local") == "local"
    else os.getenv("SUPABASE_URL")
)

# Module-level state (reset per run)
_run_xai_cost: float = 0.0
_run_perplexity_cost: float = 0.0
_run_batch_id: str = ""

__all__ = [
    "reset_run_cost_tracker",
    "record_llm_cost",
    "check_xai_run_cap",
    "check_monthly_budget",
    "get_cost_summary",
]


def _log_to_db(
    run_batch_id: str, level: str, event_type: str, message: str
) -> None:
    """
    Internal helper to log budget events to Postgres.

    Args:
        run_batch_id: UUID of the run batch.
        level: Log level.
        event_type: Event type.
        message: Log message.
    """
    if not DB_URL:
        logger.warning("Cannot log to DB: DB_URL not configured")
        return

    conn = None
    try:
        conn = psycopg2.connect(DB_URL)
        conn.autocommit = False
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO logs_events (run_batch_id, level, event_type, message)
            VALUES (%s, %s, %s, %s)
            """,
            (run_batch_id, level, event_type, message),
        )

        conn.commit()
    except Exception as e:
        if conn:
            conn.rollback()
        logger.warning(f"Failed to log budget event to DB: {e}")
    finally:
        if conn:
            conn.close()


@tool
@operation
def reset_run_cost_tracker(run_batch_id: str) -> str:
    """
    Reset the run cost tracker for a new run.

    Called by Master Agent at the start of every run before any LLM calls.

    Args:
        run_batch_id: UUID of the run batch.

    Returns:
        JSON string confirming reset with run_batch_id.
    """
    global _run_xai_cost, _run_perplexity_cost, _run_batch_id

    _run_xai_cost = 0.0
    _run_perplexity_cost = 0.0
    _run_batch_id = run_batch_id

    logger.info(f"Cost tracker reset for run batch: {run_batch_id}")

    return json.dumps({"reset": True, "run_batch_id": run_batch_id})


@tool
@operation
def record_llm_cost(
    provider: str, cost_usd: float, agent_type: str, run_batch_id: str
) -> str:
    """
    Record an LLM API cost and update the run tracker.

    Args:
        provider: LLM provider ('xai' or 'perplexity').
        cost_usd: Cost in USD.
        agent_type: Agent that made the call.
        run_batch_id: UUID of the run batch.

    Returns:
        JSON string with cost details and running totals.
    """
    global _run_xai_cost, _run_perplexity_cost

    try:
        if provider.lower() == "xai":
            _run_xai_cost += cost_usd
        elif provider.lower() == "perplexity":
            _run_perplexity_cost += cost_usd
        else:
            logger.warning(f"Unknown provider '{provider}' - cost not tracked")

        message = (
            f"{provider} +${cost_usd:.4f} | agent={agent_type} | "
            f"run_xai_total=${_run_xai_cost:.4f}"
        )

        _log_to_db(run_batch_id, "INFO", "llm_cost_recorded", message)

        logger.info(message)

        return json.dumps(
            {
                "provider": provider,
                "cost_added": cost_usd,
                "run_xai_total": _run_xai_cost,
                "run_perplexity_total": _run_perplexity_cost,
            }
        )

    except Exception as e:
        logger.error(f"Failed to record LLM cost: {e}")
        return json.dumps({"error": "record_llm_cost_failed", "detail": str(e)})


@tool
@operation
def check_xai_run_cap(run_batch_id: str) -> str:
    """
    Check if the xAI run cap has been exceeded.

    Args:
        run_batch_id: UUID of the run batch.

    Returns:
        JSON string with abort flag and cost details.
    """
    global _run_xai_cost

    try:
        if _run_xai_cost >= XAI_COST_CAP_PER_RUN:
            message = (
                f"xAI run cap ${XAI_COST_CAP_PER_RUN} reached at ${_run_xai_cost:.4f} — "
                f"aborting apply phase"
            )

            _log_to_db(run_batch_id, "CRITICAL", "budget_cap_hit", message)

            logger.critical(message)

            return json.dumps(
                {
                    "abort": True,
                    "reason": "xai_run_cap_exceeded",
                    "spent": _run_xai_cost,
                    "cap": XAI_COST_CAP_PER_RUN,
                }
            )

        remaining = XAI_COST_CAP_PER_RUN - _run_xai_cost

        logger.info(
            f"xAI run cap check: ${_run_xai_cost:.4f} / ${XAI_COST_CAP_PER_RUN} "
            f"(${remaining:.4f} remaining)"
        )

        return json.dumps(
            {
                "abort": False,
                "spent": _run_xai_cost,
                "cap": XAI_COST_CAP_PER_RUN,
                "remaining": remaining,
            }
        )

    except Exception as e:
        logger.error(f"Failed to check xAI run cap: {e}")
        return json.dumps({"error": "check_xai_run_cap_failed", "detail": str(e)})


@tool
@operation
def check_monthly_budget(run_batch_id: str) -> str:
    """
    Check if the monthly budget has been exceeded.

    Queries Postgres to calculate estimated monthly spend based on
    jobs_auto_applied * $0.0025 per application.

    Args:
        run_batch_id: UUID of the run batch.

    Returns:
        JSON string with abort flag and budget details.
    """
    if not DB_URL:
        logger.error("Cannot check monthly budget: DB_URL not configured")
        return json.dumps(
            {"error": "db_not_configured", "detail": "DB_URL not set"}
        )

    conn = None
    try:
        conn = psycopg2.connect(DB_URL)
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        cursor.execute(
            """
            SELECT COALESCE(SUM(jobs_auto_applied * 0.0025), 0) AS estimated_monthly_spend
            FROM run_batches
            WHERE run_date >= DATE_TRUNC('month', CURRENT_DATE)
            """
        )

        result = cursor.fetchone()
        estimated_monthly_spend = float(result["estimated_monthly_spend"])

        if estimated_monthly_spend >= TOTAL_MONTHLY_BUDGET:
            message = (
                f"Monthly budget cap ${TOTAL_MONTHLY_BUDGET} exceeded at "
                f"${estimated_monthly_spend:.2f} — aborting run"
            )

            _log_to_db(run_batch_id, "CRITICAL", "monthly_budget_cap_hit", message)

            logger.critical(message)

            return json.dumps(
                {
                    "abort": True,
                    "reason": "monthly_budget_exceeded",
                    "estimated_spend": estimated_monthly_spend,
                    "cap": TOTAL_MONTHLY_BUDGET,
                }
            )

        remaining = TOTAL_MONTHLY_BUDGET - estimated_monthly_spend

        logger.info(
            f"Monthly budget check: ${estimated_monthly_spend:.2f} / "
            f"${TOTAL_MONTHLY_BUDGET} (${remaining:.2f} remaining)"
        )

        return json.dumps(
            {
                "abort": False,
                "estimated_spend": estimated_monthly_spend,
                "cap": TOTAL_MONTHLY_BUDGET,
                "remaining": remaining,
            }
        )

    except Exception as e:
        logger.error(f"Failed to check monthly budget: {e}")
        return json.dumps(
            {"error": "check_monthly_budget_failed", "detail": str(e)}
        )
    finally:
        if conn:
            conn.close()


@tool
@operation
def get_cost_summary(run_batch_id: str) -> str:
    """
    Get current run cost summary.

    Args:
        run_batch_id: UUID of the run batch.

    Returns:
        JSON string with all cost details for the current run.
    """
    global _run_xai_cost, _run_perplexity_cost

    try:
        run_total_cost = _run_xai_cost + _run_perplexity_cost
        xai_cap_remaining = max(0.0, XAI_COST_CAP_PER_RUN - _run_xai_cost)

        summary = {
            "run_batch_id": run_batch_id,
            "run_xai_cost": _run_xai_cost,
            "run_perplexity_cost": _run_perplexity_cost,
            "run_total_cost": run_total_cost,
            "xai_cap": XAI_COST_CAP_PER_RUN,
            "xai_cap_remaining": xai_cap_remaining,
            "monthly_budget": TOTAL_MONTHLY_BUDGET,
        }

        logger.info(
            f"Cost summary: xAI=${_run_xai_cost:.4f}, Perplexity=${_run_perplexity_cost:.4f}, "
            f"Total=${run_total_cost:.4f}"
        )

        return json.dumps(summary)

    except Exception as e:
        logger.error(f"Failed to get cost summary: {e}")
        return json.dumps({"error": "get_cost_summary_failed", "detail": str(e)})
