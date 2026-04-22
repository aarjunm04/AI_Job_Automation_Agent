"""
Budget enforcement tools for AI Job Application Agent.

This module enforces the $10/month hard cap and the $0.38/run xAI cap.
The Master Agent calls these tools before and after every Apply Agent invocation
to ensure budget compliance.

Module-level state tracks costs per run and is reset at the start of each run.
Integrates with LiteLLM via success_callback to capture and record every LLM cost
to the PostgreSQL llm_costs table for accurate budget tracking.
"""

import os
import json
import logging
import time
from typing import Any, Optional

try:
    import psycopg2
    import psycopg2.extras
except ModuleNotFoundError:  # pragma: no cover
    psycopg2 = None  # type: ignore[assignment]

from utils.db_utils import get_db_conn

try:
    import agentops  # type: ignore
    from agentops.sdk.decorators import agent, operation  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    agentops = None  # type: ignore[assignment]
    def agent(func=None, *args, **kwargs):  # type: ignore[override]
        if callable(func):
            return func
        def decorator(f):
            return f
        return decorator
    def operation(func=None, *args, **kwargs):  # type: ignore[override]
        if callable(func):
            return func
        def decorator(f):
            return f
        return decorator

# Module-level logger
logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("LOG_LEVEL", "INFO"))

# Budget caps from environment
XAI_COST_CAP_PER_RUN = float(os.getenv("XAI_COST_CAP_PER_RUN"))
TOTAL_MONTHLY_BUDGET = float(os.getenv("TOTAL_MONTHLY_BUDGET"))

# Module-level state (reset per run)
_run_xai_cost: float = 0.0
_run_perplexity_cost: float = 0.0
_pipeline_run_id: str = ""

__all__ = [
    "init_budget_run",
    "record_llm_cost",
    "register_litellm_callback",
    "check_xai_run_cap",
    "check_monthly_budget",
    "get_cost_summary",
]

def _get_conn() -> Any:
    """Return a live Postgres connection via the central db_utils factory."""
    return get_db_conn()


def _log_to_db(
    pipeline_run_id: str, level: str, event_type: str, message: str
) -> None:
    """
    Internal helper to log budget events to Postgres.

    Args:
        pipeline_run_id: UUID of the run batch.
        level: Log level.
        event_type: Event type.
        message: Log message.
    """
    import uuid
    
    # Sanitize pipeline_run_id for Postgres UUID strictness
    valid_uuid = None
    if pipeline_run_id and pipeline_run_id != "async_call":
        try:
            # Test if it's a valid UUID
            uuid_obj = uuid.UUID(str(pipeline_run_id))
            valid_uuid = str(uuid_obj)
        except ValueError:
            valid_uuid = None
    pipeline_run_id = valid_uuid

    conn = None
    try:
        conn = _get_conn()
        if not conn:
            logger.warning("budget_tools._log_to_db: DB connection failed — skipping DB log")
            return

        conn.autocommit = False
        cursor = conn.cursor()

        if psycopg2 is None:
            cursor.execute(
                """
                INSERT INTO audit_logs (pipeline_run_id, level, event_type, message)
                VALUES (%s, %s, %s, %s)
                """,
                (pipeline_run_id, level, event_type, message),
            )
        else:
            for attempt in range(3):
                try:
                    cursor.execute(
                        """
                        INSERT INTO audit_logs (pipeline_run_id, level, event_type, message)
                        VALUES (%s, %s, %s, %s)
                        """,
                        (pipeline_run_id, level, event_type, message),
                    )
                    break
                except psycopg2.OperationalError as e:
                    if attempt == 2:
                        logger.error("DB execute failed after 3 attempts: %s", e)
                        raise
                    time.sleep(2 ** attempt)

        conn.commit()
    except Exception as e:
        if conn:
            conn.rollback()
        logger.warning(f"Failed to log budget event to DB: {e}")
    finally:
        if conn:
            conn.close()


def init_budget_run(pipeline_run_id: str) -> None:
    """
    Initialize budget tracking for a new pipeline run.

    Must be called at the start of each pipeline run with the active
    pipeline_run_id so all subsequent LLM costs are attributed correctly.

    Args:
        pipeline_run_id: UUID of the active pipeline run from pipeline_runs table.
    """
    global _run_xai_cost, _run_perplexity_cost, _pipeline_run_id
    
    _run_xai_cost = 0.0
    _run_perplexity_cost = 0.0
    _pipeline_run_id = pipeline_run_id
    
    logger.info("Budget tracking initialized for run %s", pipeline_run_id)


def record_llm_cost(
    provider: str,
    model: str,
    cost_usd: float,
    pipeline_run_id: str,
    tokens_used: int = 0,
) -> bool:
    """
    Write a single LLM call cost to the llm_costs table in Postgres.

    Args:
        provider: Provider name, e.g. 'xai', 'cerebras', 'groq'.
        model: Model name used for this call.
        cost_usd: Dollar cost of this completion as float.
        pipeline_run_id: UUID of the active pipeline run.
        tokens_used: Total tokens consumed (prompt + completion).

    Returns:
        True if the write succeeded, False on any error (fail soft).
    """
    global _run_xai_cost, _run_perplexity_cost

    conn = None
    try:
        # Update module-level accumulators
        provider_lower = provider.lower()
        if provider_lower == "xai":
            _run_xai_cost += cost_usd
        elif provider_lower == "perplexity":
            _run_perplexity_cost += cost_usd

        # Ensure llm_costs table exists
        conn = _get_conn()
        if not conn:
            logger.warning("record_llm_cost: DB connection failed")
            return False

        cursor = conn.cursor()

        # Create table if not exists
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS llm_costs (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                cost_usd NUMERIC(10,6) NOT NULL DEFAULT 0,
                pipeline_run_id TEXT,
                tokens_used INTEGER DEFAULT 0,
                recorded_at TIMESTAMPTZ DEFAULT NOW()
            );
            """
        )

        # Insert cost record (with 3 retries on transient errors)
        for attempt in range(3):
            try:
                cursor.execute(
                    """
                    INSERT INTO llm_costs 
                    (provider, model, cost_usd, pipeline_run_id, tokens_used)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (provider, model, cost_usd, pipeline_run_id, tokens_used),
                )
                conn.commit()
                logger.info(
                    "LLM cost recorded: provider=%s, model=%s, cost=$%.6f, "
                    "pipeline_run_id=%s, tokens=%d",
                    provider,
                    model,
                    cost_usd,
                    pipeline_run_id,
                    tokens_used,
                )
                return True
            except psycopg2.OperationalError as e:
                if attempt == 2:
                    logger.warning(
                        "record_llm_cost: DB error after 3 attempts: %s", e
                    )
                    if conn:
                        conn.rollback()
                    return False
                time.sleep(2 ** attempt)
            except Exception as e:
                logger.warning("record_llm_cost: Unexpected error: %s", e)
                if conn:
                    conn.rollback()
                return False

        return False

    except Exception as e:
        logger.warning("record_llm_cost: Exception during write: %s", e)
        if conn:
            try:
                conn.rollback()
            except Exception:
                pass
        return False
    finally:
        if conn:
            try:
                conn.close()
            except Exception:
                pass


def _litellm_cost_callback(
    kwargs: dict,
    completion_response: object,
    start_time: float,
    end_time: float,
) -> None:
    """
    LiteLLM success callback — captures cost after every LLM completion.

    Registered once at pipeline boot via litellm.success_callback.
    Extracts cost using litellm.completion_cost(), writes to DB via
    record_llm_cost(). Fails soft — never raises, never crashes the pipeline.

    Args:
        kwargs: LiteLLM call kwargs including model name.
        completion_response: The full LiteLLM completion response object.
        start_time: Call start timestamp.
        end_time: Call end timestamp.
    """
    try:
        # Extract model from kwargs
        model = kwargs.get("model", "unknown")

        # Infer provider
        model_lower = model.lower()
        api_base = kwargs.get("api_base", "").lower()

        if "grok" in model_lower or "xai" in model_lower:
            provider = "xai"
        elif "llama" in model_lower and "cerebras" in api_base:
            provider = "cerebras"
        elif "llama" in model_lower:
            provider = "groq"
        else:
            provider = "unknown"

        # Get cost from litellm
        cost = 0.0
        try:
            import litellm

            cost = litellm.completion_cost(completion_response=completion_response)
        except Exception as e:
            logger.debug("Failed to extract cost from completion_response: %s", e)
            cost = 0.0

        # Get token count
        tokens_used = 0
        try:
            if hasattr(completion_response, "usage") and completion_response.usage:
                tokens_used = completion_response.usage.total_tokens or 0
        except Exception as e:
            logger.debug("Failed to extract token count: %s", e)

        # Write to DB
        record_llm_cost(provider, model, cost, _pipeline_run_id, tokens_used)

    except Exception as e:
        logger.warning("_litellm_cost_callback: Unexpected error: %s", e)


def register_litellm_callback() -> None:
    """
    Register the LiteLLM success callback for cost tracking.

    Must be called ONCE at pipeline boot, before any LLM call is made.
    Safe to call multiple times — idempotent via guard check.

    Should be called in:
    - agents/apply_agent.py __init__ or boot sequence
    - agents/analyser_agent.py __init__ or boot sequence
    """
    try:
        import litellm

        if _litellm_cost_callback not in litellm.success_callback:
            litellm.success_callback.append(_litellm_cost_callback)
            logger.info(
                "LiteLLM cost callback registered — budget tracking active"
            )
        else:
            logger.debug("LiteLLM cost callback already registered")
    except Exception as e:
        logger.warning("Failed to register LiteLLM cost callback: %s", e)


def check_xai_run_cap(pipeline_run_id: str) -> str:
    """
    Check if the xAI run cap has been exceeded.

    Args:
        pipeline_run_id: UUID of the run batch.

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

            _log_to_db(pipeline_run_id, "CRITICAL", "budget_cap_hit", message)

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


def check_monthly_budget(pipeline_run_id: str) -> str:
    """
    Check if the monthly budget has been exceeded.

    Queries Postgres to calculate estimated monthly spend based on
    jobs_applied * $0.0025 per application.

    Args:
        pipeline_run_id: UUID of the run batch.

    Returns:
        JSON string with abort flag and budget details.
    """
    conn = None
    try:
        conn = _get_conn()
        if not conn:
            logger.error("budget_tools.check_monthly_budget: DB connection failed.")
            return json.dumps({"error": "db_connection_failed"})

        if psycopg2 is None:
            logger.error(
                "budget_tools.check_monthly_budget: psycopg2 is not installed in the active "
                "Python environment."
            )
            return json.dumps({"error": "psycopg2_missing"})

        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        cursor.execute(
            """
            SELECT COALESCE(SUM(jobs_applied * 0.0025), 0) AS estimated_monthly_spend
            FROM pipeline_runs
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

            _log_to_db(pipeline_run_id, "CRITICAL", "monthly_budget_cap_hit", message)

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


def get_cost_summary(pipeline_run_id: str) -> str:
    """
    Get current run cost summary.

    Args:
        pipeline_run_id: UUID of the run batch.

    Returns:
        JSON string with all cost details for the current run.
    """
    global _run_xai_cost, _run_perplexity_cost

    try:
        run_total_cost = _run_xai_cost + _run_perplexity_cost
        xai_cap_remaining = max(0.0, XAI_COST_CAP_PER_RUN - _run_xai_cost)

        summary = {
            "pipeline_run_id": pipeline_run_id,
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
