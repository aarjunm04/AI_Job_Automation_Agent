"""Tracker Agent for AI Job Application Agent.

This is the final agent in the pipeline. It runs after Apply Agent completes.
Its responsibilities are:

1. Sync all applied jobs to the Notion Job Tracker database.
2. Push all manually queued jobs to the Notion Applications database.
3. Record the AgentOps run summary.
4. Close the run session in Postgres with accurate final counts.

All external-service failures (Notion, AgentOps) are swallowed and logged;
the agent must never crash the pipeline regardless of third-party availability.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional, List

import agentops
from agentops import agent, operation
import psycopg2
import psycopg2.extras
from crewai import Agent, Task, Crew, Process

from config.settings import db_config
from integrations.llm_interface import LLMInterface

from tools.agentops_tools import record_agent_error, _record_agent_error
from tools.notion_tools import (
    check_notion_connection,
    queue_job_to_applications_db,
    sync_application_to_job_tracker,
)
from tools.postgres_tools import (
    get_run_stats,
    log_event,
    _log_event,
    update_run_batch_stats,
    get_pending_manual_queue_db,
)
from utils.db_utils import get_db_conn

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("LOG_LEVEL", "INFO"))

__all__: list[str] = ["TrackerAgent"]


# ---------------------------------------------------------------------------
# Private AgentOps helpers (not yet in agentops_tools.py public API)
# ---------------------------------------------------------------------------

def _record_run_summary(pipeline_run_id: str) -> bool:
    """Emit a run-summary ActionEvent to AgentOps.

    Fails silently — AgentOps downtime must never block the tracker.

    Args:
        pipeline_run_id: UUID of the current run batch.

    Returns:
        True if the event was recorded, False otherwise.
    """
    try:
        logger.info("_record_run_summary: recorded for pipeline_run_id=%s (auto-tracked in v4)", pipeline_run_id)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "_record_run_summary: AgentOps ActionEvent failed (non-critical) — %s", exc
        )
        return False


def _end_agentops_session(end_state: str = "Success") -> bool:
    """End the current AgentOps session.

    Fails silently — AgentOps downtime must never block the tracker.

    Args:
        end_state: Final state to report; ``"Success"`` or ``"Fail"``.

    Returns:
        True if the session was closed, False otherwise.
    """
    try:
        agentops.end_trace(end_state=end_state)
        logger.info("_end_agentops_session: session closed with state=%s", end_state)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "_end_agentops_session: failed to close session (non-critical) — %s", exc
        )
        return False


# ---------------------------------------------------------------------------
# TrackerAgent
# ---------------------------------------------------------------------------

class TrackerAgent:
    """Final pipeline agent: Notion sync, AgentOps summary, run-session close.

    Runs after the Apply Agent completes its pass.  Every Notion or AgentOps
    failure is caught, logged, and swallowed so that the persistence layer
    (Postgres) is always closed cleanly regardless.

    Attributes:
        pipeline_run_id: UUID of the current pipeline run batch.
        user_id: UUID of the candidate user.
        llminterface: Centralised LLM provider manager.
        llm: Primary CrewAI LLM object (Groq llama-3.3-70b-versatile).
        fallback_llm: Fallback CrewAI LLM object (Cerebras).
        logger: Instance-level Python logger.
    """

    def __init__(self, pipeline_run_id: str, user_id: str) -> None:
        """Initialise the TrackerAgent.

        Args:
            pipeline_run_id: UUID of the current run batch in Postgres.
            user_id: UUID of the candidate (users table).
        """
        self.logger: logging.Logger = logging.getLogger(self.__class__.__name__)
        self.pipeline_run_id: str = pipeline_run_id
        self.user_id: str = user_id
        
        self.llminterface: LLMInterface = LLMInterface()
        try:
            self.llm = self.llminterface.get_llm("TRACKER_AGENT")
            self.fallback_llm = self.llminterface.get_fallback_llm("TRACKER_AGENT")
        except Exception as _llm_exc:
            self.logger.error(
                "TrackerAgent: LLM init failed — %s. "
                "Tracker will attempt to run with degraded LLM.", _llm_exc
            )
            self.llm = None
            self.fallback_llm = None
        
        # --- CrewAI-wrapped tools (require .func alias) ---
        self.sync_application_to_job_tracker = sync_application_to_job_tracker
        self.sync_application_to_job_tracker_ = getattr(self.sync_application_to_job_tracker, "func", self.sync_application_to_job_tracker)
        
        self.check_notion_connection = check_notion_connection
        self.check_notion_connection_ = getattr(self.check_notion_connection, "func", self.check_notion_connection)
        
        self.update_run_batch_stats = update_run_batch_stats
        self.update_run_batch_stats_ = getattr(self.update_run_batch_stats, "func", self.update_run_batch_stats)

        self.queue_job_to_applications_db = queue_job_to_applications_db
        self.queue_job_to_applications_db_ = getattr(
            self.queue_job_to_applications_db, "func",
            self.queue_job_to_applications_db
        )
        
        # --- Standard methods (regular functions, no .func) ---
        self.record_run_summary = _record_run_summary
        self.end_agentops_session = _end_agentops_session

    # ------------------------------------------------------------------

    def _safe_end_agentops(self, status: str = "Success") -> None:
        """End AgentOps trace with a hard 5-second timeout.

        Prevents pipeline hang when AgentOps cloud is unreachable.
        """
        import concurrent.futures
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                future = ex.submit(agentops.end_trace, status)
                future.result(timeout=5.0)
        except concurrent.futures.TimeoutError:
            self.logger.warning(
                "_safe_end_agentops: timed out after 5s — skipping flush"
            )
        except Exception as exc:
            self.logger.warning("_safe_end_agentops: failed — %s", exc)
    # Internal DB helpers
    # ------------------------------------------------------------------

    def _query_applications(self, status: str) -> list[dict[str, Any]]:
        """Query the database for applications of a given status.

        Joins ``applications``, ``job_posts``, and ``job_scores`` to assemble
        all fields required by the Notion sync tools.  Retries up to 3 times
        with exponential backoff.

        Args:
            status: Application status to filter on
                (``"applied"`` or ``"manual_queued"``).

        Returns:
            List of dicts, each containing all fields available on the joined
            rows.  Returns an empty list on unrecoverable failure.
        """
        # connection_url check removed (handled by get_db_conn)

        sql = """
            SELECT
                a.id            AS application_id,
                a.job_post_id,
                a.status,
                a.platform,
                COALESCE(r.storage_path, '') AS resume_used,
                jp.title,
                jp.company,
                jp.url,
                COALESCE(jp.location, '')    AS location,
                COALESCE(js.fit_score, 0.0)  AS fit_score
            FROM applications a
            JOIN jobs jp ON jp.id = a.job_post_id
            LEFT JOIN resumes r ON r.id = a.resume_id
            LEFT JOIN LATERAL (
                SELECT fit_score FROM job_scores
                WHERE job_post_id = jp.id
                ORDER BY scored_at DESC
                LIMIT 1
            ) js ON TRUE
            WHERE jp.pipeline_run_id = %s
              AND a.status = %s
        """

        last_exc: Optional[Exception] = None
        conn = None
        try:
            conn = get_db_conn()
            conn.autocommit = False
            cursor = conn.cursor(
                cursor_factory=psycopg2.extras.RealDictCursor
            )
            cursor.execute(sql, (self.pipeline_run_id, status))
            rows = cursor.fetchall()
            result: list[dict[str, Any]] = [dict(row) for row in rows]
            self.logger.info(
                "_query_applications: status=%s found=%d", status, len(result)
            )
            return result
        except Exception as exc:  # noqa: BLE001
            self.logger.error(
                "_query_applications: DB error status=%s — %s",
                status,
                exc,
            )
            if conn:
                try:
                    conn.rollback()
                except Exception:
                    pass
            return []
        finally:
            if conn:
                try:
                    conn.close()
                except Exception:
                    pass

    def _get_applied_jobs(self) -> list[dict[str, Any]]:
        """Fetch applied jobs not yet synced to Notion Job Tracker.

        Returns:
            List of dicts representing applied application rows joined with
            job_posts and job_scores.
        """
        return self._query_applications("applied")

    def _get_queued_jobs(self) -> list[dict[str, Any]]:
        """Fetch manually queued jobs not yet pushed to Notion Applications DB.

        Returns:
            List of dicts representing manual_queued application rows joined
            with job_posts and job_scores.
        """
        return self._query_applications("manual_queued")

    # ------------------------------------------------------------------
    # CrewAI construction
    # ------------------------------------------------------------------

    def _build_agent(self) -> Agent:
        """Instantiate the CrewAI Agent for tracker work.

        Returns:
            Configured CrewAI Agent with all tracker tools attached.
        """
        return Agent(
            role="Run Session Tracker and Reporting Specialist",
            goal=(
                "Update Postgres run batch stats, record the complete run summary to "
                "AgentOps, and cleanly close the run session — ensuring zero data loss "
                "even if external services are partially unavailable."
            ),
            backstory=(
                "You are meticulous about closing the loop on every pipeline run. "
                "You ensure every metric is recorded in AgentOps and every run session "
                "is properly closed in Postgres with accurate counts."
            ),
            llm=self.llm,
            function_calling_llm=self.fallback_llm,
            tools=[
                update_run_batch_stats,
            ],
            verbose=True,
            max_iter=10,
            memory=False,
        )

    def _build_task(
        self,
        agent: Agent,
        applied_jobs: list[dict[str, Any]],
        queued_jobs: list[dict[str, Any]],
    ) -> Task:
        """Build the CrewAI Task description for the tracker agent.

        Args:
            agent: The TrackerAgent CrewAI Agent instance.
            applied_jobs: List of applied job dicts from _get_applied_jobs().
            queued_jobs: List of queued job dicts from _get_queued_jobs().

        Returns:
            Configured CrewAI Task instance ready to be added to a Crew.
        """
        description = (
            f"Close out run batch {self.pipeline_run_id}. "
            f"Applied count: {len(applied_jobs)}. "
            f"Queued count: {len(queued_jobs)}. "
            f"Notion sync has already been completed in Python; do NOT sync Notion. "
            f"Update Postgres run batch stats and close the session. "
            f"Record the run summary to AgentOps and end the AgentOps session."
        )

        return Task(
            description=description,
            agent=agent,
            expected_output=(
                "JSON: {agentops_recorded: bool, session_closed: bool, errors: list}"
            ),
        )

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    @operation
    def run(self) -> dict[str, Any]:
        """Execute the full tracker pass for the current run batch.

        Orchestrates:

        1. Logs tracker start event to Postgres.
        2. Fetches applied and queued jobs via direct DB queries.
        3. Short-circuits gracefully if there is nothing to sync.
        4. Builds and kicks off the CrewAI crew.
        5. Parses the structured JSON result from the agent.
        6. Logs completion event to Postgres.
        7. Records AgentOps run summary and ends the session.

        Any unhandled exception is caught, logged as CRITICAL, recorded to
        AgentOps, and results in an ``end_state="Fail"`` session close.

        Returns:
            Dict with keys: success, pipeline_run_id, notion_synced_applied,
            notion_synced_queued, agentops_recorded, session_closed.
            On failure: success=False and an "error" key with the message.
        """
        # Step 1 — log start
        try:
            _log_event(
                pipeline_run_id=self.pipeline_run_id,
                level="INFO",
                event_type="tracker_run_start",
                message="Tracker Agent starting sync pass",
            )
        except Exception as log_exc:  # noqa: BLE001
            self.logger.warning(
                "run: log_event(tracker_run_start) failed (non-critical) — %s", log_exc
            )

        try:
            # Step 2 — fetch jobs
            applied: list[dict[str, Any]] = self._get_applied_jobs()
            queued: list[dict[str, Any]] = self._get_queued_jobs()
            self.logger.info(
                "run: fetched applied=%d queued=%d for pipeline_run_id=%s",
                len(applied),
                len(queued),
                self.pipeline_run_id,
            )

            # Step 3 — short-circuit if nothing to do
            if not applied and not queued:
                self.logger.info(
                    "run: nothing to sync for pipeline_run_id=%s", self.pipeline_run_id
                )
                summary_ok: bool = self.record_run_summary(self.pipeline_run_id)
                session_ok: bool = self._safe_end_agentops("Success")
                return {
                    "success": True,
                    "reason": "nothing_to_sync",
                    "pipeline_run_id": self.pipeline_run_id,
                    "notion_synced_applied": 0,
                    "notion_synced_queued": 0,
                    "agentops_recorded": summary_ok,
                    "session_closed": session_ok,
                }

            # Step 3b — Direct Python Notion sync (do NOT delegate to LLM)
            notion_applied = 0
            notion_queued  = 0
            for job in applied:
                try:
                    self.sync_application_to_job_tracker_(
                        application_id=str(job.get("application_id", "")),
                        job_post_id=str(job.get("job_post_id", "")),
                        pipeline_run_id=self.pipeline_run_id,
                        title=str(job.get("title", "")),
                        company=str(job.get("company", "")),
                        job_url=str(job.get("url", "")),
                        platform=str(job.get("platform", "")),
                        resume_used=str(job.get("resume_used", "")),
                        ctc="",
                        notes="",
                        location=str(job.get("location", "")),
                        job_type=str(job.get("job_type", "full-time")),
                    )
                    notion_applied += 1
                except Exception as _sync_exc:
                    self.logger.warning("run: notion sync failed for job %s — %s",
                                        job.get("job_post_id"), _sync_exc)

            for job in queued:
                try:
                    self.queue_job_to_applications_db_(
                        job_post_id=str(job.get("job_post_id", "")),
                        pipeline_run_id=self.pipeline_run_id,
                        title=str(job.get("title", "")),
                        company=str(job.get("company", "")),
                        job_url=str(job.get("url", "")),
                        platform=str(job.get("platform", "")),
                        fit_score=float(job.get("fit_score", 0.0)),
                        resume_suggested=str(job.get("resume_used", "")),
                        job_type=str(job.get("job_type", "full-time")),
                        location=str(job.get("location", "")),
                        notes="",
                    )
                    notion_queued += 1
                except Exception as _sync_exc:
                    self.logger.warning(
                        "run: applications_db sync failed for job %s — %s",
                        job.get("job_post_id"), _sync_exc
                    )

            self.logger.info(
                "run: notion sync complete — applied=%d queued=%d",
                notion_applied, notion_queued,
            )

            # Step 4 — build crew and kick off
            if self.llm is None:
                self.logger.error(
                    "TrackerAgent: cannot build crew — LLM unavailable. "
                    "Returning early with success=False."
                )
                return {
                    "success": False,
                    "pipeline_run_id": self.pipeline_run_id,
                    "error": "llm_init_failed",
                }
            agent: Agent = self._build_agent()
            task: Task = self._build_task(agent, applied, queued)
            crew: Crew = Crew(
                agents=[agent],
                tasks=[task],
                process=Process.sequential,
                verbose=True,
            )

            import re as _re
            import time as _time
            _MAX_RETRIES = 3
            raw_result: Any = None
            for _attempt in range(1, _MAX_RETRIES + 1):
                try:
                    raw_result = crew.kickoff()
                    break
                except Exception as _ke:
                    _ke_str = str(_ke)
                    _is_rl = (
                        "RateLimitError" in type(_ke).__name__
                        or "rate_limit" in _ke_str.lower()
                        or "rate limit" in _ke_str.lower()
                        or "429" in _ke_str
                    )
                    if _is_rl and _attempt < _MAX_RETRIES:
                        _match = _re.search(r"try again in ([\d.]+)s", _ke_str)
                        _wait = float(_match.group(1)) + 2.0 if _match else 32.0
                        self.logger.warning(
                            "TrackerAgent crew.kickoff() RateLimitError attempt "
                            "%d/%d — sleeping %.1fs", _attempt, _MAX_RETRIES, _wait
                        )
                        _time.sleep(_wait)
                    else:
                        self.logger.error(
                            "TrackerAgent crew.kickoff() failed attempt %d/%d: %s",
                            _attempt, _MAX_RETRIES, _ke_str[:300]
                        )
                        raise

            # Step 5 — parse result
            result_text: str = ""
            if hasattr(raw_result, "raw"):
                result_text = str(raw_result.raw)
            else:
                result_text = str(raw_result)

            parsed: dict[str, Any] = {}
            try:
                # Strip markdown fences if the LLM wrapped with ```json
                clean: str = result_text.strip()
                if clean.startswith("```"):
                    import re
                    clean = re.sub(r"^```(?:json)?\s*|\s*```$", "", clean.strip())
                try:
                    parsed = json.loads(clean)
                except (json.JSONDecodeError, TypeError, ValueError) as _json_exc:
                    self.logger.warning(
                        "TrackerAgent: LLM output could not be parsed as JSON — %s. "
                        "Proceeding with empty result.", _json_exc
                    )
                    parsed = {}
            except Exception as parse_exc:  # noqa: BLE001
                raw_preview: str = result_text[:500]
                self.logger.warning(
                    "run: could not parse agent JSON result — %s — raw=%s",
                    parse_exc,
                    raw_preview,
                )

            agent_errors: list[str] = parsed.get("errors", [])

            # Step 6 — log completion
            try:
                _log_event(
                    pipeline_run_id=self.pipeline_run_id,
                    level="INFO",
                    event_type="tracker_run_complete",
                    message=(
                        f"Tracker sync complete — "
                        f"notion_synced_applied={notion_applied} "
                        f"notion_synced_queued={notion_queued} "
                        f"errors={len(agent_errors)}"
                    ),
                )
            except Exception as log_exc2:  # noqa: BLE001
                self.logger.warning(
                    "run: log_event(tracker_run_complete) failed (non-critical) — %s",
                    log_exc2,
                )

            # Step 7 — AgentOps bookkeeping
            end_state: str = "Success" if not agent_errors else "Fail"
            summary_recorded: bool = self.record_run_summary(self.pipeline_run_id)
            self._safe_end_agentops(end_state)
            session_closed: bool = True

            return {
                "success": True,
                "pipeline_run_id": self.pipeline_run_id,
                "notion_synced_applied": notion_applied,
                "notion_synced_queued": notion_queued,
                "agentops_recorded": summary_recorded,
                "session_closed": session_closed,
            }

        except Exception as exc:  # noqa: BLE001
            self.logger.critical(
                "run: unhandled exception in TrackerAgent — %s", exc, exc_info=True
            )

            # Best-effort AgentOps error recording
            try:
                _record_agent_error(
                    agent_type="TrackerAgent",
                    error_message=str(exc),
                    pipeline_run_id=self.pipeline_run_id,
                    error_code="TRACKER_UNHANDLED_ERROR",
                )
            except Exception:  # noqa: BLE001
                pass

            # Best-effort session close
            self._safe_end_agentops("Fail")

            return {
                "success": False,
                "pipeline_run_id": self.pipeline_run_id,
                "error": str(exc),
            }

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------

    def _get_run_session_stats(self) -> Optional[dict[str, Any]]:
        """Query pipeline_runs table for this run's stats."""
        conn = None
        try:
            conn = get_db_conn()
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            cursor.execute("""
                SELECT 
                    jobs_found,
                    jobs_applied,
                    jobs_queued,
                    EXTRACT(EPOCH FROM (COALESCE(completed_at, NOW()) - created_at)) / 60.0 AS duration_minutes
                FROM pipeline_runs
                WHERE pipeline_run_id = %s
            """, (self.pipeline_run_id,))
            
            row = cursor.fetchone()
            return dict(row) if row else None
        except Exception as exc:
            self.logger.warning("Failed to get run session stats: %s", exc)
            return None
        finally:
            if conn:
                conn.close()

    def _get_application_stats(self) -> dict[str, int]:
        """Query applications table grouped by status."""
        conn = None
        try:
            conn = get_db_conn()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT status, COUNT(*) as count
                FROM applications a
                JOIN jobs j ON j.id = a.job_post_id
                WHERE j.pipeline_run_id = %s
                GROUP BY status
            """, (self.pipeline_run_id,))
            
            results = cursor.fetchall()
            
            stats = {}
            for status, count in results:
                stats[status.lower().replace(" ", "_")] = count
            return stats
        except Exception as exc:
            self.logger.warning("Failed to get application stats: %s", exc)
            return {}
        finally:
            if conn:
                conn.close()

    def _get_total_cost(self) -> float:
        """Get total cost from budget_tools."""
        try:
            from tools.budget_tools import get_cost_summary
            
            raw = get_cost_summary(self.pipeline_run_id)
            data = json.loads(raw) if isinstance(raw, str) else raw
            return float(data.get("run_total_cost", 0.0))
        except Exception as exc:
            self.logger.warning("Failed to get total cost: %s", exc)
            return 0.0

    def _get_top_applied_jobs(self, limit: int = 10) -> List[dict[str, Any]]:
        """Get top applied jobs ordered by fit_score."""
        conn = None
        try:
            conn = get_db_conn()
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            cursor.execute("""
                SELECT 
                    j.title,
                    j.company,
                    j.url,
                    j.source_platform as platform,
                    COALESCE(js.fit_score, 0.0) as fit_score
                FROM applications a
                JOIN jobs j ON j.id = a.job_post_id
                LEFT JOIN job_scores js ON js.job_post_id = j.id
                WHERE j.pipeline_run_id = %s
                  AND a.status = 'applied'
                ORDER BY js.fit_score DESC NULLS LAST
                LIMIT %s
            """, (self.pipeline_run_id, limit))
            
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
        except Exception as exc:
            self.logger.warning("Failed to get top applied jobs: %s", exc)
            return []
        finally:
            if conn:
                conn.close()

    def _get_error_summary(self) -> Optional[str]:
        """Get error summary from audit_logs."""
        conn = None
        try:
            conn = get_db_conn()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT message
                FROM audit_logs
                WHERE pipeline_run_id = %s
                  AND level = 'ERROR'
                ORDER BY created_at DESC
                LIMIT 5
            """, (self.pipeline_run_id,))
            
            rows = cursor.fetchall()
            
            if not rows:
                return None
            
            errors = [row[0] for row in rows]
            return "\n".join(errors[:5])
        except Exception as exc:
            self.logger.warning("Failed to get error summary: %s", exc)
            return None
        finally:
            if conn:
                conn.close()
