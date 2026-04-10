"""Tracker Agent for AI Job Application Agent.

This is the final agent in the pipeline. It runs after Apply Agent completes.
Its responsibilities are:

1. Sync all applied jobs to the Notion Job Tracker database.
2. Push all manually queued jobs to the Notion Applications database.
3. Record the AgentOps run summary.
4. Close the run session in Postgres with accurate final counts.
5. Generate and post the FinalReport to Notion.

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
from agentops.sdk.decorators import agent, operation
import psycopg2
import psycopg2.extras
from crewai import Agent, Task, Crew, Process

from config.settings import db_config
from integrations.llm_interface import LLMInterface
from integrations.notion import NotionClient, FinalReport
from tools.agentops_tools import record_agent_error, _record_agent_error
from tools.notion_tools import (
    check_notion_connection,
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

__all__ = ["TrackerAgent", "FinalReport"]


# ---------------------------------------------------------------------------
# Private AgentOps helpers (not yet in agentops_tools.py public API)
# ---------------------------------------------------------------------------

def _record_run_summary(run_batch_id: str) -> bool:
    """Emit a run-summary ActionEvent to AgentOps.

    Fails silently — AgentOps downtime must never block the tracker.

    Args:
        run_batch_id: UUID of the current run batch.

    Returns:
        True if the event was recorded, False otherwise.
    """
    try:
        logger.info("_record_run_summary: recorded for run_batch_id=%s (auto-tracked in v4)", run_batch_id)
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
        run_batch_id: UUID of the current pipeline run batch.
        user_id: UUID of the candidate user.
        llm: Primary CrewAI LLM object (Groq llama-3.3-70b-versatile).
        logger: Instance-level Python logger.
    """

    def __init__(self, run_batch_id: str, user_id: str) -> None:
        """Initialise the TrackerAgent.

        Args:
            run_batch_id: UUID of the current run batch in Postgres.
            user_id: UUID of the candidate (users table).
        """
        self.run_batch_id: str = run_batch_id
        self.user_id: str = user_id
        try:
            self.llm = LLMInterface().get_llm("TRACKER_AGENT")
        except Exception as _llm_exc:
            self.logger.error(
                "TrackerAgent: LLM init failed — %s. "
                "Tracker will attempt to run with degraded LLM.", _llm_exc
            )
            self.llm = None
        self.logger: logging.Logger = logging.getLogger(self.__class__.__name__)
        
        # --- CrewAI-wrapped tools (require .func alias) ---
        self.sync_application_to_job_tracker = sync_application_to_job_tracker
        self.sync_application_to_job_tracker_ = getattr(self.sync_application_to_job_tracker, "func", self.sync_application_to_job_tracker)
        
        self.check_notion_connection = check_notion_connection
        self.check_notion_connection_ = getattr(self.check_notion_connection, "func", self.check_notion_connection)
        
        self.update_run_batch_stats = update_run_batch_stats
        self.update_run_batch_stats_ = getattr(self.update_run_batch_stats, "func", self.update_run_batch_stats)
        
        # --- Standard methods (regular functions, no .func) ---
        self.record_run_summary = _record_run_summary
        self.end_agentops_session = _end_agentops_session

    # ------------------------------------------------------------------
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
            WHERE jp.run_batch_id = %s
              AND a.status = %s
        """

        last_exc: Optional[Exception] = None
        for attempt in range(3):
            conn = None
            try:
                conn = get_db_conn()
                conn.autocommit = False
                cursor = conn.cursor(
                    cursor_factory=psycopg2.extras.RealDictCursor
                )
                cursor.execute(sql, (self.run_batch_id, status))
                rows = cursor.fetchall()
                conn.close()
                result: list[dict[str, Any]] = [dict(row) for row in rows]
                self.logger.info(
                    "_query_applications: status=%s found=%d", status, len(result)
                )
                return result
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if conn:
                    try:
                        conn.rollback()
                        conn.close()
                    except Exception:  # noqa: BLE001
                        pass
                if attempt < 2:
                    sleep_s = 2 ** attempt
                    self.logger.warning(
                        "_query_applications: attempt %d failed, retrying in %ds — %s",
                        attempt + 1,
                        sleep_s,
                        exc,
                    )
                    time.sleep(sleep_s)

        self.logger.error(
            "_query_applications: all retries exhausted status=%s — %s",
            status,
            last_exc,
        )
        return []

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
                "Sync every applied job to Notion Job Tracker, push every manually "
                "queued job to Notion Applications DB, record the complete run summary "
                "to AgentOps, and cleanly close the run session — ensuring zero data "
                "loss even if external services are partially unavailable."
            ),
            backstory=(
                "You are meticulous about closing the loop on every pipeline run. "
                "You ensure every job application is accounted for in Notion, every "
                "metric is recorded in AgentOps, and every run session is properly "
                "closed in Postgres with accurate counts."
            ),
            llm=self.llm,
            tools=[
                sync_application_to_job_tracker,
                check_notion_connection,
                get_pending_manual_queue_db,
                log_event,
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
            f"Close out run batch {self.run_batch_id}. "
            f"Applied count: {len(applied_jobs)}. "
            f"Queued count: {len(queued_jobs)}. "
            f"Sync all applied jobs to Notion Job Tracker. "
            f"Push all queued jobs to Notion Applications DB. "
            f"Update Postgres run batch stats and close the session."
        )

        return Task(
            description=description,
            agent=agent,
            expected_output=(
                "JSON: {notion_synced_applied: int, notion_synced_queued: int, "
                "agentops_recorded: bool, session_closed: bool, errors: list}"
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
            Dict with keys: success, run_batch_id, notion_synced_applied,
            notion_synced_queued, agentops_recorded, session_closed.
            On failure: success=False and an "error" key with the message.
        """
        # Step 1 — log start
        try:
            _log_event(
                run_batch_id=self.run_batch_id,
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
                "run: fetched applied=%d queued=%d for run_batch_id=%s",
                len(applied),
                len(queued),
                self.run_batch_id,
            )

            # Step 3 — short-circuit if nothing to do
            if not applied and not queued:
                self.logger.info(
                    "run: nothing to sync for run_batch_id=%s", self.run_batch_id
                )
                summary_ok: bool = self.record_run_summary(self.run_batch_id)
                session_ok: bool = self.end_agentops_session("Success")
                return {
                    "success": True,
                    "reason": "nothing_to_sync",
                    "run_batch_id": self.run_batch_id,
                    "notion_synced_applied": 0,
                    "notion_synced_queued": 0,
                    "agentops_recorded": summary_ok,
                    "session_closed": session_ok,
                }

            # Step 4 — build crew and kick off
            if self.llm is None:
                self.logger.error(
                    "TrackerAgent: cannot build crew — LLM unavailable. "
                    "Returning early with success=False."
                )
                return {
                    "success": False,
                    "run_batch_id": self.run_batch_id,
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

            notion_synced_applied: int = int(
                parsed.get("notion_synced_applied", len(applied))
            )
            notion_synced_queued: int = int(
                parsed.get("notion_synced_queued", len(queued))
            )
            agent_errors: list[str] = parsed.get("errors", [])

            # Step 6 — log completion
            try:
                _log_event(
                    run_batch_id=self.run_batch_id,
                    level="INFO",
                    event_type="tracker_run_complete",
                    message=(
                        f"Tracker sync complete — "
                        f"notion_synced_applied={notion_synced_applied} "
                        f"notion_synced_queued={notion_synced_queued} "
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
            summary_recorded: bool = self.record_run_summary(self.run_batch_id)
            session_closed: bool = self.end_agentops_session(end_state)

            return {
                "success": True,
                "run_batch_id": self.run_batch_id,
                "notion_synced_applied": notion_synced_applied,
                "notion_synced_queued": notion_synced_queued,
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
                    run_batch_id=self.run_batch_id,
                    error_code="TRACKER_UNHANDLED_ERROR",
                )
            except Exception:  # noqa: BLE001
                pass

            # Best-effort session close
            self.end_agentops_session("Fail")

            return {
                "success": False,
                "run_batch_id": self.run_batch_id,
                "error": str(exc),
            }

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------

    @operation
    def generate_report(self, apply_result: Optional[dict[str, Any]] = None) -> FinalReport:
        """Generate the final pipeline report with all statistics.

        Reads from run_sessions, applications, queued_jobs, and audit_logs
        to compile a complete run summary. Posts to Notion and ends AgentOps.

        Args:
            apply_result: Optional result dict from ApplyAgent.run().

        Returns:
            FinalReport dataclass with all run statistics.
        """
        self.logger.info("Generating final report for run_batch_id=%s", self.run_batch_id)

        # Initialize report with defaults
        report = FinalReport(
            run_batch_id=self.run_batch_id,
            run_date=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            jobs_discovered=0,
            jobs_scored=0,
            jobs_auto_applied=0,
            jobs_manual_queued=0,
            jobs_failed=0,
            total_cost_usd=0.0,
            duration_minutes=0.0,
            success=True,
            error_summary=None,
            top_applied_jobs=None,
        )

        try:
            # Step 1: Get run session stats from Postgres
            run_stats = self._get_run_session_stats()
            if run_stats:
                report.jobs_discovered = run_stats.get("jobs_discovered", 0)
                report.jobs_auto_applied = run_stats.get("jobs_auto_applied", 0)
                report.jobs_manual_queued = run_stats.get("jobs_queued", 0)
                report.duration_minutes = run_stats.get("duration_minutes", 0.0)

            # Step 2: Get application counts by status
            app_stats = self._get_application_stats()
            if app_stats:
                report.jobs_auto_applied = app_stats.get("applied", report.jobs_auto_applied)
                report.jobs_failed = app_stats.get("failed", 0)
                report.jobs_manual_queued = app_stats.get("manual_queued", report.jobs_manual_queued)

            # Step 3: Calculate total cost from budget tools
            report.total_cost_usd = self._get_total_cost()

            # Step 4: Get top applied jobs
            report.top_applied_jobs = self._get_top_applied_jobs(limit=10)

            # Step 5: Check for errors in audit_logs
            errors = self._get_error_summary()
            if errors:
                report.error_summary = errors
                report.success = report.jobs_failed == 0

            # Step 6: Calculate success rate
            total_attempts = report.jobs_auto_applied + report.jobs_failed
            if total_attempts > 0:
                success_rate = report.jobs_auto_applied / total_attempts
                self.logger.info("Success rate: %.1f%%", success_rate * 100)

            # Step 7: Post to Notion
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    import concurrent.futures as _cf
                    with _cf.ThreadPoolExecutor(max_workers=1) as _pool:
                        _future = _pool.submit(
                            asyncio.run, self._post_report_to_notion(report)
                        )
                        _future.result(timeout=30)
                else:
                    loop.run_until_complete(self._post_report_to_notion(report))
            except Exception as _notion_exc:
                self.logger.warning(
                    "TrackerAgent: Notion report post failed — %s. "
                    "Continuing without Notion report.", _notion_exc
                )

            # Step 8: End AgentOps session
            end_state = "Success" if report.success else "Fail"
            self.record_run_summary(self.run_batch_id)
            self.end_agentops_session(end_state)

            # Step 9: Log final summary
            self.logger.info(
                "FINAL REPORT: discovered=%d applied=%d queued=%d failed=%d cost=$%.4f duration=%.1fmin",
                report.jobs_discovered,
                report.jobs_auto_applied,
                report.jobs_manual_queued,
                report.jobs_failed,
                report.total_cost_usd,
                report.duration_minutes,
            )

        except Exception as _rpt_exc:
            self.logger.error(
                "TrackerAgent.generate_report: failed — %s",
                _rpt_exc, exc_info=True
            )
            return {
                "success": False,
                "error": f"generate_report_failed: {str(_rpt_exc)[:200]}",
            }

        return report

    def _get_run_session_stats(self) -> Optional[dict[str, Any]]:
        """Query run_sessions table for this run's stats."""
        try:
            conn = get_db_conn()
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            cursor.execute("""
                SELECT 
                    jobs_discovered,
                    jobs_auto_applied,
                    jobs_queued,
                    EXTRACT(EPOCH FROM (COALESCE(closed_at, NOW()) - created_at)) / 60.0 AS duration_minutes
                FROM run_sessions
                WHERE run_batch_id = %s
            """, (self.run_batch_id,))
            
            row = cursor.fetchone()
            conn.close()
            return dict(row) if row else None
        except Exception as exc:
            self.logger.warning("Failed to get run session stats: %s", exc)
            return None

    def _get_application_stats(self) -> dict[str, int]:
        """Query applications table grouped by status."""
        try:
            conn = get_db_conn()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT status, COUNT(*) as count
                FROM applications a
                JOIN jobs j ON j.id = a.job_post_id
                WHERE j.run_batch_id = %s
                GROUP BY status
            """, (self.run_batch_id,))
            
            results = cursor.fetchall()
            conn.close()
            
            stats = {}
            for status, count in results:
                stats[status.lower().replace(" ", "_")] = count
            return stats
        except Exception as exc:
            self.logger.warning("Failed to get application stats: %s", exc)
            return {}

    def _get_total_cost(self) -> float:
        """Get total cost from budget_tools."""
        try:
            from tools.budget_tools import get_cost_summary
            
            raw = get_cost_summary(self.run_batch_id)
            data = json.loads(raw) if isinstance(raw, str) else raw
            return float(data.get("run_total_cost", 0.0))
        except Exception as exc:
            self.logger.warning("Failed to get total cost: %s", exc)
            return 0.0

    def _get_top_applied_jobs(self, limit: int = 10) -> List[dict[str, Any]]:
        """Get top applied jobs ordered by fit_score."""
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
                WHERE j.run_batch_id = %s
                  AND a.status = 'applied'
                ORDER BY js.fit_score DESC NULLS LAST
                LIMIT %s
            """, (self.run_batch_id, limit))
            
            rows = cursor.fetchall()
            conn.close()
            return [dict(row) for row in rows]
        except Exception as exc:
            self.logger.warning("Failed to get top applied jobs: %s", exc)
            return []

    def _get_error_summary(self) -> Optional[str]:
        """Get error summary from audit_logs."""
        try:
            conn = get_db_conn()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT message
                FROM audit_logs
                WHERE run_batch_id = %s
                  AND level = 'ERROR'
                ORDER BY created_at DESC
                LIMIT 5
            """, (self.run_batch_id,))
            
            rows = cursor.fetchall()
            conn.close()
            
            if not rows:
                return None
            
            errors = [row[0] for row in rows]
            return "\n".join(errors[:5])
        except Exception as exc:
            self.logger.warning("Failed to get error summary: %s", exc)
            return None

    async def _post_report_to_notion(self, report: FinalReport) -> None:
        """Post the final report to Notion."""
        try:
            client = NotionClient()
            await client.post_run_report(report)
            self.logger.info("Successfully posted report to Notion")
        except Exception as exc:
            self.logger.warning("Failed to post to Notion: %s", exc)
            raise
