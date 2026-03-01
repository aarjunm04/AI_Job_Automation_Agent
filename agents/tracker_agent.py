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

import json
import logging
import os
import time
from typing import Any, Optional

import agentops
from agentops.sdk.decorators import agent, operation
import psycopg2
import psycopg2.extras
from crewai import Agent, Crew, Process, Task

from config.settings import db_config
from integrations.llm_interface import LLMInterface
from tools.agentops_tools import record_agent_error
from tools.notion_tools import (
    check_notion_connection,
    get_pending_manual_queue,
    queue_job_to_applications_db,
    sync_application_to_job_tracker,
)
from tools.postgres_tools import (
    get_run_stats,
    log_event,
    update_run_batch_stats,
)

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("LOG_LEVEL", "INFO"))

__all__ = ["TrackerAgent"]


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
        agentops.record(
            agentops.ActionEvent(
                action_type="run_summary",
                params={"run_batch_id": run_batch_id},
                returns={"status": "summary_recorded"},
            )
        )
        logger.info("_record_run_summary: recorded for run_batch_id=%s", run_batch_id)
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
        agentops.end_session(end_state=end_state)
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

@agent
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
        self.llm = LLMInterface().get_llm("TRACKER_AGENT")
        self.logger: logging.Logger = logging.getLogger(self.__class__.__name__)

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
        connection_url: str = db_config.connection_url
        if not connection_url:
            self.logger.error(
                "_query_applications: no DB connection_url configured"
            )
            return []

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
            JOIN job_posts jp ON jp.id = a.job_post_id
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
                conn = psycopg2.connect(connection_url)
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
                queue_job_to_applications_db,
                check_notion_connection,
                get_pending_manual_queue,
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
        applied_json: str = json.dumps(applied_jobs, default=str)
        queued_json: str = json.dumps(queued_jobs, default=str)

        description = f"""
You are closing out run batch {self.run_batch_id}.

STEP 1 — Notion health check
Call check_notion_connection(run_batch_id="{self.run_batch_id}").
Parse the JSON result. If the "connected" key is False or absent, log a WARNING
and skip ALL Notion calls below (steps 2 and 3). Do NOT raise an error.

STEP 2 — Sync applied jobs to Notion Job Tracker
For each job in the applied_jobs list below, call sync_application_to_job_tracker
with: application_id, job_post_id, run_batch_id="{self.run_batch_id}", title,
company, url (as job_url), platform, resume_used, and location.
applied_jobs (JSON):
{applied_json}

STEP 3 — Push queued jobs to Notion Applications DB
For each job in the queued_jobs list below, call queue_job_to_applications_db
with: job_post_id, run_batch_id="{self.run_batch_id}", title, company, url
(as job_url), platform, fit_score, and resume_used (as resume_suggested),
and location.
queued_jobs (JSON):
{queued_json}

STEP 4 — Update Postgres run batch stats
Call update_run_batch_stats with:
  run_batch_id="{self.run_batch_id}"
  jobs_discovered=<total jobs in applied_jobs + queued_jobs, or fetch from get_run_stats if available>
  jobs_auto_applied=<count of applied_jobs>
  jobs_queued=<count of queued_jobs>

STEP 5 — Return structured result
Return ONLY a valid JSON object with these exact keys:
{{
  "notion_synced_applied": <int: count of applied jobs successfully synced>,
  "notion_synced_queued": <int: count of queued jobs successfully pushed>,
  "agentops_recorded": true,
  "session_closed": true,
  "errors": []
}}
If some Notion syncs failed, include them in the "errors" list as strings.
"""

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
            log_event(
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
                summary_ok: bool = _record_run_summary(self.run_batch_id)
                session_ok: bool = _end_agentops_session("Success")
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
            agent: Agent = self._build_agent()
            task: Task = self._build_task(agent, applied, queued)
            crew: Crew = Crew(
                agents=[agent],
                tasks=[task],
                process=Process.sequential,
                verbose=True,
            )

            raw_result: Any = crew.kickoff()

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
                parsed = json.loads(clean)
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
                log_event(
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
            summary_recorded: bool = _record_run_summary(self.run_batch_id)
            session_closed: bool = _end_agentops_session(end_state)

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
                record_agent_error(
                    agent_type="TrackerAgent",
                    error_message=str(exc),
                    run_batch_id=self.run_batch_id,
                    error_code="TRACKER_UNHANDLED_ERROR",
                )
            except Exception:  # noqa: BLE001
                pass

            # Best-effort session close
            _end_agentops_session("Fail")

            return {
                "success": False,
                "run_batch_id": self.run_batch_id,
                "error": str(exc),
            }
