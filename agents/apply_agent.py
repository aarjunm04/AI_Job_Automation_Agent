"""Apply Agent for AI Job Application Agent.

The most operationally critical agent in the pipeline.  Receives the routing
manifest from the Analyser Agent, makes final pre-apply safety decisions,
executes Playwright form filling for every auto-route job, handles proof
capture, re-routes failed jobs to the manual queue, enforces the xAI cost
cap mid-run, and hands off all results to the Tracker Agent.

Pipeline position::

    Analyser Agent → **Apply Agent** → Tracker Agent

A single bad decision here wastes real money and real applications.  Every
action must be deliberate, logged, and reversible via manual re-queue.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from crewai import Agent, Task, Crew, Process
import agentops
import psycopg2
import psycopg2.extras

from config.settings import db_config, run_config, budget_config
from integrations.llm_interface import LLMInterface
from tools.apply_tools import (
    detect_ats_platform,
    fill_standard_form,
    check_captcha_present,
    capture_proof,
    get_apply_summary,
)
from tools.postgres_tools import (
    create_application,
    update_application_status,
    log_event,
    get_platform_config,
)
from tools.budget_tools import (
    check_xai_run_cap,
    record_llm_cost,
    get_cost_summary,
)
from tools.agentops_tools import (
    record_agent_error,
    record_fallback_event,
)
from tools.notion_tools import queue_job_to_applications_db

# Module-level logger
logger = logging.getLogger(__name__)

__all__ = ["ApplyAgent"]

# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

_DB_URL: Optional[str] = (
    os.getenv("LOCAL_POSTGRES_URL")
    if os.getenv("ACTIVE_DB", "local") == "local"
    else os.getenv("SUPABASE_URL")
)


def _get_db_conn() -> psycopg2.extensions.connection:
    """Open and return a psycopg2 connection to the active Postgres instance.

    Returns:
        Database connection with ``autocommit=False``.

    Raises:
        RuntimeError: If the DB URL environment variable is not set, or if
            the connection attempt fails.
    """
    if not _DB_URL:
        raise RuntimeError(
            "Database URL is not configured.  "
            "Set LOCAL_POSTGRES_URL or SUPABASE_URL in narad.env and "
            "ACTIVE_DB=local|supabase."
        )
    try:
        conn = psycopg2.connect(_DB_URL)
        conn.autocommit = False
        return conn
    except Exception as exc:
        raise RuntimeError(f"Postgres connection failed: {exc}") from exc


# ---------------------------------------------------------------------------
# ApplyAgent
# ---------------------------------------------------------------------------


@agentops.track_agent(agent_type="ApplyAgent")
class ApplyAgent:
    """CrewAI Apply Agent — autonomous job application executor.

    Receives the routing manifest (auto-route jobs only) from the Master
    Agent, runs safety checks on every job, executes Playwright-powered
    form filling, captures proof-of-submission, re-routes failures to the
    manual queue, and enforces the xAI cost cap mid-run.

    Attributes:
        run_batch_id: UUID of the current run batch.
        user_id: UUID of the candidate user.
        routing_manifest: List of job dicts from the AnalyserAgent routing
            manifest, pre-filtered to ``route == "auto"`` by the Master Agent.
        llm_interface: Centralised LLM provider manager.
        llm: Primary CrewAI LLM (xAI grok-4-1-fast-reasoning).
        fallback_llm_1: First fallback LLM (SambaNova Llama-3.1-70B).
        fallback_llm_2: Second fallback LLM (Cerebras llama-3.3-70b).
    """

    def __init__(
        self,
        run_batch_id: str,
        user_id: str,
        routing_manifest: List[Dict[str, Any]],
    ) -> None:
        """Initialise the Apply Agent.

        Args:
            run_batch_id: UUID of the current run batch.
            user_id: UUID of the candidate user.
            routing_manifest: List of job dicts from
                ``AnalyserAgent.get_routing_manifest()`` / the analyser
                phase's ``routing_manifest`` output, pre-filtered to
                auto-route jobs.
        """
        self.run_batch_id: str = run_batch_id
        self.user_id: str = user_id
        self.routing_manifest: List[Dict[str, Any]] = routing_manifest

        self.llm_interface: LLMInterface = LLMInterface()
        self.llm = self.llm_interface.get_llm("APPLY_AGENT")
        self.fallback_llm_1 = self.llm_interface.get_fallback_llm(
            "APPLY_AGENT", level=1
        )
        self.fallback_llm_2 = self.llm_interface.get_fallback_llm(
            "APPLY_AGENT", level=2
        )
        self._current_llm = self.llm
        self._fallback_level: int = 0

        # Live counters
        self._applied_count: int = 0
        self._failed_count: int = 0
        self._rerouted_count: int = 0
        self._budget_aborted: bool = False

        self.logger: logging.Logger = logging.getLogger(self.__class__.__name__)

        self.logger.info(
            "ApplyAgent initialised — run_batch_id=%s user_id=%s jobs=%d",
            run_batch_id,
            user_id,
            len(routing_manifest),
        )

    # ------------------------------------------------------------------
    # Internal: per-platform apply counts
    # ------------------------------------------------------------------

    def _platform_apply_counts(self) -> Dict[str, int]:
        """Query Postgres for per-platform application counts in this run.

        Returns:
            Dict mapping ``source_platform`` → count of ``status='applied'``
            applications for the current ``run_batch_id``.
        """
        conn: Optional[psycopg2.extensions.connection] = None
        try:
            conn = _get_db_conn()
            cursor = conn.cursor(
                cursor_factory=psycopg2.extras.RealDictCursor
            )
            cursor.execute(
                """
                SELECT jp.source_platform, COUNT(*) AS cnt
                FROM applications a
                JOIN job_posts jp ON jp.id = a.job_post_id
                WHERE jp.run_batch_id = %s
                  AND a.status = 'applied'
                GROUP BY jp.source_platform
                """,
                (self.run_batch_id,),
            )
            rows = cursor.fetchall()
            return {
                str(row["source_platform"]): int(str(row["cnt"]))
                for row in rows
            }
        except Exception as exc:  # noqa: BLE001
            self.logger.warning(
                "_platform_apply_counts: query failed (proceeding): %s", exc
            )
            return {}
        finally:
            if conn:
                try:
                    conn.close()
                except Exception:  # noqa: BLE001
                    pass

    # ------------------------------------------------------------------
    # Internal: pre-apply safety check
    # ------------------------------------------------------------------

    def _pre_apply_safety_check(self, job: Dict[str, Any]) -> Dict[str, Any]:
        """Run 4 safety checks before any Playwright session opens.

        Checks:
            1. **Budget gate** — ``check_xai_run_cap`` abort flag.
            2. **Platform limit** — ``max_per_run`` from ``config_limits``.
            3. **Resume exists** — fall back to ``default_resume`` if missing.
            4. **URL valid** — must start with ``http``.

        Args:
            job: Job dict from the routing manifest.

        Returns:
            ``{"proceed": True, "resume_to_use": str, "ats_platform": str}``
            on success, or ``{"proceed": False, "reason": str}`` on failure.
        """
        # Check 1 — Budget gate
        try:
            cap_raw: str = check_xai_run_cap(
                run_batch_id=self.run_batch_id
            )
            cap_result: Dict[str, Any] = {}
            try:
                cap_result = json.loads(cap_raw)
            except (json.JSONDecodeError, TypeError):
                pass

            if cap_result.get("abort", False):
                self._budget_aborted = True
                self.logger.critical(
                    "_pre_apply_safety_check: budget cap hit — aborting"
                )
                return {"proceed": False, "reason": "budget_cap_hit"}
        except Exception as exc:  # noqa: BLE001
            self.logger.warning(
                "_pre_apply_safety_check: budget check failed (proceeding): %s",
                exc,
            )

        # Check 2 — Platform limit
        platform: str = job.get("source_platform", "unknown")
        try:
            config_raw: str = get_platform_config(platform=platform)
            config: Dict[str, Any] = {}
            try:
                config = json.loads(config_raw)
            except (json.JSONDecodeError, TypeError):
                pass

            max_per_run: int = int(config.get("max_per_run", 50))
            current_counts: Dict[str, int] = self._platform_apply_counts()
            platform_count: int = current_counts.get(platform, 0)

            if platform_count >= max_per_run:
                self.logger.warning(
                    "_pre_apply_safety_check: platform %s limit reached "
                    "(%d/%d)",
                    platform,
                    platform_count,
                    max_per_run,
                )
                return {"proceed": False, "reason": "platform_limit_reached"}
        except Exception as exc:  # noqa: BLE001
            self.logger.warning(
                "_pre_apply_safety_check: platform config check failed "
                "(proceeding): %s",
                exc,
            )

        # Check 3 — Resume exists
        resume_suggested: str = job.get(
            "resume_suggested", run_config.default_resume
        )
        resume_path: Path = Path(run_config.resume_dir) / resume_suggested
        if not resume_path.exists():
            self.logger.warning(
                "_pre_apply_safety_check: resume '%s' not found — "
                "falling back to '%s'",
                resume_suggested,
                run_config.default_resume,
            )
            resume_suggested = run_config.default_resume

        # Check 4 — URL valid
        job_url: str = job.get("url", "")
        if not job_url.startswith("http"):
            self.logger.warning(
                "_pre_apply_safety_check: invalid URL '%s'", job_url
            )
            return {"proceed": False, "reason": "invalid_url"}

        return {
            "proceed": True,
            "resume_to_use": resume_suggested,
            "ats_platform": "direct",  # overridden by detect_ats_platform
        }

    # ------------------------------------------------------------------
    # Internal: apply a single job
    # ------------------------------------------------------------------

    def _apply_single_job(self, job: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the full apply flow for a single job posting.

        Runs: safety check → ATS detection → rate limit wait → form fill
        → result handling (applied / re-routed / failed).

        A failure in this method **never** propagates to the caller.
        Failed jobs are re-routed to the manual queue.

        Args:
            job: Job dict from the routing manifest with keys ``id`` (or
                ``job_post_id``), ``title``, ``company``, ``url``,
                ``source_platform``, ``fit_score``, ``resume_suggested``.

        Returns:
            Result dict with ``applied``, ``status``, ``job_post_id``,
            and context keys.
        """
        job_post_id: str = str(
            job.get("job_post_id", job.get("id", ""))
        )
        job_url: str = job.get("url", "")
        job_title: str = job.get("title", "")
        company: str = job.get("company", "")
        platform: str = job.get("source_platform", "unknown")
        fit_score: float = float(job.get("fit_score", 0.0))

        try:
            # Step 1 — Safety check
            safety: Dict[str, Any] = self._pre_apply_safety_check(job)
            if not safety.get("proceed", False):
                reason: str = safety.get("reason", "unknown_safety_failure")

                # Re-route to manual queue
                self._reroute_to_manual(
                    job, reason=reason, job_post_id=job_post_id
                )
                return {
                    "applied": False,
                    "status": "manual_queued",
                    "reason": reason,
                    "job_post_id": job_post_id,
                    "job_url": job_url,
                }

            resume_to_use: str = safety.get(
                "resume_to_use", run_config.default_resume
            )

            # Step 2 — Detect ATS platform
            ats_platform: str = "direct"
            try:
                ats_raw: str = detect_ats_platform(
                    job_url=job_url, run_batch_id=self.run_batch_id
                )
                ats_result: Dict[str, Any] = json.loads(ats_raw)
                ats_platform = ats_result.get("ats", "direct")
            except Exception as exc:  # noqa: BLE001
                self.logger.warning(
                    "_apply_single_job: ATS detection failed for %s: %s",
                    job_url,
                    exc,
                )

            # Step 3 — Rate limit wait
            try:
                config_raw: str = get_platform_config(platform=platform)
                pconfig: Dict[str, Any] = json.loads(config_raw)
                rate_limit: float = float(
                    pconfig.get("rate_limit_per_request_seconds", 3.0)
                )
            except Exception:  # noqa: BLE001
                rate_limit = 3.0

            if rate_limit > 0:
                time.sleep(rate_limit)

            # Step 4 — DRY_RUN gate at agent level
            if run_config.dry_run:
                self.logger.info(
                    "_apply_single_job: DRY_RUN=true — skipping %s at %s",
                    job_title,
                    company,
                )
                log_event(
                    run_batch_id=self.run_batch_id,
                    level="INFO",
                    event_type="dry_run_skip",
                    message=(
                        f"dry_run|{company}|{job_title}|{job_url}"
                    ),
                    job_post_id=job_post_id,
                )
                self._applied_count += 1  # count as applied for reporting
                return {
                    "applied": True,
                    "status": "applied",
                    "dry_run": True,
                    "job_post_id": job_post_id,
                    "job_url": job_url,
                }

            # Step 5 — Execute apply via fill_standard_form
            result_raw: str = fill_standard_form(
                job_url=job_url,
                job_post_id=job_post_id,
                resume_filename=resume_to_use,
                run_batch_id=self.run_batch_id,
                user_id=self.user_id,
                ats_platform=ats_platform,
            )

            result: Dict[str, Any] = {}
            try:
                result = json.loads(result_raw)
            except (json.JSONDecodeError, TypeError):
                self.logger.warning(
                    "_apply_single_job: could not parse fill_standard_form "
                    "response for %s",
                    job_url,
                )

            # Step 6 — Handle result
            if result.get("applied", False):
                self._applied_count += 1
                log_event(
                    run_batch_id=self.run_batch_id,
                    level="INFO",
                    event_type="job_applied",
                    message=(
                        f"Applied: {company} — {job_title} | "
                        f"ats={ats_platform} | "
                        f"proof={result.get('proof_confidence', 'none')}"
                    ),
                    job_post_id=job_post_id,
                )
                self.logger.info(
                    "_apply_single_job: APPLIED %s at %s (proof=%s)",
                    job_title,
                    company,
                    result.get("proof_confidence", "none"),
                )
            elif result.get("re_route") == "manual":
                self._reroute_to_manual(
                    job,
                    reason=result.get("reason", "apply_failed_rerouted"),
                    job_post_id=job_post_id,
                )
            else:
                self._failed_count += 1
                log_event(
                    run_batch_id=self.run_batch_id,
                    level="ERROR",
                    event_type="job_apply_failed",
                    message=(
                        f"Failed: {company} — {job_title} | "
                        f"reason={result.get('reason', 'unknown')}"
                    ),
                    job_post_id=job_post_id,
                )
                self.logger.error(
                    "_apply_single_job: FAILED %s at %s — %s",
                    job_title,
                    company,
                    result.get("reason", "unknown"),
                )

            # Merge job context into result
            result["job_post_id"] = job_post_id
            result["title"] = job_title
            result["company"] = company
            result["source_platform"] = platform
            return result

        except Exception as exc:  # noqa: BLE001
            self.logger.error(
                "_apply_single_job: unhandled exception for %s at %s: %s",
                job_title,
                company,
                exc,
                exc_info=True,
            )
            self._failed_count += 1
            try:
                record_agent_error(
                    agent_type="ApplyAgent",
                    error_message=str(exc),
                    run_batch_id=self.run_batch_id,
                    error_code="SINGLE_JOB_EXCEPTION",
                    job_post_id=job_post_id,
                )
            except Exception:  # noqa: BLE001
                pass
            return {
                "applied": False,
                "status": "failed",
                "error": str(exc),
                "job_post_id": job_post_id,
                "job_url": job_url,
            }

    # ------------------------------------------------------------------
    # Internal: manual queue re-routing
    # ------------------------------------------------------------------

    def _reroute_to_manual(
        self,
        job: Dict[str, Any],
        reason: str,
        job_post_id: str,
    ) -> None:
        """Re-route a job to the manual queue in Notion and Postgres.

        Increments ``_rerouted_count`` and logs a WARNING event.  Failures
        in this method are swallowed — they must never crash the caller.

        Args:
            job: Full job dict from the routing manifest.
            reason: Human-readable reason for the re-route.
            job_post_id: UUID of the job post.
        """
        try:
            self._rerouted_count += 1

            # Persist manual_queued application to Postgres
            try:
                create_application(
                    job_post_id=job_post_id,
                    resume_id="",
                    user_id=self.user_id,
                    mode="manual",
                    status="manual_queued",
                    platform=job.get("source_platform", "unknown"),
                    error_code=reason,
                )
            except Exception as db_exc:  # noqa: BLE001
                self.logger.warning(
                    "_reroute_to_manual: create_application failed: %s",
                    db_exc,
                )

            # Queue to Notion Applications DB
            try:
                queue_job_to_applications_db(
                    job_post_id=job_post_id,
                    run_batch_id=self.run_batch_id,
                    title=job.get("title", ""),
                    company=job.get("company", ""),
                    job_url=job.get("url", ""),
                    platform=job.get("source_platform", "unknown"),
                    fit_score=float(job.get("fit_score", 0.0)),
                    resume_suggested=job.get(
                        "resume_suggested", run_config.default_resume
                    ),
                    notes=f"Auto-apply re-routed: {reason}",
                    location=job.get("location", ""),
                )
            except Exception as notion_exc:  # noqa: BLE001
                self.logger.warning(
                    "_reroute_to_manual: Notion queue failed (non-critical): %s",
                    notion_exc,
                )

            log_event(
                run_batch_id=self.run_batch_id,
                level="WARNING",
                event_type="job_rerouted_to_manual",
                message=(
                    f"Re-routed: {job.get('company', '')} — "
                    f"{job.get('title', '')} | reason={reason}"
                ),
                job_post_id=job_post_id,
            )
            self.logger.warning(
                "_reroute_to_manual: %s at %s → manual queue (%s)",
                job.get("title", ""),
                job.get("company", ""),
                reason,
            )

        except Exception as exc:  # noqa: BLE001
            self.logger.error(
                "_reroute_to_manual: failed for job %s: %s",
                job_post_id,
                exc,
            )

    # ------------------------------------------------------------------
    # Internal: LLM fallback chain
    # ------------------------------------------------------------------

    def _switch_to_fallback(self, failed_provider: str) -> bool:
        """Switch the active LLM to the next fallback in the chain.

        Chain: xAI (primary) → SambaNova (fallback 1) → Cerebras (fallback 2).

        Records the fallback event to AgentOps and Postgres, then updates
        ``self._current_llm`` and ``self._fallback_level``.

        Args:
            failed_provider: Short name or model string of the provider
                that raised the exception.

        Returns:
            ``True`` if a fallback LLM is available and has been activated,
            ``False`` if the fallback chain is exhausted.
        """
        if self._fallback_level == 0 and self.fallback_llm_1 is not None:
            to_model: str = getattr(
                self.fallback_llm_1, "model", "fallback_1"
            )
            record_fallback_event(
                agent_type="ApplyAgent",
                from_provider=failed_provider,
                to_provider=str(to_model),
                run_batch_id=self.run_batch_id,
                fallback_level=1,
                reason=f"Primary provider {failed_provider} failed",
            )
            self._current_llm = self.fallback_llm_1
            self._fallback_level = 1
            self.logger.warning(
                "_switch_to_fallback: level 1 activated — switching to %s",
                to_model,
            )
            return True

        if self._fallback_level == 1 and self.fallback_llm_2 is not None:
            to_model = getattr(self.fallback_llm_2, "model", "fallback_2")
            record_fallback_event(
                agent_type="ApplyAgent",
                from_provider=failed_provider,
                to_provider=str(to_model),
                run_batch_id=self.run_batch_id,
                fallback_level=2,
                reason=f"Fallback-1 provider {failed_provider} failed",
            )
            self._current_llm = self.fallback_llm_2
            self._fallback_level = 2
            self.logger.warning(
                "_switch_to_fallback: level 2 activated — switching to %s",
                to_model,
            )
            return True

        self.logger.critical(
            "_switch_to_fallback: fallback chain exhausted at level %d — "
            "no more providers available for ApplyAgent",
            self._fallback_level,
        )
        return False

    # ------------------------------------------------------------------
    # Agent / Task builders
    # ------------------------------------------------------------------

    def _build_agent(self) -> Agent:
        """Build the CrewAI Agent instance for the apply pass.

        Returns:
            Configured ``crewai.Agent`` using the currently active LLM
            (may be primary or a fallback after ``_switch_to_fallback``).
        """
        return Agent(
            role="Senior Autonomous Job Application Specialist",
            goal=(
                "Execute high-quality, human-like job applications for every "
                "auto-routed job in the manifest, maximise successful "
                "submissions while strictly respecting platform rate limits, "
                "the xAI cost cap, and dry-run mode — re-routing any job "
                "that cannot be safely applied to the manual queue"
            ),
            backstory=(
                "You are an expert at navigating complex ATS platforms, "
                "filling application forms with precision, detecting CAPTCHAs "
                "before they waste credits, and making real-time decisions "
                "about whether to proceed or re-route. You treat every "
                "application as a real opportunity and never submit incomplete "
                "or low-quality applications."
            ),
            llm=self._current_llm,
            tools=[
                detect_ats_platform,
                fill_standard_form,
                check_captcha_present,
                get_apply_summary,
                check_xai_run_cap,
                record_llm_cost,
                get_cost_summary,
                queue_job_to_applications_db,
            ],
            verbose=True,
            max_iter=25,
            memory=False,
            max_rpm=3,
        )

    def _build_task(self, agent: Agent) -> Task:
        """Build the CrewAI Task with detailed application instructions.

        Serialises the full routing manifest JSON in the task description
        so the LLM agent operates with complete job data.

        Args:
            agent: The ``ApplyAgent`` CrewAI agent that will execute the task.

        Returns:
            Configured ``crewai.Task`` ready for crew execution.
        """
        manifest_json: str
        try:
            manifest_json = json.dumps(
                self.routing_manifest, default=str, indent=2
            )
        except Exception as exc:  # noqa: BLE001
            self.logger.warning(
                "_build_task: JSON serialisation failed: %s", exc
            )
            manifest_json = "[]"

        dry_run_flag: bool = run_config.dry_run
        xai_cap: float = budget_config.xai_cost_cap_per_run

        description: str = f"""
You are applying to {len(self.routing_manifest)} auto-routed jobs for the current run batch.
run_batch_id: {self.run_batch_id}
dry_run: {dry_run_flag}

APPLICATION INSTRUCTIONS
========================

1. For EACH job in the routing manifest below, run a pre-apply safety check
   before launching any Playwright session:
   - Check xAI budget via check_xai_run_cap
   - Verify platform rate limits
   - Confirm resume file exists
   - Validate job URL

2. If DRY_RUN={dry_run_flag}: log each job as "dry_run_skip", count as
   applied for reporting purposes, and NEVER open a browser.

3. Detect the ATS platform for each job via detect_ats_platform before
   submitting — adapt your approach per platform.

4. Check for CAPTCHA indicators before submitting. If detected, skip to
   manual queue IMMEDIATELY — do not waste credits.

5. Call fill_standard_form for each auto-route job ONE AT A TIME — NEVER
   run parallel applications.

6. After EVERY 5 applications, call check_xai_run_cap with
   run_batch_id={self.run_batch_id}. If the response contains
   "abort": true — STOP immediately and re-queue ALL remaining jobs
   to the manual queue via queue_job_to_applications_db.

7. Enforce per-platform rate limits using the platform config's
   rate_limit_per_request_seconds. Sleep between requests.

8. If a job fails to apply after 2 retries (handled by fill_standard_form) —
   re-route to manual queue via queue_job_to_applications_db.

9. After processing ALL jobs, call get_apply_summary with
   run_batch_id={self.run_batch_id} for final counts.

10. Return a complete application manifest.

Budget hard cap per run: ${xai_cap}

ROUTING MANIFEST (JSON)
=======================
{manifest_json}
"""

        return Task(
            description=description,
            expected_output=(
                'JSON: {"total_attempted": int, "applied": int, '
                '"failed": int, "rerouted_to_manual": int, '
                '"budget_aborted": bool, "platform_breakdown": dict, '
                '"dry_run": bool}'
            ),
            agent=agent,
        )

    # ------------------------------------------------------------------
    # Internal: crew output parser
    # ------------------------------------------------------------------

    def _parse_crew_output(self, crew_output: Any) -> Dict[str, Any]:
        """Extract the application results dict from a CrewAI ``CrewOutput``.

        Handles ``CrewOutput`` (with ``.raw``), plain string, and dict.

        Args:
            crew_output: Raw return value of ``Crew.kickoff()``.

        Returns:
            Parsed result dict.  Returns a safe default if parsing fails.
        """
        default: Dict[str, Any] = {
            "total_attempted": 0,
            "applied": 0,
            "failed": 0,
            "rerouted_to_manual": 0,
            "budget_aborted": False,
        }

        if crew_output is None:
            self.logger.warning("_parse_crew_output: crew_output is None")
            return default

        raw_str: str = ""
        if hasattr(crew_output, "raw"):
            raw_str = str(crew_output.raw or "")
        elif isinstance(crew_output, str):
            raw_str = crew_output
        elif isinstance(crew_output, dict):
            return crew_output
        else:
            raw_str = str(crew_output)

        if not raw_str.strip():
            self.logger.warning(
                "_parse_crew_output: crew returned empty output"
            )
            return default

        # Strip markdown code fences if present
        stripped: str = raw_str.strip()
        if stripped.startswith("```"):
            lines: List[str] = stripped.splitlines()
            inner_lines: List[str] = lines[1:] if len(lines) > 1 else lines
            if inner_lines and inner_lines[-1].strip().startswith("```"):
                inner_lines = inner_lines[:-1]
            stripped = "\n".join(inner_lines).strip()

        try:
            parsed: Any = json.loads(stripped)
            if isinstance(parsed, dict):
                return parsed
            self.logger.warning(
                "_parse_crew_output: parsed JSON is not a dict (type=%s)",
                type(parsed).__name__,
            )
            return default
        except (json.JSONDecodeError, ValueError) as exc:
            self.logger.warning(
                "_parse_crew_output: JSON decode failed: %s — snippet: %.200s",
                exc,
                raw_str,
            )
            return default

    # ------------------------------------------------------------------
    # Internal: reconcile safety net
    # ------------------------------------------------------------------

    def _reconcile_manifest(
        self, results: List[Dict[str, Any]]
    ) -> None:
        """Ensure every job in the manifest has a Postgres application record.

        Any job that has no corresponding result is created with
        ``status='manual_queued'`` as a safety net — no job left behind.

        Args:
            results: List of per-job result dicts from ``_apply_single_job``.
        """
        completed_ids: set[str] = set()
        for r in results:
            jid: str = str(r.get("job_post_id", ""))
            if jid:
                completed_ids.add(jid)

        for job in self.routing_manifest:
            job_post_id: str = str(
                job.get("job_post_id", job.get("id", ""))
            )
            if job_post_id and job_post_id not in completed_ids:
                self.logger.warning(
                    "_reconcile_manifest: job %s has no application record "
                    "— creating safety-net manual_queued entry",
                    job_post_id,
                )
                try:
                    create_application(
                        job_post_id=job_post_id,
                        resume_id="",
                        user_id=self.user_id,
                        mode="manual",
                        status="manual_queued",
                        platform=job.get("source_platform", "unknown"),
                        error_code="safety_net_no_result",
                    )
                    self._rerouted_count += 1
                except Exception as exc:  # noqa: BLE001
                    self.logger.error(
                        "_reconcile_manifest: safety-net create_application "
                        "failed for %s: %s",
                        job_post_id,
                        exc,
                    )

    # ------------------------------------------------------------------
    # Public: main run method
    # ------------------------------------------------------------------

    @agentops.track_tool
    def run(self) -> Dict[str, Any]:
        """Execute the full apply pass for all auto-route jobs.

        Lifecycle:
            1. Log run start.
            2. Early-return if manifest empty.
            3. Pre-check monthly budget.
            4. Sort manifest by ``fit_score`` descending.
            5. Build CrewAI agent and task.
            6. Execute crew with fallback chain.
            7. Parse crew result.
            8. Process each job individually via ``_apply_single_job``.
            9. Reconcile — safety-net for any orphaned jobs.
            10. Log run complete.
            11. Return structured result dict.

        Returns:
            Dict with ``success``, counts, and cost summary.
            On unhandled exception: ``{"success": False, "error": str}``.
        """
        try:
            # ----------------------------------------------------------
            # Step 1: log run start
            # ----------------------------------------------------------
            log_event(
                run_batch_id=self.run_batch_id,
                level="INFO",
                event_type="apply_run_start",
                message=(
                    f"Apply Agent starting | "
                    f"{len(self.routing_manifest)} jobs in manifest | "
                    f"dry_run={run_config.dry_run}"
                ),
            )
            self.logger.info(
                "ApplyAgent.run: starting — %d jobs | dry_run=%s",
                len(self.routing_manifest),
                run_config.dry_run,
            )

            # ----------------------------------------------------------
            # Step 2: early-return if empty
            # ----------------------------------------------------------
            if not self.routing_manifest:
                self.logger.info(
                    "ApplyAgent.run: empty manifest — returning early"
                )
                return {
                    "success": True,
                    "reason": "empty_manifest",
                    "applied": 0,
                    "failed": 0,
                    "rerouted_to_manual": 0,
                }

            # ----------------------------------------------------------
            # Step 3: pre-check monthly budget
            # ----------------------------------------------------------
            try:
                from tools.budget_tools import check_monthly_budget

                budget_raw: str = check_monthly_budget(
                    run_batch_id=self.run_batch_id
                )
                budget_result: Dict[str, Any] = {}
                try:
                    budget_result = json.loads(budget_raw)
                except (json.JSONDecodeError, TypeError):
                    pass

                if budget_result.get("abort", False):
                    self.logger.critical(
                        "ApplyAgent.run: monthly budget exceeded — "
                        "re-routing entire manifest to manual"
                    )
                    self._budget_aborted = True
                    # Re-route all jobs
                    for job in self.routing_manifest:
                        jid: str = str(
                            job.get("job_post_id", job.get("id", ""))
                        )
                        self._reroute_to_manual(
                            job,
                            reason="monthly_budget_exceeded",
                            job_post_id=jid,
                        )
                    return {
                        "success": True,
                        "run_batch_id": self.run_batch_id,
                        "total_attempted": 0,
                        "applied": 0,
                        "failed": 0,
                        "rerouted_to_manual": self._rerouted_count,
                        "budget_aborted": True,
                        "dry_run": run_config.dry_run,
                        "platform_breakdown": {},
                        "cost_summary": {},
                    }
            except Exception as exc:  # noqa: BLE001
                self.logger.warning(
                    "ApplyAgent.run: monthly budget pre-check failed "
                    "(proceeding): %s",
                    exc,
                )

            # ----------------------------------------------------------
            # Step 4: sort by fit_score descending
            # ----------------------------------------------------------
            self.routing_manifest.sort(
                key=lambda j: float(j.get("fit_score", 0.0)),
                reverse=True,
            )

            # ----------------------------------------------------------
            # Steps 5+6: build crew and execute with fallback
            # ----------------------------------------------------------
            agent: Agent = self._build_agent()
            task: Task = self._build_task(agent)
            crew = Crew(
                agents=[agent],
                tasks=[task],
                process=Process.sequential,
                verbose=True,
            )

            crew_output: Any = None

            # First attempt
            try:
                self.logger.info(
                    "ApplyAgent.run: executing crew (primary LLM)…"
                )
                crew_output = crew.kickoff()
            except Exception as primary_exc:  # noqa: BLE001
                failed_model: str = getattr(
                    self._current_llm, "model", "primary"
                )
                self.logger.error(
                    "ApplyAgent.run: primary LLM failed (%s): %s",
                    failed_model,
                    primary_exc,
                )
                switched: bool = self._switch_to_fallback(failed_model)

                if switched:
                    # Rebuild with new LLM and retry once
                    agent = self._build_agent()
                    task = self._build_task(agent)
                    retry_crew = Crew(
                        agents=[agent],
                        tasks=[task],
                        process=Process.sequential,
                        verbose=True,
                    )
                    try:
                        fallback_model: str = getattr(
                            self._current_llm,
                            "model",
                            f"fallback_{self._fallback_level}",
                        )
                        self.logger.info(
                            "ApplyAgent.run: retrying crew with fallback "
                            "LLM %s",
                            fallback_model,
                        )
                        crew_output = retry_crew.kickoff()
                    except Exception as fallback_exc:  # noqa: BLE001
                        self.logger.critical(
                            "ApplyAgent.run: fallback LLM also failed: %s",
                            fallback_exc,
                        )
                        record_agent_error(
                            agent_type="ApplyAgent",
                            error_message=str(fallback_exc),
                            run_batch_id=self.run_batch_id,
                            error_code="LLM_FALLBACK_FAILED",
                        )
                        # Fall through to programmatic execution
                        crew_output = None
                else:
                    self.logger.critical(
                        "ApplyAgent.run: no fallback available — "
                        "proceeding with programmatic execution"
                    )
                    record_agent_error(
                        agent_type="ApplyAgent",
                        error_message=str(primary_exc),
                        run_batch_id=self.run_batch_id,
                        error_code="LLM_ALL_PROVIDERS_FAILED",
                    )
                    crew_output = None

            # ----------------------------------------------------------
            # Step 7: parse crew result (informational only)
            # ----------------------------------------------------------
            if crew_output is not None:
                crew_parsed: Dict[str, Any] = self._parse_crew_output(
                    crew_output
                )
                self.logger.info(
                    "ApplyAgent.run: crew output parsed — %s",
                    {
                        k: v
                        for k, v in crew_parsed.items()
                        if k != "platform_breakdown"
                    },
                )

            # ----------------------------------------------------------
            # Step 8: process each job programmatically
            # ----------------------------------------------------------
            per_job_results: List[Dict[str, Any]] = []

            for idx, job in enumerate(self.routing_manifest, start=1):
                # Budget abort check every 5 jobs
                if self._budget_aborted:
                    self.logger.warning(
                        "ApplyAgent.run: budget aborted — re-routing "
                        "remaining %d jobs",
                        len(self.routing_manifest) - idx + 1,
                    )
                    remaining_jobs: List[Dict[str, Any]] = (
                        self.routing_manifest[idx - 1:]
                    )
                    for rjob in remaining_jobs:
                        rjid: str = str(
                            rjob.get("job_post_id", rjob.get("id", ""))
                        )
                        self._reroute_to_manual(
                            rjob,
                            reason="budget_cap_hit_mid_run",
                            job_post_id=rjid,
                        )
                    break

                # Periodic budget check every 5 jobs
                if idx > 1 and idx % 5 == 1:
                    try:
                        cap_raw: str = check_xai_run_cap(
                            run_batch_id=self.run_batch_id
                        )
                        cap_check: Dict[str, Any] = json.loads(cap_raw)
                        if cap_check.get("abort", False):
                            self._budget_aborted = True
                            self.logger.critical(
                                "ApplyAgent.run: xAI budget cap hit after "
                                "%d applications — aborting",
                                idx - 1,
                            )
                            # Re-route this job and all remaining
                            remaining = self.routing_manifest[idx - 1:]
                            for rjob in remaining:
                                rjid = str(
                                    rjob.get("job_post_id", rjob.get("id", ""))
                                )
                                self._reroute_to_manual(
                                    rjob,
                                    reason="xai_cap_hit_mid_run",
                                    job_post_id=rjid,
                                )
                            break
                    except Exception as cap_exc:  # noqa: BLE001
                        self.logger.warning(
                            "ApplyAgent.run: periodic budget check failed "
                            "(proceeding): %s",
                            cap_exc,
                        )

                # Apply single job (fail-soft)
                job_result: Dict[str, Any] = self._apply_single_job(job)
                per_job_results.append(job_result)

            # ----------------------------------------------------------
            # Step 9: reconcile — safety net
            # ----------------------------------------------------------
            self._reconcile_manifest(per_job_results)

            # ----------------------------------------------------------
            # Step 10: build platform breakdown
            # ----------------------------------------------------------
            platform_breakdown: Dict[str, Dict[str, int]] = {}
            for r in per_job_results:
                plat: str = r.get("source_platform", "unknown")
                if plat not in platform_breakdown:
                    platform_breakdown[plat] = {
                        "applied": 0,
                        "failed": 0,
                        "rerouted": 0,
                    }
                status: str = r.get("status", "failed")
                if status == "applied" or r.get("applied", False):
                    platform_breakdown[plat]["applied"] += 1
                elif status == "manual_queued":
                    platform_breakdown[plat]["rerouted"] += 1
                else:
                    platform_breakdown[plat]["failed"] += 1

            # ----------------------------------------------------------
            # Step 11: get final cost summary
            # ----------------------------------------------------------
            cost_summary: Dict[str, Any] = {}
            try:
                cost_raw: str = get_cost_summary(
                    run_batch_id=self.run_batch_id
                )
                cost_summary = json.loads(cost_raw)
            except Exception as cost_exc:  # noqa: BLE001
                self.logger.warning(
                    "ApplyAgent.run: get_cost_summary failed: %s", cost_exc
                )

            # ----------------------------------------------------------
            # Step 12: log run complete
            # ----------------------------------------------------------
            summary_msg: str = (
                f"Applied={self._applied_count} | "
                f"Rerouted={self._rerouted_count} | "
                f"Failed={self._failed_count} | "
                f"Budget_aborted={self._budget_aborted}"
            )
            log_event(
                run_batch_id=self.run_batch_id,
                level="INFO",
                event_type="apply_run_complete",
                message=summary_msg,
            )
            self.logger.info("ApplyAgent.run: %s", summary_msg)

            # ----------------------------------------------------------
            # Step 13: return result
            # ----------------------------------------------------------
            return {
                "success": True,
                "run_batch_id": self.run_batch_id,
                "total_attempted": len(per_job_results),
                "applied": self._applied_count,
                "failed": self._failed_count,
                "rerouted_to_manual": self._rerouted_count,
                "budget_aborted": self._budget_aborted,
                "dry_run": run_config.dry_run,
                "platform_breakdown": platform_breakdown,
                "cost_summary": cost_summary,
            }

        except Exception as exc:  # noqa: BLE001
            self.logger.critical(
                "ApplyAgent.run: unhandled exception: %s",
                exc,
                exc_info=True,
            )
            try:
                record_agent_error(
                    agent_type="ApplyAgent",
                    error_message=str(exc),
                    run_batch_id=self.run_batch_id,
                    error_code="APPLY_UNHANDLED_EXCEPTION",
                )
            except Exception:  # noqa: BLE001
                pass

            # Re-queue entire remaining manifest as safety net
            for job in self.routing_manifest:
                jid = str(job.get("job_post_id", job.get("id", "")))
                try:
                    create_application(
                        job_post_id=jid,
                        resume_id="",
                        user_id=self.user_id,
                        mode="manual",
                        status="manual_queued",
                        platform=job.get("source_platform", "unknown"),
                        error_code="unhandled_exception_safety_net",
                    )
                except Exception:  # noqa: BLE001
                    pass

            return {
                "success": False,
                "error": str(exc),
                "run_batch_id": self.run_batch_id,
                "applied": self._applied_count,
                "failed": self._failed_count,
                "rerouted_to_manual": self._rerouted_count,
            }

    # ------------------------------------------------------------------
    # Public: live counter accessor
    # ------------------------------------------------------------------

    def get_apply_results(self) -> Dict[str, Any]:
        """Return current live counters for phase reporting.

        Called by the Master Agent to extract intermediate results
        during or after the apply pass.

        Returns:
            Dict with ``applied``, ``failed``, ``rerouted``, and
            ``budget_aborted`` keys.
        """
        return {
            "applied": self._applied_count,
            "failed": self._failed_count,
            "rerouted": self._rerouted_count,
            "budget_aborted": self._budget_aborted,
        }
