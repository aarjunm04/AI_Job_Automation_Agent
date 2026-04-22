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
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from typing import Optional as Opt

from pydantic import BaseModel, Field

from crewai import Agent, Task, Crew, Process, LLM
import agentops
from agentops import agent, operation
import psycopg2
import psycopg2.extras

from config.settings import db_config, run_config, budget_config
from config.config_loader import config_loader as _cfg
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
    _create_application,
    update_application_status,
    _update_application_status,
    log_event,
    _log_event,
    get_platform_config,
    _get_platform_config,
    _priority_text,
)
from tools.budget_tools import (
    check_xai_run_cap,
    record_llm_cost,
    get_cost_summary,
    register_litellm_callback,
    init_budget_run,
)
from tools.agentops_tools import (
    record_agent_error,
    _record_agent_error,
    record_fallback_event,
    _record_fallback_event,
)
from utils.db_utils import get_db_conn

# Module-level logger
logger = logging.getLogger(__name__)

__all__ = ["ApplyAgent"]


class SingleJobApplyResult(BaseModel):
    """Pydantic output schema enforced by CrewAI for a single-job application.

    Setting output_pydantic=SingleJobApplyResult on the CrewAI Task forces the
    LLM to return a schema-validated object. Any free-text or hallucinated
    Final Answer that does not parse against this schema is rejected by CrewAI
    and triggers a retry, eliminating the main hallucination vector.
    """

    job_post_id: str = Field(..., description="UUID of the job post applied to")
    status: str = Field(
        ...,
        description=(
            "One of: applied, failed, manual_queued, dry_run_skip, captcha_blocked"
        ),
    )
    resume_used: str = Field(..., description="Filename of resume submitted")
    error_code: Opt[str] = Field(
        None, description="Error code string if status=failed, else null"
    )
    platform: str = Field(
        ..., description="ATS platform detected by detect_ats_platform"
    )
    applied: bool = Field(
        ..., description="True only if fill_standard_form returned applied=True"
    )


# ---------------------------------------------------------------------------
# ApplyAgent
# ---------------------------------------------------------------------------


class ApplyAgent:
    """CrewAI Apply Agent — autonomous job application executor.

    Loads auto-route jobs directly from Postgres for the given
    ``pipeline_run_id``, runs safety checks on every job, executes
    Playwright-powered form filling, captures proof-of-submission,
    re-routes failures to the manual queue, and enforces the xAI cost
    cap mid-run.

    All agent-to-agent state is passed via Postgres — no in-memory
    manifest objects are accepted from callers.

    Attributes:
        pipeline_run_id: UUID of the current run batch.
        user_id: UUID of the candidate user.
        routing_manifest: List of auto-route job dicts loaded from Postgres
            (``jobs JOIN job_scores WHERE route = 'auto'``).
        llm_interface: Centralised LLM provider manager.
        llm: Primary CrewAI LLM (xAI grok-4-1-fast-reasoning).
        fallback_llm_1: First fallback LLM (SambaNova Llama-3.1-70B).
        fallback_llm_2: Second fallback LLM (Cerebras llama-3.3-70b).
    """

    def __init__(
        self,
        pipeline_run_id: str,
        user_id: str,
    ) -> None:
        """Initialise the Apply Agent.

        Loads the routing manifest (auto-route jobs) directly from
        Postgres using ``pipeline_run_id``.  No in-memory manifest is
        accepted — all state passes via the database.

        Args:
            pipeline_run_id: UUID of the current run batch.
            user_id: UUID of the candidate user.
        """
        self.pipeline_run_id: str = pipeline_run_id
        self.user_id: str = user_id

        self.llm_interface: LLMInterface = LLMInterface()
        self.llm = self.llm_interface.get_llm("APPLY_AGENT")
        self.fallback_llm_1 = self.llm_interface.get_fallback_llm(
            "APPLY_AGENT", level=1
        )
        self.fallback_llm_2 = self.llm_interface.get_fallback_llm(
            "APPLY_AGENT", level=2
        )
        if self.fallback_llm_2 is None:
            _groq_model: str = os.getenv("GROQ_MODEL", "")
            _groq_key: str = os.getenv("GROQ_API_KEY", "")
            if _groq_model and _groq_key:
                self.fallback_llm_2 = LLM(
                    model=f"groq/{_groq_model}",
                    api_key=_groq_key,
                    temperature=float(
                        os.getenv("APPLY_LLM_TEMPERATURE", "0.1")
                    ),
                )
                logger.info(
                    "ApplyAgent.__init__: fallback_2 wired via Groq — %s",
                    _groq_model,
                )
            else:
                logger.warning(
                    "ApplyAgent.__init__: fallback_2 unavailable — "
                    "GROQ_MODEL or GROQ_API_KEY not set. "
                    "Only 1 LLM fallback active for this run."
                )
        self._current_llm = self.llm
        self._fallback_level: int = 0

        # Live counters
        self._applied_count: int = 0
        self._failed_count: int = 0
        self._rerouted_count: int = 0
        self._budget_aborted: bool = False
        self._consecutive_failures: int = 0

        self.logger: logging.Logger = logging.getLogger(self.__class__.__name__)

        # Load all runtime config from JSON — single read at init, never repeated
        self._run_cfg: dict = _cfg.get_run_config()
        self._apply_cfg: dict = _cfg.get_apply_settings()
        self._score_cfg: dict = _cfg.get_scoring_thresholds()
        self._budget_cfg: dict = _cfg.get_budget_settings()
        self._job_filters: dict = _cfg.get_job_filters()

        # Load routing manifest from Postgres (auto-route jobs only)
        self.routing_manifest: List[Dict[str, Any]] = (
            self._load_routing_manifest()
        )

        self.logger.info(
            "ApplyAgent initialised — pipeline_run_id=%s user_id=%s jobs=%d",
            pipeline_run_id,
            user_id,
            len(self.routing_manifest),
        )

        # Initialize budget tracking and register LiteLLM cost callback
        init_budget_run(pipeline_run_id)
        register_litellm_callback()

    # ------------------------------------------------------------------
    # Internal: load routing manifest from Postgres
    # ------------------------------------------------------------------

    def _load_routing_manifest(self) -> List[Dict[str, Any]]:
        """Load auto-route jobs from Postgres for the current run batch.

        Queries ``jobs JOIN job_scores`` where the analyser assigned
        ``route = 'auto'`` and the job passed eligibility.  Results are
        ordered by ``fit_score DESC`` so the highest-confidence jobs are
        processed first.

        Every job dict is guaranteed to have a non-empty ``resume_suggested``
        value — falling back to ``DEFAULT_RESUME_PATH`` env var or the first
        entry in ``config/resume_config.json`` when the DB column is null.

        Returns:
            List of job dicts ready for the apply pipeline.  Empty list
            on query failure (fail-soft).
        """
        for attempt in range(1, self._run_cfg.get("max_retries", 3) + 1):
            conn: Optional[psycopg2.extensions.connection] = None
            try:
                conn = get_db_conn()
                cursor = conn.cursor(
                    cursor_factory=psycopg2.extras.RealDictCursor
                )
                cursor.execute(
                    """
                    SELECT
                        jp.id,
                        jp.title,
                        jp.company,
                        jp.url,
                        jp.source_platform,
                        js.fit_score,
                        js.resume_id,
                        js.eligibility_pass
                    FROM jobs jp
	                    JOIN job_scores js ON js.job_post_id = jp.id
	                    WHERE jp.pipeline_run_id = %s
	                      AND js.eligibility_pass = TRUE
	                      AND js.fit_score >= %s
	                    ORDER BY js.fit_score DESC
	                    LIMIT %s
	                    """,
	                    (
	                        self.pipeline_run_id,
	                        self._score_cfg.get("fit_score_auto_min", 0.6),
	                        self._run_cfg.get("max_auto_apply_per_run", 20),
	                    ),
	                )
                rows = cursor.fetchall()
                manifest: List[Dict[str, Any]] = []
                for row in rows:
                    # BUG-FIX 3B: resolve resume_suggested — never leave empty
                    resume_suggested: str = (
                        row.get("resume_id")
                        or row.get("resume_suggested")
                        or ""
                    )
                    if not resume_suggested:
                        resume_suggested = (
                            os.getenv("DEFAULT_RESUME_PATH", "")
                            or self._get_default_resume_path()
                        )
                    manifest.append(
                        {
                            "id": str(row["id"]),
                            "job_post_id": str(row["id"]),
                            "title": row["title"],
                            "company": row["company"],
                            "url": row["url"],
                            "source_platform": row["source_platform"],
                            "fit_score": float(row["fit_score"] or 0.0),
                            "resume_suggested": resume_suggested,
                            "eligibility_pass": bool(row["eligibility_pass"]),
                            "route": "auto",
                        }
                    )
                self.logger.info(
                    "_load_routing_manifest: loaded %d auto-route jobs "
                    "for batch %s",
                    len(manifest),
                    self.pipeline_run_id,
                )
                return manifest
            except Exception as exc:  # noqa: BLE001
                max_retries = self._run_cfg.get("max_retries", 3)
                if attempt < max_retries:
                    time.sleep(min(2 ** attempt, self._run_cfg.get("max_retry_backoff_seconds", 30)))
                    self.logger.warning(
                        "_load_routing_manifest attempt %d/%d failed: %s "
                        "— retrying",
                        attempt,
                        max_retries,
                        str(exc),
                    )
                else:
                    self.logger.error(
                        "_load_routing_manifest: failed after %d attempts: %s",
                        max_retries,
                        exc,
                    )
            finally:
                if conn:
                    try:
                        conn.close()
                    except Exception:  # noqa: BLE001
                        pass
        return []

    # ------------------------------------------------------------------
    # Internal: default resume path resolver (Bug 3B helper)
    # ------------------------------------------------------------------

    def _get_default_resume_path(self) -> str:
        """Return the first available resume file path from config or env.

        Reads ``config/resume_config.json`` for a ``resumes`` list and
        returns the ``file_path`` of the first entry.  Falls back to the
        ``DEFAULT_RESUME_PATH`` env var, then a hard-coded sentinel.

        Returns:
            Non-empty resume file path string, or the sentinel value
            ``'resumes/Aarjun_AIAutomation.pdf'`` if nothing is configured.
        """
        try:
            cfg_path = os.path.join(
                os.path.dirname(__file__), "..", "config", "resume_config.json"
            )
            with open(cfg_path, "r", encoding="utf-8") as fh:
                cfg = json.load(fh)
            resumes: List[Dict[str, Any]] = cfg.get("resumes", [])
            if resumes:
                return str(resumes[0].get("file_path", ""))
        except Exception:  # noqa: BLE001
            pass
        return os.getenv("DEFAULT_RESUME_PATH", "resumes/Aarjun_AIAutomation.pdf")

    # ------------------------------------------------------------------
    # Internal: per-platform apply counts
    # ------------------------------------------------------------------

    def _platform_apply_counts(self) -> Dict[str, int]:
        """Compute per-platform application counts from Postgres for this run.

        Queries the ``applications`` table joined with ``jobs`` to count
        the number of successfully applied applications grouped by
        ``source_platform`` for the current ``pipeline_run_id``.
        Used by ``_pre_apply_safety_check`` to enforce per-platform
        rate limits (``max_per_run``).

        Returns:
            Dict mapping platform name (``str``) to count of
            ``status='applied'`` applications (``int``).  Returns an
            empty dict on query failure (fail-soft).
        """
        max_retries: int = self._run_cfg.get("max_retries", 3)
        for attempt in range(1, max_retries + 1):
            conn: Optional[psycopg2.extensions.connection] = None
            try:
                conn = get_db_conn()
                cursor = conn.cursor(
                    cursor_factory=psycopg2.extras.RealDictCursor
                )
                cursor.execute(
                    """
                    SELECT jp.source_platform, COUNT(*) AS cnt
                    FROM applications a
                    JOIN jobs jp ON jp.id = a.job_post_id
                    WHERE jp.pipeline_run_id = %s
                      AND a.status = 'applied'
                    GROUP BY jp.source_platform
                    """,
                    (self.pipeline_run_id,),
                )
                rows = cursor.fetchall()
                return {
                    str(row["source_platform"]): int(str(row["cnt"]))
                    for row in rows
                }
            except Exception as exc:  # noqa: BLE001
                if attempt < max_retries:
                    time.sleep(min(2 ** attempt, self._run_cfg.get("max_retry_backoff_seconds", 30)))
                    self.logger.warning(
                        "_platform_apply_counts attempt %d/%d failed: %s — retrying",
                        attempt, max_retries, str(exc),
                    )
                else:
                    self.logger.warning(
                        "_platform_apply_counts: query failed (proceeding): %s", exc
                    )
            finally:
                if conn:
                    try:
                        conn.close()
                    except Exception:  # noqa: BLE001
                        pass
        return {}

    # ------------------------------------------------------------------
    # Internal: ATS detection from URL
    # ------------------------------------------------------------------

    def _detect_ats(self, url: str) -> str:
        """Detect ATS platform from job URL.

        Args:
            url: Job application URL.

        Returns:
            ATS platform string: greenhouse, lever, workable, linkedin,
            or unknown.
        """
        url_lower = url.lower()
        if "greenhouse.io" in url_lower:
            return "greenhouse"
        if "lever.co" in url_lower:
            return "lever"
        if "workable.com" in url_lower:
            return "workable"
        if "linkedin.com/jobs" in url_lower:
            return "linkedin"
        return "unknown"

    # ------------------------------------------------------------------
    # Internal: atomic DB write for applications + queued_jobs
    # ------------------------------------------------------------------

    def _write_application(
        self,
        job: Dict[str, Any],
        status: str,
        mode: str,
        error_code: Optional[str] = None,
        resume_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> str:
        """Write application record atomically. Returns application UUID.

        Wraps applications INSERT + queued_jobs INSERT (when manual_queued)
        in a single transaction via get_db_conn().

        Args:
            job: Job dict with at minimum 'id' and 'url' keys.
            status: Application status (applied, failed, manual_queued).
            mode: Application mode (auto, manual).
            error_code: Optional error code for failed/queued applications.
            resume_id: Optional UUID of the resume used.
            user_id: UUID of the candidate user.  Resolved from ``self.user_id``
                when not explicitly supplied.  Never defaults to a literal string.

        Returns:
            Application UUID string, or empty string on conflict/failure.
        """
        # BUG-FIX 2B: resolve user_id from self; reject any literal placeholder
        resolved_user_id: str = user_id or self.user_id
        if not resolved_user_id or resolved_user_id == "default_user":
            self.logger.error(
                "_write_application: invalid user_id '%s' — aborting write "
                "for job %s",
                resolved_user_id,
                job.get("job_post_id", job.get("id", "")),
            )
            return ""

        conn = get_db_conn()
        try:
            with conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """INSERT INTO applications
                           (job_post_id, resume_id, user_id, mode,
                            status, platform, error_code, notion_synced,
                            notion_synced_at, proof_json)
                           VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                           ON CONFLICT DO NOTHING
                           RETURNING id""",
                        (
                            job.get("id", job.get("job_post_id", "")),
                            resume_id,
                            resolved_user_id,
                            mode,
                            status,
                            self._detect_ats(job.get("url", "")),
                            error_code,
                            False,
                            None,
                            None,
                        ),
                    )
                    row = cur.fetchone()
                    if not row:
                        return ""
                    application_id = str(row[0])
                    if status == "manual_queued":
                        cur.execute(
                            """INSERT INTO queued_jobs
                               (application_id, job_post_id, priority, notes)
                               VALUES (%s, %s, %s, %s)""",
                            (
                                application_id,
                                job.get("id", job.get("job_post_id", "")),
                                _priority_text(5),
                                error_code,
                            ),
                        )
            self.logger.info(
                "_write_application: wrote job %s status=%s user=%s",
                job.get("job_post_id", job.get("id", "")),
                status,
                resolved_user_id,
            )
            return application_id
        except Exception as exc:
            self.logger.error(
                "_write_application: transaction failed for job %s: %s",
                job.get("id", ""),
                exc,
            )
            return ""
        finally:
            conn.close()

    # ------------------------------------------------------------------

    def _count_tool_calls(self, result: Any) -> dict[str, int]:
        """Count actual tool invocations from CrewAI task output.

        Returns mapping of tool_name -> call_count.
        """
        counts: dict[str, int] = {}
        try:
            # CrewAI stores tool usage in result.tasks_output
            for task_out in getattr(result, "tasks_output", []):
                for tool_call in getattr(task_out, "tool_calls", []):
                    name: str = getattr(tool_call, "name", "") or ""
                    if name:
                        counts[name] = counts.get(name, 0) + 1
        except Exception as exc:
            self.logger.warning("_count_tool_calls: parse failed — %s", exc)
        return counts


    def _kickoff_with_timeout(
        self, crew: Any, timeout_seconds: int = 300
    ) -> Any:
        """Run crew.kickoff() with a hard wall-clock timeout.

        Raises TimeoutError if the crew does not complete in time.
        """
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            future = ex.submit(crew.kickoff)
            try:
                return future.result(timeout=float(timeout_seconds))
            except concurrent.futures.TimeoutError:
                self.logger.error(
                    "_kickoff_with_timeout: crew timed out after %ds — "
                    "aborting job", timeout_seconds
                )
                raise TimeoutError(
                    f"crew.kickoff() exceeded {timeout_seconds}s limit"
                )

    # Internal: pre-apply safety check
    # ------------------------------------------------------------------

    def _pre_apply_safety_check(self, job: Dict[str, Any]) -> Dict[str, Any]:
        """Run 4 safety checks before any Playwright session opens.

        Checks:
            1. **Budget gate** — ``check_xai_run_cap`` abort flag.
            2. **Platform limit** — ``max_per_run`` from ``system_config``.
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
                pipeline_run_id=self.pipeline_run_id
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
            config_raw: str = _get_platform_config(platform=platform)
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
        resume_dir = os.getenv("RESUME_DIR", "app/resumes")
        # Strip any leading directory prefix from filename before joining
        resume_basename = os.path.basename(resume_suggested)
        resume_path: Path = Path(resume_dir) / resume_basename
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

        Runs: DRY_RUN guard (FIRST) → ATS detection → safety check →
        rate limit wait → form fill → result handling.

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
            # ── STEP 1 — DRY_RUN guard (ALWAYS FIRST) ────────────────
            dry_run: bool = (
                os.getenv("DRY_RUN").lower()
                or self._apply_cfg.get("dry_run", False)
            )
            if dry_run:
                self.logger.info(
                    "[DRY_RUN] Would apply to %s at %s — skipping browser",
                    job.get("title"), job.get("company"),
                )
                # BUG-FIX 2C: always pass user_id=self.user_id explicitly
                self._write_application(
                    job=job,
                    status="applied",
                    mode="auto",
                    error_code="dry_run",
                    user_id=self.user_id,
                )
                self._applied_count += 1
                return {"status": "applied", "job_id": job.get("id")}

            # ── STEP 2 — ATS detection from URL ──────────────────────
            ats = self._detect_ats(job_url)
            if ats == "unknown":
                self.logger.warning(
                    "Unknown ATS for %s — routing to manual_queued",
                    job_url,
                )
                # BUG-FIX 2C: always pass user_id=self.user_id explicitly
                self._write_application(
                    job=job,
                    status="manual_queued",
                    mode="manual",
                    error_code="unknown_ats",
                    user_id=self.user_id,
                )
                self._rerouted_count += 1
                return {
                    "status": "manual_queued",
                    "job_id": job.get("id"),
                    "reason": "unknown_ats",
                }

            # ── STEP 3 — Safety check ────────────────────────────────
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

            # ── STEP 4 — Rate limit wait ─────────────────────────────
            try:
                config_raw: str = _get_platform_config(platform=platform)
                pconfig: Dict[str, Any] = json.loads(config_raw)
                rate_limit: float = float(
                    pconfig.get("rate_limit_per_request_seconds", 3.0)
                )
            except Exception:  # noqa: BLE001
                rate_limit = 3.0

            if rate_limit > 0:
                time.sleep(rate_limit)

            # ── STEP 5 — Execute apply via fill_standard_form ────────
            result_raw: str = fill_standard_form.run(
                job_url=job_url,
                job_post_id=job_post_id,
                resume_filename=resume_to_use,
                pipeline_run_id=self.pipeline_run_id,
                user_id=self.user_id,
                ats_platform=ats,
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

            # ── STEP 6 — Handle result ───────────────────────────────
            if result.get("dry_run", False):
                # fill_standard_form returned dry_run=True — count but do not
                # treat as a real application.
                self._applied_count += 1
                _log_event(
                    pipeline_run_id=self.pipeline_run_id,
                    agent="apply_agent",
                    level="INFO",
                    event_type="dry_run_skip",
                    message=(
                        f"dry_run_skip — no actual submission | "
                        f"{company} — {job_title} | ats={ats}"
                    ),
                    job_post_id=job_post_id,
                )
                self.logger.info(
                    "_apply_single_job: dry_run_skip — no actual submission "
                    "for %s at %s",
                    job_title,
                    company,
                )
            elif result.get("applied", False):
                self._applied_count += 1
                _log_event(
                    pipeline_run_id=self.pipeline_run_id,
                    agent="apply_agent",
                    level="INFO",
                    event_type="job_applied",
                    message=(
                        f"Applied: {company} — {job_title} | "
                        f"ats={ats} | "
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
                _log_event(
                    pipeline_run_id=self.pipeline_run_id,
                    agent="apply_agent",
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
                _record_agent_error(
                    agent_type="ApplyAgent",
                    error_message=str(exc),
                    pipeline_run_id=self.pipeline_run_id,
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
                _create_application(
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

            # Fallback Postgres update
            try:
                _update_application_status(
                    job_post_id=job_post_id,
                    status="manual_queued",
                    error_code=reason,
                )
            except Exception as pg_exc:  # noqa: BLE001
                self.logger.warning(
                    "_reroute_to_manual: Postgres update failed: %s",
                    pg_exc,
                )

            _log_event(
                pipeline_run_id=self.pipeline_run_id,
                    agent="apply_agent",
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
            _record_fallback_event(
                agent_type="ApplyAgent",
                from_provider=failed_provider,
                to_provider=str(to_model),
                pipeline_run_id=self.pipeline_run_id,
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
            _record_fallback_event(
                agent_type="ApplyAgent",
                from_provider=failed_provider,
                to_provider=str(to_model),
                pipeline_run_id=self.pipeline_run_id,
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
    # Internal: concurrent per-platform dispatch
    # ------------------------------------------------------------------

    def _group_by_platform(
        self, jobs: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Group routing manifest jobs by source_platform.

        Args:
            jobs: List of job dicts from the routing manifest.

        Returns:
            Dict mapping platform name to list of jobs.
        """
        groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for job in jobs:
            platform = job.get("source_platform", "unknown")
            groups[platform].append(job)
        return dict(groups)

    def _apply_platform_batch(
        self, platform: str, jobs: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Apply to all jobs for a single platform sequentially.

        Jobs within a platform are processed sequentially to respect
        per-platform rate limits. Different platforms run in parallel.

        Args:
            platform: Platform name (e.g. "linkedin", "lever").
            jobs: List of jobs for this platform.

        Returns:
            List of per-job result dicts.
        """
        results: List[Dict[str, Any]] = []
        for job in jobs:
            if self._budget_aborted:
                jid = str(job.get("job_post_id", job.get("id", "")))
                self._reroute_to_manual(
                    job,
                    reason="budget_cap_hit_mid_platform_batch",
                    job_post_id=jid,
                )
                continue
            result = self._apply_single_job(job)
            results.append(result)
        self.logger.info(
            "_apply_platform_batch: %s completed %d jobs",
            platform,
            len(results),
        )
        return results

    def _dispatch_concurrent(
        self, max_platform_workers: int = 3
    ) -> List[Dict[str, Any]]:
        """Dispatch applications concurrently across platforms.

        Groups jobs by platform, then runs up to ``max_platform_workers``
        platform batches in parallel threads. Within each platform batch,
        jobs are processed sequentially to respect rate limits.

        Args:
            max_platform_workers: Max concurrent platform threads.
                Default 3 to balance throughput with resource limits.

        Returns:
            Combined list of per-job result dicts from all platforms.
        """
        platform_groups = self._group_by_platform(self.routing_manifest)
        all_results: List[Dict[str, Any]] = []

        self.logger.info(
            "_dispatch_concurrent: %d platforms, %d total jobs",
            len(platform_groups),
            len(self.routing_manifest),
        )

        try:
            with ThreadPoolExecutor(
                max_workers=min(max_platform_workers, len(platform_groups))
            ) as executor:
                future_to_platform = {
                    executor.submit(
                        self._apply_platform_batch, platform, jobs
                    ): platform
                    for platform, jobs in platform_groups.items()
                }
                for future in as_completed(future_to_platform):
                    platform = future_to_platform[future]
                    try:
                        results = future.result(timeout=300)
                        all_results.extend(results)
                    except Exception as exc:  # noqa: BLE001
                        self.logger.error(
                            "_dispatch_concurrent: platform %s failed: %s",
                            platform,
                            exc,
                        )
                        # Safety net: reroute all jobs from failed platform
                        for job in platform_groups[platform]:
                            jid = str(
                                job.get("job_post_id", job.get("id", ""))
                            )
                            self._reroute_to_manual(
                                job,
                                reason=f"platform_dispatch_failed: {exc}",
                                job_post_id=jid,
                            )
        except Exception as exc:  # noqa: BLE001
            self.logger.error(
                "_dispatch_concurrent: thread pool failed, falling back "
                "to sequential: %s",
                exc,
            )
            # Sequential fallback
            for job in self.routing_manifest:
                if self._budget_aborted:
                    jid = str(job.get("job_post_id", job.get("id", "")))
                    self._reroute_to_manual(
                        job,
                        reason="budget_cap_fallback_sequential",
                        job_post_id=jid,
                    )
                    continue
                all_results.append(self._apply_single_job(job))

        return all_results

    # ------------------------------------------------------------------
    # Agent / Task builders
    # ------------------------------------------------------------------

    def _build_agent(
        self,
        llm_override: Any = None,
        max_iter_override: Optional[int] = None,
    ) -> Agent:
        """Build the CrewAI Agent instance for the apply pass.

        Args:
            llm_override: If provided, use this LLM instead of
                ``self._current_llm``.  Used by ``_apply_one_job_via_crew``
                to iterate through the fallback chain.
            max_iter_override: If provided, use this instead of the
                default ``max_iter=25``.  Single-job micro-crews use 6.

        Returns:
            Configured ``Agent`` using the selected LLM.
        """
        selected_llm = llm_override if llm_override is not None else self._current_llm
        selected_max_iter = max_iter_override if max_iter_override is not None else 25

        _apply_temp: float = float(os.getenv("APPLY_LLM_TEMPERATURE", "0.1"))
        _active_llm = llm_override if llm_override is not None else self.llm
        if hasattr(_active_llm, "temperature"):
            _active_llm.temperature = _apply_temp

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
            llm=_active_llm,
            tools=[
                detect_ats_platform,
                fill_standard_form,
                check_captcha_present,
                get_apply_summary,
                update_application_status,
            ],
            verbose=True,
            max_iter=selected_max_iter,
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
            Configured ``Task`` ready for crew execution.
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

        # BUG-FIX 3A: use dynamic eval for dry_run
        dry_run_eval: bool = (
            os.getenv("DRY_RUN").lower()
            or self._apply_cfg.get("dry_run", False)
        )
        self.dry_run: bool = dry_run_eval
        xai_cap: float = budget_config.xai_cost_cap_per_run

        description: str = f"""
You are applying to {len(self.routing_manifest)} auto-routed jobs for the current run batch.
pipeline_run_id: {self.pipeline_run_id}
user_id: {self.user_id}
dry_run: {str(self.dry_run).lower()}

APPLICATION INSTRUCTIONS
========================

1. For EACH job in the routing manifest below, run a pre-apply safety check
   before launching any Playwright session:
   - Check xAI budget via check_xai_run_cap
   - Verify platform rate limits
   - Confirm resume_suggested path is present (every job in this manifest
     is guaranteed to have a non-empty resume_suggested field)
   - Validate job URL

2. DRY_RUN MODE: If dry_run is true — for each job, call fill_standard_form
   with dry_run=True. The tool will simulate the apply without opening a real
   browser session and return a simulated result. You MUST still call
   fill_standard_form for EVERY job in the manifest. Reporting a Final Answer
   without having called fill_standard_form for each job is a critical
   protocol violation that will cause a pipeline failure.

3. Detect the ATS platform for each job via detect_ats_platform before
   submitting — adapt your approach per platform.

4. Check for CAPTCHA indicators before submitting. If detected, skip to
   manual queue IMMEDIATELY — do not waste credits.

5. invoke the fill_standard_form tool using fill_standard_form.run(...) for each auto-route job ONE AT A TIME — NEVER
   run parallel applications. Always pass user_id={self.user_id} and
   the job's resume_suggested value as resume_filename.

6. After EVERY 5 applications, call check_xai_run_cap with
   pipeline_run_id={self.pipeline_run_id}. If the response contains
   "abort": true — STOP immediately and re-queue ALL remaining jobs
   to the manual queue via update_application_status.

7. Enforce per-platform rate limits using the platform config's
   rate_limit_per_request_seconds. Sleep between requests.

8. If a job fails to apply after 2 retries (handled by fill_standard_form) —
   re-route to manual queue via update_application_status.

9. After processing ALL jobs, call get_apply_summary with
   pipeline_run_id={self.pipeline_run_id} for final counts.
   get_apply_summary accepts ONLY one key: pipeline_run_id (string).
   Do NOT pass the fill_standard_form response or any other dict.
   Call it as: get_apply_summary.run(pipeline_run_id='{self.pipeline_run_id}')

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

    def _build_single_job_task(self, job: Dict[str, Any]) -> str:
        """Build a hallucination-resistant task description scoped to one job.

        Unlike _build_task() which addresses the full manifest, this method
        generates a task string for a single job. Concrete argument names and
        values are injected directly into the prompt to prevent the LLM from
        guessing fill_standard_form parameter names (the primary prompt failure
        mode observed in live runs).

        Args:
            job: A single job dict from self.routing_manifest with keys:
                 job_post_id, title, company, url, source_platform,
                 resume_suggested.

        Returns:
            Task description string ready for CrewAI Task(description=...).
        """
        dry_run_eval: bool = (
            os.getenv("DRY_RUN").lower()
            or self._apply_cfg.get("dry_run", False)
        )
        dry_run_str: str = str(dry_run_eval).lower()
        per_run_cap: float = float(os.getenv("XAI_COST_CAP_PER_RUN", "0.38"))

        return (
            f"SESSION CONTEXT\n"
            f"  pipeline_run_id : {self.pipeline_run_id}\n"
            f"  user_id      : {self.user_id}\n"
            f"  dry_run      : {dry_run_str}\n"
            f"\n"
            f"JOB TO APPLY TO\n"
            f"  job_post_id  : {job['job_post_id']}\n"
            f"  title        : {job['title']}\n"
            f"  company      : {job['company']}\n"
            f"  url          : {job['url']}\n"
            f"  platform     : {job['source_platform']}\n"
            f"  resume       : {job['resume_suggested']}\n"
            f"\n"
            f"INSTRUCTIONS — follow in exact order, do not skip any step:\n"
            f"\n"
            f"1. Call detect_ats_platform with:\n"
            f"     job_url=\"{job['url']}\"\n"
            f"     pipeline_run_id=\"{self.pipeline_run_id}\"\n"
            f"   Save the returned ats_type value for use in step 2.\n"
            f"\n"
            f"2. Call fill_standard_form using fill_standard_form.run(...) with EXACTLY these arguments — no others:\n"
            f"     job_url=\"{job['url']}\"\n"
            f"     job_post_id=\"{job['job_post_id']}\"\n"
            f"     resume_filename=\"{job['resume_suggested']}\"\n"
            f"     pipeline_run_id=\"{self.pipeline_run_id}\"\n"
            f"     user_id=\"{self.user_id}\"\n"
            f"     ats_platform=<ats_type returned in step 1>\n"
            f"     dry_run={dry_run_str}\n"
            f"   THIS CALL IS MANDATORY. Writing Final Answer without calling\n"
            f"   fill_standard_form.run(...) first is a protocol violation. Your output\n"
            f"   will be discarded and the job will be marked failed.\n"
            f"\n"
            f"3. Call get_apply_summary with:\n"
            f"     pipeline_run_id=\"{self.pipeline_run_id}\"\n"
            f"   Use its output to populate your Final Answer.\n"
            f"\n"
            f"4. Write Final Answer as a JSON object with EXACTLY these keys:\n"
            f"   {{\n"
            f"     \"job_post_id\": \"{job['job_post_id']}\",\n"
            f"     \"status\": \"<applied|failed|manual_queued|dry_run_skip|captcha_blocked>\",\n"
            f"     \"resume_used\": \"{job['resume_suggested']}\",\n"
            f"     \"error_code\": \"<error string or null>\",\n"
            f"     \"platform\": \"<ats_type from step 1>\",\n"
            f"     \"applied\": <true|false>\n"
            f"   }}\n"
            f"   Do NOT add extra keys. Do NOT wrap in Observation:.\n"
            f"   Output ONLY the JSON object.\n"
            f"\n"
            f"HARD CONSTRAINTS:\n"
            f"  - You have exactly ONE job. There is no loop.\n"
            f"  - Call each tool exactly ONCE in the order listed above.\n"
            f"  - Budget hard cap for this run: ${per_run_cap:.2f}\n"
            f"  - If fill_standard_form returns captcha_blocked set status=captcha_blocked.\n"
            f"  - If fill_standard_form returns re_route=manual set status=manual_queued.\n"
        )

    def _apply_one_job_via_crew(self, job: Dict[str, Any]) -> Dict[str, Any]:
        """Apply to a single job using a scoped micro-crew.

        Instantiates a fresh CrewAI Agent and Task scoped to exactly one job.
        Uses max_iter=6 which is sufficient for the 4-step single-job task with
        two buffer iterations. Falls back through the full LLM chain on
        exception. If all LLMs fail, falls back to the direct Python path via
        _apply_single_job(). Never raises — always returns a result dict.

        Args:
            job: A single job dict from self.routing_manifest.

        Returns:
            Dict with keys: job_post_id, status, resume_used, error_code,
            platform, applied.
        """
        _default: Dict[str, Any] = {
            "job_post_id": job.get("job_post_id", ""),
            "status": "failed",
            "resume_used": job.get("resume_suggested", ""),
            "error_code": "CREW_INIT_FAILED",
            "platform": "unknown",
            "applied": False,
        }

        llm_chain: List[Any] = [
            lm for lm in [
                self.llm,
                self.fallback_llm_1,
                getattr(self, "fallback_llm_2", None),
            ]
            if lm is not None
        ]

        last_exc: Optional[Exception] = None

        for attempt, llm_instance in enumerate(llm_chain, start=1):
            try:
                logger.info(
                    "_apply_one_job_via_crew: attempt %d/%d — job_post_id=%s company=%s",
                    attempt,
                    len(llm_chain),
                    job.get("job_post_id"),
                    job.get("company"),
                )
                agent = self._build_agent(
                    llm_override=llm_instance,
                    max_iter_override=6,
                )
                task = Task(
                    description=self._build_single_job_task(job),
                    agent=agent,
                    output_pydantic=SingleJobApplyResult,
                    expected_output="JSON object matching SingleJobApplyResult schema",
                )
                crew = Crew(
                    agents=[agent],
                    tasks=[task],
                    verbose=os.getenv("LOG_LEVEL", "INFO").upper() == "DEBUG",
                    process=Process.sequential,
                )
                crew_output = crew.kickoff()

                raw_output: str = crew_output.raw if hasattr(crew_output, "raw") else str(crew_output)
                # Validate tool call sequence was followed
                if "fill_standard_form" not in raw_output and "fill_standard_form.run" not in raw_output and "dry_run_skip" not in raw_output:
                    self.logger.error(
                        "_apply_one_job_via_crew: PROTOCOL VIOLATION — LLM skipped fill_standard_form "
                        "job_post_id=%s company=%s attempt=%d — marking failed",
                        job["job_post_id"], job["company"], attempt
                    )
                    # Log to audit_logs
                    _log_event(
                        pipeline_run_id=self.pipeline_run_id,
                        event_type="apply_protocol_violation",
                        level="ERROR",
                        agent="apply_agent",
                        message=f"LLM skipped fill_standard_form for {job['company']}",
                        metadata={"job_post_id": job["job_post_id"], "raw_output_preview": raw_output[:300]}
                    )
                    # Retry with next attempt — do NOT accept this result
                    continue

                result = self._parse_crew_output(crew_output)
                logger.info(
                    "_apply_one_job_via_crew: attempt %d success — status=%s job=%s",
                    attempt,
                    result.get("status"),
                    job.get("job_post_id"),
                )
                return result

            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "_apply_one_job_via_crew: attempt %d failed — job=%s error=%s",
                    attempt,
                    job.get("job_post_id"),
                    exc,
                )
                continue

        # All LLMs exhausted — fall back to direct Python path
        logger.error(
            "_apply_one_job_via_crew: all %d LLMs failed for job=%s — "
            "falling back to _apply_single_job(). Last error: %s",
            len(llm_chain),
            job.get("job_post_id"),
            last_exc,
        )
        try:
            return self._apply_single_job(job)
        except Exception as final_exc:
            logger.error(
                "_apply_one_job_via_crew: _apply_single_job also failed for job=%s: %s",
                job.get("job_post_id"),
                final_exc,
            )
            return {**_default, "error_code": f"ALL_PATHS_FAILED:{final_exc!s}"}

    # ------------------------------------------------------------------
    # Internal: crew output parser
    # ------------------------------------------------------------------

    def _parse_crew_output(self, crew_output: Any) -> Dict[str, Any]:
        """Parse single-job crew output. Handles CrewAI Pydantic and raw JSON.

        When output_pydantic=SingleJobApplyResult is set on the Task, CrewAI
        populates crew_output.pydantic with a validated model instance. This
        method checks for that first before falling back to regex JSON extraction.

        Args:
            crew_output: Raw return value from crew.kickoff().

        Returns:
            Dict with keys: job_post_id, status, resume_used, error_code,
            platform, applied. Returns a safe default dict on any parse error.
        """
        import re

        _default: Dict[str, Any] = {
            "job_post_id": "",
            "status": "failed",
            "resume_used": "",
            "error_code": "PARSE_FAILED",
            "platform": "unknown",
            "applied": False,
        }

        try:
            # Case 1: CrewAI returned a validated Pydantic object (preferred path)
            if hasattr(crew_output, "pydantic") and crew_output.pydantic is not None:
                return crew_output.pydantic.model_dump()

            # Case 2: Raw string — strip CrewAI envelope, extract JSON
            raw: str = (
                crew_output.raw
                if hasattr(crew_output, "raw")
                else str(crew_output)
            )
            # Strip any preamble before the first opening brace
            clean: str = re.sub(
                r"^.*?(?=\{)", "", raw.strip(), flags=re.DOTALL
            )
            # Strip trailing Thought: or Observation: blocks
            clean = re.sub(
                r"\n\n(Thought|Observation):.*$", "", clean, flags=re.DOTALL
            ).strip()

            match = re.search(r"\{.*\}", clean, flags=re.DOTALL)
            if not match:
                raise ValueError(
                    f"No JSON object found in crew output: {raw[:400]!r}"
                )

            parsed: Dict[str, Any] = json.loads(match.group())

            # Warn on missing required keys and merge defaults
            required: set = {
                "job_post_id", "status", "resume_used", "platform", "applied"
            }
            missing: set = required - parsed.keys()
            if missing:
                logger.warning(
                    "_parse_crew_output: missing keys %s — merging safe defaults",
                    missing,
                )
                return {**_default, **parsed}

            return parsed

        except Exception as exc:
            logger.warning(
                "_parse_crew_output: failed to parse crew output: %s — "
                "returning safe default",
                exc,
            )
            return _default

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
                    _create_application(
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

    @operation
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
            9. Reconcile — safety net for any orphaned jobs.
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
            dry_run_eval: bool = (
                os.getenv("DRY_RUN").lower()
                or self._apply_cfg.get("dry_run", False)
            )
            _log_event(
                pipeline_run_id=self.pipeline_run_id,
                    agent="apply_agent",
                level="INFO",
                event_type="apply_run_start",
                message=(
                    f"Apply Agent starting | "
                    f"{len(self.routing_manifest)} jobs in manifest | "
                    f"dry_run={dry_run_eval}"
                ),
            )
            self.logger.info(
                "ApplyAgent.run: starting — %d jobs | dry_run=%s",
                len(self.routing_manifest),
                dry_run_eval,
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
                    pipeline_run_id=self.pipeline_run_id
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
                        "pipeline_run_id": self.pipeline_run_id,
                        "total_attempted": 0,
                        "applied": 0,
                        "failed": 0,
                        "rerouted_to_manual": self._rerouted_count,
                        "budget_aborted": True,
                        "dry_run": dry_run_eval,
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
            # ── Python-owns-loop: one micro-crew per job ──────────────────────
            per_job_results: List[Dict[str, Any]] = []
            jobs_remaining: List[Dict[str, Any]] = list(self.routing_manifest)

            for idx, job in enumerate(jobs_remaining):

                # Budget check every 5 jobs (uses existing check_xai_run_cap import)
                if idx % 5 == 0:
                    try:
                        cap_result = check_xai_run_cap(pipeline_run_id=self.pipeline_run_id)
                        if isinstance(cap_result, str):
                            cap_result = json.loads(cap_result)
                        if cap_result.get("abort"):
                            logger.warning(
                                "ApplyAgent.run: run-cap reached at job %d/%d — "
                                "aborting loop, re-routing remaining %d jobs to manual",
                                idx + 1,
                                len(jobs_remaining),
                                len(jobs_remaining) - idx,
                            )
                            self._budget_aborted = True
                            for remaining_job in jobs_remaining[idx:]:
                                self._reroute_to_manual(
                                    remaining_job,
                                    reason="BUDGET_ABORT",
                                    job_post_id=str(
                                        remaining_job.get("job_post_id",
                                                          remaining_job.get("id", ""))
                                    ),
                                )
                            break
                    except Exception as cap_exc:
                        logger.warning(
                            "ApplyAgent.run: budget cap check failed at job %d: %s",
                            idx + 1,
                            cap_exc,
                        )

                result = self._apply_one_job_via_crew(job)
                per_job_results.append(result)

                # Update run counters
                status: str = result.get("status", "failed")
                if status == "applied":
                    self._applied_count += 1
                elif status == "manual_queued":
                    self._rerouted_count += 1
                else:
                    self._failed_count += 1

                # Circuit breaker — abort if N consecutive failures detected
                _cb_threshold: int = int(
                    self._apply_cfg.get("circuit_breaker_n", 5)
                )
                if status in ("failed", "captcha_blocked"):
                    self._consecutive_failures += 1
                else:
                    self._consecutive_failures = 0

                if self._consecutive_failures >= _cb_threshold:
                    logger.error(
                        "ApplyAgent.run: circuit breaker triggered — %d consecutive "
                        "failures — aborting, re-routing %d remaining jobs to manual",
                        self._consecutive_failures,
                        len(jobs_remaining) - (idx + 1),
                    )
                    for remaining_job in jobs_remaining[idx + 1 :]:
                        self._reroute_to_manual(
                            remaining_job,
                            reason="CIRCUIT_BREAKER",
                            job_post_id=str(
                                remaining_job.get("job_post_id",
                                                  remaining_job.get("id", ""))
                            ),
                        )
                    break

                # Inter-job sleep — spaces xAI calls below 3 RPM ceiling
                import random
                delay_min = self._apply_cfg.get("apply_delay_min_seconds", 45)
                delay_max = self._apply_cfg.get("apply_delay_max_seconds", 90)
                if not dry_run_eval:
                    time.sleep(random.uniform(delay_min, delay_max))
                else:
                    time.sleep(0.3)

            # ── end loop ──────────────────────────────────────────────────────

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
                    pipeline_run_id=self.pipeline_run_id
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
            _log_event(
                pipeline_run_id=self.pipeline_run_id,
                    agent="apply_agent",
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
                "pipeline_run_id": self.pipeline_run_id,
                "total_attempted": len(per_job_results),
                "applied": self._applied_count,
                "failed": self._failed_count,
                "rerouted_to_manual": self._rerouted_count,
                "budget_aborted": self._budget_aborted,
                "dry_run": dry_run_eval,
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
                _record_agent_error(
                    agent_type="ApplyAgent",
                    error_message=str(exc),
                    pipeline_run_id=self.pipeline_run_id,
                    error_code="APPLY_UNHANDLED_EXCEPTION",
                )
            except Exception:  # noqa: BLE001
                pass

            # Re-queue entire remaining manifest as safety net
            for job in self.routing_manifest:
                jid = str(job.get("job_post_id", job.get("id", "")))
                try:
                    _create_application(
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
                "pipeline_run_id": self.pipeline_run_id,
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
