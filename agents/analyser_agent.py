"""
Analyser Agent for AI Job Application Agent.

The most logic-heavy agent in the pipeline.  Receives all job_posts discovered
by the Scraper Agent for the current ``run_batch_id``, scores each job against
15 resume variants using RAG, makes eligibility and routing decisions, and
writes scores + routes back to Postgres.

Runs after the Scraper Agent and before the Apply Agent every scheduled session.

Pipeline position:
    Scraper Agent → **Analyser Agent** → Apply Agent

Routing thresholds (from IDE_README.md SCORING AND ROUTING RULES):
    * fit_score <  0.40  → skip
    * fit_score 0.40-0.49 → manual
    * fit_score 0.50-0.74 → auto (default) or manual by form complexity
    * fit_score 0.75-0.89 → auto high-priority
    * fit_score >= 0.90   → force manual (high-stakes, no auto-apply)
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Optional

import psycopg2
import psycopg2.extras
from crewai import Agent, Task, Crew, Process
import agentops
from agentops.sdk.decorators import agent, operation

from integrations.llm_interface import LLMInterface
from tools.rag_tools import query_resume_match, get_resume_context, embed_job_description
from tools.postgres_tools import log_event, save_job_score, get_run_stats, get_platform_config
from tools.budget_tools import check_xai_run_cap, record_llm_cost, get_cost_summary
from tools.agentops_tools import record_agent_error, record_fallback_event

logger = logging.getLogger(__name__)

__all__ = ["AnalyserAgent"]

# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

_DB_URL: Optional[str] = (
    os.getenv("LOCAL_POSTGRES_URL")
    if os.getenv("ACTIVE_DB", "local") == "local"
    else os.getenv("SUPABASE_URL")
)

# LLM provider name map (model prefix → short name for cost recording)
_PROVIDER_NAME: dict[str, str] = {
    "xai": "xai",
    "sambanova": "sambanova",
    "cerebras": "cerebras",
}


def _get_db_conn() -> psycopg2.extensions.connection:
    """
    Open and return a psycopg2 connection to the active Postgres instance.

    Returns:
        Database connection with ``autocommit=False``.

    Raises:
        RuntimeError: If the DB URL environment variable is not set, or if the
            connection attempt fails.
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


def _provider_from_model(model: str) -> str:
    """
    Derive a short provider name from a CrewAI model string.

    Args:
        model: Model identifier such as ``"xai/grok-4-fast-reasoning"``.

    Returns:
        Lowercase provider prefix, e.g. ``"xai"``.
    """
    prefix = model.split("/")[0].lower() if "/" in model else model.lower()
    return _PROVIDER_NAME.get(prefix, prefix)


# ---------------------------------------------------------------------------
# AnalyserAgent
# ---------------------------------------------------------------------------


@agent
class AnalyserAgent:
    """
    CrewAI Analyser Agent — eligibility filter, RAG resume matching, routing.

    Scores every job_post in the current run batch against 15 resume variants,
    assigns a fit_score between 0.0 and 1.0, routes each job to auto-apply /
    manual-review / skip, and persists results to Postgres.
    """

    def __init__(self, run_batch_id: str, user_id: str) -> None:
        """
        Initialise the Analyser Agent for a given run batch.

        Args:
            run_batch_id: UUID of the current run batch (from ``run_batches``
                table), handed off by the Scraper Agent via the Master Agent.
            user_id: UUID of the candidate user (from ``users`` table).
        """
        self.run_batch_id = run_batch_id
        self.user_id = user_id
        self.llm_interface = LLMInterface()
        self.llm = self.llm_interface.get_llm("ANALYSER_AGENT")
        self.fallback_llm_1 = self.llm_interface.get_fallback_llm("ANALYSER_AGENT", level=1)
        self.fallback_llm_2 = self.llm_interface.get_fallback_llm("ANALYSER_AGENT", level=2)
        self.logger = logging.getLogger(self.__class__.__name__)
        self._current_llm = self.llm
        self._fallback_level: int = 0

        self.logger.info(
            "AnalyserAgent initialised — run_batch_id=%s user_id=%s",
            run_batch_id,
            user_id,
        )

    # ------------------------------------------------------------------
    # Internal: job retrieval
    # ------------------------------------------------------------------

    def _get_jobs_for_run(self) -> list[dict[str, Any]]:
        """
        Fetch all job posts for the current run batch from Postgres.

        Uses a direct psycopg2 connection (never the @tool wrapper) so that
        failures here do not interact with the CrewAI agent loop.

        Returns:
            List of job dicts, each containing id, title, company,
            source_platform, url, and location.  Empty list when no jobs
            exist for this batch.

        Raises:
            RuntimeError: If all three retry attempts are exhausted.
        """
        max_retries = 3
        last_exc: Optional[Exception] = None

        for attempt in range(max_retries):
            conn: Optional[psycopg2.extensions.connection] = None
            try:
                conn = _get_db_conn()
                cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
                cursor.execute(
                    """
                    SELECT
                        id,
                        title,
                        company,
                        source_platform,
                        url,
                        location
                    FROM job_posts
                    WHERE run_batch_id = %s
                    ORDER BY created_at ASC
                    """,
                    (self.run_batch_id,),
                )
                rows = cursor.fetchall()
                jobs: list[dict[str, Any]] = [dict(row) for row in rows]
                self.logger.info(
                    "_get_jobs_for_run: fetched %d jobs for batch %s",
                    len(jobs),
                    self.run_batch_id,
                )
                return jobs

            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt < max_retries - 1:
                    sleep_s = 2 ** attempt
                    self.logger.warning(
                        "_get_jobs_for_run attempt %d/%d failed: %s — retrying in %ds",
                        attempt + 1,
                        max_retries,
                        exc,
                        sleep_s,
                    )
                    time.sleep(sleep_s)
                else:
                    self.logger.critical(
                        "_get_jobs_for_run failed after %d attempts: %s",
                        max_retries,
                        exc,
                    )
            finally:
                if conn:
                    try:
                        conn.close()
                    except Exception:  # noqa: BLE001
                        pass

        raise RuntimeError(
            f"_get_jobs_for_run exhausted {max_retries} retries: {last_exc}"
        )

    # ------------------------------------------------------------------
    # Internal: resume UUID resolution
    # ------------------------------------------------------------------

    def _resolve_resume_id(self, resume_filename: str) -> Optional[str]:
        """
        Look up the ``resumes`` table to resolve a filename to a UUID.

        Matches on ``storage_path`` (ILIKE ``%filename%``) and falls back
        to ``label`` matching.

        Args:
            resume_filename: Resume filename such as ``"AarjunGen.pdf"``.

        Returns:
            UUID string if found, ``None`` otherwise.
        """
        if not resume_filename:
            return None

        conn: Optional[psycopg2.extensions.connection] = None
        try:
            conn = _get_db_conn()
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cursor.execute(
                """
                SELECT id FROM resumes
                WHERE storage_path ILIKE %s OR label ILIKE %s
                LIMIT 1
                """,
                (f"%{resume_filename}%", f"%{resume_filename}%"),
            )
            row = cursor.fetchone()
            if row:
                return str(row["id"])
            return None
        except Exception as exc:  # noqa: BLE001
            self.logger.warning(
                "_resolve_resume_id: lookup failed for '%s': %s", resume_filename, exc
            )
            return None
        finally:
            if conn:
                try:
                    conn.close()
                except Exception:  # noqa: BLE001
                    pass

    # ------------------------------------------------------------------
    # Internal: direct score persistence (bypasses @tool validation)
    # ------------------------------------------------------------------

    def _save_score_direct(
        self,
        job_post_id: str,
        resume_id: Optional[str],
        fit_score: float,
        eligibility_pass: bool,
        reasons_json: dict[str, Any],
    ) -> Optional[str]:
        """
        Write a job_scores row directly via psycopg2.

        Avoids the CrewAI @tool Pydantic validation layer, which does not
        accept ``None`` for a ``str``-typed ``resume_id`` field even though
        the underlying DB column is nullable.

        Args:
            job_post_id: UUID of the job post being scored.
            resume_id: UUID of the matched resume, or ``None`` if the resume
                could not be resolved from the resumes table.
            fit_score: Composite fit score in the range 0.0–1.0.
            eligibility_pass: Whether the job passed all eligibility gates.
            reasons_json: Dict carrying scoring details, route, and metadata.

        Returns:
            UUID string of the created job_score row, or ``None`` on failure.
        """
        max_retries = 3
        for attempt in range(max_retries):
            conn: Optional[psycopg2.extensions.connection] = None
            try:
                conn = _get_db_conn()
                cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
                cursor.execute(
                    """
                    INSERT INTO job_scores
                        (job_post_id, resume_id, fit_score, eligibility_pass, reasons_json)
                    VALUES (%s, %s, %s, %s, %s)
                    RETURNING id
                    """,
                    (
                        job_post_id,
                        resume_id,  # None → NULL (column is nullable FK)
                        fit_score,
                        eligibility_pass,
                        json.dumps(reasons_json),
                    ),
                )
                result = cursor.fetchone()
                conn.commit()
                score_id = str(result["id"]) if result else None
                self.logger.debug(
                    "_save_score_direct: saved job_score %s for job_post %s",
                    score_id,
                    job_post_id,
                )
                return score_id

            except Exception as exc:  # noqa: BLE001
                if conn:
                    try:
                        conn.rollback()
                    except Exception:  # noqa: BLE001
                        pass
                if attempt < max_retries - 1:
                    sleep_s = 2 ** attempt
                    self.logger.warning(
                        "_save_score_direct attempt %d/%d failed for job_post %s: "
                        "%s — retrying in %ds",
                        attempt + 1,
                        max_retries,
                        job_post_id,
                        exc,
                        sleep_s,
                    )
                    time.sleep(sleep_s)
                else:
                    self.logger.error(
                        "_save_score_direct failed after %d attempts for job_post %s: %s",
                        max_retries,
                        job_post_id,
                        exc,
                    )
            finally:
                if conn:
                    try:
                        conn.close()
                    except Exception:  # noqa: BLE001
                        pass
        return None

    # ------------------------------------------------------------------
    # Internal: routing threshold logic
    # ------------------------------------------------------------------

    @staticmethod
    def _determine_route(fit_score: float, form_complexity: Optional[str] = None) -> str:
        """
        Apply canonical routing thresholds to a fit_score.

        Args:
            fit_score: Composite fit score in the range 0.0–1.0.
            form_complexity: Optional hint from job metadata — ``"simple"``
                or ``"complex"``.  Defaults to ``"simple"`` (auto route) when
                unknown.

        Returns:
            One of ``"skip"``, ``"manual"``, or ``"auto"``.
        """
        if fit_score < 0.40:
            return "skip"
        if fit_score < 0.50:
            return "manual"
        if fit_score >= 0.90:
            return "manual"  # Force-manual regardless of complexity
        if fit_score >= 0.75:
            # High priority — route by form complexity
            if form_complexity and form_complexity.lower() == "complex":
                return "manual"
            return "auto"
        # 0.50–0.74: route by complexity, default auto
        if form_complexity and form_complexity.lower() == "complex":
            return "manual"
        return "auto"

    # ------------------------------------------------------------------
    # Internal: score a single job (programmatic RAG path)
    # ------------------------------------------------------------------

    def _score_job(self, job: dict[str, Any]) -> dict[str, Any]:
        """
        Score a single job post using the RAG resume-match tools.

        Calls ``query_resume_match`` directly (not via the LLM agent) to
        obtain a fit_score and resume suggestion, then determines the route,
        resolves the resume UUID, and persists the score row to Postgres.

        A single-job failure is always caught and returned with a ``"skip"``
        route so it never halts the full pass.

        Args:
            job: Row dict from ``job_posts`` containing at minimum ``id``,
                ``title``, ``company``, and ``url``.

        Returns:
            Dict with keys: ``job_post_id``, ``fit_score``, ``route``,
            ``resume_suggested``, ``eligibility_pass``, ``reasons_json``.
        """
        job_post_id: str = str(job.get("id", ""))
        job_title: str = job.get("title", "")
        default_resume = os.getenv("DEFAULT_RESUME", "AarjunGen.pdf")

        fallback_result: dict[str, Any] = {
            "job_post_id": job_post_id,
            "fit_score": 0.0,
            "route": "skip",
            "resume_suggested": default_resume,
            "eligibility_pass": False,
            "reasons_json": {"error": "scoring_failed"},
        }

        try:
            # Call RAG match tool directly (callable outside LLM agent context)
            raw: str = query_resume_match(
                job_description=job.get("description", ""),
                job_title=job_title,
                required_skills=job.get("required_skills", ""),
            )

            data: dict[str, Any] = {}
            try:
                data = json.loads(raw) if isinstance(raw, str) else {}
            except (json.JSONDecodeError, TypeError) as exc:
                self.logger.warning(
                    "_score_job: JSON parse failed for job %s: %s", job_post_id, exc
                )

            if data.get("error"):
                self.logger.warning(
                    "_score_job: RAG error for job %s: %s", job_post_id, data["error"]
                )
                return fallback_result

            fit_score: float = float(data.get("fit_score", 0.0) or 0.0)
            # Clamp to [0.0, 1.0]
            fit_score = max(0.0, min(1.0, fit_score))

            resume_suggested: str = (
                str(data.get("resume_suggested") or default_resume)
            )
            match_reasoning: str = str(data.get("match_reasoning", "") or "")
            talking_points: list[str] = list(data.get("talking_points", []) or [])

            route: str = self._determine_route(fit_score)
            eligibility_pass: bool = fit_score >= 0.40

            reasons_json: dict[str, Any] = {
                "fit_score": fit_score,
                "route": route,
                "resume_suggested": resume_suggested,
                "similarity_score": float(data.get("similarity_score", fit_score) or 0.0),
                "match_reasoning": match_reasoning,
                "talking_points": talking_points,
                "source": "rag_direct",
            }

            # Resolve resume UUID for FK (non-blocking)
            resume_id: Optional[str] = self._resolve_resume_id(resume_suggested)

            # Persist score row
            self._save_score_direct(
                job_post_id=job_post_id,
                resume_id=resume_id,
                fit_score=fit_score,
                eligibility_pass=eligibility_pass,
                reasons_json=reasons_json,
            )

            return {
                "job_post_id": job_post_id,
                "fit_score": fit_score,
                "route": route,
                "resume_suggested": resume_suggested,
                "eligibility_pass": eligibility_pass,
                "reasons_json": reasons_json,
            }

        except Exception as exc:  # noqa: BLE001
            self.logger.error(
                "_score_job: unhandled exception for job_post_id=%s: %s",
                job_post_id,
                exc,
                exc_info=True,
            )
            return fallback_result

    # ------------------------------------------------------------------
    # Internal: LLM fallback chain
    # ------------------------------------------------------------------

    def _switch_to_fallback(self, failed_provider: str) -> bool:
        """
        Switch the active LLM to the next fallback in the chain.

        Records the fallback event to AgentOps and Postgres, then updates
        ``self._current_llm`` and ``self._fallback_level``.

        Args:
            failed_provider: Short name or model string of the provider that
                raised the exception.

        Returns:
            ``True`` if a fallback LLM is available and has been activated,
            ``False`` if the fallback chain is exhausted.
        """
        if self._fallback_level == 0 and self.fallback_llm_1 is not None:
            to_model: str = getattr(self.fallback_llm_1, "model", "fallback_1")
            record_fallback_event(
                agent_type="AnalyserAgent",
                from_provider=failed_provider,
                to_provider=str(to_model),
                run_batch_id=self.run_batch_id,
                fallback_level=1,
                reason=f"Primary provider {failed_provider} failed",
            )
            self._current_llm = self.fallback_llm_1
            self._fallback_level = 1
            self.logger.warning(
                "_switch_to_fallback: level 1 activated — switching to %s", to_model
            )
            return True

        if self._fallback_level == 1 and self.fallback_llm_2 is not None:
            to_model = getattr(self.fallback_llm_2, "model", "fallback_2")
            record_fallback_event(
                agent_type="AnalyserAgent",
                from_provider=failed_provider,
                to_provider=str(to_model),
                run_batch_id=self.run_batch_id,
                fallback_level=2,
                reason=f"Fallback-1 provider {failed_provider} failed",
            )
            self._current_llm = self.fallback_llm_2
            self._fallback_level = 2
            self.logger.warning(
                "_switch_to_fallback: level 2 activated — switching to %s", to_model
            )
            return True

        self.logger.critical(
            "_switch_to_fallback: fallback chain exhausted at level %d — "
            "no more providers available for AnalyserAgent",
            self._fallback_level,
        )
        return False

    # ------------------------------------------------------------------
    # Agent / task builders
    # ------------------------------------------------------------------

    def _build_agent(self) -> Agent:
        """
        Build the CrewAI Agent instance for the analyser pass.

        Returns:
            Configured ``crewai.Agent`` using the currently active LLM
            (may be primary or a fallback after ``_switch_to_fallback``).
        """
        return Agent(
            role="Senior AI/ML Job Eligibility Analyst and Resume Matcher",
            goal=(
                "Score every discovered job in the current run against 15 resume "
                "variants, determine eligibility, assign a fit score between 0.0 and "
                "1.0, and route each job to auto-apply, manual review queue, or skip "
                "— with zero false positives on auto-apply routing"
            ),
            backstory=(
                "You are an expert technical recruiter and ML engineer who deeply "
                "understands what makes a candidate qualified for AI, ML, Data "
                "Science, and Automation roles.  You apply rigorous eligibility "
                "filters, use RAG-powered resume matching to find the best resume "
                "variant for each job, and make precise routing decisions that "
                "maximise application quality while respecting the $0.38/run xAI "
                "cost cap."
            ),
            llm=self._current_llm,
            tools=[
                query_resume_match,
                get_resume_context,
                embed_job_description,
                check_xai_run_cap,
                record_llm_cost,
                get_cost_summary,
            ],
            verbose=True,
            max_iter=20,
            memory=False,
        )

    def _build_task(self, agent: Agent, jobs: list[dict[str, Any]]) -> Task:
        """
        Build the CrewAI Task with detailed scoring instructions.

        Serialises the full job list as JSON inside the task description so
        the LLM agent operates with complete job data without requiring
        additional DB round-trips.

        Args:
            agent: The ``AnalyserAgent`` CrewAI agent that will execute this
                task.
            jobs: List of job post dicts fetched from ``_get_jobs_for_run``.

        Returns:
            Configured ``crewai.Task`` ready for crew execution.
        """
        jobs_json_str: str
        try:
            jobs_json_str = json.dumps(jobs, default=str, indent=2)
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("_build_task: JSON serialisation failed: %s", exc)
            jobs_json_str = "[]"

        run_xai_cap: float = float(os.getenv("XAI_COST_CAP_PER_RUN", "0.38"))

        description = f"""
You are scoring {len(jobs)} job posts for the current run batch.
run_batch_id: {self.run_batch_id}

SCORING INSTRUCTIONS
====================

1. ITERATE over every job in the JSON list below.
   For each job call `query_resume_match` with:
   - job_description: the job's description field (empty string if missing)
   - job_title: the job's title field
   - required_skills: the job's required_skills field (empty string if missing)

2. PARSE the JSON response from `query_resume_match`.
   Extract:
   - fit_score (float 0.0–1.0)
   - resume_suggested (filename string)

3. APPLY routing thresholds EXACTLY as follows:
   - fit_score < 0.40           → route = "skip"
   - fit_score 0.40 – 0.49      → route = "manual"
   - fit_score 0.50 – 0.74      → route = "auto"
     (override to "manual" if form_complexity == "complex")
   - fit_score 0.75 – 0.89      → route = "auto" (high priority)
     (override to "manual" if form_complexity == "complex")
   - fit_score >= 0.90           → route = "manual" (FORCE — never auto)

4. AFTER every 10 jobs call `check_xai_run_cap` with:
   - run_batch_id: {self.run_batch_id}
   If the response JSON contains "abort": true — STOP immediately and
   return partial results with "budget_aborted": true.

5. AFTER every xAI LLM call, estimate the cost (typically $0.001–$0.005
   per call) and call `record_llm_cost` with:
   - provider: "xai"
   - cost_usd: [estimated cost]
   - agent_type: "AnalyserAgent"
   - run_batch_id: {self.run_batch_id}
   Budget hard cap per run: ${run_xai_cap}

6. RETURN a complete routing manifest as a single JSON object (no
   markdown, no code blocks — raw JSON only).

JOB LIST (JSON)
===============
{jobs_json_str}
"""

        return Task(
            description=description,
            expected_output=(
                'JSON object: {{"scored": int, "auto_route": int, '
                '"manual_route": int, "skip_route": int, '
                '"budget_aborted": bool, '
                '"results": [{{"job_post_id": "...", "fit_score": 0.0, '
                '"route": "auto|manual|skip", '
                '"resume_suggested": "filename.pdf"}}]}}'
            ),
            agent=agent,
        )

    # ------------------------------------------------------------------
    # Public: main run method
    # ------------------------------------------------------------------

    @operation
    def run(self) -> dict[str, Any]:
        """
        Execute the full Analyser Agent scoring pass for the current run batch.

        Lifecycle:
        1. Log run start.
        2. Fetch all job posts for ``run_batch_id`` from Postgres.
        3. Early-return if no jobs found.
        4. Build CrewAI agent + task.
        5. Execute crew (with LLM fallback on provider failure).
        6. Parse routing manifest from crew output.
        7. Persist each scored result to ``job_scores`` via ``_save_score_direct``.
        8. Log run complete with summary counts.
        9. Return structured result dict.

        Returns:
            Dict with keys: ``success``, ``run_batch_id``, ``total_scored``,
            ``auto_route``, ``manual_route``, ``skipped``,
            ``budget_aborted``, ``routing_manifest``.
            On failure: ``{"success": False, "error": str}``.
        """
        try:
            # ----------------------------------------------------------
            # Step 1: log start
            # ----------------------------------------------------------
            log_event(
                run_batch_id=self.run_batch_id,
                level="INFO",
                event_type="analyser_run_start",
                message="Analyser Agent starting scoring pass",
            )
            self.logger.info(
                "AnalyserAgent.run: starting pass for batch %s", self.run_batch_id
            )

            # ----------------------------------------------------------
            # Step 2: fetch jobs
            # ----------------------------------------------------------
            jobs: list[dict[str, Any]] = self._get_jobs_for_run()

            # ----------------------------------------------------------
            # Step 3: early-return if nothing to score
            # ----------------------------------------------------------
            if not jobs:
                self.logger.info(
                    "AnalyserAgent.run: no jobs found for batch %s — returning early",
                    self.run_batch_id,
                )
                log_event(
                    run_batch_id=self.run_batch_id,
                    level="INFO",
                    event_type="analyser_run_complete",
                    message="Analyser pass complete — 0 jobs found",
                )
                return {
                    "success": True,
                    "scored": 0,
                    "reason": "no_jobs_found",
                }

            self.logger.info(
                "AnalyserAgent.run: fetched %d jobs — building crew", len(jobs)
            )

            # ----------------------------------------------------------
            # Steps 4 + 5: build and execute crew with fallback
            # ----------------------------------------------------------
            agent: Agent = self._build_agent()
            task: Task = self._build_task(agent, jobs)
            crew = Crew(
                agents=[agent],
                tasks=[task],
                process=Process.sequential,
                verbose=True,
            )

            crew_output: Any = None

            # First attempt
            try:
                self.logger.info("AnalyserAgent.run: executing crew (primary LLM)…")
                crew_output = crew.kickoff()
            except Exception as primary_exc:  # noqa: BLE001
                failed_model: str = getattr(self._current_llm, "model", "primary")
                self.logger.error(
                    "AnalyserAgent.run: primary LLM failed (%s): %s",
                    failed_model,
                    primary_exc,
                )
                switched: bool = self._switch_to_fallback(failed_model)
                if not switched:
                    self.logger.critical(
                        "AnalyserAgent.run: no fallback available — aborting"
                    )
                    record_agent_error(
                        agent_type="AnalyserAgent",
                        error_message=str(primary_exc),
                        run_batch_id=self.run_batch_id,
                        error_code="LLM_ALL_PROVIDERS_FAILED",
                    )
                    return {
                        "success": False,
                        "error": "all_llm_providers_failed",
                        "detail": str(primary_exc),
                    }

                # Rebuild agent with new LLM and retry once
                agent = self._build_agent()
                task = self._build_task(agent, jobs)
                retry_crew = Crew(
                    agents=[agent],
                    tasks=[task],
                    process=Process.sequential,
                    verbose=True,
                )
                try:
                    fallback_model: str = getattr(
                        self._current_llm, "model", f"fallback_{self._fallback_level}"
                    )
                    self.logger.info(
                        "AnalyserAgent.run: retrying crew with fallback LLM %s",
                        fallback_model,
                    )
                    crew_output = retry_crew.kickoff()
                except Exception as fallback_exc:  # noqa: BLE001
                    self.logger.critical(
                        "AnalyserAgent.run: fallback LLM also failed: %s", fallback_exc
                    )
                    record_agent_error(
                        agent_type="AnalyserAgent",
                        error_message=str(fallback_exc),
                        run_batch_id=self.run_batch_id,
                        error_code="LLM_FALLBACK_FAILED",
                    )
                    return {
                        "success": False,
                        "error": "llm_fallback_failed",
                        "detail": str(fallback_exc),
                    }

            # ----------------------------------------------------------
            # Step 6: parse routing manifest from crew output
            # ----------------------------------------------------------
            manifest_data: dict[str, Any] = self._parse_crew_output(crew_output)
            results_raw: list[dict[str, Any]] = manifest_data.get("results", [])
            budget_aborted: bool = bool(manifest_data.get("budget_aborted", False))

            self.logger.info(
                "AnalyserAgent.run: crew returned %d scored results "
                "(budget_aborted=%s)",
                len(results_raw),
                budget_aborted,
            )

            # Build index of crew results keyed by job_post_id for easy lookup
            result_index: dict[str, dict[str, Any]] = {
                str(r.get("job_post_id", "")): r for r in results_raw
            }

            # ----------------------------------------------------------
            # Step 7: persist each result to job_scores
            # ----------------------------------------------------------
            auto_count = 0
            manual_count = 0
            skip_count = 0
            routing_manifest: list[dict[str, Any]] = []

            for job in jobs:
                job_post_id: str = str(job.get("id", ""))
                crew_result: Optional[dict[str, Any]] = result_index.get(job_post_id)

                if crew_result:
                    fit_score: float = float(
                        crew_result.get("fit_score", 0.0) or 0.0
                    )
                    fit_score = max(0.0, min(1.0, fit_score))
                    route: str = str(crew_result.get("route", "skip") or "skip").lower()
                    resume_suggested: str = str(
                        crew_result.get("resume_suggested", "")
                        or os.getenv("DEFAULT_RESUME", "AarjunGen.pdf")
                    )
                else:
                    # Job was not scored by crew (e.g. budget abort mid-batch)
                    # Fall back to direct programmatic scoring
                    self.logger.debug(
                        "AnalyserAgent.run: job %s missing from crew results — "
                        "running _score_job fallback",
                        job_post_id,
                    )
                    fallback_scored: dict[str, Any] = self._score_job(job)
                    fit_score = float(fallback_scored.get("fit_score", 0.0))
                    route = str(fallback_scored.get("route", "skip"))
                    resume_suggested = str(
                        fallback_scored.get(
                            "resume_suggested",
                            os.getenv("DEFAULT_RESUME", "AarjunGen.pdf"),
                        )
                    )

                eligibility_pass: bool = fit_score >= 0.40

                reasons_json: dict[str, Any] = {
                    "fit_score": fit_score,
                    "route": route,
                    "resume_suggested": resume_suggested,
                    "source": "crew_output" if crew_result else "rag_direct_fallback",
                }

                resume_id: Optional[str] = self._resolve_resume_id(resume_suggested)

                self._save_score_direct(
                    job_post_id=job_post_id,
                    resume_id=resume_id,
                    fit_score=fit_score,
                    eligibility_pass=eligibility_pass,
                    reasons_json=reasons_json,
                )

                if route == "auto":
                    auto_count += 1
                elif route == "manual":
                    manual_count += 1
                else:
                    skip_count += 1

                routing_manifest.append(
                    {
                        "job_post_id": job_post_id,
                        "title": job.get("title", ""),
                        "company": job.get("company", ""),
                        "url": job.get("url", ""),
                        "source_platform": job.get("source_platform", ""),
                        "fit_score": fit_score,
                        "route": route,
                        "resume_suggested": resume_suggested,
                        "eligibility_pass": eligibility_pass,
                    }
                )

            total_scored: int = len(jobs)

            # ----------------------------------------------------------
            # Step 8: log completion
            # ----------------------------------------------------------
            summary_msg: str = (
                f"Analyser pass complete — total={total_scored} "
                f"auto={auto_count} manual={manual_count} skip={skip_count} "
                f"budget_aborted={budget_aborted}"
            )
            self.logger.info("AnalyserAgent.run: %s", summary_msg)
            log_event(
                run_batch_id=self.run_batch_id,
                level="INFO",
                event_type="analyser_run_complete",
                message=summary_msg,
            )

            # ----------------------------------------------------------
            # Step 9: return result dict
            # ----------------------------------------------------------
            return {
                "success": True,
                "run_batch_id": self.run_batch_id,
                "total_scored": total_scored,
                "auto_route": auto_count,
                "manual_route": manual_count,
                "skipped": skip_count,
                "budget_aborted": budget_aborted,
                "routing_manifest": routing_manifest,
            }

        except Exception as exc:  # noqa: BLE001
            self.logger.critical(
                "AnalyserAgent.run: unhandled exception: %s", exc, exc_info=True
            )
            try:
                record_agent_error(
                    agent_type="AnalyserAgent",
                    error_message=str(exc),
                    run_batch_id=self.run_batch_id,
                    error_code="ANALYSER_UNHANDLED_EXCEPTION",
                )
            except Exception:  # noqa: BLE001
                pass
            return {"success": False, "error": str(exc)}

    # ------------------------------------------------------------------
    # Internal: crew output parser
    # ------------------------------------------------------------------

    def _parse_crew_output(self, crew_output: Any) -> dict[str, Any]:
        """
        Extract the routing manifest dict from a CrewAI ``CrewOutput`` object.

        CrewAI may return a ``CrewOutput`` (with a ``.raw`` attribute), a plain
        string, or a dict depending on version.  This method handles all three
        cases gracefully.

        Args:
            crew_output: Raw return value of ``Crew.kickoff()``.

        Returns:
            Parsed routing manifest dict.  Returns
            ``{"results": [], "budget_aborted": False}`` as a safe default if
            parsing fails.
        """
        default: dict[str, Any] = {"results": [], "budget_aborted": False}

        if crew_output is None:
            self.logger.warning("_parse_crew_output: crew_output is None")
            return default

        # Try .raw attribute (CrewOutput in newer crewai versions)
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
            self.logger.warning("_parse_crew_output: crew returned empty output")
            return default

        # Strip markdown code fences if present
        stripped: str = raw_str.strip()
        if stripped.startswith("```"):
            lines = stripped.splitlines()
            # Drop first and last fence lines
            inner_lines = lines[1:] if len(lines) > 1 else lines
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
                "_parse_crew_output: JSON decode failed: %s — raw snippet: %.200s",
                exc,
                raw_str,
            )
            return default

    # ------------------------------------------------------------------
    # Public: routing manifest for Apply Agent
    # ------------------------------------------------------------------

    def get_routing_manifest(self) -> list[dict[str, Any]]:
        """
        Return the routing manifest for all eligibility-passing jobs in this
        run batch.

        Queries Postgres for all job_posts in the current ``run_batch_id``
        where the corresponding ``job_scores`` row has ``eligibility_pass =
        TRUE``, ordered by ``fit_score DESC`` so the Apply Agent processes
        highest-confidence jobs first.

        Called by the Master Agent to hand off work to the Apply Agent after
        this scoring pass completes.

        Returns:
            List of dicts with keys: ``id``, ``title``, ``company``, ``url``,
            ``source_platform``, ``fit_score``, ``resume_id``,
            ``eligibility_pass``.  Empty list on query failure.
        """
        max_retries = 3
        last_exc: Optional[Exception] = None

        for attempt in range(max_retries):
            conn: Optional[psycopg2.extensions.connection] = None
            try:
                conn = _get_db_conn()
                cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
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
                    FROM job_posts jp
                    JOIN job_scores js ON js.job_post_id = jp.id
                    WHERE jp.run_batch_id = %s
                      AND js.eligibility_pass = TRUE
                    ORDER BY js.fit_score DESC
                    """,
                    (self.run_batch_id,),
                )
                rows = cursor.fetchall()
                manifest: list[dict[str, Any]] = []
                for row in rows:
                    manifest.append(
                        {
                            "id": str(row["id"]),
                            "title": row["title"],
                            "company": row["company"],
                            "url": row["url"],
                            "source_platform": row["source_platform"],
                            "fit_score": float(row["fit_score"] or 0.0),
                            "resume_id": str(row["resume_id"]) if row["resume_id"] else None,
                            "eligibility_pass": bool(row["eligibility_pass"]),
                        }
                    )
                self.logger.info(
                    "get_routing_manifest: returning %d eligible jobs for batch %s",
                    len(manifest),
                    self.run_batch_id,
                )
                return manifest

            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt < max_retries - 1:
                    sleep_s = 2 ** attempt
                    self.logger.warning(
                        "get_routing_manifest attempt %d/%d failed: %s — retry in %ds",
                        attempt + 1,
                        max_retries,
                        exc,
                        sleep_s,
                    )
                    time.sleep(sleep_s)
                else:
                    self.logger.error(
                        "get_routing_manifest failed after %d attempts: %s",
                        max_retries,
                        exc,
                    )
            finally:
                if conn:
                    try:
                        conn.close()
                    except Exception:  # noqa: BLE001
                        pass

        self.logger.error(
            "get_routing_manifest: returning empty list after exhausted retries. "
            "Last error: %s",
            last_exc,
        )
        return []
