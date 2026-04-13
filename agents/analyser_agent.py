"""
Analyser Agent for AI Job Application Agent.

The most logic-heavy agent in the pipeline.  Receives all job_posts discovered
by the Scraper Agent for the current ``pipeline_run_id``, scores each job against
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

# POST-AUDIT SURGICAL FIX — 2026-04-06 | S1:bootstrap S2:fetch S3:RAG S4:LLM S5:writeback S6:dedup | S7:skipped(agentops hold)

from __future__ import annotations

import asyncio
import json
import re
import logging
import os
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import islice
from typing import Any, Optional

import httpx
import math
import psycopg2
import psycopg2.extras
from crewai import Agent, Task, Crew, Process
import agentops
from agentops import agent, operation, tool

from integrations.llm_interface import LLMInterface
from tools.rag_tools import query_resume_match, _query_resume_match, get_resume_context, embed_job_description
from tools.postgres_tools import _log_event, save_job_score, get_run_stats, get_platform_config
from tools.budget_tools import check_xai_run_cap, record_llm_cost, get_cost_summary
from tools.agentops_tools import record_agent_error, record_fallback_event
from utils.db_utils import get_db_conn

logger = logging.getLogger(__name__)

__all__ = ["AnalyserAgent"]

# --- S3: RAG rate-limit constant (1.1s per RAG server enforcement) ---
RAG_CALL_DELAY: float = float(os.getenv("RAG_CALL_DELAY_SECONDS", "1.6"))
MAX_RAG_RETRIES: int = 3

# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _chunk(lst: list, n: int):
    """Yield successive n-sized chunks from lst.

    Args:
        lst: The list to split.
        n: Maximum items per chunk.

    Yields:
        Non-overlapping sublists of length <= n.
    """
    it = iter(lst)
    chunk = list(islice(it, n))
    while chunk:
        yield chunk
        chunk = list(islice(it, n))

# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

# LLM provider name map (model prefix → short name for cost recording)
_PROVIDER_NAME: dict[str, str] = {
    "xai": "xai",
    "sambanova": "sambanova",
    "cerebras": "cerebras",
}


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


class AnalyserAgent:
    """
    CrewAI Analyser Agent — eligibility filter, RAG resume matching, routing.

    Scores every job_post in the current run batch against 15 resume variants,
    assigns a fit_score between 0.0 and 1.0, routes each job to auto-apply /
    manual-review / skip, and persists results to Postgres.
    """

    def __init__(self, pipeline_run_id: str, user_id: str) -> None:
        """
        Initialise the Analyser Agent for a given run batch.

        Args:
            pipeline_run_id: UUID of the current run batch (from ``pipeline_runs``
                table), handed off by the Scraper Agent via the Master Agent.
            user_id: UUID of the candidate user (from ``users`` table).
        """
        self.pipeline_run_id = pipeline_run_id
        self.user_id = user_id
        self.llm_interface = LLMInterface()
        self.llm = self.llm_interface.get_llm("ANALYSER_AGENT")
        self.fallback_llm_1 = self.llm_interface.get_fallback_llm("ANALYSER_AGENT", level=1)
        self.fallback_llm_2 = self.llm_interface.get_fallback_llm("ANALYSER_AGENT", level=2)
        self.logger = logging.getLogger(self.__class__.__name__)
        self._current_llm = self.llm
        self._fallback_level: int = 0

        self.logger.info(
            "AnalyserAgent initialised — pipeline_run_id=%s user_id=%s",
            pipeline_run_id,
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
                conn = get_db_conn()
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
                    FROM jobs
                    WHERE pipeline_run_id = %s
                      AND id NOT IN (SELECT job_post_id FROM job_scores)
                    ORDER BY created_at ASC
                    """,
                    (self.pipeline_run_id,),
                )
                rows = cursor.fetchall()
                jobs: list[dict[str, Any]] = [dict(row) for row in rows]
                self.logger.info(
                    "_get_jobs_for_run: fetched %d jobs for batch %s",
                    len(jobs),
                    self.pipeline_run_id,
                )
                
                # BUG-07: Deduplication on (company.lower().strip(), title.lower().strip())
                seen = set()
                deduped = []
                dropped = 0
                for job in jobs:
                    company = str(job.get("company", "")).lower().strip()
                    title = str(job.get("title", "")).lower().strip()
                    key = (company, title)
                    if key not in seen:
                        seen.add(key)
                        deduped.append(job)
                    else:
                        dropped += 1
                        self.logger.info(f"Dedup: dropped duplicate job '{job.get('title')}' @ '{job.get('company')}'")
                
                self.logger.info(f"_get_jobs_for_run: {len(deduped)} unique jobs after dedup (dropped {dropped})")
                return deduped

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
            resume_filename: Resume filename such as ``"Aarjun_Gen.pdf"``.

        Returns:
            UUID string if found, ``None`` otherwise.
        """
        if not resume_filename:
            return None

        for attempt in range(1, 4):
            conn: Optional[psycopg2.extensions.connection] = None
            try:
                conn = get_db_conn()
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
                if attempt < 3:
                    time.sleep(2 ** attempt)
                    self.logger.warning(
                        "_resolve_resume_id attempt %d/3 failed: %s — retrying",
                        attempt, str(exc),
                    )
                else:
                    self.logger.error(
                        "_resolve_resume_id failed after 3 attempts: %s", str(exc)
                    )
            finally:
                if conn:
                    try:
                        conn.close()
                    except Exception:  # noqa: BLE001
                        pass
        return None

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
        conn: Optional[Any] = None,
    ) -> Optional[str]:
        """
        Write a job_scores row directly via psycopg2.

        Avoids the CrewAI @tool Pydantic validation layer, which does not
        accept ``None`` for a ``str``-typed ``resume_id`` field even though
        the underlying DB column is nullable.

        S5: ON CONFLICT (job_post_id) DO UPDATE SET added for idempotency.
        S5: Single connection + single commit for atomicity.

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
            _conn: Optional[psycopg2.extensions.connection] = None
            _is_local = False
            try:
                if conn is not None:
                    _conn = conn
                else:
                    _conn = get_db_conn()  # S2.4: uses get_db_conn() from utils.db_utils
                    _is_local = True
                cursor = _conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
                resume_used_value: Optional[str] = None
                try:
                    resume_used_value = (
                        str(reasons_json.get("resume_suggested"))
                        if reasons_json.get("resume_suggested") is not None
                        else None
                    )
                except Exception:  # noqa: BLE001
                    resume_used_value = None
                # S5.2: table name = job_scores (schema-verified), ON CONFLICT added
                cursor.execute(
                    """
                    INSERT INTO job_scores
                        (job_post_id, resume_id, fit_score, eligibility_pass, reasons_json, routed_at, resume_used)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (job_post_id) DO UPDATE SET
                        resume_id = EXCLUDED.resume_id,
                        fit_score = EXCLUDED.fit_score,
                        eligibility_pass = EXCLUDED.eligibility_pass,
                        reasons_json = EXCLUDED.reasons_json,
                        routed_at = EXCLUDED.routed_at,
                        resume_used = EXCLUDED.resume_used,
                        scored_at = NOW()
                    RETURNING id
                    """,
                    (
                        job_post_id,
                        resume_id,  # None → NULL (column is nullable FK)
                        fit_score,
                        eligibility_pass,
                        json.dumps(reasons_json),
                        datetime.utcnow(),
                        resume_used_value,
                    ),
                )
                result = cursor.fetchone()
                if _is_local:
                    _conn.commit()  # S5.3: single commit
                score_id = str(result["id"]) if result else None
                # BUG-05: Requirement: Log every successful INSERT to job_scores
                if score_id:
                    self.logger.info(f"job_scores saved: {job_post_id} score={fit_score}")

                self.logger.debug(
                    "_save_score_direct: saved job_score %s for job_post %s",
                    score_id,
                    job_post_id,
                )
                return score_id

            except Exception as exc:  # noqa: BLE001
                if _is_local and _conn:
                    try:
                        _conn.rollback()  # S5.3: rollback on error
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
                if _is_local and _conn:
                    try:
                        _conn.close()  # S5.3: always close
                    except Exception:  # noqa: BLE001
                        pass
        return None

    # ------------------------------------------------------------------
    # Internal: routing threshold logic
    # ------------------------------------------------------------------

    # S6: Older _create_manual_application_record DELETED (had bad schema: auditlogs/job_id columns).
    # Canonical version retained at bottom of file (L1602+).

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
        if fit_score < float(os.getenv("ANALYSER_MANUAL_THRESHOLD", "0.50")):
            return "skip"
        if fit_score < 0.60:
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
        default_resume = os.getenv("DEFAULT_RESUME", "Aarjun_Gen.pdf")

        fallback_result: dict[str, Any] = {
            "job_id": job.get("id"),
            "resume_id": None,
            "rag_score": 0.0,
            "form_complexity": None,
            "route": "manual_queue",
            "raw_response": "",
            "error": "scoring_failed",
            "job_post_id": job_post_id,
            "fit_score": 0.0,
            "resume_suggested": default_resume,
            "eligibility_pass": False,
            "reasons_json": {"error": "scoring_failed"},
        }

        try:
            # Call RAG match tool directly via .run()
            raw: str = query_resume_match.run(
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
            score_id = self._save_score_direct(
                job_post_id=job_post_id,
                resume_id=resume_id,
                fit_score=fit_score,
                eligibility_pass=eligibility_pass,
                reasons_json=reasons_json,
            )

            # === SURGICAL HANDOFF: Ensure manual-routed jobs reach TrackerAgent via applications table ===
            if route == "manual_queue":
                self._create_manual_application_record(
                    job_post_id=job_post_id,
                    source_platform=str(job.get("source_platform", "unknown")),
                    reason="analyser_manual_route",
                    resume_suggested=resume_suggested,
                )
            # === END SURGICAL HANDOFF ===

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
                "_score_job: unhandled exception for job %s: %s", job_post_id, exc
            )
            return fallback_result

    # ------------------------------------------------------------------
    # Internal: parallel scoring with ThreadPoolExecutor
    # ------------------------------------------------------------------

    def _score_jobs_parallel(
        self,
        jobs: list[dict[str, Any]],
        max_workers: int = 4,
    ) -> list[dict[str, Any]]:
        """Score multiple jobs concurrently using a thread pool.

        Falls back to sequential scoring if parallel execution fails.
        Each job is scored independently via ``_score_job``.

        Args:
            jobs: List of job dicts to score.
            max_workers: Number of concurrent scoring threads.
                Default 4 to balance throughput with rate limits.

        Returns:
            List of score result dicts, one per job.
        """
        results: list[dict[str, Any]] = []
        try:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_job = {}
                for job in jobs:
                    future_to_job[executor.submit(self._score_job, job)] = job
                    time.sleep(0.4)  # Throttling to protect RAG server
                
                for future in as_completed(future_to_job):
                    try:
                        result = future.result(timeout=60)
                        results.append(result)
                    except Exception as exc:  # noqa: BLE001
                        job = future_to_job[future]
                        self.logger.error(
                            "_score_jobs_parallel: job %s failed: %s",
                            job.get("id", "?"),
                            exc,
                        )
                        results.append({
                            "job_id": job.get("id"),
                            "resume_id": None,
                            "rag_score": 0.0,
                            "form_complexity": None,
                            "route": "manual_queue",
                            "raw_response": "",
                            "error": str(exc),
                            "job_post_id": str(job.get("id", "")),
                            "fit_score": 0.0,
                            "resume_suggested": os.getenv("DEFAULT_RESUME", "Aarjun_Gen.pdf"),
                            "eligibility_pass": False,
                            "reasons_json": {"error": str(exc)},
                        })
        except Exception as exc:  # noqa: BLE001
            self.logger.warning(
                "_score_jobs_parallel: thread pool failed, falling back to "
                "sequential: %s",
                exc,
            )
            for job in jobs:
                results.append(self._score_job(job))

        return results

    # ------------------------------------------------------------------
    # Internal: LLM fallback chain (Fix 4)
    # ------------------------------------------------------------------

    async def _call_llm(
        self,
        prompt: str,
        purpose: str = "analyser_job_scoring",
        max_tokens: int = 2048,
        pipeline_run_id: Optional[str] = None,
    ) -> str:
        """Call the configured LLM provider chain with exponential backoff.

        Attempts ANALYSER_LLM_PRIMARY first, then ANALYSER_LLM_FALLBACK_1,
        then ANALYSER_LLM_FALLBACK_2. Logs a warning between every attempt.
        Raises RuntimeError if all providers fail.

        Args:
            prompt: The prompt string to send to the LLM.
            purpose: A string identifying the purpose of the call for logging/budgeting.
            max_tokens: The maximum number of tokens for the response.

        Returns:
            Raw response string from the first successful provider.

        Raises:
            RuntimeError: When all configured providers are exhausted.
        """
        last_error: Optional[Exception] = None
        
        for attempt in range(3): # 0=primary, 1=fallback1, 2=fallback2
            try:
                response = await self.llm_interface.complete(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    purpose=purpose,
                    pipeline_run_id=pipeline_run_id or self.pipeline_run_id,
                )
                if attempt > 0:
                    self.logger.info(
                        "LLM call succeeded on attempt %d (fallback level %d)",
                        attempt + 1, attempt
                    )
                return response
            except Exception as exc:
                last_error = exc
                self.logger.warning(
                    "LLM attempt %d failed: %s",
                    attempt + 1,
                    exc,
                )
                
                # Switch to next fallback LLM for the *next* complete() call
                # The llm_interface should handle the actual switching logic
                # This agent's responsibility is to retry
                can_fallback = self._switch_to_fallback(
                    failed_provider=str(self._current_llm)
                )

                if not can_fallback:
                    self.logger.error("All LLM providers failed. Last error: %s", last_error)
                    raise RuntimeError("LLM fallback chain exhausted") from last_error
                
                await asyncio.sleep(2 ** attempt)

        self.logger.error("All LLM providers failed after retries. Last: %s", last_error)
        raise RuntimeError("LLM fallback chain exhausted") from last_error

    # ------------------------------------------------------------------
    # Internal: per-job scoring fallback (Fix 3)
    # ------------------------------------------------------------------

    async def _score_jobs_individually(
        self,
        jobs: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Score jobs one at a time using _call_llm when batch parse fails.

        Args:
            jobs: List of job dicts to score individually.

        Returns:
            List of score dicts (same format as batch parse output).
        """
        results: list[dict[str, Any]] = []
        for idx, job in enumerate(jobs, 1):
            prompt = (
                "SYSTEM: YOU ARE A JSON-ONLY RESPONSE ENGINE. DO NOT TALK. DO NOT EXPLAIN. "
                "OUTPUT ONLY RAW JSON. IF YOU ADD CONVERSATIONAL TEXT, THE SYSTEM WILL CRASH.\n\n"
                "HARD GATE — APPLY BEFORE ALL OTHER SCORING:\n"
                "If the job title and full description do NOT explicitly mention at least one of:\n"
                "[AI, Artificial Intelligence, Machine Learning, ML, Data Science, LLM, "
                "NLP, Computer Vision, Automation, MLOps, Data Engineering, GenAI, "
                "Deep Learning, Neural Network, Python in an AI/DS/ML context]\n"
                "then fit_score MUST be set to 0.20 or below. No exceptions.\n"
                "This applies regardless of company prestige, compensation, or other factors.\n"
                "Manufacturing, hardware engineering, full-stack web, sales, GTM, telecom "
                "infrastructure, audit, compliance, and program management roles that do not "
                "mention the above keywords must score 0.20 or below.\n\n"
                "CANDIDATE CONSTRAINTS (NON-NEGOTIABLE):\n"
                "- Experience Level: Fresher (0.5 Years Total).\n"
                "- Hard Exclusions: If job title contains 'Senior', 'Staff', 'Principal', 'Lead', 'Manager', "
                "'Director', or 'VP', YOU MUST score it < 0.40 and set route to 'skip'.\n"
                "- Reject jobs requiring 2+ years of experience immediately (score < 0.40).\n\n"
                "You are a job fit scorer. Score this job for candidate fit.\n"
                "Return ONLY a valid JSON object (no markdown, no explanation).\n"
                'Format: {"job_index": 1, "fit_score": 0.85, '
                '"route": "auto", "reason": "strong Python/ML match"}\n\n'
                "Rules:\n"
                "  fit_score: 0.0 to 1.0\n"
                "  route: manual (fit_score>=0.90),auto (fit_score>=0.60), "
                "manual (0.40-0.60), skip (<0.40)\n\n"
                f"Job to score:\n"
                f"[{idx}] {job.get('title', 'N/A')} at {job.get('company', 'N/A')}\n"
                f"Description: {str(job.get('description', ''))[:800]}\n"
            )
            try:
                response_text = await self._call_llm(
                    prompt=prompt,
                    purpose="analyser_job_scoring_individual",
                    pipeline_run_id=self.pipeline_run_id,
                )
                score = json.loads(response_text)
                score["job_index"] = idx
                results.append(score)
            except Exception as exc:
                self.logger.warning(
                    "_score_jobs_individually: failed for job %s: %s — skipping",
                    job.get("id", "?"),
                    exc,
                )
        return results


    def _get_candidate_context(self) -> str:
        """
        Load candidate profile from userprofile.json and format context strings.
        Used by batch and recursive halving scoring pipelines.
        """
        # ── Load candidate profile from ConfigLoader (fail-soft) ────────────
        try:
            from config.config_loader import config_loader as _cfg
            profile: dict = _cfg.user or {}
        except Exception as _e:
            self.logger.warning(
                "_get_candidate_context: could not load user profile, "
                "scoring will use safe defaults: %s", _e,
            )
            profile = {}

        # ── Extract profile fields with safe defaults ────────────────────────
        candidate_name:   str       = profile.get("personal", {}).get("name", "Candidate")
        years_exp:        int       = int(profile.get("formanswers", {}).get("yearsofexperience", 0) or 0)
        target_roles:     list[str] = profile.get("preferences", {}).get("targetroles", []) or []
        work_type:        str       = profile.get("preferences", {}).get("worktype", "remote") or "remote"
        blacklisted_cos:  list[str] = profile.get("preferences", {}).get("blacklistedcompanies", []) or []
        salary_floor_usd: int       = int(profile.get("preferences", {}).get("salaryfloorusd", 0) or 0)
        salary_floor_inr: int       = int(profile.get("preferences", {}).get("salaryfloorinr", 0) or 0)

        # ── Derive seniority tier dynamically from YOE ───────────────────────
        if years_exp == 0:
            seniority_label    = "Entry-level / Fresher (0 YOE)"
            hard_reject_titles = [
                "Senior", "Staff", "Principal", "Lead", "Director",
                "Manager", "Head of", "VP", "Vice President", "C-level",
            ]
            max_required_yoe = 1
        elif years_exp <= 2:
            seniority_label    = f"Junior ({years_exp} YOE)"
            hard_reject_titles = [
                "Staff", "Principal", "Director", "Manager",
                "Head of", "VP", "Vice President", "C-level",
            ]
            max_required_yoe = 3
        else:
            seniority_label    = f"Mid-level ({years_exp} YOE)"
            hard_reject_titles = ["Director", "VP", "Vice President", "C-level"]
            max_required_yoe   = years_exp + 2

        # ── Build candidate context block ───────────────────────────────────
        return (
            "CANDIDATE PROFILE — READ THIS BEFORE SCORING ANY JOB\n"
            "======================================================\n"
            f"Name            : {candidate_name}\n"
            f"Seniority       : {seniority_label}\n"
            "Target roles    : "
            f"{', '.join(target_roles) if target_roles else 'Any AI/ML/Data role'}\n"
            f"Preferred work  : {work_type}\n"
            f"Salary floor    : ${salary_floor_usd:,} USD  /  "
            f"₹{salary_floor_inr:,} INR (0 = no floor)\n\n"
            "HARD REJECT — set fit_score ≤ 0.30 immediately, no exceptions:\n"
            f"  • Job title contains any of: {', '.join(hard_reject_titles)}\n"
            f"  • Description requires more than {max_required_yoe} years experience\n"
            "  • Company is blacklisted: "
            f"{', '.join(blacklisted_cos) if blacklisted_cos else 'none'}\n\n"
            "SCORING BONUSES:\n"
            "  • +0.05 if position is remote and preferred work type is remote\n"
            "  • +0.10 if job title directly matches one of the target roles\n\n"
            "SCORING CALIBRATION:\n"
            "  • 0.60+ means candidate is realistically eligible to apply NOW\n"
            "  • Do NOT inflate score on skills alone if seniority is a blocker\n"
        )


    def _build_batch_prompt(self, jobs: list[dict[str, Any]]) -> str:
        """Serializes a chunk of jobs into a schema-verified batch prompt."""
        candidate_context = self._get_candidate_context()

        batch_prompt: str = (
            "SYSTEM: YOU ARE A JSON-ONLY RESPONSE ENGINE. DO NOT TALK. "
            "DO NOT EXPLAIN. OUTPUT ONLY RAW JSON. "
            "IF YOU ADD CONVERSATIONAL TEXT, THE SYSTEM WILL CRASH.\n\n"
            f"{candidate_context}\n"
            "HARD GATE — APPLY BEFORE ALL OTHER SCORING:\n"
            "If the job title and description do NOT explicitly mention at least one of:\n"
            "[AI, Artificial Intelligence, Machine Learning, ML, Data Science, LLM, "
            "NLP, Computer Vision, Automation, MLOps, Data Engineering, GenAI, "
            "Deep Learning, Neural Network, Python in an AI/DS/ML context]\n"
            "then fit_score MUST be set to 0.20 or below. No exceptions.\n"
            "Manufacturing, hardware, full-stack web, sales, GTM, telecom, "
            "audit, compliance, and program management roles without the above "
            "keywords must score 0.20 or below.\n\n"
            "Score each job for candidate fit. "
            "Return ONLY a valid JSON array, one object per job, same order as input.\n"
            'Format: [{"job_index": 1, "fit_score": 0.85, '
            '"route": "auto", "reason": "strong Python/ML match"}, ...]\n\n'
            "Routing rules:\n"
            "  fit_score >= 0.90  → route = manual_queue  (too senior, always review)\n"
            "  fit_score >= 0.60  → route = auto\n"
            "  fit_score >= 0.45  → route = manual_queue\n"
            "  fit_score <  0.45  → route = skip\n\n"
            "Jobs to score:\n"
        )
        for idx, job in enumerate(jobs, 1):
            batch_prompt += (
                f"[{idx}] {job.get('title', 'N/A')} at {job.get('company', 'N/A')}\n"
                f"Description: {str(job.get('description', ''))[:800]}\n---\n"
            )
        return batch_prompt


    async def _score_chunk_with_halving(
        self, jobs: list[dict[str, Any]], depth: int = 0
    ) -> list[dict[str, Any]]:
        """
        Recursively scores chunks, halving on JSON parse failure or token limit truncation.
        Falls back to individual scoring at depth 4 or when 1 job remains.
        """
        if len(jobs) == 1 or depth >= 4:
            return await self._score_jobs_individually(jobs)

        try:
            prompt = self._build_batch_prompt(jobs)
            response_text: str = await self._call_llm(
                prompt=prompt,
                purpose="analyser_job_scoring_batch",
                pipeline_run_id=self.pipeline_run_id,
            )
            scores = json.loads(response_text)
            if not isinstance(scores, list):
                raise ValueError("Recursive halving: response is not a JSON array")
            return scores
        except (json.JSONDecodeError, ValueError, RuntimeError) as exc:
            mid = len(jobs) // 2
            self.logger.info(
                "Halving batch depth=%d: %d → %d+%d (Reason: %s)",
                depth, len(jobs), mid, len(jobs) - mid, exc
            )
            left = await self._score_chunk_with_halving(jobs[:mid], depth + 1)
            right = await self._score_chunk_with_halving(jobs[mid:], depth + 1)
            return left + right

    # ------------------------------------------------------------------
    # Internal: batch scoring via LLM (Fix 3)
    # ------------------------------------------------------------------

    async def _run_batch_scoring(
        self,
        jobs: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Score a list of jobs in batches using _call_llm with the xAI
        fallback chain. Injects candidate profile context from userprofile.json
        into every batch prompt. Falls back to _score_jobs_individually on
        JSON parse failure. Fail-soft on all errors.

        Args:
            jobs: Full list of raw job dicts for the current run.

        Returns:
            List of dicts each containing job_post_id, fit_score, route,
            and reason — ready for DB write.
        """
        # ── Batch loop ───────────────────────────────────────────────────────
        _BATCH_SIZE: int = 20
        all_results: list[dict[str, Any]] = []

        for batch in _chunk(jobs, _BATCH_SIZE):
            # ── LLM call with recursive halving fallback ─────────────────────
            try:
                batch_prompt = self._build_batch_prompt(batch)
                response_text: str = await self._call_llm(
                    prompt=batch_prompt,
                    purpose="analyser_job_scoring_batch",
                    pipeline_run_id=self.pipeline_run_id,
                )
                scores = json.loads(response_text)
                if not isinstance(scores, list):
                    raise ValueError("Main batch loop: response is not a JSON array")
            except (json.JSONDecodeError, ValueError, RuntimeError) as exc:
                self.logger.warning(
                    "_run_batch_scoring: batch scoring failed (%s) "
                    "— triggering recursive halving",
                    exc,
                )
                scores = await self._score_chunk_with_halving(batch, depth=0)

            # ── Map results back to jobs ──────────────────────────────────────
            _VALID_ROUTES: frozenset[str] = frozenset({"auto", "manual_queue", "skip"})
            batch_scored_ids: set[str] = set()

            for result in scores:
                if not isinstance(result, dict):
                    continue

                try:
                    job_idx: int = int(result.get("job_index", -1)) - 1
                except (TypeError, ValueError):
                    job_idx = -1

                if job_idx < 0 or job_idx >= len(batch):
                    self.logger.warning(
                        "_run_batch_scoring: invalid job_index=%s in LLM "
                        "result — skipping entry",
                        result.get("job_index", "MISSING"),
                    )
                    continue

                job         = batch[job_idx]
                job_post_id = str(job.get("id", ""))
                fit_score   = max(0.0, min(1.0, float(result.get("fit_score", 0.0) or 0.0)))
                route       = str(result.get("route", "skip") or "skip").strip()

                # ── Route validator ──────────────────────────────────────────
                if route == "manual":
                    self.logger.info(
                        "_run_batch_scoring: normalised 'manual' → "
                        "'manual_queue' for job %s", job_post_id,
                    )
                    route = "manual_queue"
                elif route not in _VALID_ROUTES:
                    self.logger.warning(
                        "_run_batch_scoring: unknown route '%s' → 'skip' "
                        "for job %s", route, job_post_id,
                    )
                    route = "skip"

                # Hard floor — only catches extreme contradictions
                if fit_score < 0.30 and route == "auto":
                    self.logger.warning(
                        "_run_batch_scoring: score=%.2f too low for auto "
                        "→ manual_queue for job %s", fit_score, job_post_id,
                    )
                    route = "manual_queue"
                if fit_score >= 0.85 and route == "skip":
                    self.logger.warning(
                        "_run_batch_scoring: score=%.2f too high to skip "
                        "→ manual_queue for job %s", fit_score, job_post_id,
                    )
                    route = "manual_queue"
                # ── End route validator ──────────────────────────────────────

                reason = str(result.get("reason", "") or "")
                all_results.append({
                    "job_post_id": job_post_id,
                    "fit_score":   fit_score,
                    "route":       route,
                    "reason":      reason,
                })
                batch_scored_ids.add(job_post_id)

            # ── Fail-soft for jobs LLM did not return a result for ───────────
            for unmapped in batch:
                unmapped_id = str(unmapped.get("id") or unmapped.get("job_post_id", ""))
                if unmapped_id not in batch_scored_ids:
                    self.logger.warning(
                        "_run_batch_scoring: job_id=%s got no LLM score "
                        "— defaulting to manual_queue",
                        unmapped_id,
                    )
                    all_results.append({
                        "job_post_id": unmapped_id,
                        "fit_score":   0.0,
                        "route":       "manual_queue",
                        "reason":      "unbatch_no_llm_result",
                    })

        return all_results

    # ------------------------------------------------------------------
    # Internal: LLM fallback chain (CrewAI-level switch)
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
            getattr(record_fallback_event, "run", record_fallback_event)(
                agent_type="AnalyserAgent",
                from_provider=failed_provider,
                to_provider=str(to_model),
                pipeline_run_id=self.pipeline_run_id,
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
            getattr(record_fallback_event, "run", record_fallback_event)(
                agent_type="AnalyserAgent",
                from_provider=failed_provider,
                to_provider=str(to_model),
                pipeline_run_id=self.pipeline_run_id,
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

    _ANALYSER_FALLBACK_CHAIN: list[tuple[str, str]] = [
        (os.getenv("XAI_API_KEY", ""),
         os.getenv("XAI_DEFAULT_MODEL")),
        (os.getenv("SAMBANOVA_API_KEY", ""),
         os.getenv("SAMBANOVA_MODEL")),
        (os.getenv("CEREBRAS_API_KEY", ""),
         os.getenv("CEREBRAS_MODEL")),
    ]

    async def _call_with_fallback(
        self,
        prompt: str,
        pipeline_run_id: str,
        purpose: str = "analyser_fallback_call"
    ) -> str:
        """
        Wrap the primary LLM call with a retry and fallback mechanism.
        
        This method is the designated entry point for direct LLM calls
        from the analyser's `run` method, ensuring resilience.
        """
        last_exc: Exception | None = None
        for attempt in range(3): # 3 attempts total
            try:
                # Use the new _call_llm which has the fallback logic built-in
                response = await self._call_llm(
                    prompt=prompt,
                    purpose=f"{purpose}_attempt_{attempt+1}",
                )
                return response
            except Exception as exc:
                last_exc = exc
                wait = 2.0 ** attempt
                self.logger.warning(
                    "Analyser _call_with_fallback attempt %d failed: %s — retrying in %.1fs",
                    attempt + 1, exc, wait,
                )
                await asyncio.sleep(wait)
        
        self.logger.critical(
            "Analyser _call_with_fallback failed after all attempts. Last error: %s",
            last_exc
        )
        raise RuntimeError(
            "Analyser LLM call failed after all retries and fallbacks."
        ) from last_exc

    # ------------------------------------------------------------------
    # Agent / task builders
    # ------------------------------------------------------------------

    def _build_agent(self) -> Agent:
        self.llm = self.llm_interface.get_llm("ANALYSER_AGENT")
        if self._fallback_level == 1:
            self._current_llm = self.llm_interface.get_fallback_llm("ANALYSER_AGENT", level=1)
        elif self._fallback_level == 2:
            self._current_llm = self.llm_interface.get_fallback_llm("ANALYSER_AGENT", level=2)
        else:
            self._current_llm = self.llm
        
        """
        Build the CrewAI Agent instance for the analyser pass.

        Returns:
            Configured ``Agent`` using the currently active LLM
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
            Configured ``Task`` ready for crew execution.
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
pipeline_run_id: {self.pipeline_run_id}

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
   - pipeline_run_id: {self.pipeline_run_id}
   If the response JSON contains "abort": true — STOP immediately and
   return partial results with "budget_aborted": true.

5. AFTER every xAI LLM call, estimate the cost (typically $0.001–$0.005
   per call) and call `record_llm_cost` with:
   - provider: "xai"
   - cost_usd: [estimated cost]
   - agent_type: "AnalyserAgent"
   - pipeline_run_id: {self.pipeline_run_id}
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
        2. Fetch all job posts for ``pipeline_run_id`` from Postgres.
        3. Early-return if no jobs found.
        4. Build CrewAI agent + task.
        5. Execute crew (with LLM fallback on provider failure).
        6. Parse routing manifest from crew output.
        7. Persist each scored result to ``job_scores`` via ``_save_score_direct``.
        8. Log run complete with summary counts.
        9. Return structured result dict.

        Returns:
            Dict with keys: ``success``, ``pipeline_run_id``, ``total_scored``,
            ``auto_route``, ``manual_route``, ``skipped``,
            ``budget_aborted``, ``routing_manifest``.
            On failure: ``{"success": False, "error": str}``.
        """
        try:
            # ----------------------------------------------------------
            # Step 1: log start
            # ----------------------------------------------------------
            _log_event(
                pipeline_run_id=self.pipeline_run_id,
                level="INFO",
                event_type="analyser_run_start",
                message="Analyser Agent starting scoring pass",
            )
            self.logger.info(
                "AnalyserAgent.run: starting pass for batch %s", self.pipeline_run_id
            )

            # ----------------------------------------------------------
            # S4.4: Budget gate — check ONCE before the job loop
            # ----------------------------------------------------------
            _budget_check_raw: str = check_xai_run_cap(pipeline_run_id=self.pipeline_run_id)
            try:
                _budget_check: dict = json.loads(_budget_check_raw)
            except (json.JSONDecodeError, TypeError, ValueError):
                _budget_check = {"abort": False}
            _budget_ok: bool = not bool(_budget_check.get("abort", False))
            if not _budget_ok:
                self.logger.warning(
                    "xAI budget cap reached — all jobs routed manual_queue run_id=%s",
                    self.pipeline_run_id,
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
                    self.pipeline_run_id,
                )
                _log_event(
                    pipeline_run_id=self.pipeline_run_id,
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
            self.logger.info("Passing %d jobs to batch scoring engine...", len(jobs))
            
            import asyncio
            import nest_asyncio
            nest_asyncio.apply()
            
            # Execute batch scoring directly without CrewAI overhead
            results_raw: list[dict[str, Any]] = asyncio.run(
                self._run_batch_scoring(jobs)
            )
            budget_aborted: bool = False

            self.logger.info(
                "AnalyserAgent.run: LLM returned %d scored results "
                "(budget_aborted=%s)",
                len(results_raw),
                budget_aborted,
            )

            # Build index of LLM results keyed by job_post_id for easy lookup
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
                llm_result: Optional[dict[str, Any]] = result_index.get(job_post_id)

                fit_score = 0.0
                route = "skip"
                reason = "scoring_failed"

                # S4.4: Budget cap gate — bypass LLM if budget exceeded
                if not _budget_ok:
                    route, fit_score, reason = "manual_queue", 0.0, "budget_cap_reached"
                elif llm_result:
                    fit_score = float(llm_result.get("fit_score", 0.0))
                    fit_score = max(0.0, min(1.0, fit_score))
                    reason = str(llm_result.get("reason", ""))
                else:
                    self.logger.debug(
                        "AnalyserAgent.run: job %s missing from LLM results",
                        job_post_id,
                    )

                # S4.2: NaN/Inf guard on fit_score
                try:
                    fit_score = float(fit_score)
                    if math.isnan(fit_score) or math.isinf(fit_score):
                        fit_score = 0.0
                except (ValueError, TypeError):
                    fit_score = 0.0

                # S4.3: Hard threshold block — overrides LLM route
                _SKIP_THRESHOLD   = float(os.getenv("ANALYSER_SKIP_THRESHOLD",   "0.40"))
                _MANUAL_THRESHOLD = float(os.getenv("ANALYSER_MANUAL_THRESHOLD", "0.50"))

                if fit_score < _SKIP_THRESHOLD:
                    route = "skip"
                elif fit_score < _MANUAL_THRESHOLD:
                    route = "manual_queue"
                elif fit_score >= 0.90:
                    route = "manual_queue"
                else:
                    # 0.60–0.89: canonical _determine_route (handles complexity)
                    route = self._determine_route(fit_score)

                eligibility_pass = fit_score >= 0.40

                default_resume = os.getenv("DEFAULT_RESUME", "Aarjun_Gen.pdf")
                resume_suggested = default_resume
                match_reasoning = reason
                similarity_score = fit_score

                # S3: Sequential RAG gate — strictly one job at a time, 1.1s rate limit enforced
                if fit_score >= 0.50:
                    try:

                        # S3 HARD LOCK: 1.1s delay BEFORE each RAG call
                        # (placed here so it always fires even on retry exit)
                        rag_url: str = os.getenv("RAG_SERVER_URL", "http://ai_rag_server:8090")
                        rag_headers = {
                            # S3.3: correct header key
                            "X-RAG-API-Key": os.getenv("RAG_API_KEY", ""),
                            "Content-Type": "application/json",
                        }
                        # S3.4: payload with session_id, correct field names, env-driven top_k
                        rag_payload: dict[str, Any] = {
                            "session_id": str(self.pipeline_run_id),
                            "job_description": str(job.get("description") or job.get("title", "")),
                            "query": str(job.get("title", "")),
                            "top_k": int(os.getenv("RAG_TOP_K", "3")),
                        }

                        rag_result: Optional[dict[str, Any]] = None
                        last_rag_error: Optional[str] = None

                        for rag_attempt in range(MAX_RAG_RETRIES):
                            try:
                                # S3.5: httpx (replaces any requests usage)
                                resp = httpx.post(
                                    rag_url + "/rag/query",
                                    headers=rag_headers,
                                    json=rag_payload,
                                    timeout=30.0,
                                )
                                resp.raise_for_status()
                                rag_result = resp.json()
                                self.logger.debug("RAG raw result: %s", rag_result)

                                if not rag_result.get("success"):
                                    self.logger.warning(
                                        "RAG returned success=false for job_id=%s — response: %s",
                                        job_post_id, rag_result
                                    )
                                    raise ValueError(f"RAG success=false for job_id={job_post_id}")
                                break
                            except httpx.TimeoutException as e:
                                last_rag_error = f"timeout:{e}"
                                self.logger.warning(
                                    "RAG timeout job_id=%s attempt=%d/%d",
                                    job_post_id, rag_attempt + 1, MAX_RAG_RETRIES,
                                )
                            except httpx.HTTPStatusError as e:
                                last_rag_error = f"http_{e.response.status_code}"
                                self.logger.warning(
                                    "RAG HTTP error job_id=%s status=%d attempt=%d/%d",
                                    job_post_id, e.response.status_code,
                                    rag_attempt + 1, MAX_RAG_RETRIES,
                                )
                                self.logger.warning(
                                    "RAG HTTP error body: %s",
                                    e.response.text[:500] if hasattr(e.response, "text") else "no body"
                                )
                                if e.response.status_code == 429:
                                    time.sleep(2.0 * (rag_attempt + 1))
                            except httpx.ConnectError as e:
                                last_rag_error = f"connect:{e}"
                                self.logger.warning(
                                    "RAG connect error job_id=%s attempt=%d/%d",
                                    job_post_id, rag_attempt + 1, MAX_RAG_RETRIES,
                                )
                            except Exception as e:  # noqa: BLE001
                                last_rag_error = f"unexpected:{e}"
                                self.logger.error(
                                    "RAG unexpected error job_id=%s error=%s",
                                    job_post_id, e,
                                )
                            if rag_attempt < MAX_RAG_RETRIES - 1:
                                time.sleep(1.6 * (2 ** rag_attempt))  # backoff: 1.6s → 3.2s → 6.4s

                        # S3 HARD LOCK: 1.1s rate limit — ALWAYS fires, every job
                        time.sleep(RAG_CALL_DELAY)

                        if rag_result is None:
                            self.logger.error(
                                "RAG failed all retries job_id=%s last_error=%s using fallback",
                                job_post_id, last_rag_error,
                            )
                            # fallback: keep existing fit_score, default resume
                        else:
                            chunks = rag_result.get("chunks", [])
                            if not chunks:
                                self.logger.warning(
                                    "RAG returned empty chunks for job_id=%s — applying fallback",
                                    job_post_id
                                )
                                raise ValueError(f"RAG empty chunks for job_id={job_post_id}")

                            best_chunk    = chunks[0]
                            rag_score     = float(best_chunk.get("score", 0.0))
                            best_resume_id = (
                                best_chunk.get("resume_id")
                                or best_chunk.get("metadata", {}).get("resume_id")
                                or ""
                            )
                            matched_label     = best_chunk.get("metadata", {}).get("label", "")
                            matched_role_focus = best_chunk.get("metadata", {}).get("role_focus", "")
                            similarity_score  = rag_score

                            # Weighted aggregation across all chunks — top chunk weighted 60%, rest split 40%
                            if len(chunks) > 1:
                                top_weight   = 0.60
                                rest_weight  = 0.40 / (len(chunks) - 1)
                                similarity_score = top_weight * rag_score + rest_weight * sum(
                                    float(c.get("score", 0.0)) for c in chunks[1:]
                                )
                                self.logger.debug(
                                    "RAG multi-chunk aggregation: job_id=%s chunks=%d final_score=%.4f",
                                    job_post_id, len(chunks), similarity_score,
                                )

                            self.logger.info(
                                "RAG match: job_id=%s resume=%s label=%s score=%.4f processing_ms=%s",
                                job_post_id,
                                best_resume_id,
                                matched_label,
                                similarity_score,
                                rag_result.get("processing_time_ms", "n/a"),
                            )

                            if best_resume_id and best_resume_id != default_resume:
                                resume_suggested = best_resume_id

                            # Fallback for match_reasoning
                            match_reasoning = str(best_chunk.get("metadata", {}).get("match_reasoning") or (rag_result.get("metadata", {}) or {}).get("match_reasoning") or reason)

                    except Exception as e:  # noqa: BLE001
                        self.logger.error("RAG pipeline failed for job %s: %s", job_post_id, e)

                reasons_json: dict[str, Any] = {
                    "fit_score": fit_score,
                    "route": route,
                    "resume_suggested": resume_suggested,
                    "reason": reason,
                    "match_reasoning": match_reasoning,
                    "similarity_score": similarity_score,
                    "source": "llm_batch_then_rag",
                }

                manifest_entry: dict[str, Any] = {
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

                resume_id: Optional[str] = self._resolve_resume_id(resume_suggested)
                auto_app_written: Optional[str] = None

                _shared_conn = None
                try:
                    _shared_conn = get_db_conn()
                    _shared_conn.autocommit = False

                    score_ok = self._save_score_direct(
                        job_post_id=job_post_id,
                        resume_id=resume_id,
                        fit_score=fit_score,
                        eligibility_pass=eligibility_pass,
                        reasons_json=reasons_json,
                        conn=_shared_conn
                    )

                    # S6: manual_queue handoff — condition is route == "manual_queue" OR route == "manual"
                    if route in ("manual_queue", "manual"):
                        self._create_manual_application_record(
                            job_post_id=job_post_id,
                            source_platform=str(job.get("source_platform", "unknown")),
                            reason="analyser_manual_route",
                            resume_id=resume_id,
                            conn=_shared_conn
                        )

                    _shared_conn.commit()
                    self.logger.debug("Atomic write committed — job_post_id=%s", job_post_id)
                except Exception as _atomic_exc:
                    if _shared_conn:
                        try:
                            _shared_conn.rollback()
                        except Exception:
                            pass
                    self.logger.error(
                        "Atomic write failed — job_post_id=%s error=%s — job NOT persisted",
                        job_post_id, _atomic_exc
                    )
                finally:
                    if _shared_conn:
                        try:
                            _shared_conn.close()
                        except Exception:
                            pass

                if route == "auto":
                    auto_count += 1
                elif route in ("manual", "manual_queue"):
                    manual_count += 1
                else:
                    skip_count += 1

                routing_manifest.append(manifest_entry)

            total_scored: int = len(jobs)

            # ----------------------------------------------------------
            # Top-100 filter: cap routing manifest to 100 highest-scoring
            # ----------------------------------------------------------
            _MAX_MANIFEST: int = 100
            max_manifest = _MAX_MANIFEST
            routing_manifest.sort(
                key=lambda x: float(x.get("fit_score", 0.0)),
                reverse=True,
            )
            if len(routing_manifest) > max_manifest:
                overflow = len(routing_manifest) - max_manifest
                self.logger.info(
                    "Top-%d filter: capping manifest from %d (dropping %d low-score)",
                    max_manifest,
                    len(routing_manifest),
                    overflow,
                )
                routing_manifest = routing_manifest[:max_manifest]

            # ----------------------------------------------------------
            # Step 8: log completion
            # ----------------------------------------------------------
            summary_msg: str = (
                f"Analyser pass complete — total={total_scored} "
                f"auto={auto_count} manual={manual_count} skip={skip_count} "
                f"budget_aborted={budget_aborted}"
            )
            self.logger.info("AnalyserAgent.run: %s", summary_msg)
            _log_event(
                pipeline_run_id=self.pipeline_run_id,
                level="INFO",
                event_type="analyser_run_complete",
                message=summary_msg,
            )

            # ----------------------------------------------------------
            # Step 9: return result dict
            # ----------------------------------------------------------
            return {
                "success": True,
                "pipeline_run_id": self.pipeline_run_id,
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
                getattr(record_agent_error, "run", record_agent_error)(
                    agent_type="AnalyserAgent",
                    error_message=str(exc),
                    pipeline_run_id=self.pipeline_run_id,
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

        raw_str = str(crew_output.raw if hasattr(crew_output, "raw") else crew_output)
        
        # Robust Regex Extraction: Find the first JSON object or array
        match = re.search(r'(\{.*\}|\[.*\])', raw_str, re.DOTALL)
        stripped = match.group(1) if match else raw_str.strip()

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

        Queries Postgres for all job_posts in the current ``pipeline_run_id``
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
                conn = get_db_conn()
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
                    FROM jobs jp
                    JOIN job_scores js ON js.job_post_id = jp.id
                    WHERE jp.pipeline_run_id = %s
                      AND js.eligibility_pass = TRUE
                    ORDER BY js.fit_score DESC
                    """,
                    (self.pipeline_run_id,),
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
                    self.pipeline_run_id,
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

    # ------------------------------------------------------------------
    # Internal: manual application persistence (S6 canonical)
    # ------------------------------------------------------------------


    def _create_manual_application_record(
        self,
        job_post_id: str,
        source_platform: str,
        reason: str = "analyser_manual_route",
        resume_id: Optional[str] = None,
        conn: Optional[Any] = None,
    ) -> Optional[str]:
        """
        Create an application record for a manually routed job (S6 canonical).

        Schema-verified against database/schema.sql.
        applications columns: id, job_post_id, resume_id, user_id, mode, status, platform,
                               submitted_at, error_code.
        S5.4/S6: Uses correct column names. mode is required (CHECK: 'auto'|'manual').
        Duplicate guard via SELECT before INSERT for idempotency (no unique index exists
        on job_post_id+user_id in schema for ON CONFLICT).

        Args:
            job_post_id: UUID of the job post.
            source_platform: The platform the job was sourced from (e.g., \"linkedin\").
            reason: A code indicating why it was manually routed.

        Returns:
            UUID of the created application record, or None on failure.
        """
        max_retries = 3
        last_exc: Optional[Exception] = None

        for attempt in range(max_retries):
            _conn: Optional[psycopg2.extensions.connection] = None
            _is_local = False
            try:
                if conn is not None:
                    _conn = conn
                else:
                    _conn = get_db_conn()
                    _is_local = True
                cursor = _conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

                # Duplicate guard — check before insert (no unique constraint in schema)
                cursor.execute(
                    "SELECT id FROM applications WHERE job_post_id = %s AND user_id = %s LIMIT 1",
                    (job_post_id, self.user_id),
                )
                if cursor.fetchone():
                    self.logger.debug(
                        "_create_manual_application_record: record already exists for job=%s",
                        job_post_id,
                    )
                    return None  # already exists, not an error

                # Persist routing context for downstream agents.
                cursor.execute(
                    """
                    INSERT INTO applications
                        (job_post_id, user_id, mode, status, platform, resume_id, error_code)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                    """,
                    (
                        job_post_id,
                        self.user_id,
                        "manual",          # mode CHECK: 'auto' | 'manual'
                        "manual_queued",   # status CHECK: 'applied' | 'failed' | 'manual_queued'
                        source_platform,   # platform column (not source_platform)
                        resume_id,         # UUID id of the resume
                        reason,            # error_code column reused for routing reason
                    ),
                )
                result = cursor.fetchone()
                if _is_local:
                    _conn.commit()             # S5.3: single commit
                app_id = str(result["id"]) if result else None
                if app_id:
                    self.logger.info(
                        "_create_manual_application_record: created application %s for job %s",
                        app_id,
                        job_post_id,
                    )
                else:
                    # Log to audit_logs on silent failure (schema-verified columns)
                    try:
                        audit_conn = get_db_conn()
                        audit_cursor = audit_conn.cursor()
                        audit_cursor.execute(
                            """
                            INSERT INTO audit_logs
                                (pipeline_run_id, job_post_id, level, event_type, message)
                            VALUES (%s, %s, %s, %s, %s)
                            """,
                            (
                                self.pipeline_run_id,
                                job_post_id,
                                "WARNING",
                                "manual_queue_persist_silent",
                                f"INSERT returned no id for job_post_id={job_post_id}",
                            ),
                        )
                        audit_conn.commit()
                        audit_conn.close()
                    except Exception:  # noqa: BLE001
                        pass
                return app_id
            except Exception as exc:
                last_exc = exc
                if _is_local and _conn:
                    try:
                        _conn.rollback()
                    except Exception:  # noqa: BLE001
                        pass
                if attempt < max_retries - 1:
                    sleep_s = 2 ** attempt
                    self.logger.warning(
                        "_create_manual_application_record attempt %d/%d failed for job %s: %s — retrying in %ds",
                        attempt + 1,
                        max_retries,
                        job_post_id,
                        exc,
                        sleep_s,
                    )
                    time.sleep(sleep_s)
                else:
                    self.logger.error(
                        "_create_manual_application_record failed after %d attempts for job %s: %s",
                        max_retries,
                        job_post_id,
                        exc,
                    )
                    # S5.5: audit_logs schema-verified columns
                    try:
                        audit_conn = get_db_conn()
                        audit_cursor = audit_conn.cursor()
                        audit_cursor.execute(
                            """
                            INSERT INTO audit_logs
                                (pipeline_run_id, job_post_id, level, event_type, message)
                            VALUES (%s, %s, %s, %s, %s)
                            """,
                            (
                                self.pipeline_run_id,
                                job_post_id,
                                "ERROR",
                                "manual_queue_persist_failure",
                                f"error={exc} reason={reason}",
                            ),
                        )
                        audit_conn.commit()
                        audit_conn.close()
                    except Exception:  # noqa: BLE001
                        pass
            finally:
                if _is_local and _conn:
                    try:
                        _conn.close()
                    except Exception:  # noqa: BLE001
                        pass
        return None

    def _resolve_resume_id(self, label_or_path: str) -> Optional[str]:
        """
        Resolve a resume label or storage_path to a UUID id.
        """
        if not label_or_path:
            return None
        try:
            conn = get_db_conn()
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id FROM resumes WHERE (label = %s OR storage_path = %s) AND user_id = %s LIMIT 1",
                (label_or_path, label_or_path, self.user_id)
            )
            row = cursor.fetchone()
            conn.close()
            return str(row[0]) if row else None
        except Exception as e:
            self.logger.warning("_resolve_resume_id failed for %s: %s", label_or_path, e)
            return None
