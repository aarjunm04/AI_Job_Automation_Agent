"""Developer Agent — autonomous post-session pattern analysis and
improvement suggestion engine.

READ-ONLY analysis of Postgres + AgentOps data. Writes ONLY to
``developer_suggestions`` table. Never modifies production code.
Runs after each weekly batch session via GitHub Actions schedule
or manual trigger via FastAPI ``/dev-agent`` endpoint (Phase 2).

Analysis pipeline:
    1. Platform failure rate analysis
    2. Selector drift detection (NAV_FAIL patterns)
    3. Proof confidence scoring
    4. LLM budget burn monitoring
    5. Resume variant performance comparison
    6. Error code distribution analysis

LLM usage (summary generation only):
    Primary: xAI Grok → Fallback 1: Groq Llama → Fallback 2: Cerebras
    → Rule-based text fallback.
    Token budget: ``MAX_TOKENS_PER_RUN = 8000`` per analysis run.
    Monthly budget cap: ``$5 xAI`` (env ``XAI_MONTHLY_BUDGET_USD``).
"""

from __future__ import annotations

import os
import json
import uuid
import logging
import asyncio
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from typing import Optional, Any, Dict, List, Tuple

import agentops
import psycopg2
import psycopg2.extras

logger = logging.getLogger(__name__)

__all__ = ["DeveloperAgent", "DeveloperAgentResult"]


# ======================================================================
# Dataclasses
# ======================================================================


@dataclass
class Suggestion:
    """A single improvement suggestion to be written to Postgres.

    Attributes:
        suggestion_type: Category — ``"selector_fix"`` |
            ``"platform_skip"`` | ``"resume_swap"`` |
            ``"schedule_adjust"`` | ``"budget_alert"`` | ``"general"``.
        platform: Target platform name or ``"all"``.
        priority: Urgency — ``"critical"`` | ``"high"`` | ``"medium"``
            | ``"low"``.
        title: Short human-readable title (max 256 chars).
        description: Full explanation with context.
        evidence_json: Supporting data dict.
        suggested_fix: Concrete actionable fix text.
    """

    suggestion_type: str
    platform: str
    priority: str
    title: str
    description: str
    evidence_json: Dict[str, Any]
    suggested_fix: str


@dataclass
class DeveloperAgentResult:
    """Result returned by ``DeveloperAgent.run()``.

    Attributes:
        run_id: Unique ID for this analysis run.
        suggestions_written: Count of new suggestions written.
        suggestions_skipped: Count skipped (duplicates).
        analysis_summary: Natural language summary from LLM.
        token_usage: Total tokens consumed across all LLM calls.
        budget_remaining_usd: Estimated remaining xAI monthly budget.
        error: Error message if run failed, else None.
    """

    run_id: str
    suggestions_written: int
    suggestions_skipped: int
    analysis_summary: str
    token_usage: int
    budget_remaining_usd: float
    error: Optional[str] = None


# ======================================================================
# Developer Agent
# ======================================================================


@agentops.track_agent(name="DeveloperAgent")
class DeveloperAgent:
    """Autonomous post-session analysis and improvement suggestion agent.

    Analyses Postgres application outcomes and ``audit_logs`` to
    identify failure patterns, selector drift, budget anomalies, and
    platform performance issues. Writes structured suggestions to
    ``developer_suggestions`` table for human review.

    **READ-ONLY analysis. No code writes. No production modifications.**

    Args:
        lookback_days: Number of days of data to analyse. Default 7.
        dry_run: If True, compute suggestions but do NOT write to DB.
    """

    # -- Class-level constants --
    MAX_TOKENS_PER_RUN: int = 8000
    FAILURE_RATE_THRESHOLD: float = 0.40
    LOW_CONFIDENCE_THRESHOLD: float = 0.70
    MIN_SAMPLES_FOR_ANALYSIS: int = 5

    def __init__(
        self,
        lookback_days: int = 7,
        dry_run: bool = False,
    ) -> None:
        self.lookback_days: int = lookback_days
        self.dry_run: bool = dry_run
        self.run_id: str = (
            f"dev_agent_"
            f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        )
        self._token_usage: int = 0
        self._suggestions: List[Suggestion] = []
        self._conn: Optional[Any] = None

    # ------------------------------------------------------------------
    # DB Connection
    # ------------------------------------------------------------------

    def _get_db_connection(self) -> Any:
        """Return a Postgres connection using ``ACTIVE_DB`` env var.

        Returns:
            ``psycopg2`` connection with ``RealDictCursor`` factory.
        """
        active_db: str = os.getenv("ACTIVE_DB", "local")
        url: str = (
            os.getenv("LOCAL_POSTGRES_URL", "")
            if active_db == "local"
            else os.getenv("SUPABASE_URL", "")
        )
        return psycopg2.connect(
            url, cursor_factory=psycopg2.extras.RealDictCursor
        )

    # ------------------------------------------------------------------
    # Main Run
    # ------------------------------------------------------------------

    async def run(self) -> DeveloperAgentResult:
        """Execute the full Developer Agent analysis pipeline.

        Sequence:
            1. Connect to Postgres.
            2. Run all 6 analysis methods (each appends to
               ``self._suggestions``).
            3. Deduplicate suggestions vs existing DB records.
            4. Generate LLM summary of findings (with fallback chain).
            5. Write new suggestions to ``developer_suggestions`` table.
            6. Return ``DeveloperAgentResult``.

        Returns:
            DeveloperAgentResult with counts, summary, and budget info.
        """
        logger.info(
            "DeveloperAgent run started: run_id=%s lookback=%dd "
            "dry_run=%s",
            self.run_id,
            self.lookback_days,
            self.dry_run,
        )

        try:
            self._conn = self._get_db_connection()
        except Exception as e:
            logger.error(
                "DeveloperAgent: DB connection failed: %s", str(e)
            )
            return DeveloperAgentResult(
                run_id=self.run_id,
                suggestions_written=0,
                suggestions_skipped=0,
                analysis_summary="",
                token_usage=0,
                budget_remaining_usd=self._get_budget_remaining(0),
                error=f"DB connection failed: {str(e)}",
            )

        try:
            # READ-ONLY analysis methods — each appends to
            # self._suggestions. Failures in individual methods are
            # caught internally and do not abort the run.
            await self._analyse_platform_failure_rates()
            await self._analyse_selector_drift()
            await self._analyse_proof_confidence()
            await self._analyse_budget_burn()
            await self._analyse_resume_performance()
            await self._analyse_error_code_distribution()

            # Deduplicate against existing pending suggestions
            new_suggestions: List[Suggestion]
            skipped: int
            new_suggestions, skipped = (
                await self._deduplicate_suggestions()
            )

            # LLM summary (with fallback chain)
            summary: str = await self._generate_analysis_summary(
                new_suggestions
            )

            # Write to DB (or simulate in dry_run)
            written: int = 0
            if not self.dry_run:
                written = await self._write_suggestions(new_suggestions)
            else:
                written = len(new_suggestions)
                logger.info(
                    "[DRY_RUN] Would write %d suggestions", written
                )

            budget_remaining: float = self._get_budget_remaining(
                self._token_usage
            )

            logger.info(
                "DeveloperAgent run complete: written=%d skipped=%d "
                "tokens=%d budget_remaining=$%.4f",
                written,
                skipped,
                self._token_usage,
                budget_remaining,
            )

            return DeveloperAgentResult(
                run_id=self.run_id,
                suggestions_written=written,
                suggestions_skipped=skipped,
                analysis_summary=summary,
                token_usage=self._token_usage,
                budget_remaining_usd=budget_remaining,
                error=None,
            )

        except Exception as e:
            logger.error(
                "DeveloperAgent run failed: %s", str(e), exc_info=True
            )
            return DeveloperAgentResult(
                run_id=self.run_id,
                suggestions_written=0,
                suggestions_skipped=0,
                analysis_summary="",
                token_usage=self._token_usage,
                budget_remaining_usd=self._get_budget_remaining(
                    self._token_usage
                ),
                error=str(e),
            )
        finally:
            if self._conn:
                self._conn.close()

    # ==================================================================
    # Analysis Method 1: Platform Failure Rates
    # ==================================================================

    @agentops.track_tool(name="analyse_platform_failure_rates")
    async def _analyse_platform_failure_rates(self) -> None:
        """Analyse application success/failure rates per platform.

        Queries ``applications`` + ``jobs`` tables for the lookback
        window. For each platform with ``>= MIN_SAMPLES``: calculates
        failure rate. If failure rate exceeds thresholds, appends a
        suggestion.

        Suggestion types generated:
            - ``"platform_skip"`` — if failure rate > 70%.
            - ``"general"``       — if failure rate 40–70%.

        READ-ONLY. No DB writes here.
        """
        # READ-ONLY analysis. No code writes.
        try:
            cutoff: datetime = datetime.now(timezone.utc) - timedelta(
                days=self.lookback_days
            )
            with self._conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                        j.platform,
                        COUNT(*) AS total,
                        SUM(CASE WHEN a.status LIKE 'applied%%'
                            THEN 1 ELSE 0 END) AS success_count,
                        SUM(CASE WHEN a.status IN
                            ('failed', 'error', 'manual_queued')
                            THEN 1 ELSE 0 END) AS failure_count,
                        AVG(a.proof_confidence) AS avg_confidence
                    FROM applications a
                    JOIN jobs j ON a.job_id = j.id
                    WHERE a.applied_at >= %s
                    GROUP BY j.platform
                    ORDER BY total DESC
                    """,
                    (cutoff,),
                )
                rows = cur.fetchall()

            for row in rows:
                platform: str = row["platform"] or "unknown"
                total: int = int(row["total"])
                failures: int = int(row["failure_count"] or 0)
                successes: int = int(row["success_count"] or 0)

                if total < self.MIN_SAMPLES_FOR_ANALYSIS:
                    continue

                failure_rate: float = failures / total

                if failure_rate > 0.70:
                    self._suggestions.append(
                        Suggestion(
                            suggestion_type="platform_skip",
                            platform=platform,
                            priority="critical",
                            title=(
                                f"{platform}: {failure_rate:.0%} "
                                "failure rate — consider disabling"
                            ),
                            description=(
                                f"Platform '{platform}' has a "
                                f"{failure_rate:.0%} failure rate "
                                f"over the last "
                                f"{self.lookback_days} days "
                                f"({failures}/{total} applications "
                                "failed). Disabling this platform "
                                "would conserve budget and improve "
                                "overall pipeline efficiency."
                            ),
                            evidence_json={
                                "platform": platform,
                                "total": total,
                                "failures": failures,
                                "successes": successes,
                                "failure_rate": round(
                                    failure_rate, 4
                                ),
                                "lookback_days": self.lookback_days,
                            },
                            suggested_fix=(
                                "In config/platforms.json, set "
                                f"'{platform}.enabled' to false. "
                                "Re-enable after fixing selectors."
                            ),
                        )
                    )
                    logger.warning(
                        "Platform '%s': critical failure rate %.0f%%",
                        platform,
                        failure_rate * 100,
                    )

                elif failure_rate > self.FAILURE_RATE_THRESHOLD:
                    self._suggestions.append(
                        Suggestion(
                            suggestion_type="general",
                            platform=platform,
                            priority="high",
                            title=(
                                f"{platform}: elevated failure rate "
                                f"{failure_rate:.0%}"
                            ),
                            description=(
                                f"Platform '{platform}' shows "
                                f"{failure_rate:.0%} failure rate "
                                f"({failures}/{total}). Investigate "
                                "selector health and CAPTCHA "
                                "frequency."
                            ),
                            evidence_json={
                                "platform": platform,
                                "total": total,
                                "failures": failures,
                                "failure_rate": round(
                                    failure_rate, 4
                                ),
                            },
                            suggested_fix=(
                                f"Run dry_run=True session on "
                                f"'{platform}' and inspect "
                                "audit_logs for dominant error_code."
                            ),
                        )
                    )

        except Exception as e:
            logger.error(
                "Platform failure rate analysis error: %s", str(e)
            )

    # ==================================================================
    # Analysis Method 2: Selector Drift
    # ==================================================================

    @agentops.track_tool(name="analyse_selector_drift")
    async def _analyse_selector_drift(self) -> None:
        """Detect selector drift by analysing ``NAV_FAIL`` error patterns.

        Queries ``audit_logs`` for ``NAV_FAIL`` errors in the lookback
        window. Groups by platform. If a platform has ``>= 3``
        ``NAV_FAIL`` events with selector-related error messages:
        suggests selector review.

        READ-ONLY. No DB writes here.
        """
        # READ-ONLY analysis. No code writes.
        try:
            cutoff: datetime = datetime.now(timezone.utc) - timedelta(
                days=self.lookback_days
            )
            with self._conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                        metadata_json->>'platform' AS platform,
                        error_code,
                        COUNT(*) AS error_count,
                        array_agg(
                            metadata_json->>'reroute_reason'
                            ORDER BY created_at DESC
                        ) AS reasons
                    FROM audit_logs
                    WHERE error_code = 'NAV_FAIL'
                      AND created_at >= %s
                      AND metadata_json IS NOT NULL
                    GROUP BY
                        metadata_json->>'platform',
                        error_code
                    HAVING COUNT(*) >= 3
                    ORDER BY error_count DESC
                    """,
                    (cutoff,),
                )
                rows = cur.fetchall()

            selector_keywords: List[str] = [
                "not found",
                "selector",
                "timeout",
                "fingerprint",
                "button not found",
                "did not load",
                "modal did not",
            ]

            for row in rows:
                platform: str = row["platform"] or "unknown"
                error_count: int = int(row["error_count"])
                reasons: List[Optional[str]] = row["reasons"] or []

                selector_failures: List[str] = [
                    r
                    for r in reasons
                    if r
                    and any(
                        kw in r.lower() for kw in selector_keywords
                    )
                ]

                if len(selector_failures) >= 2:
                    self._suggestions.append(
                        Suggestion(
                            suggestion_type="selector_fix",
                            platform=platform,
                            priority=(
                                "critical"
                                if error_count >= 10
                                else "high"
                            ),
                            title=(
                                f"{platform}: {error_count} NAV_FAIL "
                                "errors — possible selector drift"
                            ),
                            description=(
                                f"Platform '{platform}' has "
                                f"{error_count} NAV_FAIL errors in "
                                f"the last {self.lookback_days} days "
                                "with selector-related failure "
                                "messages. The ATS may have updated "
                                "its DOM — selectors need review."
                            ),
                            evidence_json={
                                "platform": platform,
                                "nav_fail_count": error_count,
                                "sample_reasons": selector_failures[
                                    :5
                                ],
                                "lookback_days": self.lookback_days,
                            },
                            suggested_fix=(
                                f"Manually inspect {platform} job "
                                "page DOM. Update selectors in "
                                f"auto_apply/platforms/"
                                f"{platform}.py. Use browser DevTools "
                                "to find current selectors."
                            ),
                        )
                    )
                    logger.warning(
                        "Selector drift detected on '%s': "
                        "%d NAV_FAILs",
                        platform,
                        error_count,
                    )

        except Exception as e:
            logger.error(
                "Selector drift analysis error: %s", str(e)
            )

    # ==================================================================
    # Analysis Method 3: Proof Confidence
    # ==================================================================

    @agentops.track_tool(name="analyse_proof_confidence")
    async def _analyse_proof_confidence(self) -> None:
        """Analyse average proof confidence scores per platform.

        Low proof confidence indicates the platform's submission
        confirmation signals are unreliable — may need new proof
        capture strategy.

        Threshold: avg ``proof_confidence < 0.70``.
        Minimum samples: ``MIN_SAMPLES_FOR_ANALYSIS``.

        READ-ONLY. No DB writes here.
        """
        # READ-ONLY analysis. No code writes.
        try:
            cutoff: datetime = datetime.now(timezone.utc) - timedelta(
                days=self.lookback_days
            )
            with self._conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                        j.platform,
                        COUNT(*) AS sample_count,
                        AVG(a.proof_confidence) AS avg_confidence,
                        MIN(a.proof_confidence) AS min_confidence,
                        SUM(CASE WHEN a.proof_confidence IS NULL
                            THEN 1 ELSE 0 END) AS null_confidence_count
                    FROM applications a
                    JOIN jobs j ON a.job_id = j.id
                    WHERE a.applied_at >= %s
                      AND a.status LIKE 'applied%%'
                    GROUP BY j.platform
                    HAVING COUNT(*) >= %s
                    """,
                    (cutoff, self.MIN_SAMPLES_FOR_ANALYSIS),
                )
                rows = cur.fetchall()

            for row in rows:
                platform: str = row["platform"] or "unknown"
                avg_conf: float = float(
                    row["avg_confidence"] or 0.0
                )
                null_count: int = int(
                    row["null_confidence_count"] or 0
                )
                samples: int = int(row["sample_count"])

                if avg_conf < self.LOW_CONFIDENCE_THRESHOLD:
                    self._suggestions.append(
                        Suggestion(
                            suggestion_type="general",
                            platform=platform,
                            priority=(
                                "high"
                                if avg_conf < 0.50
                                else "medium"
                            ),
                            title=(
                                f"{platform}: low proof confidence "
                                f"{avg_conf:.0%} — improve "
                                "verification"
                            ),
                            description=(
                                f"Platform '{platform}' has average "
                                f"proof confidence of {avg_conf:.0%} "
                                f"across {samples} applications. "
                                "Submission confirmations are "
                                "uncertain. Add more specific "
                                "confirmation selectors."
                            ),
                            evidence_json={
                                "platform": platform,
                                "avg_confidence": round(avg_conf, 4),
                                "min_confidence": float(
                                    row["min_confidence"] or 0
                                ),
                                "null_confidence_count": null_count,
                                "sample_count": samples,
                            },
                            suggested_fix=(
                                f"In auto_apply/platforms/"
                                f"{platform}.py, add additional "
                                "confirmation selectors to "
                                "_capture_submission_proof or "
                                "_handle_*_submit method. Inspect "
                                "live platform for new confirmation "
                                "DOM."
                            ),
                        )
                    )

        except Exception as e:
            logger.error(
                "Proof confidence analysis error: %s", str(e)
            )

    # ==================================================================
    # Analysis Method 4: Budget Burn
    # ==================================================================

    @agentops.track_tool(name="analyse_budget_burn")
    async def _analyse_budget_burn(self) -> None:
        """Monitor LLM API token usage and budget burn rate.

        Reads ``audit_logs`` for LLM call events with token counts.
        Projects monthly spend. Alerts if projected spend exceeds
        90% of monthly budget cap (``$5 xAI``).

        READ-ONLY. No DB writes here.
        """
        # READ-ONLY analysis. No code writes.
        try:
            cutoff_30d: datetime = datetime.now(
                timezone.utc
            ) - timedelta(days=30)
            with self._conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                        metadata_json->>'provider' AS provider,
                        SUM(
                            CAST(
                                metadata_json->>'tokens_used'
                                AS FLOAT
                            )
                        ) AS total_tokens,
                        COUNT(*) AS call_count
                    FROM audit_logs
                    WHERE event_type = 'llm_call'
                      AND created_at >= %s
                      AND metadata_json->>'tokens_used' IS NOT NULL
                    GROUP BY metadata_json->>'provider'
                    """,
                    (cutoff_30d,),
                )
                rows = cur.fetchall()

            xai_budget: float = float(
                os.getenv("XAI_MONTHLY_BUDGET_USD", "5.0")
            )
            cost_per_1k: float = float(
                os.getenv("XAI_COST_PER_1K_TOKENS", "0.005")
            )

            for row in rows:
                provider: str = row["provider"] or "unknown"
                total_tokens: float = float(
                    row["total_tokens"] or 0
                )
                cost: float = (total_tokens / 1000) * cost_per_1k
                threshold_usd: float = xai_budget * 0.90

                if cost > threshold_usd:
                    self._suggestions.append(
                        Suggestion(
                            suggestion_type="budget_alert",
                            platform="all",
                            priority="critical",
                            title=(
                                f"Budget alert: {provider} spend "
                                f"${cost:.2f} exceeds 90% of "
                                f"${xai_budget:.2f} monthly cap"
                            ),
                            description=(
                                f"LLM provider '{provider}' consumed "
                                f"{total_tokens:,.0f} tokens "
                                f"(~${cost:.3f}) in 30 days. "
                                f"Exceeds 90% of "
                                f"${xai_budget:.2f} budget cap. "
                                "Reduce LLM call frequency or "
                                "switch to cheaper providers."
                            ),
                            evidence_json={
                                "provider": provider,
                                "total_tokens_30d": round(
                                    total_tokens
                                ),
                                "estimated_cost_usd": round(
                                    cost, 4
                                ),
                                "budget_cap_usd": xai_budget,
                                "threshold_usd": threshold_usd,
                                "call_count": int(
                                    row["call_count"]
                                ),
                            },
                            suggested_fix=(
                                "1. Reduce Analyser Agent LLM calls "
                                "by increasing ChromaDB similarity "
                                "threshold. "
                                "2. Switch to Groq (free tier) as "
                                "primary for non-critical analysis. "
                                "3. Reduce MAX_TOKENS_PER_RUN in "
                                "DeveloperAgent to 4000."
                            ),
                        )
                    )
                    logger.warning(
                        "Budget alert: %s spend $%.3f > "
                        "threshold $%.2f",
                        provider,
                        cost,
                        threshold_usd,
                    )

        except Exception as e:
            logger.error(
                "Budget burn analysis error: %s", str(e)
            )

    # ==================================================================
    # Analysis Method 5: Resume Performance
    # ==================================================================

    @agentops.track_tool(name="analyse_resume_performance")
    async def _analyse_resume_performance(self) -> None:
        """Analyse which resume variants correlate with successful applies.

        Compares application success rates per ``resume_used`` value.
        If one resume has ``> 20%`` gap vs the best: suggest swap.

        Minimum samples per resume: ``MIN_SAMPLES_FOR_ANALYSIS``.

        READ-ONLY. No DB writes here.
        """
        # READ-ONLY analysis. No code writes.
        try:
            cutoff: datetime = datetime.now(timezone.utc) - timedelta(
                days=self.lookback_days
            )
            with self._conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                        a.resume_used,
                        j.platform,
                        COUNT(*) AS total,
                        SUM(CASE
                            WHEN a.status LIKE 'applied%%'
                             AND (a.proof_confidence IS NULL
                                  OR a.proof_confidence > 0.70)
                            THEN 1 ELSE 0
                        END) AS successes
                    FROM applications a
                    JOIN jobs j ON a.job_id = j.id
                    WHERE a.applied_at >= %s
                      AND a.resume_used IS NOT NULL
                    GROUP BY a.resume_used, j.platform
                    HAVING COUNT(*) >= %s
                    ORDER BY j.platform, total DESC
                    """,
                    (cutoff, self.MIN_SAMPLES_FOR_ANALYSIS),
                )
                rows = cur.fetchall()

            platform_resumes: Dict[
                str, List[Dict[str, Any]]
            ] = defaultdict(list)
            for row in rows:
                total_val: int = int(row["total"])
                success_val: int = int(row["successes"])
                platform_resumes[row["platform"]].append(
                    {
                        "resume": row["resume_used"],
                        "total": total_val,
                        "successes": success_val,
                        "rate": (
                            success_val / total_val
                            if total_val > 0
                            else 0.0
                        ),
                    }
                )

            for platform, resume_list in platform_resumes.items():
                if len(resume_list) < 2:
                    continue
                best: Dict[str, Any] = max(
                    resume_list, key=lambda x: x["rate"]
                )
                worst: Dict[str, Any] = min(
                    resume_list, key=lambda x: x["rate"]
                )
                gap: float = best["rate"] - worst["rate"]

                if gap > 0.20:
                    self._suggestions.append(
                        Suggestion(
                            suggestion_type="resume_swap",
                            platform=platform,
                            priority="medium",
                            title=(
                                f"{platform}: resume "
                                f"'{worst['resume']}' "
                                f"underperforming by {gap:.0%}"
                            ),
                            description=(
                                f"On '{platform}', resume "
                                f"'{worst['resume']}' has "
                                f"{worst['rate']:.0%} success vs "
                                f"'{best['resume']}' at "
                                f"{best['rate']:.0%}. Consider "
                                "swapping to the better performer."
                            ),
                            evidence_json={
                                "platform": platform,
                                "best_resume": best,
                                "worst_resume": worst,
                                "performance_gap": round(gap, 4),
                            },
                            suggested_fix=(
                                "In config/platforms.json, set "
                                f"'{platform}.preferred_resume' to "
                                f"'{best['resume']}'. Or update "
                                "ChromaDB weights for this platform."
                            ),
                        )
                    )

        except Exception as e:
            logger.error(
                "Resume performance analysis error: %s", str(e)
            )

    # ==================================================================
    # Analysis Method 6: Error Code Distribution
    # ==================================================================

    @agentops.track_tool(name="analyse_error_code_distribution")
    async def _analyse_error_code_distribution(self) -> None:
        """Analyse distribution of error codes in ``audit_logs``.

        Identifies top error codes per platform and generates targeted
        suggestions:
            - ``CAPTCHA``    → proxy rotation / platform skip.
            - ``TIMEOUT``    → increase wait times.
            - ``UPLOAD_FAIL`` → check resume path.

        READ-ONLY. No DB writes here.
        """
        # READ-ONLY analysis. No code writes.
        try:
            cutoff: datetime = datetime.now(timezone.utc) - timedelta(
                days=self.lookback_days
            )
            with self._conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                        metadata_json->>'platform' AS platform,
                        error_code,
                        COUNT(*) AS count
                    FROM audit_logs
                    WHERE error_code IS NOT NULL
                      AND created_at >= %s
                    GROUP BY
                        metadata_json->>'platform',
                        error_code
                    ORDER BY count DESC
                    LIMIT 30
                    """,
                    (cutoff,),
                )
                rows = cur.fetchall()

            captcha_map: Dict[str, int] = {}
            timeout_map: Dict[str, int] = {}
            upload_fail_total: int = 0

            for row in rows:
                platform: str = row["platform"] or "unknown"
                code: str = row["error_code"] or "UNKNOWN"
                count: int = int(row["count"])

                if code == "CAPTCHA":
                    captcha_map[platform] = (
                        captcha_map.get(platform, 0) + count
                    )
                elif code == "TIMEOUT":
                    timeout_map[platform] = (
                        timeout_map.get(platform, 0) + count
                    )
                elif code == "UPLOAD_FAIL":
                    upload_fail_total += count

            # CAPTCHA alerts
            for platform, count in captcha_map.items():
                if count >= 3:
                    self._suggestions.append(
                        Suggestion(
                            suggestion_type="general",
                            platform=platform,
                            priority=(
                                "critical"
                                if count >= 10
                                else "high"
                            ),
                            title=(
                                f"{platform}: {count} CAPTCHA "
                                "events — review proxy rotation"
                            ),
                            description=(
                                f"{count} CAPTCHA events on "
                                f"'{platform}' in "
                                f"{self.lookback_days} days. "
                                "Current proxy pool may be flagged. "
                                "Rotate Webshare proxies or reduce "
                                "apply frequency."
                            ),
                            evidence_json={
                                "platform": platform,
                                "captcha_count": count,
                                "lookback_days": self.lookback_days,
                            },
                            suggested_fix=(
                                "1. Rotate Webshare proxy pool. "
                                "2. Reduce session frequency. "
                                "3. Add random delay 3–8s between "
                                "applies in config/platforms.json."
                            ),
                        )
                    )

            # TIMEOUT alerts
            for platform, count in timeout_map.items():
                if count >= 5:
                    self._suggestions.append(
                        Suggestion(
                            suggestion_type="general",
                            platform=platform,
                            priority="medium",
                            title=(
                                f"{platform}: {count} TIMEOUT "
                                "events — increase wait times"
                            ),
                            description=(
                                f"{count} TIMEOUT errors on "
                                f"'{platform}'. Page load or element "
                                "wait timeouts may be too aggressive."
                            ),
                            evidence_json={
                                "platform": platform,
                                "timeout_count": count,
                            },
                            suggested_fix=(
                                f"In auto_apply/platforms/"
                                f"{platform}.py, increase "
                                "STEP_TIMEOUT from 20000 to 30000ms "
                                "and FIELD_TIMEOUT from 8000 to "
                                "12000ms."
                            ),
                        )
                    )

            # UPLOAD_FAIL alert (global)
            if upload_fail_total >= 3:
                self._suggestions.append(
                    Suggestion(
                        suggestion_type="general",
                        platform="all",
                        priority="high",
                        title=(
                            f"{upload_fail_total} resume UPLOAD_FAIL "
                            "events — check RESUME_DIR path"
                        ),
                        description=(
                            f"{upload_fail_total} resume upload "
                            f"failures in {self.lookback_days} days. "
                            "RESUME_DIR may be misconfigured or "
                            "resume PDFs may be missing."
                        ),
                        evidence_json={
                            "upload_fail_count": upload_fail_total,
                            "lookback_days": self.lookback_days,
                        },
                        suggested_fix=(
                            "1. Verify RESUME_DIR in ~/narad.env "
                            "points to correct path. "
                            "2. Confirm all resume PDFs exist. "
                            "3. Check DEFAULT_RESUME filename."
                        ),
                    )
                )

        except Exception as e:
            logger.error(
                "Error code distribution analysis error: %s",
                str(e),
            )

    # ==================================================================
    # Deduplication
    # ==================================================================

    async def _deduplicate_suggestions(
        self,
    ) -> Tuple[List[Suggestion], int]:
        """Remove suggestions that already exist as pending in DB.

        Deduplication key: ``(suggestion_type, platform, title)``.
        Only checks ``status='pending'`` — approved/dismissed can
        be re-raised.

        Returns:
            Tuple of ``(new_suggestions, skipped_count)``.
        """
        try:
            with self._conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT suggestion_type, platform, title
                    FROM developer_suggestions
                    WHERE status = 'pending'
                      AND created_at >= NOW() - INTERVAL '30 days'
                    """
                )
                existing = {
                    (
                        r["suggestion_type"],
                        r["platform"],
                        r["title"],
                    )
                    for r in cur.fetchall()
                }
        except Exception as e:
            logger.warning(
                "Deduplication query failed: %s — skipping dedup",
                str(e),
            )
            return self._suggestions, 0

        new: List[Suggestion] = []
        skipped: int = 0
        for s in self._suggestions:
            key = (s.suggestion_type, s.platform, s.title)
            if key in existing:
                skipped += 1
                logger.debug(
                    "Duplicate suggestion skipped: %s", s.title
                )
            else:
                new.append(s)

        return new, skipped

    # ==================================================================
    # Write Suggestions (ONLY DB write in this agent)
    # ==================================================================

    async def _write_suggestions(
        self, suggestions: List[Suggestion]
    ) -> int:
        """Write new suggestions to ``developer_suggestions`` table.

        All writes in a single atomic transaction. On failure, rolls
        back and raises.

        Args:
            suggestions: List of ``Suggestion`` objects to insert.

        Returns:
            Count of rows inserted.
        """
        if not suggestions:
            return 0

        self._conn.autocommit = False
        try:
            with self._conn.cursor() as cur:
                for s in suggestions:
                    cur.execute(
                        """
                        INSERT INTO developer_suggestions (
                            id, run_id, suggestion_type, platform,
                            priority, title, description,
                            evidence_json, suggested_fix,
                            status, created_at
                        ) VALUES (
                            %s, %s, %s, %s, %s, %s, %s,
                            %s, %s, %s, %s
                        )
                        """,
                        (
                            str(uuid.uuid4()),
                            self.run_id,
                            s.suggestion_type,
                            s.platform,
                            s.priority,
                            s.title[:256],
                            s.description,
                            psycopg2.extras.Json(s.evidence_json),
                            s.suggested_fix,
                            "pending",
                            datetime.now(timezone.utc),
                        ),
                    )
            self._conn.commit()
            logger.info(
                "Written %d suggestions to developer_suggestions",
                len(suggestions),
            )
            return len(suggestions)
        except Exception as e:
            self._conn.rollback()
            logger.error(
                "Failed to write suggestions to DB: %s", str(e)
            )
            raise

    # ==================================================================
    # LLM Summary (Fallback Chain)
    # ==================================================================

    async def _generate_analysis_summary(
        self, suggestions: List[Suggestion]
    ) -> str:
        """Generate a natural language summary via LLM fallback chain.

        Token budget: max 2000 tokens (input) + 500 (output) per call.
        Fallback chain: xAI Grok → Groq Llama → Cerebras → rule-based.

        Args:
            suggestions: List of new suggestions from this run.

        Returns:
            Natural language summary string.
        """
        if not suggestions:
            return (
                "No new improvement suggestions generated this run. "
                "All systems operating within normal parameters."
            )

        # Build prompt — keep under 2000 tokens
        suggestion_lines: str = "\n".join(
            [
                f"- [{s.priority.upper()}] {s.platform}: {s.title}"
                for s in suggestions[:15]
            ]
        )
        prompt: str = (
            "You are analysing a job application automation system. "
            f"Here are {len(suggestions)} improvement suggestions "
            f"from the last {self.lookback_days} days:\n\n"
            f"{suggestion_lines}\n\n"
            "Write a concise 3–5 sentence summary for the developer. "
            "Focus on: most critical issues, overall system health, "
            "and top priority action item. Plain text only."
        )

        # Primary: xAI Grok
        summary: str = await self._call_llm_with_retry(
            provider="xai",
            api_key=os.getenv("XAI_API_KEY", ""),
            model=os.getenv("XAI_MODEL", "grok-2-latest"),
            base_url="https://api.x.ai/v1",
            prompt=prompt,
        )
        if summary:
            return summary

        # Fallback 1: Groq Llama
        summary = await self._call_llm_with_retry(
            provider="groq",
            api_key=os.getenv("GROQ_API_KEY", ""),
            model=os.getenv(
                "GROQ_MODEL", "llama-3.3-70b-versatile"
            ),
            base_url="https://api.groq.com/openai/v1",
            prompt=prompt,
        )
        if summary:
            return summary

        # Fallback 2: Cerebras
        summary = await self._call_llm_with_retry(
            provider="cerebras",
            api_key=os.getenv("CEREBRAS_API_KEY", ""),
            model=os.getenv("CEREBRAS_MODEL", "llama3.1-8b"),
            base_url="https://api.cerebras.ai/v1",
            prompt=prompt,
        )
        if summary:
            return summary

        # Rule-based fallback — no LLM available
        critical_count: int = sum(
            1 for s in suggestions if s.priority == "critical"
        )
        high_count: int = sum(
            1 for s in suggestions if s.priority == "high"
        )
        return (
            f"Developer Agent analysis complete. "
            f"{len(suggestions)} new suggestions generated: "
            f"{critical_count} critical, {high_count} high priority. "
            f"Top item: "
            f"{suggestions[0].title if suggestions else 'None'}. "
            "LLM summary unavailable — all providers failed."
        )

    async def _call_llm_with_retry(
        self,
        provider: str,
        api_key: str,
        model: str,
        base_url: str,
        prompt: str,
        max_retries: int = 3,
    ) -> str:
        """Call an OpenAI-compatible LLM endpoint with retry logic.

        Uses ``openai.AsyncOpenAI`` client pointed at ``base_url``.
        Respects ``MAX_TOKENS_PER_RUN`` budget cap. Retries up to
        ``max_retries`` with exponential backoff. Auth errors (401/403)
        abort immediately.

        Args:
            provider: Provider name for logging/tracking.
            api_key: API key string.
            model: Model identifier string.
            base_url: OpenAI-compatible API base URL.
            prompt: User prompt string.
            max_retries: Max retry attempts.

        Returns:
            Response text or ``""`` on failure.
        """
        if not api_key:
            logger.debug(
                "LLM provider '%s': no API key configured", provider
            )
            return ""

        # Budget guard
        if self._token_usage >= self.MAX_TOKENS_PER_RUN:
            logger.warning(
                "Token budget exhausted (%d/%d) — skipping %s call",
                self._token_usage,
                self.MAX_TOKENS_PER_RUN,
                provider,
            )
            return ""

        try:
            from openai import (
                AsyncOpenAI,
                APIStatusError,
                APIConnectionError,
            )
        except ImportError:
            logger.error("openai package not installed")
            return ""

        client = AsyncOpenAI(api_key=api_key, base_url=base_url)

        for attempt in range(1, max_retries + 1):
            try:
                response = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=500,
                    temperature=0.3,
                )
                tokens_used: int = (
                    response.usage.total_tokens
                    if response.usage
                    else 0
                )
                self._token_usage += tokens_used

                logger.info(
                    "LLM call success: provider=%s tokens=%d "
                    "total=%d",
                    provider,
                    tokens_used,
                    self._token_usage,
                )
                return (
                    response.choices[0].message.content or ""
                )

            except APIStatusError as e:
                logger.warning(
                    "LLM %s attempt %d/%d APIStatusError: %s",
                    provider,
                    attempt,
                    max_retries,
                    str(e),
                )
                if e.status_code in (401, 403):
                    break  # Auth error — no retry
                await asyncio.sleep(float(2**attempt))

            except APIConnectionError as e:
                logger.warning(
                    "LLM %s attempt %d/%d connection error: %s",
                    provider,
                    attempt,
                    max_retries,
                    str(e),
                )
                await asyncio.sleep(float(2**attempt))

            except Exception as e:
                logger.warning(
                    "LLM %s attempt %d/%d unexpected error: %s",
                    provider,
                    attempt,
                    max_retries,
                    str(e),
                )
                await asyncio.sleep(float(2**attempt))

        logger.error(
            "LLM provider '%s' failed after %d attempts",
            provider,
            max_retries,
        )
        return ""

    # ==================================================================
    # Budget Helper
    # ==================================================================

    def _get_budget_remaining(self, tokens_used: int) -> float:
        """Estimate remaining xAI monthly budget after this run.

        Args:
            tokens_used: Total tokens consumed in this run.

        Returns:
            Estimated remaining budget in USD.
        """
        budget: float = float(
            os.getenv("XAI_MONTHLY_BUDGET_USD", "5.0")
        )
        cost_per_1k: float = float(
            os.getenv("XAI_COST_PER_1K_TOKENS", "0.005")
        )
        run_cost: float = (tokens_used / 1000) * cost_per_1k
        return round(max(0.0, budget - run_cost), 6)
