"""Master Agent for AI Job Application Agent.

The Master Agent is the brain and orchestrator of the entire pipeline. It boots
the system, owns the run lifecycle from first heartbeat to final session close,
gates every sub-agent behind budget checks, handles all inter-agent handoffs,
and is the single point of failure recovery.

Pipeline flow::

    Master (boot) → Scraper → Analyser → Apply → Tracker → Master (close)

Everything lives or dies by this file.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from crewai import Agent, Task, Crew, Process
import agentops
from agentops.sdk.decorators import operation

from config.settings import db_config, run_config, budget_config, api_config
from integrations.llm_interface import LLMInterface
from agents.scraper_agent import ScraperAgent
from agents.analyser_agent import AnalyserAgent
from agents.tracker_agent import TrackerAgent
from tools.postgres_tools import (
    create_run_batch,
    update_run_batch_stats,
    log_event,
    get_run_stats,
    _create_run_batch,
    _update_run_batch_stats,
    _log_event,
)
from tools.budget_tools import (
    reset_run_cost_tracker,
    check_xai_run_cap,
    check_monthly_budget,
    get_cost_summary,
)
from tools.agentops_tools import (
    record_agent_error,
    record_fallback_event,
    _record_agent_error,
    _record_fallback_event,
)
from utils.normalise_dedupe import deduplicate_jobs_fuzzy
from utils.db_utils import get_db_conn

# Module-level logger
logger = logging.getLogger(__name__)

__all__ = ["MasterAgent"]


# ---------------------------------------------------------------------------
# AgentOps session helpers (mirrors tracker_agent.py pattern)
# ---------------------------------------------------------------------------

def _start_agentops_session(run_batch_id: str, run_index_in_week: int) -> bool:
    """Initialise an AgentOps session for the current pipeline run.

    Fails silently — AgentOps downtime must never block the master.

    Args:
        run_batch_id: UUID of the current run batch.
        run_index_in_week: Schedule index (1, 2, or 3).

    Returns:
        ``True`` if the session was started, ``False`` otherwise.
    """
    try:
        api_key: str = api_config.agentops_api_key
        if not api_key:
            logger.warning(
                "_start_agentops_session: AGENTOPS_API_KEY not set — skipping"
            )
            return False

        agentops.init(api_key=api_key, auto_start_session=False)
        agentops.start_trace(
            tags=[
                f"run_batch_id:{run_batch_id}",
                f"run_index:{run_index_in_week}",
            ]
        )
        logger.info(
            "_start_agentops_session: session started for run_batch_id=%s",
            run_batch_id,
        )
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "_start_agentops_session: failed (non-critical) — %s", exc
        )
        return False


def _end_agentops_session(run_batch_id: str, end_state: str = "Success") -> bool:
    """End the current AgentOps session.

    Fails silently — AgentOps downtime must never block the master.

    Args:
        run_batch_id: UUID of the current run batch (for logging context).
        end_state: Final state to report; ``"Success"`` or ``"Fail"``.

    Returns:
        ``True`` if the session was closed, ``False`` otherwise.
    """
    try:
        agentops.end_trace(end_state=end_state)
        logger.info(
            "_end_agentops_session: session closed state=%s run=%s",
            end_state,
            run_batch_id,
        )
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "_end_agentops_session: failed (non-critical) — %s", exc
        )
        return False


def _record_run_summary(run_batch_id: str) -> bool:
    """Emit a run-summary ActionEvent to AgentOps.

    Fails silently — AgentOps downtime must never block the master.

    Args:
        run_batch_id: UUID of the current run batch.

    Returns:
        ``True`` if the event was recorded, ``False`` otherwise.
    """
    try:
        logger.info(
            "_record_run_summary: recorded for run_batch_id=%s (auto-tracked in v4)", run_batch_id
        )
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "_record_run_summary: AgentOps ActionEvent failed (non-critical) — %s",
            exc,
        )
        return False


# ---------------------------------------------------------------------------
# MasterAgent
# ---------------------------------------------------------------------------


@agentops.track_agent(name="MasterAgent")
class MasterAgent:
    """CrewAI Master Agent — full pipeline orchestrator.

    The Master Agent boots the system, creates the run session, executes
    each pipeline phase in sequence (Scraper → Analyser → Apply → Tracker),
    gates every sub-agent behind budget checks, handles all error recovery,
    and produces the final run report.

    Attributes:
        mode: Pipeline execution mode — ``"full"``, ``"scrape_only"``,
            ``"analyse_only"``, ``"apply_only"``, or ``"dry_run"``.
        run_batch_id: UUID generated at __init__ time, replaced by the
            Postgres-assigned UUID after ``_create_run_session``.
        run_index_in_week: Schedule index (1 = Mon, 2 = Thu, 3 = Sat).
        user_id: UUID of the candidate user.
        llm_interface: Centralised LLM provider manager.
        llm: Primary CrewAI LLM (Groq llama-3.3-70b-versatile).
        fallback_llm: Fallback LLM (Cerebras llama-3.3-70b).
    """

    def __init__(self, mode: str = "full") -> None:
        """Initialise the Master Agent.

        Args:
            mode: Execution mode. One of ``"full"``, ``"scrape_only"``,
                ``"analyse_only"``, ``"apply_only"``, ``"dry_run"``.
                ``"dry_run"`` sets the ``DRY_RUN`` environment variable
                to ``"true"`` and then runs the full pipeline.
        """
        # dry_run sets env flag, still runs full pipeline
        if mode == "dry_run":
            os.environ["DRY_RUN"] = "true"

        self.mode: str = mode if mode != "dry_run" else "full"
        self.run_batch_id: str = str(uuid.uuid4())
        self.run_index_in_week: int = self._calculate_run_index()
        from config.config_loader import config_loader as _cl
        self.user_id: str = (
            _cl.get_user_metadata().get("email", "default_user")
        )

        self.llm_interface: LLMInterface = LLMInterface()
        self.llm = self.llm_interface.get_llm("MASTER_AGENT")
        self.fallback_llm = self.llm_interface.get_fallback_llm(
            "MASTER_AGENT", level=1
        )
        self._current_llm = self.llm
        self._run_state: Dict[str, Any] = {}

        self.logger: logging.Logger = logging.getLogger(self.__class__.__name__)

        self.logger.info(
            "MasterAgent initialised — mode=%s run_batch_id=%s index=%d",
            self.mode,
            self.run_batch_id,
            self.run_index_in_week,
        )

    # ------------------------------------------------------------------
    # Internal: run index calculation
    # ------------------------------------------------------------------

    def _calculate_run_index(self) -> int:
        """Determine the run index in the week based on current weekday.

        Returns:
            ``1`` for Monday, ``2`` for Thursday, ``3`` for Saturday.
            Defaults to ``1`` for any other weekday.
        """
        weekday: int = datetime.utcnow().weekday()  # Mon=0 … Sun=6
        mapping: Dict[int, int] = {0: 1, 3: 2, 5: 3}
        return mapping.get(weekday, 1)

    # ------------------------------------------------------------------
    # Internal: system boot
    # ------------------------------------------------------------------

    def _boot_system(self) -> bool:
        """Perform full system health check before any run begins.

        Steps:
            1. Validate all required environment variables.
            2. Test database connection.
            3. Test LLM provider connections (non-critical).
            4. Check default resume file exists (non-critical).
            5. Log full boot report.

        Returns:
            ``True`` if the database is reachable and critical env vars are
            present; ``False`` otherwise (run must not proceed).
        """
        from config.config_loader import config_loader
        self.logger.info("=== SYSTEM BOOT CHECK START ===")

        # Step 1 — Validate critical environment variables
        critical_keys: Dict[str, str] = {
            "XAI_API_KEY": api_config.xai_api_key,
            "GROQ_API_KEY": api_config.groq_api_key,
        }
        # At least one DB URL must be configured
        has_db_url: bool = bool(
            db_config.local_postgres_url or db_config.supabase_url
        )

        missing_keys: List[str] = [
            key for key, value in critical_keys.items() if not value.strip()
        ]
        if not has_db_url:
            missing_keys.append("LOCAL_POSTGRES_URL or SUPABASE_URL")

        if missing_keys:
            self.logger.critical(
                "Boot failed — missing critical env vars: %s",
                ", ".join(missing_keys),
            )
            return False

        self.logger.info("Boot step 1 PASSED — all critical env vars present")

        # Step 2 — Test database connection
        dry_run: bool = os.getenv("DRY_RUN", "false").strip().lower() == "true"
        if dry_run:
            self.logger.warning(
                "Boot step 2 SKIPPED — DRY_RUN=true, "
                "database connectivity not required"
            )
        else:
            try:
                conn = get_db_conn()
                conn.close()
                self.logger.info("Boot step 2 PASSED — database connection OK")
            except Exception as exc:  # noqa: BLE001
                self.logger.critical(
                    "Boot step 2 FAILED — database connection error: %s", exc
                )
                return False

        # Step 3 — Test LLM connections (non-critical)
        for agent_type in ("MASTER_AGENT", "SCRAPER_AGENT"):
            try:
                result = self.llm_interface.test_connection(agent_type)
                if result.get("reachable"):
                    self.logger.info(
                        "Boot step 3 — LLM %s reachable (%.0fms)",
                        agent_type,
                        result.get("latency_ms", 0),
                    )
                else:
                    self.logger.warning(
                        "Boot step 3 — LLM %s NOT reachable: %s",
                        agent_type,
                        result.get("error"),
                    )
            except Exception as exc:  # noqa: BLE001
                self.logger.warning(
                    "Boot step 3 — LLM test for %s failed: %s",
                    agent_type,
                    exc,
                )

        # Step 4 — Check resume files exist (non-critical)
        resume_path: Path = (
            Path(os.getenv("RESUME_DIR", "app/resumes"))
            / os.getenv("DEFAULT_RESUME", "Aarjun_Gen.pdf")
        )
        if resume_path.exists():
            self.logger.info(
                "Boot step 4 PASSED — default resume found: %s", resume_path
            )
        else:
            self.logger.warning(
                "Boot step 4 WARNING — default resume NOT found: %s",
                resume_path,
            )

        # Step 5 — Boot report summary
        self.logger.info(
            "=== BOOT COMPLETE | DB=%s | mode=%s | run_index=%d | "
            "resume_dir=%s | dry_run=%s ===",
            db_config.active_db,
            self.mode,
            self.run_index_in_week,
            os.getenv("RESUME_DIR", "app/resumes"),
            os.getenv("DRY_RUN", "false"),
        )

        return True

    # ------------------------------------------------------------------
    # Internal: run session creation
    # ------------------------------------------------------------------

    def _create_run_session(self) -> str:
        """Create the run batch in Postgres and initialise cost tracking.

        Creates a ``run_batches`` row, resets the per-run cost tracker,
        and starts the AgentOps session.

        Returns:
            The Postgres-assigned ``run_batch_id`` UUID string.
        """
        # Create run batch in Postgres
        raw_result: str = _create_run_batch(run_index_in_week=self.run_index_in_week)
        parsed: Dict[str, Any] = {}
        try:
            parsed = json.loads(raw_result)
        except (json.JSONDecodeError, TypeError) as exc:
            self.logger.warning(
                "_create_run_session: could not parse create_run_batch "
                "response — %s — raw=%s",
                exc,
                raw_result[:200],
            )

        if parsed.get("run_batch_id"):
            self.run_batch_id = parsed["run_batch_id"]

        # Reset per-run cost tracker
        reset_run_cost_tracker(run_batch_id=self.run_batch_id)

        # Start AgentOps session
        _start_agentops_session(self.run_batch_id, self.run_index_in_week)

        # Log session creation
        _log_event(
            run_batch_id=self.run_batch_id,
            level="INFO",
            event_type="run_session_created",
            message=(
                f"Run {self.run_batch_id} started | mode={self.mode} "
                f"| index={self.run_index_in_week}"
            ),
        )

        self.logger.info(
            "Run session created: run_batch_id=%s", self.run_batch_id
        )

        return self.run_batch_id

    # ------------------------------------------------------------------
    # Internal: phase runners
    # ------------------------------------------------------------------

    def _run_scraper_phase(self) -> Dict[str, Any]:
        """Execute the Scraper Agent phase.

        Budget gate: checks monthly budget before proceeding.

        Returns:
            Phase result dict with success/failure status and job counts.
        """
        try:
            # Budget gate — monthly budget check
            budget_raw: str = check_monthly_budget(run_batch_id=self.run_batch_id)
            budget_result: Dict[str, Any] = {}
            try:
                budget_result = json.loads(budget_raw)
            except (json.JSONDecodeError, TypeError):
                pass

            if budget_result.get("abort", False):
                self.logger.critical(
                    "Scraper phase aborted — monthly budget exceeded: %s",
                    budget_result.get("reason", "unknown"),
                )
                return {
                    "phase": "scraper",
                    "aborted": True,
                    "reason": "monthly_budget_exceeded",
                    "success": False,
                }

            # Instantiate and run scraper
            scraper_agent = ScraperAgent(run_batch_id=self.run_batch_id)
            result: Dict[str, Any] = scraper_agent.run()

            if not result.get("success", False):
                self.logger.error(
                    "Scraper phase failed: %s", result.get("error", "unknown")
                )
                _record_agent_error(
                    agent_type="ScraperAgent",
                    error_message=result.get("error", "scraper_run_failed"),
                    run_batch_id=self.run_batch_id,
                    error_code="SCRAPER_PHASE_FAILED",
                )
                result["phase"] = "scraper"
                result["aborted"] = True
                return result

            # ----------------------------------------------------------
            # Dedup: filter out jobs already processed in this run_batch_id
            # Guards against crash-restart re-processing duplicates.
            # Uses fuzzy dedup (title+company sim ≥ 0.85) on top of URL hash.
            # ----------------------------------------------------------
            try:
                db_conn = get_db_conn()
                try:
                    with db_conn.cursor() as cur:
                        cur.execute(
                            "SELECT url FROM jobs WHERE run_batch_id = %s",
                            (self.run_batch_id,)
                        )
                        already_processed = {row[0] for row in cur.fetchall()}

                    original_count = result.get("total_jobs", 0)
                    removed = len(already_processed)
                    if removed:
                        self.logger.info(
                            "run_batch_id dedup: %d URLs already in DB | "
                            "original=%d",
                            removed,
                            original_count,
                        )
                finally:
                    db_conn.close()
            except Exception as e:
                self.logger.warning(
                    "run_batch dedup query failed: %s — "
                    "proceeding with full unfiltered batch",
                    str(e),
                )
                # Fail-soft: do NOT abort run on dedup failure

            # Store result in run state
            self._run_state["scraper"] = result

            _log_event(
                run_batch_id=self.run_batch_id,
                level="INFO",
                event_type="scraper_phase_complete",
                message=(
                    f"Discovered {result.get('total_jobs', 0)} jobs"
                ),
            )

            self.logger.info(
                "Scraper phase complete: %d jobs discovered",
                result.get("total_jobs", 0),
            )

            return result

        except Exception as exc:  # noqa: BLE001
            self.logger.error(
                "Scraper phase unhandled exception: %s", exc, exc_info=True
            )
            _record_agent_error(
                agent_type="ScraperAgent",
                error_message=str(exc),
                run_batch_id=self.run_batch_id,
                error_code="SCRAPER_PHASE_EXCEPTION",
            )
            return {
                "phase": "scraper",
                "aborted": True,
                "success": False,
                "error": str(exc),
            }

    def _run_analyser_phase(self) -> Dict[str, Any]:
        """Execute the Analyser Agent phase.

        Requires scraper phase success with jobs discovered.
        Budget gate: checks xAI run cap before proceeding.

        Returns:
            Phase result dict with scoring/routing counts.
        """
        try:
            # Requires scraper phase with jobs
            scraper_result: Dict[str, Any] = self._run_state.get("scraper", {})
            if scraper_result.get("total_jobs", 0) == 0:
                self.logger.info(
                    "Analyser phase skipped — no jobs to analyse"
                )
                return {
                    "phase": "analyser",
                    "aborted": True,
                    "reason": "no_jobs_to_analyse",
                    "success": True,
                }

            # Budget gate — xAI run cap
            cap_raw: str = check_xai_run_cap(run_batch_id=self.run_batch_id)
            cap_result: Dict[str, Any] = {}
            try:
                cap_result = json.loads(cap_raw)
            except (json.JSONDecodeError, TypeError):
                pass

            if cap_result.get("abort", False):
                self.logger.critical(
                    "Analyser phase aborted — xAI run cap exceeded"
                )
                return {
                    "phase": "analyser",
                    "aborted": True,
                    "reason": "xai_run_cap_exceeded",
                    "success": False,
                }

            # Instantiate and run analyser
            analyser_agent = AnalyserAgent(
                run_batch_id=self.run_batch_id,
                user_id=self.user_id,
            )
            result: Dict[str, Any] = analyser_agent.run()

            if not result.get("success", False):
                self.logger.error(
                    "Analyser phase failed: %s", result.get("error", "unknown")
                )
                _record_agent_error(
                    agent_type="AnalyserAgent",
                    error_message=result.get("error", "analyser_run_failed"),
                    run_batch_id=self.run_batch_id,
                    error_code="ANALYSER_PHASE_FAILED",
                )
                result["phase"] = "analyser"
                result["aborted"] = True
                return result

            # Store result in run state
            self._run_state["analyser"] = result

            _log_event(
                run_batch_id=self.run_batch_id,
                level="INFO",
                event_type="analyser_phase_complete",
                message=(
                    f"Scored {result.get('total_scored', 0)} jobs | "
                    f"auto={result.get('auto_route', 0)} "
                    f"manual={result.get('manual_route', 0)} "
                    f"skip={result.get('skipped', 0)}"
                ),
            )

            self.logger.info(
                "Analyser phase complete: scored=%d auto=%d manual=%d skip=%d",
                result.get("total_scored", 0),
                result.get("auto_route", 0),
                result.get("manual_route", 0),
                result.get("skipped", 0),
            )

            return result

        except Exception as exc:  # noqa: BLE001
            self.logger.error(
                "Analyser phase unhandled exception: %s", exc, exc_info=True
            )
            _record_agent_error(
                agent_type="AnalyserAgent",
                error_message=str(exc),
                run_batch_id=self.run_batch_id,
                error_code="ANALYSER_PHASE_EXCEPTION",
            )
            return {
                "phase": "analyser",
                "aborted": True,
                "success": False,
                "error": str(exc),
            }

    def _check_run_cost_veto(self) -> bool:
        """Check if projected run cost exceeds the per-run veto threshold.

        Uses ``RUN_COST_VETO`` env var (default $0.10) as the hard cap.
        Queries the cost tracker for current spend.

        Returns:
            ``True`` if the run should be vetoed (cost exceeded),
            ``False`` if within budget.
        """
        from config.config_loader import config_loader
        veto_threshold: float = float(
            config_loader.get_budget_settings().get("xai_cost_cap_per_run_usd", 0.38)
        )
        try:
            cost_raw: str = get_cost_summary(run_batch_id=self.run_batch_id)
            cost_data: Dict[str, Any] = json.loads(cost_raw)
            total_spent: float = float(cost_data.get("run_total_cost", 0.0))
            if total_spent >= veto_threshold:
                self.logger.critical(
                    "RUN COST VETO: $%.4f spent ≥ $%.4f threshold — vetoing",
                    total_spent,
                    veto_threshold,
                )
                return True
        except Exception as exc:  # noqa: BLE001
            self.logger.warning(
                "_check_run_cost_veto: cost check failed (proceeding): %s", exc
            )
        return False

    def _run_apply_phase(self) -> Dict[str, Any]:
        """Execute the Apply Agent phase.

        Requires analyser phase and ``AUTO_APPLY_ENABLED=true``.
        Budget gate: checks xAI run cap AND per-run cost veto before proceeding.
        Imports ``ApplyAgent`` inline to avoid circular import issues.

        Returns:
            Phase result dict with application counts.
        """
        try:
            # Check if auto-apply is enabled
            if not run_config.auto_apply_enabled:
                self.logger.info(
                    "Apply phase skipped — auto_apply_enabled=false"
                )
                return {
                    "phase": "apply",
                    "skipped": True,
                    "reason": "auto_apply_disabled",
                    "success": True,
                }

            # Budget gate — xAI run cap
            cap_raw: str = check_xai_run_cap(run_batch_id=self.run_batch_id)
            cap_result: Dict[str, Any] = {}
            try:
                cap_result = json.loads(cap_raw)
            except (json.JSONDecodeError, TypeError):
                pass

            if cap_result.get("abort", False):
                self.logger.critical(
                    "Apply phase aborted — xAI run cap exceeded"
                )
                return {
                    "phase": "apply",
                    "aborted": True,
                    "reason": "xai_run_cap_exceeded",
                    "success": False,
                }

            # Budget gate — per-run cost veto ($0.10 default)
            if self._check_run_cost_veto():
                return {
                    "phase": "apply",
                    "aborted": True,
                    "reason": "run_cost_veto",
                    "success": False,
                }

            # Get routing manifest — jobs with route="auto"
            analyser_result: Dict[str, Any] = self._run_state.get(
                "analyser", {}
            )
            manifest: List[Dict[str, Any]] = analyser_result.get(
                "routing_manifest", []
            )

            # Filter to auto-route jobs only
            auto_jobs: List[Dict[str, Any]] = [
                j for j in manifest
                if j.get("route") == "auto"
            ]

            if not auto_jobs:
                self.logger.info(
                    "Apply phase skipped — no auto-route jobs in manifest"
                )
                return {
                    "phase": "apply",
                    "skipped": True,
                    "reason": "no_auto_route_jobs",
                    "success": True,
                }

            # Import ApplyAgent inline to avoid circular imports
            from agents.apply_agent import ApplyAgent  # type: ignore[import-untyped]

            apply_agent = ApplyAgent(
                run_batch_id=self.run_batch_id,
                user_id=self.user_id,
                routing_manifest=auto_jobs,
            )
            result: Dict[str, Any] = apply_agent.run()

            # Store result in run state
            self._run_state["apply"] = result

            applied_count: int = result.get("applied", 0)
            failed_count: int = result.get("failed", 0)

            _log_event(
                run_batch_id=self.run_batch_id,
                level="INFO",
                event_type="apply_phase_complete",
                message=(
                    f"Applied={applied_count} Failed={failed_count}"
                ),
            )

            self.logger.info(
                "Apply phase complete: applied=%d failed=%d",
                applied_count,
                failed_count,
            )

            return result

        except ImportError:
            self.logger.warning(
                "Apply phase skipped — ApplyAgent not yet implemented"
            )
            return {
                "phase": "apply",
                "skipped": True,
                "reason": "apply_agent_not_implemented",
                "success": True,
            }
        except Exception as exc:  # noqa: BLE001
            self.logger.error(
                "Apply phase unhandled exception: %s", exc, exc_info=True
            )
            _record_agent_error(
                agent_type="ApplyAgent",
                error_message=str(exc),
                run_batch_id=self.run_batch_id,
                error_code="APPLY_PHASE_EXCEPTION",
            )
            return {
                "phase": "apply",
                "aborted": True,
                "success": False,
                "error": str(exc),
            }

    def _run_tracker_phase(self) -> Dict[str, Any]:
        """Execute the Tracker Agent phase.

        This phase **always** runs regardless of other phase outcomes.
        The tracker must always close the session.

        Returns:
            Phase result dict with Notion sync counts.
        """
        try:
            tracker_agent = TrackerAgent(
                run_batch_id=self.run_batch_id,
                user_id=self.user_id,
            )
            result: Dict[str, Any] = tracker_agent.run()

            # Store result in run state
            self._run_state["tracker"] = result

            _log_event(
                run_batch_id=self.run_batch_id,
                level="INFO",
                event_type="tracker_phase_complete",
                message=(
                    f"Notion synced: applied={result.get('notion_synced_applied', 0)} "
                    f"queued={result.get('notion_synced_queued', 0)}"
                ),
            )

            self.logger.info(
                "Tracker phase complete: synced_applied=%d synced_queued=%d",
                result.get("notion_synced_applied", 0),
                result.get("notion_synced_queued", 0),
            )

            return result

        except Exception as exc:  # noqa: BLE001
            self.logger.error(
                "Tracker phase unhandled exception: %s", exc, exc_info=True
            )
            return {
                "phase": "tracker",
                "success": False,
                "error": str(exc),
            }

    # ------------------------------------------------------------------
    # Internal: final report
    # ------------------------------------------------------------------

    def _build_final_report(self, started_at: datetime) -> Dict[str, Any]:
        """Aggregate all phase results into a complete run report.

        Queries Postgres for final run stats and the cost summary module
        for final spend figures.

        Args:
            started_at: UTC datetime when the run started.

        Returns:
            Comprehensive run report dictionary.
        """
        completed_at: datetime = datetime.utcnow()
        duration_minutes: float = (
            (completed_at - started_at).total_seconds() / 60.0
        )

        # Determine which phases completed vs aborted
        phases_completed: List[str] = []
        phases_aborted: List[str] = []

        for phase_name in ("scraper", "analyser", "apply", "tracker"):
            phase_result: Dict[str, Any] = self._run_state.get(phase_name, {})
            if phase_result.get("aborted", False):
                phases_aborted.append(phase_name)
            elif phase_result.get("skipped", False):
                pass  # skipped phases are neither completed nor aborted
            elif phase_result:
                phases_completed.append(phase_name)

        # Get final Postgres stats
        run_stats: Dict[str, Any] = {}
        try:
            raw_stats: str = get_run_stats.run(
                json.dumps({"run_batch_id": self.run_batch_id})
            )
            run_stats = json.loads(raw_stats)
        except Exception as exc:  # noqa: BLE001
            self.logger.warning(
                "_build_final_report: get_run_stats failed — %s", exc
            )

        # Get final cost summary
        cost_data: Dict[str, Any] = {}
        try:
            raw_cost: str = get_cost_summary(run_batch_id=self.run_batch_id)
            cost_data = json.loads(raw_cost)
        except Exception as exc:  # noqa: BLE001
            self.logger.warning(
                "_build_final_report: get_cost_summary failed — %s", exc
            )

        # Extract counts from phase results
        scraper_result: Dict[str, Any] = self._run_state.get("scraper", {})
        analyser_result: Dict[str, Any] = self._run_state.get("analyser", {})
        apply_result: Dict[str, Any] = self._run_state.get("apply", {})

        jobs_discovered: int = scraper_result.get(
            "total_jobs",
            run_stats.get("jobs_discovered", 0),
        )
        jobs_auto_applied: int = apply_result.get(
            "applied",
            run_stats.get("jobs_auto_applied", 0),
        )
        jobs_manual_queued: int = analyser_result.get(
            "manual_route",
            run_stats.get("jobs_queued", 0),
        )
        jobs_skipped: int = analyser_result.get("skipped", 0)
        jobs_failed: int = apply_result.get("failed", 0)

        # Notion sync count from tracker
        tracker_result: Dict[str, Any] = self._run_state.get("tracker", {})
        notion_synced: int = (
            tracker_result.get("notion_synced_applied", 0)
            + tracker_result.get("notion_synced_queued", 0)
        )

        # Cost figures
        xai_cost: float = float(cost_data.get("run_xai_cost", 0.0))
        perplexity_cost: float = float(
            cost_data.get("run_perplexity_cost", 0.0)
        )
        total_cost: float = float(cost_data.get("run_total_cost", 0.0))
        budget_remaining: float = float(
            cost_data.get("xai_cap_remaining", budget_config.xai_cost_cap_per_run)
        )

        # Safety net / fallback counts
        safety_net_triggered: bool = scraper_result.get(
            "safety_net_triggered", False
        )

        # Count fallback events (approximate from analyser state)
        llm_fallback_events: int = 0
        if analyser_result.get("fallback_activated", False):
            llm_fallback_events += 1

        # Determine overall success
        success: bool = bool(
            "scraper" not in phases_aborted
            and "analyser" not in phases_aborted
        )

        report: Dict[str, Any] = {
            "run_batch_id": self.run_batch_id,
            "run_index_in_week": self.run_index_in_week,
            "mode": self.mode,
            "dry_run": os.getenv("DRY_RUN", "false").lower() == "true",
            "started_at": started_at.isoformat(),
            "completed_at": completed_at.isoformat(),
            "duration_minutes": round(duration_minutes, 2),
            "phases_completed": phases_completed,
            "phases_aborted": phases_aborted,
            "jobs_discovered": jobs_discovered,
            "jobs_auto_applied": jobs_auto_applied,
            "jobs_manual_queued": jobs_manual_queued,
            "jobs_skipped": jobs_skipped,
            "jobs_failed": jobs_failed,
            "notion_synced": notion_synced,
            "xai_cost_usd": round(xai_cost, 4),
            "perplexity_cost_usd": round(perplexity_cost, 4),
            "total_cost_usd": round(total_cost, 4),
            "budget_remaining_usd": round(budget_remaining, 4),
            "safety_net_triggered": safety_net_triggered,
            "llm_fallback_events": llm_fallback_events,
            "success": success,
        }

        return report

    # ------------------------------------------------------------------
    # Public: main entry point
    # ------------------------------------------------------------------

    @operation
    def run(self) -> Dict[str, Any]:
        """Execute the full pipeline run.

        This is the single public entry point called by ``main.py``.
        Orchestrates all phases in sequence based on ``self.mode``,
        handles early termination on phase abort, and always ensures
        the Tracker phase runs.

        Returns:
            Complete run report dictionary. On unhandled exception,
            returns a minimal dict with ``success=False`` and the error.
        """
        started_at: datetime = datetime.utcnow()
        pipeline_aborted: bool = False
        unhandled_exception: Optional[Exception] = None
        report: Dict[str, Any] = {}

        try:
            # Step 1 — Boot system
            if not self._boot_system():
                self._run_state["boot_failed"] = True
                pipeline_aborted = True
                raise RuntimeError("Boot failed — services unhealthy")

            # Step 2 — Create run session
            self._create_run_session()

            # Step 3 — Log pipeline start banner
            _log_event(
                run_batch_id=self.run_batch_id,
                level="INFO",
                event_type="pipeline_start",
                message=(
                    f"=== PIPELINE START | run={self.run_batch_id} "
                    f"| mode={self.mode} "
                    f"| target={run_config.jobs_per_run_target} jobs ==="
                ),
            )

            self.logger.info(
                "=== PIPELINE START | run=%s | mode=%s | target=%d jobs ===",
                self.run_batch_id,
                self.mode,
                run_config.jobs_per_run_target,
            )

            # Step 4 — Execute phases based on mode
            if self.mode in ("full",):
                # Full pipeline: scraper → analyser → apply → tracker

                # Scraper phase
                scraper_result: Dict[str, Any] = self._run_scraper_phase()
                if scraper_result.get("aborted", False):
                    pipeline_aborted = True

                # Per-run cost veto between phases
                if not pipeline_aborted and self._check_run_cost_veto():
                    pipeline_aborted = True
                    self.logger.critical(
                        "Pipeline aborted — per-run cost veto after scraper"
                    )

                # Analyser phase (only if scraper did not abort)
                if not pipeline_aborted:
                    analyser_result: Dict[str, Any] = (
                        self._run_analyser_phase()
                    )
                    if analyser_result.get("aborted", False):
                        pipeline_aborted = True

                # Per-run cost veto between phases
                if not pipeline_aborted and self._check_run_cost_veto():
                    pipeline_aborted = True
                    self.logger.critical(
                        "Pipeline aborted — per-run cost veto after analyser"
                    )

                # Apply phase (only if analyser did not abort)
                if not pipeline_aborted:
                    self._run_apply_phase()

            elif self.mode == "scrape_only":
                scraper_result = self._run_scraper_phase()
                if scraper_result.get("aborted", False):
                    pipeline_aborted = True

            elif self.mode == "analyse_only":
                # Score jobs from previous run batch stored in DB
                analyser_result = self._run_analyser_phase()
                if analyser_result.get("aborted", False):
                    pipeline_aborted = True

            elif self.mode == "apply_only":
                # Apply phase loads routing manifest from DB
                apply_result = self._run_apply_phase()
                if apply_result.get("aborted", False):
                    pipeline_aborted = True

        except Exception as exc:
            unhandled_exception = exc
            pipeline_aborted = True
            self.logger.critical(
                "Master Agent unhandled exception: %s", exc, exc_info=True
            )

            try:
                _record_agent_error(
                    agent_type="MasterAgent",
                    error_message=str(exc),
                    run_batch_id=self.run_batch_id,
                    error_code="CRITICAL",
                )
            except Exception as record_exc:  # noqa: BLE001
                self.logger.warning(
                    "Failed to record agent error (non-critical): %s", record_exc
                )

        finally:
            # Tracker phase — ALWAYS runs, never skipped
            try:
                self._run_tracker_phase()
            except Exception as tracker_exc:  # noqa: BLE001
                self.logger.error(
                    "TrackerAgent failed in finally block: %s",
                    tracker_exc,
                    exc_info=True,
                )

            # Build final report after tracker (so tracker state is included)
            try:
                report = self._build_final_report(started_at)
            except Exception as report_exc:  # noqa: BLE001
                self.logger.error(
                    "Failed to build final report: %s", report_exc, exc_info=True
                )
                report = {
                    "success": False,
                    "error": str(report_exc),
                    "run_batch_id": self.run_batch_id,
                }

            if unhandled_exception is not None:
                report["success"] = False
                report["error"] = str(unhandled_exception)
            elif pipeline_aborted and report.get("success", True):
                report["success"] = False

            # Update run batch stats
            try:
                _update_run_batch_stats(
                    run_batch_id=self.run_batch_id,
                    jobs_discovered=report["jobs_discovered"],
                    jobs_auto_applied=report["jobs_auto_applied"],
                    jobs_queued=report["jobs_manual_queued"],
                )
            except Exception as exc:  # noqa: BLE001
                self.logger.warning(
                    "Failed to update run_batch_stats: %s", exc
                )

            # Log pipeline close banner
            try:
                _log_event(
                    run_batch_id=self.run_batch_id,
                    level="INFO",
                    event_type="pipeline_complete",
                    message=(
                        f"=== PIPELINE COMPLETE "
                        f"| applied={report.get('jobs_auto_applied', 0)} "
                        f"| queued={report.get('jobs_manual_queued', 0)} "
                        f"| cost=${float(report.get('total_cost_usd', 0.0)):.4f} ==="
                    ),
                )
            except Exception as exc:  # noqa: BLE001
                self.logger.warning(
                    "Failed to log pipeline_complete event: %s", exc
                )

            self.logger.info(
                "=== PIPELINE COMPLETE | applied=%d | queued=%d | cost=$%.4f ===",
                int(report.get("jobs_auto_applied", 0)),
                int(report.get("jobs_manual_queued", 0)),
                float(report.get("total_cost_usd", 0.0)),
            )

            # Record run summary and end AgentOps session
            _record_run_summary(self.run_batch_id)
            end_state = "Success" if report.get("success") else "Fail"
            _end_agentops_session(self.run_batch_id, end_state)

            # Write JSON report to logs/
            try:
                self._write_run_report(report)
            except Exception as exc:  # noqa: BLE001
                self.logger.warning("Failed to write run report: %s", exc)

        return report

    # ------------------------------------------------------------------
    # Class method: CLI convenience constructor
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Internal: write JSON report to logs/
    # ------------------------------------------------------------------

    def _write_run_report(self, report: Dict[str, Any]) -> None:
        """Write the run report as JSON to the logs directory.

        Creates both ``latest_run.json`` (overwritten each run) and a
        timestamped file for audit history.

        Args:
            report: Complete run report dictionary.
        """
        try:
            logs_dir = Path("logs")
            logs_dir.mkdir(parents=True, exist_ok=True)

            # Write latest run
            latest = logs_dir / "latest_run.json"
            latest.write_text(json.dumps(report, indent=2, default=str))

            # Write timestamped archive
            ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            archive = logs_dir / f"run_{ts}_{self.run_batch_id[:8]}.json"
            archive.write_text(json.dumps(report, indent=2, default=str))

            self.logger.info(
                "Run report written to %s and %s", latest, archive
            )
        except Exception as exc:  # noqa: BLE001
            self.logger.warning(
                "_write_run_report: failed to write report: %s", exc
            )

    @classmethod
    def from_cli(cls, mode: str = "full") -> "MasterAgent":
        """Convenience constructor for ``main.py`` CLI usage.

        Reads ``LOG_LEVEL`` from environment and configures Python
        ``logging.basicConfig`` before returning the agent instance.

        Args:
            mode: Pipeline execution mode (same values as ``__init__``).

        Returns:
            Configured ``MasterAgent`` instance ready to call ``.run()``.
        """
        log_level: str = os.getenv("LOG_LEVEL", "INFO").upper()
        logging.basicConfig(
            level=getattr(logging, log_level, logging.INFO),
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )

        return cls(mode=mode)
