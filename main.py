"""AI Job Application Agent — single CLI entry point.

Invoked by GitHub Actions cron (``--mode full``), manual developer runs
(``--mode dry_run``), and Docker CMD.  Boots logging, validates env,
instantiates :class:`~agents.master_agent.MasterAgent`, runs the full
pipeline, prints the final JSON report, and exits with the correct
POSIX code.

Includes:
- SessionBootstrapper: Verifies all 7 Docker services healthy before work
- BudgetEnforcer: Checks $9.50 monthly cap before any LLM calls
- SIGTERM handler: Graceful shutdown with Postgres cleanup
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import signal
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Optional

import httpx
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# MODULE-LEVEL SETUP  (runs at import time)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# ENVIRONMENT LOADING
# ---------------------------------------------------------------------------
_env_path = os.getenv("DOTENV_PATH", os.path.expanduser("~/java.env"))
if os.path.exists(_env_path):
    load_dotenv(dotenv_path=_env_path, override=False)

# Ensure logs/ directory exists before any FileHandler is created.
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/pipeline.log", mode="a", encoding="utf-8"),
    ],
)

logger: logging.Logger = logging.getLogger("main")

# Deferred imports — placed here so env is loaded first.
from config.settings import api_config, budget_config, db_config, run_config  # noqa: E402
from agents.master_agent import MasterAgent  # noqa: E402

__all__ = [
    "main",
    "SessionBootstrapper",
    "BudgetEnforcer",
    "PipelineRunner",
]


# ---------------------------------------------------------------------------
# SERVICE HEALTH DEFINITIONS
# ---------------------------------------------------------------------------

@dataclass
class ServiceHealth:
    """Definition of a Docker service health endpoint."""
    name: str
    url: str
    port: int
    timeout: float = 5.0


# Service host/port resolution via environment with Compose defaults
POSTGRES_HOST = os.getenv("LOCAL_POSTGRES_HOST","ai_postgres")
POSTGRES_PORT = int(os.getenv("LOCAL_POSTGRES_PORT", "5432"))

REDIS_HOST = os.getenv("REDIS_HOST","ai_redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))

CHROMA_HOST = os.getenv("CHROMA_HOST", "ai_chromadb")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8001"))

RAG_HOST = os.getenv("RAG_SERVER_HOST", "ai_rag_server")
RAG_PORT = int(os.getenv("RAG_SERVER_PORT", "8090"))

FASTAPI_HOST = os.getenv("FASTAPI_HOST", "ai_api_server")
FASTAPI_PORT = int(os.getenv("FASTAPI_PORT", "8000"))

PLAYWRIGHT_SCRAPER_HOST = os.getenv("PLAYWRIGHT_SCRAPER_HOST",  "ai_playwright_scraper")
PLAYWRIGHT_SCRAPER_PORT = int(os.getenv("PLAYWRIGHT_SCRAPER_PORT", "8001"))

PLAYWRIGHT_APPLY_HOST = os.getenv("PLAYWRIGHT_APPLY_HOST", "ai_playwright_apply")
PLAYWRIGHT_APPLY_PORT = int(os.getenv("PLAYWRIGHT_APPLY_PORT", "8003"))


# All 7 services that must be healthy before pipeline starts
REQUIRED_SERVICES: list[ServiceHealth] = [
    ServiceHealth("database", f"{POSTGRES_HOST}:{POSTGRES_PORT}", POSTGRES_PORT, timeout=3.0),
    ServiceHealth("cache", f"{REDIS_HOST}:{REDIS_PORT}", REDIS_PORT, timeout=3.0),
    ServiceHealth("chromadb", f"http://{CHROMA_HOST}:{CHROMA_PORT}/api/v2/heartbeat", CHROMA_PORT),
    ServiceHealth("rag_server", f"http://{RAG_HOST}:{RAG_PORT}/health", RAG_PORT),
    ServiceHealth("api_server", f"http://{FASTAPI_HOST}:{FASTAPI_PORT}/health", FASTAPI_PORT),
    ServiceHealth("pw_scraper", f"http://{PLAYWRIGHT_SCRAPER_HOST}:{PLAYWRIGHT_SCRAPER_PORT}/health", PLAYWRIGHT_SCRAPER_PORT),
    ServiceHealth("pw_apply", f"http://{PLAYWRIGHT_APPLY_HOST}:{PLAYWRIGHT_APPLY_PORT}/health", PLAYWRIGHT_APPLY_PORT),
]


# ---------------------------------------------------------------------------
# SESSION BOOTSTRAPPER
# ---------------------------------------------------------------------------

class SessionBootstrapper:
    """Verify all Docker services are healthy before pipeline work starts.
    
    Checks each service with 3 retries and 30-second exponential backoff.
    On failure, sends Notion alert and exits with code 1.
    """
    
    def __init__(self, max_retries: int = 3, backoff_base: float = 30.0) -> None:
        """Initialize the session bootstrapper.
        
        Args:
            max_retries: Maximum retry attempts per service.
            backoff_base: Base backoff duration in seconds.
        """
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.logger = logging.getLogger(self.__class__.__name__)
    
    async def _check_tcp_service(self, host: str, port: int, timeout: float) -> bool:
        """Check TCP-only services (Postgres, Redis) via socket connection."""
        import socket
        try:
            # Extract host from URL if needed
            if host.startswith("http://"):
                host = host.replace("http://", "").split(":")[0]
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            result = sock.connect_ex((host, port))
            sock.close()
            return result == 0
        except Exception as exc:
            self.logger.debug("TCP check failed for %s:%d: %s", host, port, exc)
            return False
    
    async def _check_http_service(self, url: str, timeout: float) -> bool:
        """Check HTTP services via GET /health endpoint."""
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.get(url)
                return resp.status_code == 200
        except Exception as exc:
            self.logger.debug("HTTP check failed for %s: %s", url, exc)
            return False
    
    async def _check_service(self, service: ServiceHealth) -> bool:
        """Check a single service with retries and backoff."""
        for attempt in range(1, self.max_retries + 1):
            self.logger.info(
                "Checking %s (attempt %d/%d)...",
                service.name, attempt, self.max_retries
            )
            
            # TCP-only for Postgres and Redis
            if service.name in ("database", "cache"):
                host = service.url.replace("http://", "").split(":")[0]
                healthy = await self._check_tcp_service(host, service.port, service.timeout)
            else:
                healthy = await self._check_http_service(service.url, service.timeout)
            
            if healthy:
                self.logger.info("✅ %s is healthy", service.name)
                return True
            
            if attempt < self.max_retries:
                wait = self.backoff_base * attempt
                self.logger.warning(
                    "⚠️ %s unhealthy, retrying in %.0fs...", service.name, wait
                )
                await asyncio.sleep(wait)
        
        self.logger.critical("❌ %s failed after %d attempts", service.name, self.max_retries)
        return False
    
    async def verify_all_services(self) -> tuple[bool, list[str]]:
        """Verify all required services are healthy.
        
        Returns:
            Tuple of (all_healthy, list_of_failed_service_names)
        """
        failed_services: list[str] = []
        
        for service in REQUIRED_SERVICES:
            if not await self._check_service(service):
                failed_services.append(service.name)
        
        all_healthy = len(failed_services) == 0
        return all_healthy, failed_services
    
    async def boot(self) -> bool:
        """Run the full boot sequence.
        
        Returns:
            True if all services healthy, False otherwise.
            On failure, sends Notion alert before returning.
        """
        self.logger.info("=" * 60)
        self.logger.info("SESSION BOOTSTRAPPER — Verifying Docker Services")
        self.logger.info("=" * 60)
        
        all_healthy, failed = await self.verify_all_services()
        
        if not all_healthy:
            self.logger.critical(
                "Boot failed — unhealthy services: %s", ", ".join(failed)
            )
            # Send Notion alert
            await self._send_failure_alert(failed)
            return False
        
        self.logger.info("=" * 60)
        self.logger.info("✅ All %d services healthy — proceeding", len(REQUIRED_SERVICES))
        self.logger.info("=" * 60)
        return True
    
    async def _send_failure_alert(self, failed_services: list[str]) -> None:
        """Send Notion alert about boot failure."""
        try:
            from integrations.notion import NotionClient
            client = NotionClient()
            await client.post_alert(
                message=f"Pipeline boot failed — unhealthy services: {', '.join(failed_services)}",
                level="CRITICAL"
            )
        except Exception as exc:
            self.logger.warning("Failed to send Notion alert: %s", exc)


# ---------------------------------------------------------------------------
# BUDGET ENFORCER
# ---------------------------------------------------------------------------

class BudgetEnforcer:
    """Enforce monthly budget cap before pipeline runs.
    
    Checks budget_tools.get_monthly_spend() > $9.50 → abort with Notion alert.
    """
    
    from config.config_loader import config_loader
    MONTHLY_CAP_USD: float = float(
        config_loader.get_budget_settings().get("total_monthly_budget_usd", 9.50)
    )
    
    def __init__(self) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def check_budget(self) -> tuple[bool, float, float]:
        """Check if budget allows pipeline execution.
        
        Returns:
            Tuple of (can_proceed, spent_usd, remaining_usd)
        """
        try:
            from tools.budget_tools import get_cost_summary
            
            raw = get_cost_summary(run_batch_id="budget_check")
            summary = json.loads(raw) if isinstance(raw, str) else raw
            
            # Calculate monthly spend from the summary
            monthly_spent = float(summary.get("monthly_total_cost", 0.0))
            remaining = self.MONTHLY_CAP_USD - monthly_spent
            can_proceed = monthly_spent < self.MONTHLY_CAP_USD
            
            self.logger.info(
                "Budget check: $%.2f spent / $%.2f cap ($%.2f remaining)",
                monthly_spent, self.MONTHLY_CAP_USD, remaining
            )
            
            return can_proceed, monthly_spent, remaining
            
        except Exception as exc:
            self.logger.warning("Budget check failed, assuming OK: %s", exc)
            return True, 0.0, self.MONTHLY_CAP_USD
    
    async def enforce(self) -> bool:
        """Enforce budget cap. Returns True if pipeline can proceed."""
        can_proceed, spent, remaining = self.check_budget()
        
        if not can_proceed:
            self.logger.critical(
                "BUDGET CAP EXCEEDED: $%.2f spent > $%.2f cap",
                spent, self.MONTHLY_CAP_USD
            )
            # Send Notion alert
            try:
                from integrations.notion import NotionClient
                client = NotionClient()
                await client.post_alert(
                    message=f"Monthly budget cap exceeded: ${spent:.2f} spent > ${self.MONTHLY_CAP_USD:.2f} cap. Pipeline aborted.",
                    level="CRITICAL"
                )
            except Exception as exc:
                self.logger.warning("Failed to send budget alert: %s", exc)
            return False
        
        return True
    
    def veto_llm_heavy(self, projected_cost: float = 0.10) -> bool:
        """Check if LLM-heavy steps should be skipped due to budget.
        
        Args:
            projected_cost: Projected cost of the next operation.
            
        Returns:
            True if should veto (skip LLM-heavy steps).
        """
        can_proceed, spent, remaining = self.check_budget()
        
        if remaining < projected_cost:
            self.logger.warning(
                "Vetoing LLM-heavy step: $%.2f remaining < $%.2f projected",
                remaining, projected_cost
            )
            return True
        return False


# ---------------------------------------------------------------------------
# PIPELINE RUNNER
# ---------------------------------------------------------------------------

@dataclass
class PipelineResult:
    """Result of a full pipeline run."""
    success: bool
    run_batch_id: str
    jobs_discovered: int = 0
    jobs_scored: int = 0
    jobs_auto_applied: int = 0
    jobs_manual_queued: int = 0
    jobs_failed: int = 0
    total_cost_usd: float = 0.0
    duration_minutes: float = 0.0
    error: Optional[str] = None
    interrupted: bool = False


class PipelineRunner:
    """Execute the full pipeline with proper session management."""
    
    def __init__(
        self,
        mode: str = "full",
        session_id: Optional[str] = None,
        budget_override: Optional[float] = None,
        dry_run: bool = False,
    ) -> None:
        """Initialize the pipeline runner.
        
        Args:
            mode: Execution mode (full, scrape-only, apply-only, dry-run).
            session_id: Optional session UUID override.
            budget_override: Optional per-run budget override.
            dry_run: If True, run through scoring but skip apply.
        """
        self.mode = mode
        self.session_id = session_id or str(uuid.uuid4())
        self.budget_override = budget_override
        self.dry_run = dry_run or os.getenv("DRY_RUN", "").lower() == "true"
        self.logger = logging.getLogger(self.__class__.__name__)
        self._shutdown_requested = False
        self._current_run_batch_id: Optional[str] = None
    
    def request_shutdown(self) -> None:
        """Request graceful shutdown."""
        self._shutdown_requested = True
        self.logger.warning("Shutdown requested — will complete current operation")
    
    def _create_run_session(self) -> str:
        """Create a run_sessions record in Postgres."""
        from tools.postgres_tools import _create_run_batch
        
        result = _create_run_batch(
            run_index_in_week=self._get_run_index(),
        )
        data = json.loads(result) if isinstance(result, str) else result
        run_batch_id = data.get("run_batch_id", self.session_id)
        self._current_run_batch_id = run_batch_id
        self.logger.info("Created run session: %s", run_batch_id)
        return run_batch_id
    
    def _close_run_session(self, result: PipelineResult) -> None:
        """Close the run session with final stats."""
        if not self._current_run_batch_id:
            return
        
        try:
            from tools.postgres_tools import _update_run_batch_stats
            
            _update_run_batch_stats(
                run_batch_id=self._current_run_batch_id,
                jobs_discovered=result.jobs_discovered,
                jobs_auto_applied=result.jobs_auto_applied,
                jobs_queued=result.jobs_manual_queued,
            )
            self.logger.info("Closed run session: %s", self._current_run_batch_id)
        except Exception as exc:
            self.logger.error("Failed to close run session: %s", exc)
    
    def _get_run_index(self) -> int:
        """Get run index (1, 2, or 3) based on weekday."""
        weekday = datetime.now(timezone.utc).weekday()
        # Mon=0, Wed=2, Fri=4 → 1, 2, 3
        mapping = {0: 1, 2: 2, 4: 3}
        return mapping.get(weekday, 1)
    
    async def run_full(self) -> PipelineResult:
        """Execute the full pipeline.
        
        Steps:
        1. Create run_sessions record (Postgres)
        2. Call ScraperAgent.run() → List[Job]
        3. Call AnalyserAgent.run(jobs) → List[JobScore]
        4. Call ApplyAgent.run(scores) → ApplyResult (unless dry_run)
        5. Call TrackerAgent.run(result) → FinalReport
        6. Close run_sessions (closed_at=NOW)
        """
        start_time = datetime.now(timezone.utc)
        result = PipelineResult(
            success=False,
            run_batch_id=self.session_id,
        )
        
        try:
            # Step 1: Create run session
            run_batch_id = self._create_run_session()
            result.run_batch_id = run_batch_id
            
            if self._shutdown_requested:
                result.interrupted = True
                return result
            
            # Use MasterAgent for orchestration
            master = MasterAgent.from_cli(mode=self.mode)
            
            # Check for dry run
            if self.dry_run:
                self.logger.info("DRY RUN MODE — will skip apply phase")
                os.environ["DRY_RUN"] = "true"
            
            # Run the pipeline via MasterAgent
            master_result = master.run()
            
            # Extract results
            result.success = master_result.get("success", False)
            result.jobs_discovered = master_result.get("jobs_discovered", 0)
            result.jobs_scored = master_result.get("jobs_scored", 0)
            result.jobs_auto_applied = master_result.get("jobs_auto_applied", 0)
            result.jobs_manual_queued = master_result.get("jobs_manual_queued", 0)
            result.jobs_failed = master_result.get("jobs_failed", 0)
            result.total_cost_usd = master_result.get("total_cost_usd", 0.0)
            result.error = master_result.get("error")
            
        except Exception as exc:
            self.logger.critical("Pipeline failed: %s", exc, exc_info=True)
            result.success = False
            result.error = str(exc)
        
        finally:
            # Calculate duration
            end_time = datetime.now(timezone.utc)
            result.duration_minutes = (end_time - start_time).total_seconds() / 60.0
            
            # Close run session
            self._close_run_session(result)
            
            # Post Notion report if interrupted
            if result.interrupted:
                try:
                    from integrations.notion import NotionClient
                    client = NotionClient()
                    await client.post_alert(
                        message=f"Pipeline interrupted (SIGTERM). Run: {result.run_batch_id}",
                        level="WARNING"
                    )
                except Exception:
                    pass
        
        return result


# ---------------------------------------------------------------------------
# SIGTERM HANDLER
# ---------------------------------------------------------------------------

_pipeline_runner: Optional[PipelineRunner] = None
_shutdown_in_progress = False


def _sigterm_handler(signum: int, frame: Any) -> None:
    """Handle SIGTERM for graceful shutdown."""
    global _shutdown_in_progress
    
    if _shutdown_in_progress:
        logger.warning("Force shutdown requested — exiting immediately")
        sys.exit(1)
    
    _shutdown_in_progress = True
    logger.warning("SIGTERM received — initiating graceful shutdown")
    
    if _pipeline_runner:
        _pipeline_runner.request_shutdown()
    
    # Give 10 seconds for graceful shutdown before Docker sends SIGKILL
    # The pipeline should complete current Postgres write within this time


def setup_signal_handlers() -> None:
    """Setup SIGTERM and SIGINT handlers for graceful shutdown."""
    signal.signal(signal.SIGTERM, _sigterm_handler)
    signal.signal(signal.SIGINT, _sigterm_handler)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse and return CLI arguments for the pipeline entry point.

    Returns:
        Parsed :class:`argparse.Namespace` containing all recognised flags.
    """
    parser = argparse.ArgumentParser(
        description="AI Job Application Agent — Autonomous Pipeline"
    )
    parser.add_argument(
        "--mode",
        choices=["full", "scrape-only", "scrape_only", "apply-only", "apply_only", "dry-run", "dry_run"],
        default="full",
        help="Pipeline execution mode (default: full)",
    )
    parser.add_argument(
        "--session-id",
        type=str,
        default=None,
        help="Override session UUID (default: auto-generated)",
    )
    parser.add_argument(
        "--budget-override",
        type=float,
        default=None,
        help="Override per-run budget cap in USD",
    )
    parser.add_argument(
        "--run-index",
        type=int,
        default=0,
        help="Override run index 1/2/3 (0=auto-detect from weekday)",
    )
    parser.add_argument(
        "--budget-check",
        action="store_true",
        help="Print current budget status and exit",
    )
    parser.add_argument(
        "--health-check",
        action="store_true",
        help="Run boot health check and exit without running pipeline",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Fill all forms but skip final submit. No real applications sent.",
    )
    parser.add_argument(
        "--skip-boot-check",
        action="store_true",
        default=False,
        help="Skip Docker service health checks (for local dev)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# SHORTCUTS
# ---------------------------------------------------------------------------


def print_budget_status() -> None:
    """Print the current run cost summary to stdout and exit 0.

    Calls :func:`~tools.budget_tools.get_cost_summary` with a dummy
    ``run_batch_id`` so it can be invoked outside of an active run.
    """
    from tools.budget_tools import get_cost_summary  # inline — thin entry point

    raw: str = get_cost_summary(run_batch_id="budget_check")
    try:
        summary: dict = json.loads(raw)
        print("\n=== BUDGET STATUS ===")
        print(f"  Run xAI cost      : ${float(summary.get('run_xai_cost', 0)):.4f}")
        print(f"  Run Perplexity    : ${float(summary.get('run_perplexity_cost', 0)):.4f}")
        print(f"  Run total         : ${float(summary.get('run_total_cost', 0)):.4f}")
        print(f"  xAI cap           : ${float(summary.get('xai_cap', budget_config.xai_cost_cap_per_run)):.4f}")
        print(f"  xAI cap remaining : ${float(summary.get('xai_cap_remaining', 0)):.4f}")
        print(f"  Monthly budget    : ${float(summary.get('monthly_budget', budget_config.total_monthly_budget)):.2f}")
        print(f"  Monthly cap       : $9.50")
        print("=====================\n")
    except (json.JSONDecodeError, TypeError, ValueError):
        print(raw)
    sys.exit(0)


async def run_health_check_async() -> bool:
    """Run async health check of all Docker services."""
    bootstrapper = SessionBootstrapper()
    return await bootstrapper.boot()


def run_health_check() -> bool:
    """Run the health check and exit with the result code.

    Exits ``0`` on pass, ``1`` on failure — never returns to the caller.
    """
    passed = asyncio.run(run_health_check_async())
    if passed:
        logger.info("Health check PASSED — all systems go")
        print("✅ HEALTH CHECK PASSED")
        sys.exit(0)
    else:
        logger.error("Health check FAILED")
        print("❌ HEALTH CHECK FAILED")
        sys.exit(1)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------


async def async_main(args: argparse.Namespace) -> int:
    """Async main function for the pipeline.
    
    Args:
        args: Parsed CLI arguments.
        
    Returns:
        Exit code (0 success, 1 failure, 130 interrupt).
    """
    global _pipeline_runner
    from config.config_loader import config_loader
    
    # ------------------------------------------------------------------
    # Startup banner
    # ------------------------------------------------------------------
    logger.info("=" * 70)
    logger.info("AI JOB APPLICATION AGENT — PIPELINE START")
    logger.info(
        "Mode: %s | Time: %sZ",
        args.mode.upper(),
        datetime.now(timezone.utc).isoformat(),
    )
    logger.info(
        "Target: %d jobs | Budget: $%.2f/month",
        config_loader.get_run_config().get("jobs_per_run_target", 100),
        config_loader.get_budget_settings().get("total_monthly_budget_usd", 10.0),
    )
    logger.info(
        "DRY RUN: %s | AUTO APPLY: %s",
        args.dry_run or config_loader.get_apply_settings().get("dry_run", False),
        config_loader.get_apply_settings().get("auto_apply_enabled", False),
    )
    logger.info("=" * 70)

    # ------------------------------------------------------------------
    # Critical env check — fail fast before any agent is instantiated.
    # ------------------------------------------------------------------
    missing: list[str] = []
    if not api_config.groq_api_key:
        missing.append("GROQ_API_KEY")
    if not api_config.xai_api_key:
        missing.append("XAI_API_KEY")
    if not db_config.local_postgres_url and not db_config.supabase_url:
        missing.append("LOCAL_POSTGRES_URL or SUPABASE_URL")

    try:
        from config.config_loader import config_loader as _cl
        _cl.get_run_config()
    except Exception as _cfg_exc:
        logger.critical("Config JSON load failed: %s", _cfg_exc)
        missing.append("config/platform_settings.json (unreadable)")

    if missing:
        logger.critical("Missing required env vars: %s", missing)
        logger.critical("Cannot start pipeline. Fill ~/java.env and retry.")
        print(f"❌ MISSING ENV VARS: {missing}")
        return 1

    # ------------------------------------------------------------------
    # Step 1: Session Bootstrapper — verify all Docker services
    # ------------------------------------------------------------------
    if not args.skip_boot_check:
        bootstrapper = SessionBootstrapper()
        boot_ok = await bootstrapper.boot()
        if not boot_ok:
            logger.critical("Session boot failed — services unhealthy")
            print("❌ BOOT FAILED — Docker services unhealthy")
            return 1
    else:
        logger.warning("Skipping boot check (--skip-boot-check flag)")

    # ------------------------------------------------------------------
    # Step 2: Budget Enforcer — check $9.50 monthly cap
    # ------------------------------------------------------------------
    enforcer = BudgetEnforcer()
    budget_ok = await enforcer.enforce()
    if not budget_ok:
        logger.critical("Budget cap exceeded — pipeline aborted")
        print("❌ BUDGET CAP EXCEEDED — pipeline aborted")
        return 1

    # ------------------------------------------------------------------
    # Step 3: Setup signal handlers for graceful shutdown
    # ------------------------------------------------------------------
    setup_signal_handlers()

    # ------------------------------------------------------------------
    # Step 4: Initialize and run pipeline
    # ------------------------------------------------------------------
    # Normalize mode
    mode = args.mode.replace("-", "_")
    dry_run = args.dry_run or mode == "dry_run"
    
    if dry_run:
        os.environ["DRY_RUN"] = "true"
        logger.info("DRY RUN MODE ACTIVE — no real submissions")

    runner = PipelineRunner(
        mode=mode,
        session_id=args.session_id,
        budget_override=args.budget_override,
        dry_run=dry_run,
    )
    _pipeline_runner = runner  # For SIGTERM handler access
    
    try:
        result = await runner.run_full()
    except KeyboardInterrupt:
        logger.warning("Pipeline interrupted by user (KeyboardInterrupt)")
        print("\n⚠️  Pipeline interrupted. Partial results may exist in Postgres.")
        return 130
    except Exception as exc:
        logger.critical("Unhandled exception in main(): %s", exc, exc_info=True)
        print(f"❌ PIPELINE CRASHED: {exc}")
        return 1

    # ------------------------------------------------------------------
    # Final report
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE — FINAL REPORT")
    print("=" * 70)
    
    result_dict = {
        "success": result.success,
        "runbatchid": result.run_batch_id,
        "mode": args.mode,
        "jobsdiscovered": result.jobs_discovered,
        "jobsscored": result.jobs_scored,
        "jobsautoapplied": result.jobs_auto_applied,
        "jobsmanualqueued": result.jobs_manual_queued,
        "jobsfailed": result.jobs_failed,
        "totalcostusd": result.total_cost_usd,
        "durationminutes": result.duration_minutes,
        "interrupted": result.interrupted,
        "error": result.error,
    }
    print(json.dumps(result_dict, indent=2, default=str))
    print("=" * 70)

    if result.success:
        logger.info(
            "PIPELINE SUCCESS | applied=%d | queued=%d | cost=$%.4f | duration=%.1fmin",
            result.jobs_auto_applied,
            result.jobs_manual_queued,
            result.total_cost_usd,
            result.duration_minutes,
        )
        print(
            f"\n✅ SUCCESS — Applied: {result.jobs_auto_applied} | "
            f"Queued: {result.jobs_manual_queued} | "
            f"Cost: ${result.total_cost_usd:.4f}"
        )
        
        # Post success report to Notion
        try:
            from integrations.notion import NotionClient
            client = NotionClient()
            await client.post_run_report_simple(
                run_batch_id=result.run_batch_id,
                jobs_discovered=result.jobs_discovered,
                jobs_applied=result.jobs_auto_applied,
                jobs_queued=result.jobs_manual_queued,
                jobs_failed=result.jobs_failed,
                cost_usd=result.total_cost_usd,
                duration_mins=result.duration_minutes,
            )
        except Exception as exc:
            logger.warning("Failed to post Notion report: %s", exc)
        
        return 0
    else:
        logger.error(
            "PIPELINE FAILED | reason=%s", result.error or "unknown"
        )
        print(f"\n❌ PIPELINE FAILED — {result.error or 'unknown'}")
        return 1


def main() -> int:
    """Run the full AI Job Application Agent pipeline.

    Parses CLI arguments, validates critical environment variables,
    instantiates :class:`~agents.master_agent.MasterAgent`, executes the
    pipeline, prints the final JSON report to stdout, and returns the
    correct POSIX exit code.

    Returns:
        ``0`` on pipeline success, ``1`` on failure or missing env vars,
        ``130`` on :exc:`KeyboardInterrupt` (Unix convention).
    """
    args: argparse.Namespace = parse_args()

    # Budget check shortcut — print and exit, never starts pipeline.
    if args.budget_check:
        print_budget_status()
        return 0  # unreachable; print_budget_status() calls sys.exit(0)

    # Health check shortcut — boots system, prints result, exits.
    if args.health_check:
        run_health_check()
        return 0  # unreachable; run_health_check() calls sys.exit()

    # Run async main
    return asyncio.run(async_main(args))


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sys.exit(main())
