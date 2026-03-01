"""AI Job Application Agent — single CLI entry point.

Invoked by GitHub Actions cron (``--mode full``), manual developer runs
(``--mode dry_run``), and Docker CMD.  Boots logging, validates env,
instantiates :class:`~agents.master_agent.MasterAgent`, runs the full
pipeline, prints the final JSON report, and exits with the correct
POSIX code.

No agent logic, no tool calls, and no direct DB access live here.
This module is intentionally thin.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# MODULE-LEVEL SETUP  (runs at import time)
# ---------------------------------------------------------------------------

# ~/narad.env is the single source of truth for all secrets/config.
# override=False so values already in the process environment win.
load_dotenv(dotenv_path=os.path.expanduser("~/narad.env"), override=False)

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

__all__ = ["main"]


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
        choices=["full", "scrape_only", "analyse_only", "apply_only", "dry_run"],
        default="full",
        help="Pipeline execution mode (default: full)",
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

    raw: str = get_cost_summary("budget_check")
    try:
        summary: dict = json.loads(raw)
        print("\n=== BUDGET STATUS ===")
        print(f"  Run xAI cost      : ${float(summary.get('run_xai_cost', 0)):.4f}")
        print(f"  Run Perplexity    : ${float(summary.get('run_perplexity_cost', 0)):.4f}")
        print(f"  Run total         : ${float(summary.get('run_total_cost', 0)):.4f}")
        print(f"  xAI cap           : ${float(summary.get('xai_cap', budget_config.xai_cost_cap_per_run)):.4f}")
        print(f"  xAI cap remaining : ${float(summary.get('xai_cap_remaining', 0)):.4f}")
        print(f"  Monthly budget    : ${float(summary.get('monthly_budget', budget_config.total_monthly_budget)):.2f}")
        print("=====================\n")
    except (json.JSONDecodeError, TypeError, ValueError):
        print(raw)
    sys.exit(0)


def run_health_check() -> bool:
    """Run the MasterAgent boot health check and exit with the result code.

    Instantiates :class:`~agents.master_agent.MasterAgent` in dry-run mode
    and calls ``_boot_system()`` directly.  Exits ``0`` on pass, ``1`` on
    failure — never returns to the caller.

    Returns:
        This function always calls :func:`sys.exit` and never returns.
    """
    master: MasterAgent = MasterAgent(mode="dry_run")
    passed: bool = master._boot_system()
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

    # ------------------------------------------------------------------
    # Startup banner
    # ------------------------------------------------------------------
    logger.info("=" * 70)
    logger.info("AI JOB APPLICATION AGENT — PIPELINE START")
    logger.info(
        "Mode: %s | Time: %sZ",
        args.mode.upper(),
        datetime.utcnow().isoformat(),
    )
    logger.info(
        "Target: %d jobs | Budget: $%.2f/month",
        run_config.jobs_per_run_target,
        budget_config.total_monthly_budget,
    )
    logger.info(
        "DRY RUN: %s | AUTO APPLY: %s",
        run_config.dry_run,
        run_config.auto_apply_enabled,
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

    if missing:
        logger.critical("Missing required env vars: %s", missing)
        logger.critical("Cannot start pipeline. Fill ~/narad.env and retry.")
        print(f"❌ MISSING ENV VARS: {missing}")
        return 1

    # ------------------------------------------------------------------
    # Instantiate and run pipeline
    # ------------------------------------------------------------------
    try:
        agent: MasterAgent = MasterAgent.from_cli(mode=args.mode)
        result: dict = agent.run()
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
    print(json.dumps(result, indent=2, default=str))
    print("=" * 70)

    if result.get("success"):
        logger.info(
            "PIPELINE SUCCESS | applied=%d | queued=%d | cost=$%.4f | duration=%.1fmin",
            result.get("jobs_auto_applied", 0),
            result.get("jobs_manual_queued", 0),
            result.get("total_cost_usd", 0),
            result.get("duration_minutes", 0),
        )
        print(
            f"\n✅ SUCCESS — Applied: {result.get('jobs_auto_applied', 0)} | "
            f"Queued: {result.get('jobs_manual_queued', 0)} | "
            f"Cost: ${result.get('total_cost_usd', 0):.4f}"
        )
        return 0
    else:
        logger.error(
            "PIPELINE FAILED | reason=%s", result.get("error", "unknown")
        )
        print(f"\n❌ PIPELINE FAILED — {result.get('error', 'unknown')}")
        return 1


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sys.exit(main())
