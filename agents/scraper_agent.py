"""
Scraper Agent for AI Job Application Agent.

This CrewAI agent orchestrates job discovery across all configured platforms,
normalizes job data, removes duplicates, and persists results to Postgres.
All platform configuration is read from config/platform_settings.json.
"""

import os
import json
import logging
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

import httpx
from crewai import Agent, Task, Crew, Process
import agentops
from agentops import agent, operation, track_agent

from integrations.llm_interface import LLMInterface
from tools.scraper_tools import (
    run_jobspy_scrape,
    run_rest_api_scrape,
    run_serpapi_scrape,
    run_safety_net_scrape,
    normalise_and_dedup,
    get_scrape_summary,
)
from tools.postgres_tools import (
    log_event,
    _log_event,
    create_run_batch,
    update_run_batch_stats,
    get_platform_config,
    _fetch_user_config,
)
from tools.budget_tools import record_llm_cost, check_monthly_budget, check_xai_run_cap

# Module-level logger
logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("LOG_LEVEL", "INFO"))

# Scraper service connection — set in docker-compose environment block
SCRAPER_SERVICE_URL: str = os.getenv("SCRAPER_SERVICE_URL", "http://ai_playwright_scraper:8001")
SCRAPER_API_KEY: str = os.getenv("SCRAPER_SERVICE_API_KEY", "")

# Module-level platform config cache
_platform_config: Optional[Dict[str, Any]] = None

__all__ = ["ScraperAgent"]


from config.config_loader import config_loader as _cfg

def _load_platform_config() -> Dict[str, Any]:
    """
    Load platform configuration from config_loader.

    Returns:
        Dictionary containing platform configuration.
    """
    global _platform_config

    if _platform_config is not None:
        return _platform_config

    _platform_config = _cfg.settings.get("platform_settings", {})

    logger.info("Loaded platform configuration from config_loader")
    return _platform_config


@track_agent(name="ScraperAgent")
class ScraperAgent:
    """
    CrewAI Scraper Agent for job discovery.

    Orchestrates scraping across all configured platforms, normalizes data,
    removes duplicates, and persists results to Postgres.
    """

    def __init__(self, pipeline_run_id: str) -> None:
        """
        Initialize the Scraper Agent.

        Args:
            pipeline_run_id: UUID of the current run batch.
        """
        self.pipeline_run_id = pipeline_run_id
        self.llm_interface = LLMInterface()
        self.llm = self.llm_interface.get_llm("SCRAPER_AGENT")
        self.platform_config = _load_platform_config()
        self.run_state: Dict[str, Any] = {}
        self.logger = logging.getLogger(self.__class__.__name__)

        self.logger.info(
            f"ScraperAgent initialized for run batch: {pipeline_run_id}"
        )

    def _build_agent(self) -> Agent:
        """
        Build the CrewAI Agent with role, goal, and tools.

        Returns:
            Configured CrewAI Agent instance.
        """
        return Agent(
            role="Senior Job Discovery Specialist",
            goal=(
                "Discover, normalise, and deduplicate the maximum number of relevant "
                "remote AI, ML, Data Science, and Automation engineering jobs across "
                "all configured platforms in a single run session"
            ),
            backstory=(
                "You are an expert at systematically scraping job platforms, normalising "
                "inconsistent job data into clean structured records, and ensuring zero "
                "duplicate entries make it into the pipeline. You know exactly which "
                "platforms to hit, in what order, and when to trigger safety-net sources."
            ),
            llm=self.llm,
            tools=[
                run_jobspy_scrape,
                run_rest_api_scrape,
                run_serpapi_scrape,
                run_safety_net_scrape,
                normalise_and_dedup,
                get_scrape_summary,
            ],
            verbose=True,
            max_iter=15,
            memory=False,
        )

    def _build_task(self, agent: Agent) -> Task:
        """
        Build the CrewAI Task with detailed instructions.

        Args:
            agent: The agent that will execute this task.

        Returns:
            Configured CrewAI Task instance.
        """
        # Get search query from environment or use default
        search_query = ""

        # Get minimum jobs target
        min_jobs = int(os.getenv("JOBS_PER_RUN_TARGET", "100"))

        description = f"""
Execute a comprehensive job discovery run across all configured platforms.

STEP 1: JobSpy Scrape (LinkedIn + Indeed)
- Call run_jobspy_scrape with:
  - pipeline_run_id: {self.pipeline_run_id}
  The tool reads all search queries from config/user_profile.json automatically.

STEP 2: REST API Scrape (RemoteOK + Himalayas)
- Call run_rest_api_scrape with:
  - pipeline_run_id: {self.pipeline_run_id}
  - platforms: "remoteok,himalayas"

STEP 3: SerpAPI Scrape (Google Jobs)
- Call run_serpapi_scrape with:
  - pipeline_run_id: {self.pipeline_run_id}
  - search_query: "{search_query}"
  - location: "Remote"
  - results_wanted: 25

STEP 4: Playwright Scrape (Delegated to playwright-scraper container)
Playwright scraping is handled by the dedicated ai_playwright_scraper container
via HTTP. This step is executed automatically — do NOT call any tool for this.
The playwright results will be merged into the job pool before normalization.

STEP 5: Check Total Job Count
- Call get_scrape_summary with pipeline_run_id: {self.pipeline_run_id}
- Parse the total_jobs count from the response

STEP 6: Safety Net Activation (Conditional)
- If total_jobs < {min_jobs}:
  - Call run_safety_net_scrape with:
    - pipeline_run_id: {self.pipeline_run_id}
    - current_job_count: [total from step 5]

STEP 7: Normalize and Deduplicate
- Call normalise_and_dedup with pipeline_run_id: {self.pipeline_run_id}
- This removes duplicate URLs and cleans the dataset

STEP 8: Final Summary
- Call get_scrape_summary with pipeline_run_id: {self.pipeline_run_id}
- Return this final summary as your output

IMPORTANT:
- Execute steps sequentially
- Log any platform failures but continue with remaining platforms
- Always complete normalization and deduplication
- Return the final summary JSON from step 8
"""

        return Task(
            description=description,
            expected_output=(
                "JSON object with total jobs discovered by platform, duplicates removed, "
                "and whether minimum threshold was met"
            ),
            agent=agent,
        )

    def _run_playwright_via_http(self) -> Dict[str, Any]:
        """Delegate playwright scraping to the ai_playwright_scraper container.

        Fetches platform and query config from Postgres via ``_fetch_user_config``
        and sends a single HTTP POST to ``{SCRAPER_SERVICE_URL}/scrape``.  Retries
        up to 3 times with exponential back-off on connection errors.

        Returns:
            Dict with keys: pipeline_run_id, jobs_found, jobs (list[dict]),
            platforms_scraped, duration_seconds on success; or
            {success, reason, total_jobs, jobs, aborted} on failure.
        """
        playwright_platforms: List[str] = ["wellfound", "arcdev"]
        search_queries: List[str] = []
        max_jobs: int = int(os.getenv("JOBS_PER_RUN_TARGET", "150"))

        try:
            cfg: Dict[str, Any] = _fetch_user_config()
            platform_settings: Dict[str, Any] = cfg.get("platform_settings", {})
            custom_platforms: List[str] = platform_settings.get("playwright_platforms", [])
            if custom_platforms:
                playwright_platforms = custom_platforms

            user_profile_path = "config_loader.get_job_preferences()"
            user_profile = {"job_preferences": _cfg.get_job_preferences()}
            custom_queries: List[str] = (
                user_profile.get("job_preferences", {}).get("search_queries", [])
            )
            if custom_queries:
                search_queries = custom_queries
            target: int = int(
                cfg.get("user_settings", {}).get(
                    "jobs_per_run_target",
                    os.getenv("JOBS_PER_RUN_TARGET", "150"),
                )
            )
            max_jobs = target
        except Exception as cfg_exc:  # noqa: BLE001
            self.logger.warning(
                "_run_playwright_via_http: config fetch failed (%s), using env defaults",
                cfg_exc,
            )

        for attempt in range(3):
            try:
                with httpx.Client(timeout=60.0) as client:
                    resp = client.post(
                        f"{SCRAPER_SERVICE_URL}/scrape",
                        headers={
                            "X-API-Key": SCRAPER_API_KEY,
                            "Content-Type": "application/json",
                        },
                        json={
                            "pipeline_run_id": self.pipeline_run_id,
                            "search_queries": search_queries,
                            "platforms": playwright_platforms,
                            "max_jobs": max_jobs,
                        },
                    )
                if resp.status_code == 200:
                    data: Dict[str, Any] = resp.json()
                    self.logger.info(
                        "_run_playwright_via_http: %d jobs from playwright-scraper in %.2fs",
                        data.get("jobs_found", 0),
                        data.get("duration_seconds", 0.0),
                    )
                    return data
                self.logger.error(
                    "_run_playwright_via_http: scraper service returned HTTP %d (attempt %d/3)",
                    resp.status_code,
                    attempt + 1,
                )
            except httpx.RequestError as exc:
                self.logger.error(
                    "_run_playwright_via_http: connection error (attempt %d/3): %s",
                    attempt + 1,
                    exc,
                )
            if attempt < 2:
                time.sleep(2 ** attempt)

        self.logger.error("scraper_service_unreachable: all 3 attempts failed")
        return {
            "success": False,
            "reason": "scraper_service_unreachable",
            "total_jobs": 0,
            "jobs": [],
            "aborted": True,
        }

    def _fallback_scrape_sequence(self) -> Dict[str, Any]:
        """
        Hardcoded sequential fallback if LLM is unavailable or out of budget.
        Mimics the exact steps given in the CrewAI task prompt.
        """
        self.logger.info("Initiating hardcoded fallback scrape sequence...")
        
        search_query = ""
        min_jobs = int(os.getenv("JOBS_PER_RUN_TARGET", "100"))

        try:
            self.logger.info("Fallback Step 1: JobSpy Scrape")
            run_jobspy_scrape.run(
                pipeline_run_id=self.pipeline_run_id,
            )
        except Exception as e:
            if isinstance(e, (KeyboardInterrupt, SystemExit)):
                raise
            self.logger.error(f"Fallback JobSpy failed: {e}")

        try:
            self.logger.info("Fallback Step 2: REST API Scrape")
            run_rest_api_scrape.run(
                pipeline_run_id=self.pipeline_run_id,
                platforms="remoteok,himalayas"
            )
        except Exception as e:
            self.logger.error(f"Fallback REST API failed: {e}")

        try:
            self.logger.info("Fallback Step 3: SerpAPI Scrape")
            run_serpapi_scrape.run(
                pipeline_run_id=self.pipeline_run_id,
                query=search_query,
                location="Remote",
                results_wanted=25
            )
        except Exception as e:
            self.logger.error(f"Fallback SerpAPI failed: {e}")

        # Playwright scraping delegated to the ai_playwright_scraper container via HTTP
        self.logger.info("Fallback Step 4: Playwright Scrape — delegating to playwright-scraper service")
        playwright_result = self._run_playwright_via_http()
        if not playwright_result.get("aborted"):
            self.run_state["scraped_jobs"] = playwright_result.get("jobs", [])
        else:
            self.logger.error(
                "_fallback_scrape_sequence: playwright-scraper unreachable — continuing without playwright jobs"
            )

        try:
            self.logger.info("Fallback Step 5: Check Total Job Count")
            summary_str = get_scrape_summary.run(pipeline_run_id=self.pipeline_run_id)
            summary = json.loads(summary_str)
            total_jobs = summary.get("total_jobs", 0)

            if total_jobs < min_jobs:
                self.logger.info("Fallback Step 6: Safety Net Activation")
                run_safety_net_scrape.run(
                    pipeline_run_id=self.pipeline_run_id,
                    current_job_count=total_jobs
                )
        except Exception as e:
            self.logger.error(f"Fallback Summary/Safety Net failed: {e}")

        try:
            self.logger.info("Fallback Step 7: Normalize and Deduplicate")
            normalise_and_dedup.run(pipeline_run_id=self.pipeline_run_id)
        except Exception as e:
            self.logger.error(f"Fallback Deduplication failed: {e}")

        try:
            self.logger.info("Fallback Step 8: Final Summary")
            final_summary_str = get_scrape_summary.run(pipeline_run_id=self.pipeline_run_id)
            final_summary = json.loads(final_summary_str)
        except Exception as e:
            self.logger.error(f"Fallback Final Summary failed: {e}")
            final_summary = {"total_jobs": 0, "by_platform": {}, "error": str(e)}

        return final_summary

    @operation
    def run(self) -> Dict[str, Any]:
        """
        Execute the scraper agent run.

        This is the main entry point called by the Master Agent.

        Returns:
            Dictionary with run results including success status, total jobs,
            platform breakdown, and any errors.
        """
        try:
            # Log run start
            _log_event(
                pipeline_run_id=self.pipeline_run_id,
                level="INFO",
                event_type="scraper_run_start",
                message=f"ScraperAgent run started for batch {self.pipeline_run_id}",
            )

            self.logger.info(f"Starting scraper run for batch: {self.pipeline_run_id}")

            # Check monthly budget + xAI run cap before proceeding
            budget_check = check_monthly_budget(pipeline_run_id=self.pipeline_run_id)
            budget_result = json.loads(budget_check)

            xai_check = check_xai_run_cap(pipeline_run_id=self.pipeline_run_id)
            xai_result = json.loads(xai_check)

            budget_abort = budget_result.get("abort", False)
            xai_abort = xai_result.get("abort", False)

            used_fallback = False

            if budget_abort or xai_abort:
                abort_reason = (
                    budget_result.get("reason", "monthly_budget")
                    if budget_abort
                    else xai_result.get("reason", "xai_run_cap")
                )
                self.logger.warning(
                    f"Budget cap hit ({abort_reason}). "
                    "Bypassing LLM orchestration and using fallback scrape sequence."
                )
                _log_event(
                    pipeline_run_id=self.pipeline_run_id,
                    level="WARNING",
                    event_type="scraper_budget_cap",
                    message=f"Budget cap hit {abort_reason}. Using hardcoded scrape fallback.",
                )
                result_data = self._fallback_scrape_sequence()
                used_fallback = True
            else:
                # Build agent and task
                agent = self._build_agent()
                task = self._build_task(agent)

                # Create and execute crew
                crew = Crew(
                    agents=[agent],
                    tasks=[task],
                    process=Process.sequential,
                    verbose=True,
                )

                self.logger.info("Executing CrewAI crew...")
                try:
                    result = crew.kickoff()
                except Exception as llm_e:
                    self.logger.warning(
                        f"LLM execution failed: {llm_e}. "
                        "Bypassing LLM orchestration and using fallback scrape sequence."
                    )
                    _log_event(
                        pipeline_run_id=self.pipeline_run_id,
                        level="WARNING",
                        event_type="scraper_llm_failure",
                        message=f"LLM failed {llm_e}. Using hardcoded scrape fallback.",
                    )
                    result_data = self._fallback_scrape_sequence()
                    used_fallback = True

            if not used_fallback:
                # Parse result - CrewAI may return string or dict
                if isinstance(result, str):
                    try:
                        result_data = json.loads(result)
                    except json.JSONDecodeError:
                        self.logger.warning(
                            f"Could not parse crew result as JSON: {result}"
                        )
                        result_data = {"raw_output": result}
                else:
                    result_data = result if isinstance(result, dict) else {"result": str(result)}

            # Extract metrics for database update
            total_jobs = result_data.get("total_jobs", 0)
            by_platform = result_data.get("by_platform", {})
            
            # Calculate jobs by category
            jobs_applied = 0  # Will be set by Apply Agent later
            jobs_queued = 0  # Will be set by Analyser Agent later

            # Update run batch stats
            update_run_batch_stats.run(
                pipeline_run_id=self.pipeline_run_id,
                jobs_found=total_jobs,
                jobs_applied=jobs_applied,
                jobs_queued=jobs_queued,
            )

            # Log completion
            _log_event(
                pipeline_run_id=self.pipeline_run_id,
                level="INFO",
                event_type="scraper_run_complete",
                message=f"Scraper complete — {total_jobs} jobs discovered",
            )

            return {
                "success": True,
                "pipeline_run_id": self.pipeline_run_id,
                "total_jobs": total_jobs,
                "by_platform": by_platform,
                "safety_net_triggered": result_data.get("safety_net_triggered", False),
                "duplicates_removed": result_data.get("duplicates_removed", 0),
                "minimum_met": result_data.get("minimum_met", False),
            }

        except Exception as e:
            self.logger.error(f"Scraper run failed with exception: {e}", exc_info=True)

            # Log critical error
            _log_event(
                pipeline_run_id=self.pipeline_run_id,
                level="ERROR",
                event_type="scraper_phase_exception",
                message=f"Scraper run phase exception — detail: {str(e)}",
            )

    def get_enabled_platforms(self, category: str = "primary") -> List[str]:
        """
        Get list of enabled platforms for a given category.

        Args:
            category: Platform category (primary, safety_net, supplementary, phase2).

        Returns:
            List of enabled platform names.
        """
        platforms = self.platform_config.get("platforms", {})
        enabled = []

        for platform_name, config in platforms.items():
            if (
                config.get("category") == category
                and config.get("enabled", False)
            ):
                enabled.append(platform_name)

        return enabled

    def get_platform_rate_limit(self, platform: str) -> float:
        """
        Get rate limit for a specific platform.

        Args:
            platform: Platform name.

        Returns:
            Rate limit in seconds per request (default: 3.0).
        """
        platforms = self.platform_config.get("platforms", {})
        platform_config = platforms.get(platform, {})
        return float(platform_config.get("rate_limit_per_request_seconds", 3.0))
