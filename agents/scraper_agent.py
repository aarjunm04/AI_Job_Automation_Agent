"""
Scraper Agent for AI Job Application Agent.

This CrewAI agent orchestrates job discovery across all configured platforms,
normalizes job data, removes duplicates, and persists results to Postgres.
All platform configuration is read from config/platforms.json.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

from crewai import Agent, Task, Crew, Process
import agentops
from agentops.sdk.decorators import agent, operation

from integrations.llm_interface import LLMInterface
from tools.scraper_tools import (
    run_jobspy_scrape,
    run_rest_api_scrape,
    run_playwright_scrape,
    run_serpapi_scrape,
    run_safety_net_scrape,
    normalise_and_dedup,
    get_scrape_summary,
)
from tools.postgres_tools import (
    log_event,
    create_run_batch,
    update_run_batch_stats,
    get_platform_config,
)
from tools.budget_tools import record_llm_cost, check_monthly_budget

# Module-level logger
logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("LOG_LEVEL", "INFO"))

# Module-level platform config cache
_platform_config: Optional[Dict[str, Any]] = None

__all__ = ["ScraperAgent"]


def _load_platform_config() -> Dict[str, Any]:
    """
    Load platform configuration from config/platforms.json.

    Returns:
        Dictionary containing platform configuration.

    Raises:
        FileNotFoundError: If config/platforms.json does not exist.
    """
    global _platform_config

    if _platform_config is not None:
        return _platform_config

    config_path = Path(__file__).parent.parent / "config" / "platforms.json"

    if not config_path.exists():
        raise FileNotFoundError(
            f"Platform configuration not found at {config_path}. "
            "Create config/platforms.json with platform definitions."
        )

    with open(config_path, "r", encoding="utf-8") as f:
        _platform_config = json.load(f)

    logger.info(f"Loaded platform configuration from {config_path}")
    return _platform_config


@agent
class ScraperAgent:
    """
    CrewAI Scraper Agent for job discovery.

    Orchestrates scraping across all configured platforms, normalizes data,
    removes duplicates, and persists results to Postgres.
    """

    def __init__(self, run_batch_id: str) -> None:
        """
        Initialize the Scraper Agent.

        Args:
            run_batch_id: UUID of the current run batch.
        """
        self.run_batch_id = run_batch_id
        self.llm_interface = LLMInterface()
        self.llm = self.llm_interface.get_llm("SCRAPER_AGENT")
        self.platform_config = _load_platform_config()
        self.logger = logging.getLogger(self.__class__.__name__)

        self.logger.info(
            f"ScraperAgent initialized for run batch: {run_batch_id}"
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
                run_playwright_scrape,
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
        search_query = os.getenv(
            "SEARCH_QUERY", "AI ML Data Science Automation Engineer"
        )

        # Get minimum jobs target
        min_jobs = int(os.getenv("JOBS_PER_RUN_MINIMUM", "100"))

        description = f"""
Execute a comprehensive job discovery run across all configured platforms.

STEP 1: JobSpy Scrape (LinkedIn + Indeed)
- Call run_jobspy_scrape with:
  - run_batch_id: {self.run_batch_id}
  - search_query: "{search_query}"
  - location: "Remote"
  - results_wanted: 50

STEP 2: REST API Scrape (RemoteOK + Himalayas)
- Call run_rest_api_scrape with:
  - run_batch_id: {self.run_batch_id}
  - platforms: "remoteok,himalayas"

STEP 3: SerpAPI Scrape (Google Jobs)
- Call run_serpapi_scrape with:
  - run_batch_id: {self.run_batch_id}
  - search_query: "{search_query}"
  - location: "Remote"
  - results_wanted: 25

STEP 4: Playwright Scrape (Primary Platforms)
Run these platforms ONE AT A TIME in this exact order:
1. wellfound
2. weworkremotely
3. ycombinator
4. arc
5. turing
6. crossover

For each platform, call run_playwright_scrape with:
- run_batch_id: {self.run_batch_id}
- platform: [platform_name]
- max_jobs: 30

STEP 5: Check Total Job Count
- Call get_scrape_summary with run_batch_id: {self.run_batch_id}
- Parse the total_jobs count from the response

STEP 6: Safety Net Activation (Conditional)
- If total_jobs < {min_jobs}:
  - Call run_safety_net_scrape with:
    - run_batch_id: {self.run_batch_id}
    - current_job_count: [total from step 5]

STEP 7: Normalize and Deduplicate
- Call normalise_and_dedup with run_batch_id: {self.run_batch_id}
- This removes duplicate URLs and cleans the dataset

STEP 8: Final Summary
- Call get_scrape_summary with run_batch_id: {self.run_batch_id}
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
            log_event(
                run_batch_id=self.run_batch_id,
                level="INFO",
                event_type="scraper_run_start",
                message="Scraper Agent starting job discovery",
            )

            self.logger.info(f"Starting scraper run for batch: {self.run_batch_id}")

            # Check monthly budget before proceeding
            budget_check = check_monthly_budget(run_batch_id=self.run_batch_id)
            budget_result = json.loads(budget_check)

            if budget_result.get("abort", False):
                self.logger.critical(
                    f"Monthly budget exceeded: {budget_result.get('reason')}"
                )
                log_event(
                    run_batch_id=self.run_batch_id,
                    level="CRITICAL",
                    event_type="scraper_run_aborted",
                    message=f"Monthly budget exceeded: {budget_result.get('reason')}",
                )
                return {
                    "success": False,
                    "reason": "monthly_budget_exceeded",
                    "run_batch_id": self.run_batch_id,
                }

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
            result = crew.kickoff()

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
            jobs_auto_applied = 0  # Will be set by Apply Agent later
            jobs_queued = 0  # Will be set by Analyser Agent later

            # Update run batch stats
            update_run_batch_stats(
                run_batch_id=self.run_batch_id,
                jobs_discovered=total_jobs,
                jobs_auto_applied=jobs_auto_applied,
                jobs_queued=jobs_queued,
            )

            # Log completion
            log_event(
                run_batch_id=self.run_batch_id,
                level="INFO",
                event_type="scraper_run_complete",
                message=f"Scraper complete — {total_jobs} jobs discovered",
            )

            self.logger.info(
                f"Scraper run completed: {total_jobs} jobs discovered across "
                f"{len(by_platform)} platforms"
            )

            return {
                "success": True,
                "run_batch_id": self.run_batch_id,
                "total_jobs": total_jobs,
                "by_platform": by_platform,
                "safety_net_triggered": result_data.get("safety_net_triggered", False),
                "duplicates_removed": result_data.get("duplicates_removed", 0),
                "minimum_met": result_data.get("minimum_met", False),
            }

        except Exception as e:
            self.logger.error(f"Scraper run failed with exception: {e}", exc_info=True)

            # Log critical error
            log_event(
                run_batch_id=self.run_batch_id,
                level="CRITICAL",
                event_type="scraper_run_failed",
                message=f"Scraper run failed: {str(e)}",
            )

            return {
                "success": False,
                "error": str(e),
                "run_batch_id": self.run_batch_id,
            }

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
