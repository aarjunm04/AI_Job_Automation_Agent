"""Centralised configuration settings for the AI Job Application Agent.

All environment variable reads are consolidated here into typed, frozen
dataclass instances. Every other module should import the module-level
singletons (``db_config``, ``run_config``, ``budget_config``,
``api_config``) from this module instead of calling ``os.getenv()``
directly.

Secrets are loaded exclusively from environment variables (typically
injected via ``--env-file ~/narad.env``). No values are hard-coded.
"""

import logging
import os
from dataclasses import dataclass, field

__all__ = [
    "db_config",
    "run_config",
    "budget_config",
    "api_config",
    "get_settings",
    "DBConfig",
    "RunConfig",
    "BudgetConfig",
    "APIConfig",
]


@dataclass(frozen=True)
class DBConfig:
    """Database connection configuration.

    Attributes:
        active_db: Which DB backend to use; ``"local"`` (Docker Postgres)
            or ``"supabase"``.
        local_postgres_url: SQLAlchemy connection URL for the local Docker
            Postgres instance.
        supabase_url: Supabase project URL (production DB).
        supabase_key: Supabase service-role key.
        redis_url: Redis connection URL used for caching / queuing.
        chromadb_path: Filesystem path to the ChromaDB data directory.
    """

    active_db: str = field(
        default_factory=lambda: os.getenv("ACTIVE_DB", "local")
    )
    local_postgres_url: str = field(
        default_factory=lambda: os.getenv("LOCAL_POSTGRES_URL", "")
    )
    supabase_url: str = field(
        default_factory=lambda: os.getenv("SUPABASE_URL", "")
    )
    supabase_key: str = field(
        default_factory=lambda: os.getenv("SUPABASE_KEY", "")
    )
    redis_url: str = field(
        default_factory=lambda: os.getenv("REDIS_URL", "redis://localhost:6379")
    )
    chromadb_path: str = field(
        default_factory=lambda: os.getenv("CHROMADB_PATH", "app/chromadb")
    )

    @property
    def connection_url(self) -> str:
        """Return the active Postgres connection URL based on ``active_db``.

        Returns:
            ``local_postgres_url`` when ``active_db == "local"``, otherwise
            ``supabase_url``.
        """
        return (
            self.local_postgres_url
            if self.active_db == "local"
            else self.supabase_url
        )


@dataclass(frozen=True)
class RunConfig:
    """Runtime behaviour configuration for each pipeline run.

    Attributes:
        jobs_per_run_target: Ideal number of jobs to discover per run.
        jobs_per_run_minimum: Minimum acceptable jobs; triggers safety-net
            platforms if not reached.
        max_playwright_sessions: Maximum concurrent Playwright browser
            sessions.
        auto_apply_enabled: When ``False``, all jobs are routed to the
            manual queue regardless of score.
        dry_run: When ``True``, form submissions are simulated and no real
            applications are submitted.
        resume_dir: Directory containing resume PDF variants.
        default_resume: Filename of the general-purpose resume to use when
            no domain-specific variant is selected.
        search_query: Default job-search query string passed to scrapers.
        log_level: Python ``logging`` level string (e.g. ``"INFO"``).
    """

    jobs_per_run_target: int = field(
        default_factory=lambda: int(os.getenv("JOBS_PER_RUN_TARGET", "150"))
    )
    jobs_per_run_minimum: int = field(
        default_factory=lambda: int(os.getenv("JOBS_PER_RUN_MINIMUM", "100"))
    )
    max_playwright_sessions: int = field(
        default_factory=lambda: int(os.getenv("MAX_PLAYWRIGHT_SESSIONS", "5"))
    )
    auto_apply_enabled: bool = field(
        default_factory=lambda: os.getenv(
            "AUTO_APPLY_ENABLED", "true"
        ).lower()
        == "true"
    )
    dry_run: bool = field(
        default_factory=lambda: os.getenv("DRY_RUN", "false").lower() == "true"
    )
    resume_dir: str = field(
        default_factory=lambda: os.getenv("RESUME_DIR", "resumes")
    )
    default_resume: str = field(
        default_factory=lambda: os.getenv("DEFAULT_RESUME", "AarjunGen.pdf")
    )
    search_query: str = field(
        default_factory=lambda: os.getenv(
            "SEARCH_QUERY",
            "AI ML Data Science Automation Engineer remote",
        )
    )
    log_level: str = field(
        default_factory=lambda: os.getenv("LOG_LEVEL", "INFO")
    )


@dataclass(frozen=True)
class BudgetConfig:
    """Cost and budget enforcement configuration.

    Attributes:
        xai_cost_cap_per_run: Maximum USD spend on xAI APIs allowed in a
            single pipeline run before the Master Agent aborts.
        total_monthly_budget: Hard monthly budget cap in USD across all
            paid API providers.
    """

    xai_cost_cap_per_run: float = field(
        default_factory=lambda: float(
            os.getenv("XAI_COST_CAP_PER_RUN", "0.38")
        )
    )
    total_monthly_budget: float = field(
        default_factory=lambda: float(
            os.getenv("TOTAL_MONTHLY_BUDGET", "10.00")
        )
    )


@dataclass(frozen=True)
class APIConfig:
    """Third-party API key configuration.

    Attributes:
        groq_api_key: Groq API key (Master & Tracker agents).
        xai_api_key: xAI API key (Analyser & Apply agents).
        perplexity_api_key: Perplexity API key (Scraper agent).
        cerebras_api_key: Cerebras API key (fallback for multiple agents).
        sambanova_api_key: SambaNova API key (fallback for Analyser/Apply).
        nvidia_nim_api_key: NVIDIA NIM API key (ChromaDB embeddings).
        gemini_api_key: Gemini API key (fallback embeddings).
        agentops_api_key: AgentOps API key (tracing & monitoring).
        notion_api_key: Notion integration token (Tracker agent).
        serpapi_keys: List of SerpAPI keys (up to 4 accounts, round-robin).
    """

    groq_api_key: str = field(
        default_factory=lambda: os.getenv("GROQ_API_KEY", "")
    )
    xai_api_key: str = field(
        default_factory=lambda: os.getenv("XAI_API_KEY", "")
    )
    perplexity_api_key: str = field(
        default_factory=lambda: os.getenv("PERPLEXITY_API_KEY", "")
    )
    cerebras_api_key: str = field(
        default_factory=lambda: os.getenv("CEREBRAS_API_KEY", "")
    )
    sambanova_api_key: str = field(
        default_factory=lambda: os.getenv("SAMBANOVA_API_KEY", "")
    )
    nvidia_nim_api_key: str = field(
        default_factory=lambda: os.getenv("NVIDIA_NIM_API_KEY", "")
    )
    gemini_api_key: str = field(
        default_factory=lambda: os.getenv("GEMINI_API_KEY", "")
    )
    agentops_api_key: str = field(
        default_factory=lambda: os.getenv("AGENTOPS_API_KEY", "")
    )
    notion_api_key: str = field(
        default_factory=lambda: os.getenv("NOTION_API_KEY", "")
    )
    serpapi_keys: list[str] = field(
        default_factory=lambda: [
            v
            for k in [
                "SERPAPI_API_KEY_1",
                "SERPAPI_API_KEY_2",
                "SERPAPI_API_KEY_3",
                "SERPAPI_API_KEY_4",
            ]
            for v in [os.getenv(k, "")]
            if v
        ]
    )


def get_settings() -> tuple[DBConfig, RunConfig, BudgetConfig, APIConfig]:
    """Initialise logging and build all configuration singletons.

    Reads environment variables (typically sourced from ``~/narad.env``)
    and returns a tuple of frozen dataclass instances representing every
    subsystem's configuration.  This function is called once at module
    import time; the results are stored as module-level singletons.

    Returns:
        A four-element tuple ``(db_config, run_config, budget_config,
        api_config)``, each a frozen dataclass populated exclusively from
        environment variables.
    """
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
    return DBConfig(), RunConfig(), BudgetConfig(), APIConfig()


db_config, run_config, budget_config, api_config = get_settings()
