"""Utility modules for AI Job Application Agent.

Provides shared database, proxy, normalisation, and helper
utilities used across agents, tools, and scrapers.
"""

from .db_utils import get_db_conn
from .proxy_rate_limit import (
    ProxyPool,
    get_next_proxy,
    get_proxy_dict,
    get_playwright_proxy,
    get_httpx_proxy,
    mark_proxy_dead,
    mark_proxy_success,
    reset_cycle,
    get_proxy_stats,
    is_proxy_pool_healthy,
)

from .normalise_dedupe import (
    normalise_job_post,
    clean_description,
    deduplicate_jobs_fuzzy,
    canonical_url,
    compute_similarity,
    days_old,
    upsert_jobs_postgres,
)

__all__ = [
    "get_db_conn",
    "get_proxy_dict",
    "get_next_proxy",
    "reset_cycle",
    "ProxyRateLimiter",
    "normalise_job_post",
    "clean_description",
    "deduplicate_jobs_fuzzy",
    "canonical_url",
    "compute_similarity",
    "days_old",
    "upsert_jobs_postgres",
]
