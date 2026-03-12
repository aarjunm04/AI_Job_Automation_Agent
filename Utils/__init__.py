"""Utility modules for AI Job Application Agent.

Provides shared database, proxy, normalisation, and helper
utilities used across agents, tools, and scrapers.
"""

from .db_utils import get_db_conn
from .proxy_ratelimit import (
    get_proxy_dict,
    get_next_proxy,
    reset_proxy_cycle,
    ProxyRateLimiter,
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
    "reset_proxy_cycle",
    "ProxyRateLimiter",
    "normalise_job_post",
    "clean_description",
    "deduplicate_jobs_fuzzy",
    "canonical_url",
    "compute_similarity",
    "days_old",
    "upsert_jobs_postgres",
]
