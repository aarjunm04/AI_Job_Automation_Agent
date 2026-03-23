"""SerpAPI Google Jobs tool for supplementary job discovery.

Implements 4-key round-robin rotation across SERP_API_KEY_1..4 to
avoid quota hotspots. Used as a safety-net when primary scrapers
return fewer than the minimum jobs per run.
"""

from __future__ import annotations

import itertools
import json
import logging
import os
import threading
import time
from typing import Any, Optional

import agentops
from serpapi import GoogleSearch

logger = logging.getLogger(__name__)

_SERP_KEYS: list[str] = [
    k.strip()
    for k in [
        os.getenv("SERP_API_KEY_1", "") or "",
        os.getenv("SERP_API_KEY_2", "") or "",
        os.getenv("SERP_API_KEY_3", "") or "",
        os.getenv("SERP_API_KEY_4", "") or "",
    ]
    if (k or "").strip()
]

_key_lock = threading.Lock()
_key_cycle = itertools.cycle(_SERP_KEYS) if _SERP_KEYS else None


def _next_key() -> str:
    if not _SERP_KEYS or _key_cycle is None:
        logger.error("SerpAPI: no SERP_API_KEY_1..4 keys configured")
        raise RuntimeError("serpapi_no_keys_configured")
    with _key_lock:
        return next(_key_cycle)


@agentops.track_tool
def search_google_jobs(
    query: str = "",
    location: str = "Remote",
    num_results: int = 20,
    **kwargs: Any,
) -> str:
    """Search Google Jobs via SerpAPI with exponential backoff + key rotation."""
    params = {
        "engine": "google_jobs",
        "q": query,
        "location": location,
        "num": num_results,
    }
    params.update(kwargs)

    last_exc: Optional[Exception] = None
    results: dict[str, Any] | None = None

    for attempt in range(3):
        try:
            params["api_key"] = _next_key()
            results = GoogleSearch(params).get_dict()
            if isinstance(results, dict) and "error" in results:
                raise RuntimeError(str(results["error"]))
            break
        except Exception as e:  # noqa: BLE001
            last_exc = e
            if attempt == 2:
                logger.error("SerpAPI GoogleSearch failed after 3 attempts: %s", e)
                raise
            delay = 2 ** attempt  # 1s, 2s
            logger.warning(
                "SerpAPI attempt %d failed, retrying in %ds: %s",
                attempt + 1,
                delay,
                e,
            )
            time.sleep(delay)

    if not results:
        raise RuntimeError(f"serpapi_no_results: {last_exc}")

    raw_jobs = results.get("jobs_results", []) if isinstance(results, dict) else []
    normalised: list[dict[str, Any]] = []
    for job in raw_jobs:
        related = job.get("related_links") or [{}]
        normalised.append(
            {
                "title": job.get("title", ""),
                "company": job.get("company_name", ""),
                "job_url": related[0].get("link", ""),
                "location": job.get("location", location),
                "description": job.get("description", "")[:500],
                "platform": "serpapi",
                "source": "google_jobs",
            }
        )

    return json.dumps(normalised)


__all__ = ["search_google_jobs"]

