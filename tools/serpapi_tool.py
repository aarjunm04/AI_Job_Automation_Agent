from __future__ import annotations
"""SerpAPI Google Jobs tool for supplementary job discovery.

Implements 4-key round-robin rotation across SERPAPI_API_KEY_1..4.
Used as fallback when primary scrapers return < 100 jobs per run.
READ-ONLY discovery tool — never modifies any data.
"""


import threading
import json
import logging
import os
import time
import agentops
from agentops.sdk.decorators import operation
from serpapi import GoogleSearch
from typing import List, Dict, Any, Optional
import hashlib
from pathlib import Path

logger = logging.getLogger(__name__)

def _retry_call(fn, *args, max_retries: int = 3, **kwargs):
    """Execute fn with exponential backoff retry."""
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            last_exc = exc
            wait = 2.0 ** attempt
            logger.warning(
                "Attempt %d/%d failed: %s — retrying in %.1fs",
                attempt + 1, max_retries, exc, wait,
            )
            time.sleep(wait)
    raise RuntimeError(
        f"All {max_retries} attempts failed. Last error: {last_exc}"
    )

# Load all 4 keys at import time, filter empty strings
_SERPAPI_KEYS: list[str] = [
    k for k in [
        os.getenv("SERPAPI_API_KEY_1", ""),
        os.getenv("SERPAPI_API_KEY_2", ""),
        os.getenv("SERPAPI_API_KEY_3", ""),
        os.getenv("SERPAPI_API_KEY_4", ""),
    ] if k.strip()
]
_key_lock = threading.Lock()
_key_index: int = 0

if not _SERPAPI_KEYS:
    logger.warning(
        "serpapi_tool: no SERPAPI_API_KEY_* env vars set — "
        "SerpAPI discovery disabled"
    )

def _get_next_key() -> Optional[str]:
    """Thread-safe round-robin key rotation across _SERPAPI_KEYS.

    Returns:
        Next available API key string, or None if pool is empty.
    """
    global _key_index
    with _key_lock:
        if not _SERPAPI_KEYS:
            return None
        key = _SERPAPI_KEYS[_key_index % len(_SERPAPI_KEYS)]
        _key_index += 1
        return key

@operation
def search_google_jobs(
    query: str = "",
    location: str = "Remote",
    num_results: int = 20,
    **kwargs
) -> str:
    """Search Google Jobs via SerpAPI with 4-key round-robin rotation.

    Queries the Google Jobs engine via SerpAPI. Rotates across up to
    4 API keys to maximise monthly credit budget. Returns normalised
    job list as JSON string. Called as supplementary discovery only
    when primary scrapers return fewer than 100 jobs in a run.

    Args:
        query: Job search query e.g. "ML Engineer remote India".
        location: Location filter e.g. "Remote" or "India".
        num_results: Max results to fetch per call. Default 20.

    Returns:
        JSON string — list of normalised job dicts on success:
          [{"title", "company", "job_url", "location",
            "description", "platform", "source"}, ...]
        JSON string — error dict on total failure:
          {"error": "<reason>"}
    """
    if not _SERPAPI_KEYS:
        return json.dumps({"error": "no_serpapi_keys_configured"})

    max_attempts = min(3, len(_SERPAPI_KEYS))

    def _do_search(
        query_params: dict[str, str],
        api_key: str,
    ) -> list[dict[str, Any]]:
        """Performs the SerpAPI search and normalises results."""
        params = {
            "engine": "google_jobs",
            "api_key": api_key,
            **query_params,
        }

        search = GoogleSearch(params)
        response = search.get_dict()

        if "error" in response:
            raise RuntimeError(f"SerpAPI error: {response['error']}")

        raw_jobs = response.get("jobs_results", [])
        normalised_jobs = []
        for job in raw_jobs:
            title = job["title"]
            company_name = job["company_name"]
            job_id = job.get("job_id", "")
            if not job_id:
                # Create a stable job_id if one isn't provided.
                job_id_str = f"{title}{company_name}"
                job_id = hashlib.sha256(job_id_str.encode('utf-8')).hexdigest()

            apply_link = ""
            if apply_options := job.get("apply_options", []):
                if isinstance(apply_options, list) and apply_options:
                    apply_link = apply_options[0].get("link", "")

            normalised_jobs.append({
                "title": title,
                "company_name": company_name,
                "location": job["location"],
                "description": job.get("description", "")[:3000],
                "job_id": job_id,
                "apply_link": apply_link,
                "source": "serpapi",
            })
        return normalised_jobs

    def _do_search():
        key = _get_next_key()
        if not key:
            raise RuntimeError("serpapi_all_keys_exhausted")
        params = {
            "engine": "google_jobs",
            "q": query,
            "location": "United States",
            "ltype": "1",
            "api_key": key,
            "num": 10,
        }
        results = GoogleSearch(params).get_dict()
        if "error" in results:
            raise RuntimeError(str(results["error"]))
        return results

    try:
        results = _retry_call(_do_search, max_retries=max_attempts)
    except Exception as e:
        logger.error(
            "SerpAPI: all %d key attempts failed for query='%s'",
            max_attempts, query
        )
        return json.dumps({"error": "serpapi_all_keys_failed"})

    raw_jobs = results.get("jobs_results", [])
    normalised = []
    for job in raw_jobs:
        related = job.get("related_links") or [{}]
        normalised.append({
            "title":       job.get("title", ""),
            "company":     job.get("company_name", ""),
            "job_url":     related[0].get("link", ""),
            "location":    job.get("location", location),
            "description": job.get("description", "")[:500],
            "platform":    "serpapi",
            "source":      "google_jobs"
        })
    logger.info(
        "SerpAPI: %d jobs returned for query='%s'",
        len(normalised), query
    )
    return json.dumps(normalised)

__all__ = ["search_google_jobs"]
