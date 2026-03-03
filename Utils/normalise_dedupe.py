"""Utility functions for job post normalisation and deduplication.
Called by scraper_agent.py and scraper_tools.py after raw job collection."""

import re
import logging
import hashlib
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["normalise_job_post", "deduplicate_jobs", "generate_url_hash", "clean_description"]


def clean_description(text: str, max_chars: int = 2000) -> str:
    """Clean and truncate a raw job description string.

    Strips HTML tags using BeautifulSoup, normalises whitespace,
    and truncates at word boundary. Falls back to regex on error.

    Args:
        text: Raw description string, may contain HTML.
        max_chars: Maximum character length. Default 2000.

    Returns:
        Cleaned, truncated plain text string.
    """
    if not text:
        return ""
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(text, "html.parser")
        cleaned = soup.get_text(separator=" ")
    except Exception as e:
        logger.warning(
            "BeautifulSoup HTML clean failed: %s — using regex fallback",
            str(e)
        )
        import re
        cleaned = re.sub(r"<[^>]+>", " ", text)

    import re
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    if len(cleaned) <= max_chars:
        return cleaned
    truncated = cleaned[:max_chars]
    last_space = truncated.rfind(" ")
    if last_space > int(max_chars * 0.8):
        truncated = truncated[:last_space]
    return truncated + "..."


def generate_url_hash(url: str) -> str:
    """Generate SHA256 hex digest for normalising and deduplicating job URLs."""
    url_stripped = str(url).strip().lower() if url else ""
    return hashlib.sha256(url_stripped.encode("utf-8")).hexdigest()


def normalise_job_post(raw: dict[str, Any]) -> dict[str, Any]:
    """Format raw job dictionary to standard structure."""
    title = str(raw.get("title") or raw.get("job_title") or "").strip()
    company = str(raw.get("company") or raw.get("company_name") or "").strip()
    url = str(raw.get("url") or raw.get("job_url") or raw.get("apply_url") or "").strip()
    source_platform = str(raw.get("source_platform") or raw.get("platform") or raw.get("site") or "").strip().lower()
    location = str(raw.get("location") or raw.get("job_location") or "").strip()
    job_type = str(raw.get("job_type") or raw.get("employment_type") or "").strip().lower()
    
    desc_raw = raw.get("description") or raw.get("job_description") or raw.get("body") or ""
    description = clean_description(desc_raw)
    
    required_skills = str(raw.get("required_skills") or raw.get("skills") or "").strip()
    salary_range = str(raw.get("salary_range") or raw.get("compensation") or raw.get("salary") or "").strip()
    
    posted_at = raw.get("posted_at") or raw.get("date_posted") or raw.get("date") or ""
    
    is_remote_val = raw.get("is_remote")
    is_remote = False
    if is_remote_val is True or str(is_remote_val).lower() == "true":
        is_remote = True
    elif "remote" in location.lower():
        is_remote = True
    elif "remote" in title.lower():
        is_remote = True

    if not url:
        logger.warning(f"normalise_job_post: missing url for job title={title}")

    url_hash = generate_url_hash(url)

    return {
        "title": title,
        "company": company,
        "url": url,
        "source_platform": source_platform,
        "location": location,
        "job_type": job_type,
        "description": description,
        "required_skills": required_skills,
        "salary_range": salary_range,
        "posted_at": posted_at,
        "is_remote": is_remote,
        "url_hash": url_hash
    }


def deduplicate_jobs(jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deduplicate list of jobs based on the url_hash."""
    seen_hashes = set()
    result = []
    
    for job in jobs:
        url_hash = job.get("url_hash")
        if not url_hash:
            url = job.get("url") or ""
            url_hash = generate_url_hash(url)
            job["url_hash"] = url_hash
            
        if url_hash not in seen_hashes:
            seen_hashes.add(url_hash)
            result.append(job)
            
    removed = len(jobs) - len(result)
    logger.info(f"deduplicate_jobs: {len(jobs)} input → {len(result)} after dedup ({removed} removed)")
    
    return result
