"""Utility functions for job post normalisation and deduplication.

Called by scraper_agent.py and scraper_tools.py after raw job collection.
Provides URL-hash dedup, fuzzy title+company dedup (Levenshtein ≥0.85),
URL canonicalisation, timestamp→days_old, and Postgres batch upsert.

Includes:
- Job dataclass for type-safe job handling
- Levenshtein similarity for fuzzy matching
- Company slug normalization
- Title normalization (strip Senior/Jr/Remote/URGENT)
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from difflib import SequenceMatcher
from typing import Any, Optional, List
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

logger = logging.getLogger(__name__)

__all__ = [
    "Job",
    "normalise_job_post",
    "normalize_job",
    "deduplicate_jobs",
    "deduplicate_jobs_fuzzy",
    "deduplicate_jobs_levenshtein",
    "generate_url_hash",
    "canonical_url",
    "clean_description",
    "compute_similarity",
    "levenshtein_similarity",
    "days_old",
    "upsert_jobs_postgres",
    "company_slug",
    "normalize_title",
    "DedupeStats",
]


# ---------------------------------------------------------------------------
# Job Dataclass
# ---------------------------------------------------------------------------

@dataclass
class Job:
    """Normalized job data structure."""
    title: str
    company: str
    url: str
    source_platform: str = ""
    location: str = ""
    job_type: str = ""
    description: str = ""
    required_skills: str = ""
    salary_range: str = ""
    posted_at: str = ""
    is_remote: bool = False
    url_hash: str = ""
    days_old: int = -1
    fit_score: float = 0.0
    # Normalized fields for deduplication
    title_normalized: str = ""
    company_slug: str = ""

    def __post_init__(self) -> None:
        """Generate derived fields after init."""
        if not self.url_hash and self.url:
            self.url_hash = generate_url_hash(self.url)
        if not self.title_normalized and self.title:
            self.title_normalized = normalize_title(self.title)
        if not self.company_slug and self.company:
            self.company_slug = company_slug(self.company)


@dataclass
class DedupeStats:
    """Statistics from a deduplication run."""
    input_count: int
    output_count: int
    url_removed: int = 0
    fuzzy_removed: int = 0
    levenshtein_merged: int = 0

    @property
    def total_removed(self) -> int:
        return self.input_count - self.output_count


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


# ---------------------------------------------------------------------------
# URL canonicalisation
# ---------------------------------------------------------------------------

# Tracking query params to strip during canonicalisation
_TRACKING_PARAMS: frozenset[str] = frozenset({
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "ref", "refId", "source", "fbclid", "gclid", "mc_cid", "mc_eid",
    "trk", "trkInfo", "trackingId", "gh_jid",
})


def canonical_url(url: str) -> str:
    """Canonicalise a job URL for deduplication.

    Strips tracking query parameters, lowercases the host, removes trailing
    slashes, and normalises the scheme to https.

    Args:
        url: Raw URL string.

    Returns:
        Canonical URL string. Empty string if input is falsy.
    """
    if not url:
        return ""
    url = url.strip()
    try:
        parsed = urlparse(url)
        scheme = "https" if parsed.scheme in ("http", "https", "") else parsed.scheme
        netloc = (parsed.netloc or "").lower().rstrip(".")
        path = parsed.path.rstrip("/") or "/"
        # Strip tracking params
        qp = parse_qs(parsed.query, keep_blank_values=False)
        filtered = {k: v for k, v in qp.items() if k.lower() not in _TRACKING_PARAMS}
        query = urlencode(filtered, doseq=True) if filtered else ""
        return urlunparse((scheme, netloc, path, "", query, ""))
    except Exception:
        return url.strip().lower()


# ---------------------------------------------------------------------------
# Fuzzy similarity
# ---------------------------------------------------------------------------


def compute_similarity(a: str, b: str) -> float:
    """Compute normalised string similarity using SequenceMatcher.

    Returns a float in [0.0, 1.0] where 1.0 is an exact match.
    Lowercases and strips whitespace before comparison.

    Args:
        a: First string.
        b: Second string.

    Returns:
        Similarity ratio.
    """
    a_clean = re.sub(r"\s+", " ", (a or "").strip().lower())
    b_clean = re.sub(r"\s+", " ", (b or "").strip().lower())
    if not a_clean or not b_clean:
        return 0.0
    return SequenceMatcher(None, a_clean, b_clean).ratio()


# ---------------------------------------------------------------------------
# Timestamp helper
# ---------------------------------------------------------------------------


def days_old(posted_at: Any) -> int:
    """Compute number of days since a job was posted.

    Accepts ISO-8601 strings, ``datetime`` objects, or None.  Returns
    ``-1`` when the posted date is unparseable.

    Args:
        posted_at: Posting timestamp.

    Returns:
        Integer days since posting, or -1 if unknown.
    """
    if not posted_at:
        return -1
    try:
        if isinstance(posted_at, datetime):
            dt = posted_at
        else:
            raw = str(posted_at).strip()
            # Try ISO-8601 first
            dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        delta = datetime.now(timezone.utc) - dt
        return max(0, delta.days)
    except Exception:
        return -1


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
        logger.warning("normalise_job_post: missing url for job title=%s", title)

    url = canonical_url(url)
    url_hash = generate_url_hash(url)
    age_days = days_old(posted_at)

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
        "url_hash": url_hash,
        "days_old": age_days,
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
    logger.info("deduplicate_jobs: %d input → %d after dedup (%d removed)", len(jobs), len(result), removed)
    
    return result


# ---------------------------------------------------------------------------
# Fuzzy deduplication (title + company similarity)
# ---------------------------------------------------------------------------


def deduplicate_jobs_fuzzy(
    jobs: list[dict[str, Any]],
    threshold: float = 0.85,
) -> list[dict[str, Any]]:
    """Deduplicate jobs using URL hash AND fuzzy title+company similarity.

    First applies exact URL-hash dedup, then checks remaining jobs for
    title+company pairs with SequenceMatcher ratio ≥ threshold.  When
    a fuzzy duplicate is found, the earlier (first-seen) entry is kept.

    Args:
        jobs: List of normalised job dicts (must have ``url_hash``,
            ``title``, and ``company`` keys).
        threshold: Minimum combined similarity to consider a duplicate.
            Default 0.85.

    Returns:
        Deduplicated list preserving insertion order.
    """
    # Phase 1 — exact URL-hash dedup
    url_deduped = deduplicate_jobs(jobs)

    # Phase 2 — fuzzy title+company dedup (O(n²) but n ≤ ~200 per run)
    kept: list[dict[str, Any]] = []
    fuzzy_removed = 0

    for candidate in url_deduped:
        c_title = candidate.get("title", "")
        c_company = candidate.get("company", "")
        is_dup = False

        for existing in kept:
            title_sim = compute_similarity(c_title, existing.get("title", ""))
            company_sim = compute_similarity(c_company, existing.get("company", ""))
            # Weighted: 60% title, 40% company — title matters more
            combined = 0.6 * title_sim + 0.4 * company_sim
            if combined >= threshold:
                is_dup = True
                fuzzy_removed += 1
                logger.debug(
                    "Fuzzy dup: '%s @ %s' ~ '%s @ %s' (sim=%.3f)",
                    c_title, c_company,
                    existing.get("title", ""), existing.get("company", ""),
                    combined,
                )
                break

        if not is_dup:
            kept.append(candidate)

    if fuzzy_removed:
        logger.info(
            "deduplicate_jobs_fuzzy: removed %d fuzzy duplicates | %d → %d",
            fuzzy_removed, len(url_deduped), len(kept),
        )

    return kept


# ---------------------------------------------------------------------------
# Postgres batch upsert
# ---------------------------------------------------------------------------


def upsert_jobs_postgres(
    jobs: list[dict[str, Any]],
    pipeline_run_id: str,
) -> dict[str, Any]:
    """Batch upsert normalised jobs into the Postgres ``jobs`` table.

    Uses ``ON CONFLICT (url) DO UPDATE`` to handle re-scrapes gracefully.
    Requires a live DB connection via ``utils.db_utils.get_db_conn``.

    Args:
        jobs: List of normalised job dicts from ``normalise_job_post``.
        pipeline_run_id: UUID of the current run batch.

    Returns:
        Dict with ``inserted``, ``updated``, and ``errors`` counts.
    """
    from utils.db_utils import get_db_conn

    inserted = 0
    updated = 0
    errors = 0

    if not jobs:
        return {"inserted": 0, "updated": 0, "errors": 0}

    conn: Optional[Any] = None
    try:
        conn = get_db_conn()
        cursor = conn.cursor()

        for job in jobs:
            url = job.get("url", "")
            if not url:
                errors += 1
                continue

            try:
                cursor.execute(
                    """
                    INSERT INTO jobs (pipeline_run_id, source_platform, title,
                        company, location, url, posted_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (url) DO UPDATE SET
                        pipeline_run_id = EXCLUDED.pipeline_run_id,
                        title = EXCLUDED.title,
                        company = EXCLUDED.company,
                        location = EXCLUDED.location,
                        posted_at = EXCLUDED.posted_at
                    RETURNING (xmax = 0) AS is_insert
                    """,
                    (
                        pipeline_run_id,
                        job.get("source_platform", "unknown"),
                        job.get("title", ""),
                        job.get("company", ""),
                        job.get("location", ""),
                        url,
                        job.get("posted_at") or None,
                    ),
                )
                row = cursor.fetchone()
                if row and row[0]:
                    inserted += 1
                else:
                    updated += 1
            except Exception as exc:
                conn.rollback()
                errors += 1
                logger.warning("upsert_jobs_postgres: failed for url=%s: %s", url, exc)
                # Re-open transaction after rollback
                continue

        conn.commit()
    except Exception as exc:
        logger.error("upsert_jobs_postgres: batch upsert failed: %s", exc)
        if conn:
            try:
                conn.rollback()
            except Exception:
                pass
        return {"inserted": 0, "updated": 0, "errors": len(jobs)}
    finally:
        if conn:
            try:
                conn.close()
            except Exception:
                pass

    logger.info(
        "upsert_jobs_postgres: inserted=%d updated=%d errors=%d",
        inserted, updated, errors,
    )
    return {"inserted": inserted, "updated": updated, "errors": errors}


# ---------------------------------------------------------------------------
# Title and Company Normalization
# ---------------------------------------------------------------------------

# Patterns to strip from job titles for comparison
_TITLE_STRIP_PATTERNS: tuple[str, ...] = (
    r"\b(senior|sr\.?|junior|jr\.?|lead|staff|principal)\b",
    r"\b(remote|hybrid|onsite|on-site)\b",
    r"\b(urgent|immediate|asap|hiring now)\b",
    r"\b(i+|ii+|iii+|iv|v|vi)\b",  # Roman numerals
    r"\b(level\s*\d+|l\d+)\b",  # Level markers
    r"[()[\]{}]",
    r"[-–—/\\|,]",
    r"\s+",
)


def normalize_title(title: str) -> str:
    """Normalize job title for comparison.
    
    Strips common prefixes like Senior/Jr/Remote/URGENT,
    removes punctuation, and lowercases.
    
    Args:
        title: Raw job title.
        
    Returns:
        Normalized title string.
    """
    if not title:
        return ""
    
    result = title.lower().strip()
    
    for pattern in _TITLE_STRIP_PATTERNS:
        result = re.sub(pattern, " ", result, flags=re.IGNORECASE)
    
    # Collapse multiple spaces
    result = re.sub(r"\s+", " ", result).strip()
    
    return result


def company_slug(company: str) -> str:
    """Generate a normalized company slug for comparison.
    
    Lowercases, removes non-alphanumeric characters,
    strips common suffixes like Inc, LLC, Corp.
    
    Args:
        company: Raw company name.
        
    Returns:
        Normalized company slug.
    """
    if not company:
        return ""
    
    result = company.lower().strip()
    
    # Strip common company suffixes
    suffixes = (
        r"\b(inc\.?|llc\.?|ltd\.?|corp\.?|corporation|company|co\.?)\b",
        r"\b(technologies|technology|tech|systems|solutions|services)\b",
        r"\b(group|holdings|partners)\b",
    )
    for suffix in suffixes:
        result = re.sub(suffix, "", result, flags=re.IGNORECASE)
    
    # Keep only alphanumeric
    result = re.sub(r"[^a-z0-9]", "", result)
    
    return result


# ---------------------------------------------------------------------------
# Levenshtein Similarity
# ---------------------------------------------------------------------------

def levenshtein_distance(s1: str, s2: str) -> int:
    """Compute Levenshtein (edit) distance between two strings.
    
    Uses dynamic programming with O(min(m,n)) space.
    
    Args:
        s1: First string.
        s2: Second string.
        
    Returns:
        Edit distance as integer.
    """
    if len(s1) < len(s2):
        s1, s2 = s2, s1
    
    if not s2:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            # Cost is 0 if characters match, 1 otherwise
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


def levenshtein_similarity(s1: str, s2: str) -> float:
    """Compute Levenshtein similarity ratio.
    
    Returns 1.0 for identical strings, 0.0 for completely different.
    
    Args:
        s1: First string.
        s2: Second string.
        
    Returns:
        Similarity ratio in [0.0, 1.0].
    """
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    
    max_len = max(len(s1), len(s2))
    distance = levenshtein_distance(s1, s2)
    
    return 1.0 - (distance / max_len)


# ---------------------------------------------------------------------------
# Normalize Job (returns Job dataclass)
# ---------------------------------------------------------------------------

def normalize_job(raw: dict[str, Any]) -> Job:
    """Normalize raw job dict to Job dataclass.
    
    Cleans title, canonical_url, company_slug, days_old.
    
    Args:
        raw: Raw job dictionary from scraper.
        
    Returns:
        Normalized Job dataclass instance.
    """
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
    
    url = canonical_url(url)
    url_hash = generate_url_hash(url)
    age_days = days_old(posted_at)
    
    return Job(
        title=title,
        company=company,
        url=url,
        source_platform=source_platform,
        location=location,
        job_type=job_type,
        description=description,
        required_skills=required_skills,
        salary_range=salary_range,
        posted_at=str(posted_at) if posted_at else "",
        is_remote=is_remote,
        url_hash=url_hash,
        days_old=age_days,
        title_normalized=normalize_title(title),
        company_slug=company_slug(company),
    )


# ---------------------------------------------------------------------------
# Levenshtein-based Deduplication
# ---------------------------------------------------------------------------

def deduplicate_jobs_levenshtein(
    jobs: list[dict[str, Any]],
    threshold: float = 0.85,
) -> tuple[list[dict[str, Any]], DedupeStats]:
    """Deduplicate jobs using Levenshtein similarity.
    
    Three-phase deduplication:
    1. URL-hash exact dedup
    2. Company slug + normalized title grouping
    3. Cross-group Levenshtein similarity > threshold → merge (keep highest fit_score)
    
    Args:
        jobs: List of job dicts.
        threshold: Levenshtein similarity threshold (default 0.85).
        
    Returns:
        Tuple of (deduplicated list, DedupeStats).
    """
    if not jobs:
        return [], DedupeStats(0, 0)
    
    input_count = len(jobs)
    
    # Phase 1: URL-hash dedup
    url_seen: set[str] = set()
    url_deduped: list[dict[str, Any]] = []
    
    for job in jobs:
        url = job.get("url", "")
        url_hash = job.get("url_hash") or generate_url_hash(url)
        
        if url_hash not in url_seen:
            url_seen.add(url_hash)
            # Ensure job has normalized fields
            if "title_normalized" not in job:
                job["title_normalized"] = normalize_title(job.get("title", ""))
            if "company_slug" not in job:
                job["company_slug"] = company_slug(job.get("company", ""))
            url_deduped.append(job)
    
    url_removed = input_count - len(url_deduped)
    
    # Phase 2: Group by company_slug + title_normalized
    groups: dict[str, list[dict[str, Any]]] = {}
    
    for job in url_deduped:
        key = f"{job.get('company_slug', '')}:{job.get('title_normalized', '')}"
        if key not in groups:
            groups[key] = []
        groups[key].append(job)
    
    # Within each group, keep only the newest
    phase2_kept: list[dict[str, Any]] = []
    for key, group in groups.items():
        if len(group) == 1:
            phase2_kept.append(group[0])
        else:
            # Keep newest (lowest days_old) or highest fit_score
            best = max(
                group,
                key=lambda j: (
                    -j.get("days_old", 999),  # Prefer newer
                    j.get("fit_score", 0),    # Then higher score
                )
            )
            phase2_kept.append(best)
    
    fuzzy_removed = len(url_deduped) - len(phase2_kept)
    
    # Phase 3: Cross-group Levenshtein similarity check
    # Compare normalized titles with similarity > threshold
    final_kept: list[dict[str, Any]] = []
    levenshtein_merged = 0
    
    for candidate in phase2_kept:
        c_title = candidate.get("title_normalized", "")
        c_company = candidate.get("company_slug", "")
        is_dup = False
        
        for existing in final_kept:
            e_title = existing.get("title_normalized", "")
            e_company = existing.get("company_slug", "")
            
            # Must have similar company
            company_sim = levenshtein_similarity(c_company, e_company)
            if company_sim < 0.7:
                continue
            
            # Check title similarity
            title_sim = levenshtein_similarity(c_title, e_title)
            
            # Combined: "ML Engineer" vs "Machine Learning Engineer" should merge
            combined = 0.6 * title_sim + 0.4 * company_sim
            
            if combined >= threshold:
                is_dup = True
                levenshtein_merged += 1
                
                # Merge: keep the one with higher fit_score
                if candidate.get("fit_score", 0) > existing.get("fit_score", 0):
                    final_kept.remove(existing)
                    final_kept.append(candidate)
                
                logger.debug(
                    "Levenshtein merge: '%s @ %s' ~ '%s @ %s' (sim=%.3f)",
                    candidate.get("title", ""), candidate.get("company", ""),
                    existing.get("title", ""), existing.get("company", ""),
                    combined,
                )
                break
        
        if not is_dup:
            final_kept.append(candidate)
    
    stats = DedupeStats(
        input_count=input_count,
        output_count=len(final_kept),
        url_removed=url_removed,
        fuzzy_removed=fuzzy_removed,
        levenshtein_merged=levenshtein_merged,
    )
    
    logger.info(
        "deduplicate_jobs_levenshtein: %d→%d (url=%d fuzzy=%d lev=%d)",
        input_count, len(final_kept), url_removed, fuzzy_removed, levenshtein_merged,
    )
    
    return final_kept, stats
