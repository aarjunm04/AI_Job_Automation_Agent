"""
core/jobspy_adapter.py

ENTERPRISE-GRADE JOBSPY WRAPPER
================================

Production-ready abstraction over JobSpy for:
- LinkedIn
- Indeed
- ZipRecruiter
- Glassdoor

Responsibilities:
├── Patch JobSpy country parsing to avoid invalid-country crashes
├── Enforce strict site allowlist
├── Enforce per-site job limits (config-driven)
├── Async-friendly integration with ScraperEngine
├── Structured logging for observability
└── Return RAW job dicts only (no normalization/dedupe)

This module does NOT:
├── Normalize data
├── Deduplicate data
└── Apply business rules
"""

from __future__ import annotations

import asyncio
import logging
from typing import List, Dict, Any, Iterable

import math
import pandas as pd

# ================================================================================
# LOGGING
# ================================================================================

LOG = logging.getLogger("jobspy_adapter")

if not LOG.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    )
    handler.setFormatter(formatter)
    LOG.addHandler(handler)

LOG.setLevel(logging.INFO)

# ================================================================================
# JOBSPY IMPORTS + PATCHING
# ================================================================================

try:
    import jobspy  # noqa: F401
    from jobspy import scrape_jobs
    from jobspy.model import Country, Site

    JOBSPY_AVAILABLE = True
except Exception as e:  # pragma: no cover
    JOBSPY_AVAILABLE = False
    LOG.error("JobSpy not available: %s", e)

# ---- Country.from_string PATCH --------------------------------------------------
# Prevent crashes from invalid country strings (e.g. "armenia")

if JOBSPY_AVAILABLE:
    _real_from_string = Country.from_string

    @classmethod  # type: ignore[override]
    def _safe_from_string(cls, raw):
        try:
            return _real_from_string(raw)
        except Exception:
            # Treat any invalid country as worldwide
            return _real_from_string("worldwide")

    Country.from_string = _safe_from_string  # type: ignore[attr-defined]
    LOG.info("Patched JobSpy Country.from_string() successfully.")

# ================================================================================
# CONSTANTS
# ================================================================================

ALLOWED_SITES: Iterable["Site"] = (
    Site.LINKEDIN,
    Site.INDEED,
    Site.ZIP_RECRUITER,
    Site.GLASSDOOR,
)

# Default country used when JobSpy requires a concrete country.
# Glassdoor MUST NOT be called with "worldwide" or invalid countries.
DEFAULT_COUNTRY = "united states"


# ================================================================================
# HELPERS
# ================================================================================


def _safe_str(value: Any, default: str = "") -> str:
    """
    Convert value to a safe string.

    - None          -> default
    - NaN/NaT/etc.  -> default
    - Other values  -> str(value)
    """
    if value is None:
        return default

    # Handle pandas / numpy NaN
    try:
        if isinstance(value, float) and math.isnan(value):
            return default
    except Exception:
        # If math.isnan itself fails, fall through to str
        pass

    # Some libraries may expose their own NaN-like objects
    if isinstance(value, (pd.Timestamp,)):
        # For timestamps we generally do not want string conversion here
        # (they are handled separately), so just default.
        return default

    text = str(value)
    # Guard against whitespace-only strings
    text = text.strip()
    return text or default


def _sanitize_location(value: Any) -> str:
    """
    Normalize a raw location value to a safe, non-empty string.

    Requirements:
    - Never crash on NaN / floats.
    - Default invalid or missing locations to "Remote".
    """
    text = _safe_str(value, default="Remote")
    if not text:
        return "Remote"

    # Very short numeric-only values are not meaningful locations
    if text.isdigit() and len(text) <= 3:
        return "Remote"

    return text


def _build_job_dict(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a raw JobSpy record into a safe raw job dict without
    applying any business rules.
    """
    title_raw = record.get("job_title") or record.get("title")
    job_url_raw = record.get("job_url") or record.get("url")

    title = _safe_str(title_raw)
    job_url = _safe_str(job_url_raw)

    if not title or not job_url:
        # Upstream caller should skip empty results
        return {}

    company = _safe_str(record.get("company") or record.get("company_name"))
    location = _sanitize_location(record.get("location"))
    description = _safe_str(record.get("description") or record.get("snippet"))
    source = _safe_str(record.get("site") or "jobspy").lower()

    job: Dict[str, Any] = {
        "title": title,
        "company": company,
        "location": location,
        "job_url": job_url,
        "description": description,
        "source": source,
    }

    # Optional structured fields
    posted_date = (
        record.get("date_posted") or record.get("posted_date") or record.get("date")
    )
    if posted_date is not None:
        job["posted_date"] = posted_date

    salary = record.get("salary") or record.get("compensation")
    if salary is not None:
        job["salary"] = salary

    job_type = record.get("job_type") or record.get("employment_type")
    if job_type is not None:
        job["job_type"] = job_type

    company_url = record.get("company_url") or record.get("company_link")
    if company_url is not None:
        job["company_url"] = company_url

    # Additional, potentially useful metadata (non-business-logic)
    for key in ("industry", "experience_level", "company_size", "benefits"):
        if key in record and record[key] is not None:
            job[key] = record[key]

    return job


# ================================================================================
# ADAPTER
# ================================================================================


class JobSpyAdapter:
    """
    Thin async-friendly wrapper around JobSpy.

    Used by ScraperEngine as a single "jobspy" scraper which internally hits:
    - LinkedIn
    - Indeed
    - ZipRecruiter
    - Glassdoor

    Parameters
    ----------
    jobs_per_site : int
        Max number of jobs per JobSpy site (LinkedIn, Indeed, etc.).
    concurrency : int
        Reserved for future use / parity, JobSpy itself runs synchronously.
    hours_old : int
        Limit to jobs posted in the last `hours_old` hours.
    allowed_countries : Optional[list[str]]
        Optional list of allowed countries used to choose a Glassdoor country.
    """

    name = "jobspy"

    def __init__(
        self,
        jobs_per_site: int = 20,
        concurrency: int = 4,
        hours_old: int = 10,
        allowed_countries: List[str] | None = None,
    ) -> None:
        if not JOBSPY_AVAILABLE:
            raise RuntimeError("JobSpy is not installed or failed to import")

        self.jobs_per_site = int(jobs_per_site)
        self.concurrency = int(concurrency)
        self.hours_old = int(hours_old)

        # Country handling: Glassdoor must always have a real country.
        # If allowed_countries is provided (from YAML locations.allowed_countries),
        # choose the first one; otherwise fall back to DEFAULT_COUNTRY.
        if allowed_countries and isinstance(allowed_countries, list):
            self.glassdoor_country = str(allowed_countries[0]).lower()
        else:
            self.glassdoor_country = DEFAULT_COUNTRY

        LOG.info(
            "JobSpyAdapter initialized | sites=%s | jobs_per_site=%s | hours_old=%s | glassdoor_country=%s",
            [s.value for s in ALLOWED_SITES],
            self.jobs_per_site,
            self.hours_old,
            self.glassdoor_country,
        )

    # ---------------------------------------------------------------------- #
    # PUBLIC API
    # ---------------------------------------------------------------------- #

    async def run(self, playwright_manager=None) -> List[Dict[str, Any]]:  # noqa: ARG002
        """
        Async entry point called by ScraperEngine.

        Parameters
        ----------
        playwright_manager : Any
            Ignored. Present for interface compatibility with Playwright scrapers.

        Returns
        -------
        List[Dict[str, Any]]
            Raw job dictionaries with keys:
            - title
            - company
            - location
            - job_url
            - description
            - source
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._run_sync)

    # ---------------------------------------------------------------------- #
    # INTERNAL SYNC EXECUTION
    # ---------------------------------------------------------------------- #

    def _run_sync(self) -> List[Dict[str, Any]]:
        """
        Synchronous execution wrapper around JobSpy.

        Notes
        -----
        - JobSpy is synchronous internally, so this runs in a threadpool when
          invoked from the async ScraperEngine.
        - Returns RAW job dictionaries without normalization or dedupe.
        - Extracts all available fields from JobSpy DataFrame for comprehensive data.
        - Runs each ALLOWED_SITES entry independently so a failure on one site
          does not impact others.
        """
        if not JOBSPY_AVAILABLE:
            LOG.error("JobSpy is not available at runtime.")
            return []

        rows: List[Dict[str, Any]] = []

        for site in ALLOWED_SITES:
            site_name = getattr(site, "value", str(site))
            try:
                country_arg = self._resolve_country_for_site(site)
                LOG.info(
                    "Running JobSpy for site=%s | results=%d | hours_old=%d | country_indeed=%s",
                    site_name,
                    self.jobs_per_site,
                    self.hours_old,
                    country_arg,
                )

                df: pd.DataFrame = scrape_jobs(
                    site_name=[site],
                    search_term=None,  # Broad discovery; engine filters later
                    is_remote=True,
                    results_wanted=self.jobs_per_site,
                    country_indeed=country_arg,
                    hours_old=self.hours_old,
                    verbose=0,
                )

                if df is None or df.empty:
                    LOG.warning("JobSpy returned empty DataFrame for site=%s", site_name)
                    continue

                records = df.to_dict(orient="records")
                site_count = 0
                for record in records:
                    job_dict = _build_job_dict(record)
                    if not job_dict:
                        continue
                    rows.append(job_dict)
                    site_count += 1

                LOG.info("JobSpy site=%s returned %d usable raw jobs", site_name, site_count)

            except ImportError as e:
                LOG.error("JobSpy import failed for site=%s: %s", site_name, e)
            except AttributeError as e:
                LOG.error(
                    "JobSpy attribute error for site=%s: %s. Version may be incompatible.",
                    site_name,
                    e,
                    exc_info=True,
                )
            except ValueError as e:
                LOG.error(
                    "JobSpy value error for site=%s: %s. Check parameters.",
                    site_name,
                    e,
                    exc_info=True,
                )
            except Exception as e:  # pragma: no cover - safety net
                LOG.error(
                    "JobSpy scrape failed for site=%s with unexpected error: %s",
                    site_name,
                    e,
                    exc_info=True,
                )

        LOG.info("JobSpy returned %d total raw jobs across all sites", len(rows))
        return rows

    # ------------------------------------------------------------------ #
    # INTERNALS
    # ------------------------------------------------------------------ #

    def _resolve_country_for_site(self, site: "Site") -> str:
        """
        Decide which `country_indeed` argument to use for a given JobSpy site.

        Requirements:
        - Glassdoor must always use a real country (never worldwide).
        - LinkedIn, Indeed, ZipRecruiter may use worldwide.
        """
        site_name = getattr(site, "value", "").lower()

        if site_name in {"linkedin", "indeed", "zip_recruiter", "ziprecruiter"}:
            # Let JobSpy treat this as worldwide; invalid country strings are
            # safely patched by Country.from_string.
            return "worldwide"

        if site_name == "glassdoor":
            # Enforce a specific, real country to satisfy Glassdoor.
            return self.glassdoor_country

        # Fallback for any unexpected site
        return DEFAULT_COUNTRY

