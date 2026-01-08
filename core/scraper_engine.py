"""
core/scraper_engine.py

ENTERPRISE-GRADE JOB SCRAPER ENGINE
===================================

Production-ready, deterministic job scraper engine for job discovery and
pre-qualification.

Supported Sources:
- JobSpy (LinkedIn, Indeed, Glassdoor, ZipRecruiter)
- Jooble Official API
- Remotive Official API
- Google Jobs via SerpAPI (optional)
- Playwright-based scrapers (external modules) with optional proxy support

Key Features:
- Comprehensive normalization to unified schema
- Deterministic filtering based on job_filters.yaml
- Rule-based relevance scoring (0-100) driven entirely from YAML
- Deduplication using title|company|url hash
- Resource-aware scraping (SerpAPI credit tracking)
- Multiple output formats (DataFrame, JSON, ingestion payload)
- Structured logging and fail-soft error handling
- Static proxy pool discovery & rotation via WEBSHARE_PROXY_* env vars
"""

from __future__ import annotations

import os
import json
import http.client
import asyncio
import logging
import hashlib
import time
import re
from datetime import datetime, timezone
from typing import Dict, List, Any, Set, Optional, Tuple
from dataclasses import dataclass, asdict, field
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv
import yaml
import requests
import pandas as pd
from dateutil import parser as date_parser

from core.jobspy_adapter import JobSpyAdapter

# ================================================================================
# LOGGING
# ================================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler()],
)

LOG = logging.getLogger("scraper_engine")
LOG.setLevel(logging.INFO)

# ================================================================================
# PATHS
# ================================================================================

BASE_DIR = Path(__file__).resolve().parent.parent

# NOTE: job_filters.yaml lives in core/ in your project; adjust path accordingly.
FILTERS_PATH = BASE_DIR / "core" / "job_filters.yaml"

OUTPUT_DIR = BASE_DIR / "logs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LATEST_RUN_PATH = OUTPUT_DIR / "latest_run.json"
LATEST_METRICS_PATH = OUTPUT_DIR / "latest_metrics.json"
SERPAPI_USAGE_PATH = OUTPUT_DIR / "serpapi_usage.json"

# ================================================================================
# RESOURCE MANAGER
# ================================================================================


class ResourceManager:
    """Manages SerpAPI credit tracking with monthly quota management."""

    def __init__(self, monthly_quota: int = 250, usage_file: Path = SERPAPI_USAGE_PATH):
        self.monthly_quota = monthly_quota
        self.usage_file = usage_file
        self.usage_data = self._load_usage()
        self._check_reset()

    def _load_usage(self) -> Dict[str, Any]:
        """Load usage data from file or initialize."""
        if not self.usage_file.exists():
            return {
                "current_month": datetime.now(timezone.utc).strftime("%Y-%m"),
                "credits_used": 0,
                "credits_remaining": self.monthly_quota,
                "last_reset": datetime.now(timezone.utc).isoformat(),
                "next_reset": self._get_next_reset_date().isoformat(),
                "runs_this_month": 0,
            }

        try:
            with self.usage_file.open("r", encoding="utf-8") as f:
                data = json.load(f)
            return data
        except Exception as e:  # pragma: no cover - safety net
            LOG.warning(
                "Failed to load SerpAPI usage data: %s. Initializing fresh.", e
            )
            return {
                "current_month": datetime.now(timezone.utc).strftime("%Y-%m"),
                "credits_used": 0,
                "credits_remaining": self.monthly_quota,
                "last_reset": datetime.now(timezone.utc).isoformat(),
                "next_reset": self._get_next_reset_date().isoformat(),
                "runs_this_month": 0,
            }

    def _get_next_reset_date(self) -> datetime:
        """Get next reset date (1st of next month, UTC)."""
        now = datetime.now(timezone.utc)
        if now.month == 12:
            next_month = datetime(now.year + 1, 1, 1, tzinfo=timezone.utc)
        else:
            next_month = datetime(now.year, now.month + 1, 1, tzinfo=timezone.utc)
        return next_month

    def _check_reset(self) -> None:
        """Check if monthly reset is needed and perform reset."""
        now = datetime.now(timezone.utc)
        current_month = now.strftime("%Y-%m")
        if self.usage_data.get("current_month") != current_month:
            LOG.info("Monthly SerpAPI quota reset detected. Resetting usage.")
            self.usage_data = {
                "current_month": current_month,
                "credits_used": 0,
                "credits_remaining": self.monthly_quota,
                "last_reset": now.isoformat(),
                "next_reset": self._get_next_reset_date().isoformat(),
                "runs_this_month": 0,
            }
            self._save_usage()

    def can_use_credits(self, credits: int) -> bool:
        """Check if credits can be used."""
        return self.usage_data.get("credits_remaining", 0) >= credits

    def use_credits(self, credits: int) -> bool:
        """Use credits and update tracking."""
        if not self.can_use_credits(credits):
            return False
        self.usage_data["credits_used"] = self.usage_data.get("credits_used", 0) + credits
        self.usage_data["credits_remaining"] = self.monthly_quota - self.usage_data[
            "credits_used"
        ]
        self.usage_data["runs_this_month"] = self.usage_data.get("runs_this_month", 0) + 1
        self._save_usage()
        return True

    def _save_usage(self) -> None:
        """Save usage data to file."""
        try:
            with self.usage_file.open("w", encoding="utf-8") as f:
                json.dump(self.usage_data, f, indent=2)
        except Exception as e:  # pragma: no cover
            LOG.error("Failed to save SerpAPI usage data: %s", e)

    def get_status(self) -> Dict[str, Any]:
        """Get current usage status."""
        return {
            "credits_used": self.usage_data.get("credits_used", 0),
            "credits_remaining": self.usage_data.get("credits_remaining", 0),
            "current_month": self.usage_data.get("current_month"),
            "next_reset": self.usage_data.get("next_reset"),
        }


# ================================================================================
# PROXY POOL
# ================================================================================


class ProxyPool:
    """
    Static proxy manager driven by WEBSHARE_PROXY_* environment variables.

    Behavior:
    - Discovers all env vars whose names start with WEBSHARE_PROXY_.
    - Provides round-robin proxies for browser-based scrapers.
    - Tracks consecutive failures per proxy.
    - If all proxies exceed the failure threshold, falls back to direct mode.
    - Never raises exceptions to callers; failures are logged and proxied
      calls simply receive `None` so they can run without a proxy.
    """

    def __init__(self, failure_threshold: int = 3) -> None:
        self.failure_threshold = int(failure_threshold)
        self.proxies: List[str] = self._discover_proxies()
        self.failures: Dict[str, int] = {p: 0 for p in self.proxies}
        self._index: int = 0
        self.direct_fallback: bool = False

        if self.proxies:
            LOG.info(
                "ProxyPool initialized with %d proxies | failure_threshold=%d",
                len(self.proxies),
                self.failure_threshold,
            )
        else:
            LOG.info("ProxyPool initialized with no proxies. Running direct-only.")

    def _discover_proxies(self) -> List[str]:
        proxies: List[str] = []
        for key, value in os.environ.items():
            if key.startswith("WEBSHARE_PROXY_") and value:
                proxies.append(value.strip())
        return proxies

    def get_next_proxy(self) -> Optional[str]:
        """
        Return the next proxy URL in round-robin order.

        - If in direct fallback mode, returns None.
        - If no proxies are configured, returns None.
        """
        if self.direct_fallback or not self.proxies:
            return None

        # Filter out proxies over failure threshold
        healthy = [p for p in self.proxies if self.failures.get(p, 0) < self.failure_threshold]
        if not healthy:
            LOG.warning(
                "All proxies exceeded failure threshold. Entering direct fallback mode."
            )
            self.direct_fallback = True
            return None

        # Round-robin across healthy proxies
        self._index = (self._index + 1) % len(healthy)
        proxy = healthy[self._index]
        return proxy

    def report_failure(self, proxy: Optional[str]) -> None:
        """Record a failure for a given proxy."""
        if proxy is None or proxy not in self.failures:
            return
        self.failures[proxy] = self.failures.get(proxy, 0) + 1
        LOG.warning(
            "Proxy failure recorded | proxy=%s | failures=%d",
            proxy,
            self.failures[proxy],
        )
        # If all proxies are now unhealthy, switch to direct mode.
        if all(count >= self.failure_threshold for count in self.failures.values()):
            LOG.error(
                "All proxies unhealthy (>= %d failures). Switching to direct fallback.",
                self.failure_threshold,
            )
            self.direct_fallback = True

    def report_success(self, proxy: Optional[str]) -> None:
        """
        Optionally allow scrapers to reset failure counters on success.
        """
        if proxy is None or proxy not in self.failures:
            return
        if self.failures[proxy] > 0:
            self.failures[proxy] = 0
            LOG.info("Proxy success | resetting failure counter | proxy=%s", proxy)


# ================================================================================
# METRICS
# ================================================================================


@dataclass
class ScrapeMetrics:
    total_jobs_raw: int = 0
    total_jobs_unique: int = 0
    total_jobs_filtered: int = 0
    deduped_jobs: int = 0

    scrapers_succeeded: int = 0
    scrapers_failed: int = 0

    execution_time_ms: float = 0.0
    jobs_per_minute: float = 0.0

    # name -> {"count": int, "runtime_ms": float, "errors": int}
    sites_scraped: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    resource_usage: Dict[str, Any] = field(default_factory=dict)
    score_distribution: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ================================================================================
# NORMALIZATION
# ================================================================================


def _normalize_string(s: str) -> str:
    """Normalize string for hashing: lowercase, strip, remove special chars."""
    if not s:
        return ""
    s = " ".join(s.lower().strip().split())
    s = re.sub(r"[^\w\s]", "", s)
    return s


def _normalize_url(url: str) -> str:
    """Normalize URL: remove query params and fragments for deduplication."""
    if not url:
        return ""
    try:
        parsed = urlparse(url)
        normalized = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
        normalized = normalized.rstrip("/")
        return normalized.lower()
    except Exception:
        return url.lower()


def _generate_job_hash(title: str, company: str, url: str) -> str:
    """Generate deterministic hash for job deduplication."""
    normalized_title = _normalize_string(title)
    normalized_company = _normalize_string(company)
    normalized_url = _normalize_url(url)
    key = f"{normalized_title}|{normalized_company}|{normalized_url}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def _parse_date(date_str: Optional[str]) -> Optional[datetime]:
    """Parse date string to datetime (UTC)."""
    if not date_str:
        return None
    try:
        dt = date_parser.parse(str(date_str))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def _extract_experience(text: str) -> Tuple[Optional[int], Optional[int]]:
    """Extract experience range from text (years)."""
    if not text:
        return None, None
    text_lower = str(text).lower()
    patterns = [
        r"(\d+)\s*-\s*(\d+)\s*years?",
        r"(\d+)\s*\+\s*years?",
        r"(\d+)\s*years?",
    ]
    for pattern in patterns:
        match = re.search(pattern, text_lower)
        if match:
            if len(match.groups()) == 2:
                return int(match.group(1)), int(match.group(2))
            if "+" in match.group(0):
                return int(match.group(1)), None
            years = int(match.group(1))
            return years, years
    return None, None


def _extract_salary(text: str) -> Tuple[Optional[float], Optional[float], Optional[str]]:
    """Extract salary range and currency from text."""
    if not text:
        return None, None, None
    text_lower = str(text).lower()

    currency_map = {
        "$": "USD",
        "usd": "USD",
        "€": "EUR",
        "eur": "EUR",
        "£": "GBP",
        "gbp": "GBP",
        "₹": "INR",
        "inr": "INR",
        "cad": "CAD",
        "aud": "AUD",
        "sgd": "SGD",
        "chf": "CHF",
    }

    currency = None
    for symbol, curr in currency_map.items():
        if symbol in text_lower:
            currency = curr
            break

    patterns = [
        r"\$?\s*(\d+(?:,\d{3})*(?:\.\d+)?)\s*-\s*\$?\s*(\d+(?:,\d{3})*(?:\.\d+)?)",
        r"\$?\s*(\d+(?:,\d{3})*(?:\.\d+)?)\s*k",
        r"\$?\s*(\d+(?:,\d{3})*(?:\.\d+)?)",
    ]

    for pattern in patterns:
        match = re.search(pattern, text_lower)
        if not match:
            continue
        groups = match.groups()
        if len(groups) == 2:
            min_sal = float(groups[0].replace(",", ""))
            max_sal = float(groups[1].replace(",", ""))
            if min_sal < 1000:
                min_sal *= 1000
            if max_sal < 1000:
                max_sal *= 1000
            return min_sal, max_sal, currency or "USD"
        if "k" in match.group(0):
            sal = float(groups[0].replace(",", "")) * 1000
            return sal, None, currency or "USD"
        sal = float(groups[0].replace(",", ""))
        if sal < 1000:
            sal *= 1000
        return sal, None, currency or "USD"

    return None, None, None


def _detect_remote_type(location: str) -> str:
    """Detect remote type from location string."""
    if not location:
        return "unknown"
    location_lower = location.lower()
    if any(term in location_lower for term in ["remote", "work from home", "wfh", "anywhere"]):
        return "remote"
    if "hybrid" in location_lower:
        return "hybrid"
    return "onsite"


class Normalizer:
    """Comprehensive job normalizer to unified schema."""

    @staticmethod
    def normalize(raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Normalize raw job data to unified schema."""
        try:
            title = str(raw.get("title") or "").strip()
            company = str(
                raw.get("company") or raw.get("company_name") or ""
            ).strip()
            url = str(raw.get("job_url") or raw.get("url") or "").strip()
            location = str(raw.get("location") or "Remote").strip()
            description = str(
                raw.get("description") or raw.get("snippet") or ""
            ).strip()
            source = raw.get("source", "unknown")

            if not title or not url:
                return None

            job_id = _generate_job_hash(title, company, url)

            posted_date = _parse_date(
                raw.get("posted_date") or raw.get("date_posted")
            )
            if not posted_date:
                posted_date = _parse_date(raw.get("posted_at"))

            exp_text = (
                raw.get("experience")
                or raw.get("experience_level")
                or description
            )
            exp_min, exp_max = _extract_experience(exp_text)

            salary_text = raw.get("salary") or raw.get("compensation") or description
            salary_min, salary_max, salary_currency = _extract_salary(salary_text)

            remote_type = _detect_remote_type(location)

            job_type = raw.get("job_type") or raw.get("employment_type") or "full-time"
            if isinstance(job_type, str):
                job_type = job_type.lower()

            required_skills = raw.get("required_skills") or raw.get("skills") or []
            if isinstance(required_skills, str):
                required_skills = []

            preferred_skills = raw.get("preferred_skills") or []

            normalized = {
                # Core fields
                "job_id": job_id,
                "title": title[:200],
                "company": company[:100],
                "location": location[:100],
                "job_url": url,
                "description": description[:2500],
                "source": source,
                "scraped_at": datetime.now(timezone.utc).isoformat(),
                # Compensation
                "salary_min": salary_min,
                "salary_max": salary_max,
                "salary_currency": salary_currency,
                # Experience
                "experience_min": exp_min,
                "experience_max": exp_max,
                "experience_level": raw.get("experience_level"),
                # Job details
                "job_type": job_type,
                "posted_date": posted_date.isoformat() if posted_date else None,
                "application_deadline": None,
                "remote_type": remote_type,
                "employment_type": job_type,
                # Company info
                "company_size": raw.get("company_size"),
                "company_url": raw.get("company_url"),
                "industry": raw.get("industry"),
                # Application
                "application_url": raw.get("application_url") or url,
                "application_method": raw.get("application_method"),
                # Skills
                "required_skills": required_skills
                if isinstance(required_skills, list)
                else [],
                "preferred_skills": preferred_skills
                if isinstance(preferred_skills, list)
                else [],
                "education_required": raw.get("education_required"),
                # Additional
                "benefits": raw.get("benefits") or [],
                "visa_sponsorship": raw.get("visa_sponsorship"),
            }

            return normalized
        except Exception as e:  # pragma: no cover
            LOG.warning("Failed to normalize job: %s", e)
            return None


# ================================================================================
# FILTER ENGINE
# ================================================================================


class FilterEngine:
    """
    Deterministic filtering based on job_filters.yaml.

    All business rules (hard filters, thresholds) are derived from YAML.
    """

    def __init__(self, filters_path: Path):
        self.filters_data = self._load_filters(filters_path)

        # Search criteria
        search_criteria = self.filters_data.get("search_criteria", {})
        self.required_keywords = [
            k.lower()
            for k in search_criteria.get("required_keywords", [])
            if isinstance(k, str)
        ]

        # Exclusions
        exclusions = self.filters_data.get("exclusions", {})
        self.exclude_if_rules: List[str] = [
            str(rule).lower() for rule in exclusions.get("exclude_if", [])
        ]
        self.technical_exclusions: List[str] = [
            str(rule).lower() for rule in exclusions.get("technical_exclusions", [])
        ]

        # Experience rules
        experience_cfg = self.filters_data.get("experience", {})
        self.experience_min_years: Optional[int] = experience_cfg.get("minimum_years")
        self.experience_max_years: Optional[int] = experience_cfg.get("maximum_years")
        # Soft extension: allow up to +2 years if scoring is high enough.
        self.experience_soft_extension: int = 2

        # Compensation rules
        compensation_cfg = self.filters_data.get("compensation", {})
        self.salary_minimum: Optional[float] = compensation_cfg.get("minimum_salary")
        self.salary_preferred_minimum: Optional[float] = compensation_cfg.get(
            "preferred_minimum"
        )

        # Location rules
        locations_cfg = self.filters_data.get("locations", {})
        self.allowed_countries: List[str] = locations_cfg.get("allowed_countries", [])
        self.excluded_countries: List[str] = locations_cfg.get(
            "excluded_countries", []
        )
        self.remote_only: bool = bool(locations_cfg.get("remote_only", False))
        self.hybrid_acceptable: bool = bool(
            locations_cfg.get("hybrid_acceptable", True)
        )
        self.on_site_acceptable: bool = bool(
            locations_cfg.get("on_site_acceptable", False)
        )

        # Freshness rules derived from exclusions "posted > N days ago" style.
        self.max_days_old: Optional[int] = self._derive_max_days_old()

        LOG.info(
            "FilterEngine initialized | required_keywords=%d | salary_min=%s | max_days_old=%s",
            len(self.required_keywords),
            self.salary_minimum,
            self.max_days_old,
        )

    def _load_filters(self, path: Path) -> Dict[str, Any]:
        """Load job filters from YAML."""
        if not path.exists():
            LOG.warning("job_filters.yaml not found at %s. Using empty config.", path)
            return {}
        try:
            with path.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            return data or {}
        except Exception as e:  # pragma: no cover
            LOG.error("Failed to parse job_filters.yaml: %s. Using empty config.", e)
            return {}

    def _derive_max_days_old(self) -> Optional[int]:
        """
        Infer maximum allowed job age in days from exclusion rules.

        Example rule: "posted > 21 days ago" -> max_days_old = 21
        """
        candidates: List[int] = []
        for rule in self.exclude_if_rules:
            match = re.search(r"posted\s*>\s*(\d+)\s*days", rule)
            if match:
                candidates.append(int(match.group(1)))
        if not candidates:
            return None
        return min(candidates)

    # ------------------------------------------------------------------ #
    # HARD FILTERING
    # ------------------------------------------------------------------ #

    def should_exclude(self, job: Dict[str, Any]) -> bool:
        """
        Check if job should be excluded by hard filters only.

        Business rules:
        - Missing title or URL -> exclude.
        - Age beyond configured days -> exclude.
        - Violates exclusion rules (e.g., sales/marketing, non-technical).
        - Lacks required keywords.
        - Salary below hard floor (INR only, as defined in YAML).
        - Experience above hard maximum (years > 5).
        - Remote / location constraints.
        """
        if not job.get("title") or not job.get("job_url"):
            return True

        # Date filter
        if self.max_days_old is not None and job.get("posted_date"):
            posted_date = _parse_date(job["posted_date"])
            if posted_date:
                days_old = (datetime.now(timezone.utc) - posted_date).days
                if days_old > self.max_days_old:
                    return True

        text = f"{job.get('title', '')} {job.get('description', '')}".lower()

        # Exclusion rules: look for strongly negative signals
        for rule in self.exclude_if_rules + self.technical_exclusions:
            # Very lightweight: if a meaningful token is present, exclude.
            tokens = [tok for tok in re.split(r"\s+", rule) if tok and tok.isalpha()]
            if not tokens:
                continue
            if all(tok in text for tok in tokens[:2]):
                return True

        # Required keywords
        if self.required_keywords:
            if not any(keyword in text for keyword in self.required_keywords):
                return True

        # Salary hard floor (INR only, per exclusions rule in YAML)
        if self.salary_minimum is not None:
            salary_min = job.get("salary_min")
            currency = job.get("salary_currency") or "INR"
            if (
                salary_min is not None
                and currency.upper() == "INR"
                and salary_min < float(self.salary_minimum)
            ):
                return True

        # Experience hard maximum
        if self.experience_max_years is not None:
            exp_max = job.get("experience_max")
            if exp_max is not None and exp_max > int(self.experience_max_years):
                return True

        # Location / remote constraints
        remote_type = (job.get("remote_type") or "unknown").lower()
        if self.remote_only and remote_type == "onsite":
            return True
        if not self.hybrid_acceptable and remote_type == "hybrid":
            return True

        return False


# ================================================================================
# SCORING ENGINE
# ================================================================================


class ScoringEngine:
    """
    YAML-driven relevance scoring on a 0-100 scale.

    All weights and thresholds are pulled from `ai_scoring` in job_filters.yaml.
    """

    def __init__(self, filters_data: Dict[str, Any]) -> None:
        ai_scoring = filters_data.get("ai_scoring", {})
        weights_cfg = ai_scoring.get("weights", {})

        self.job_title_weight: float = float(weights_cfg.get("job_title_match", 0.3))
        self.skills_weight: float = float(weights_cfg.get("skills_match", 0.25))
        self.salary_weight: float = float(weights_cfg.get("salary_match", 0.15))
        self.location_weight: float = float(weights_cfg.get("location_match", 0.10))
        self.company_weight: float = float(weights_cfg.get("company_match", 0.05))
        self.experience_weight: float = float(weights_cfg.get("experience_match", 0.10))
        self.industry_weight: float = float(weights_cfg.get("industry_match", 0.05))

        self.min_application_score: float = float(
            ai_scoring.get("minimum_application_score", 60)
        )
        self.auto_apply_threshold: float = float(
            ai_scoring.get("auto_apply_threshold", 80)
        )
        self.manual_review_threshold: float = float(
            ai_scoring.get("manual_review_threshold", 70)
        )

        locations_cfg = filters_data.get("locations", {})
        self.preferred_locations: List[str] = [
            str(loc).lower()
            for loc in locations_cfg.get("preferred", [])
            if isinstance(loc, str)
        ]

        industries_cfg = filters_data.get("industries", {})
        self.preferred_industries: List[str] = [
            str(ind).lower()
            for ind in industries_cfg.get("preferred", [])
            if isinstance(ind, str)
        ]

        companies_cfg = filters_data.get("companies", {})
        self.priority_companies: List[str] = [
            str(c).lower() for c in companies_cfg.get("priority_companies", [])
        ]

        search_criteria = filters_data.get("search_criteria", {})
        self.job_titles: List[str] = [
            str(t).lower() for t in search_criteria.get("job_titles", [])
        ]
        self.preferred_keywords: List[str] = [
            str(k).lower() for k in search_criteria.get("preferred_keywords", [])
        ]

        LOG.info(
            "ScoringEngine initialized | auto_apply=%s | manual_review=%s | min_score=%s",
            self.auto_apply_threshold,
            self.manual_review_threshold,
            self.min_application_score,
        )

    # ------------------------------------------------------------------ #
    # SCORING
    # ------------------------------------------------------------------ #

    def calculate_score(self, job: Dict[str, Any]) -> float:
        """
        Calculate relevance score (0-100) using YAML-driven weights.

        Criteria:
        - Title match against configured job_titles.
        - Skills match from preferred_keywords.
        - Salary proximity to preferred minimum / target.
        - Location alignment against preferred locations and remote type.
        - Company match (priority companies / tech-y names).
        - Experience alignment with 0-5 year band and soft 4-5 year extension.
        - Industry match with preferred industries.
        """
        score = 0.0

        title = (job.get("title") or "").lower()
        description = (job.get("description") or "").lower()
        location = (job.get("location") or "").lower()
        company = (job.get("company") or "").lower()
        industry = (job.get("industry") or "").lower()
        remote_type = (job.get("remote_type") or "").lower()

        # Title match
        title_match = 0.0
        if self.job_titles:
            if any(t in title for t in self.job_titles):
                title_match = 1.0
        score += title_match * self.job_title_weight * 100.0

        # Skills match (preferred keywords within description)
        skills_match_ratio = 0.0
        if self.preferred_keywords:
            hits = sum(1 for kw in self.preferred_keywords if kw in description)
            skills_match_ratio = min(1.0, hits / max(5, len(self.preferred_keywords)))
        score += skills_match_ratio * self.skills_weight * 100.0

        # Salary match (soft preference)
        salary_min = job.get("salary_min")
        salary_currency = (job.get("salary_currency") or "INR").upper()
        salary_match_ratio = 0.0
        # Use same thresholds as FilterEngine for consistency
        # Hard minimum is enforced earlier; here we only award proximity.
        if salary_min and isinstance(salary_min, (int, float)):
            if salary_currency == "INR":
                # Assume 800000 is "good", 1500000 is "target" from YAML.
                if salary_min >= 1500000:
                    salary_match_ratio = 1.0
                elif salary_min >= 800000:
                    salary_match_ratio = 0.7
                else:
                    salary_match_ratio = 0.3
            else:
                salary_match_ratio = 0.5
        score += salary_match_ratio * self.salary_weight * 100.0

        # Location & remote match
        location_ratio = 0.0
        if remote_type == "remote":
            location_ratio = 1.0
        elif remote_type == "hybrid":
            location_ratio = 0.7
        else:
            location_ratio = 0.3

        # Boost if location string contains any preferred location
        if self.preferred_locations and any(
            pref.lower() in location for pref in self.preferred_locations
        ):
            location_ratio = min(1.0, location_ratio + 0.2)

        score += location_ratio * self.location_weight * 100.0

        # Company match
        company_ratio = 0.0
        if self.priority_companies and any(
            priority in company for priority in self.priority_companies
        ):
            company_ratio = 1.0
        elif any(term in company for term in ["labs", "ai", "ml", "data", "tech"]):
            company_ratio = 0.7
        score += company_ratio * self.company_weight * 100.0

        # Experience alignment
        exp_max = job.get("experience_max")
        experience_ratio = 0.0
        if exp_max is not None:
            if exp_max <= 1:
                experience_ratio = 1.0
            elif exp_max <= 3:
                experience_ratio = 0.8
            elif exp_max <= 5:
                experience_ratio = 0.6
            else:
                experience_ratio = 0.2
        score += experience_ratio * self.experience_weight * 100.0

        # Industry match
        industry_ratio = 0.0
        if self.preferred_industries and any(
            ind in industry for ind in self.preferred_industries
        ):
            industry_ratio = 1.0
        score += industry_ratio * self.industry_weight * 100.0

        # Clamp
        return max(0.0, min(100.0, score))

    # ------------------------------------------------------------------ #
    # DECISION
    # ------------------------------------------------------------------ #

    def classify_decision(self, score: float) -> str:
        """
        Decide application action based on YAML thresholds.

        Returns: "auto_apply", "manual_review", or "reject".
        """
        if score >= self.auto_apply_threshold:
            return "auto_apply"
        if score >= self.manual_review_threshold:
            return "manual_review"
        if score >= self.min_application_score:
            return "manual_review"
        return "reject"


# ================================================================================
# API SCRAPERS
# ================================================================================


class BaseAPIScraper:
    """Base class for API scrapers."""

    name: str = "base_api"

    def __init__(self, jobs_per_site: int):
        self.jobs_per_site = jobs_per_site

    async def run(self) -> List[Dict[str, Any]]:
        """Async entry point."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._run_sync)

    def _run_sync(self) -> List[Dict[str, Any]]:
        """Synchronous implementation."""
        raise NotImplementedError


class JoobleAPIScraper(BaseAPIScraper):
    """Jooble Official API scraper."""

    name = "jooble"

    def __init__(
        self,
        jobs_per_site: int,
        keywords: str = "AI engineer machine learning data scientist",
    ):
        super().__init__(jobs_per_site)
        self.api_key = os.getenv("JOOBLE_API_KEY")
        self.keywords = keywords

    def _run_sync(self) -> List[Dict[str, Any]]:
        """Scrape jobs from Jooble API."""
        if not self.api_key:
            LOG.warning("Jooble API key missing (env JOOBLE_API_KEY). Skipping.")
            return []

        host = "jooble.org"
        conn = None
        try:
            conn = http.client.HTTPConnection(host, timeout=20)
            headers = {"Content-type": "application/json"}
            body_dict = {"keywords": self.keywords, "location": ""}
            body = json.dumps(body_dict)
            conn.request("POST", f"/api/{self.api_key}", body, headers)
            response = conn.getresponse()
            if response.status != 200:
                LOG.error("Jooble HTTP %s %s", response.status, response.reason)
                return []
            data_bytes = response.read()
            data = json.loads(data_bytes.decode("utf-8"))
            jobs = data.get("jobs", [])[: self.jobs_per_site]
            results: List[Dict[str, Any]] = []
            for j in jobs:
                title = j.get("title", "")
                company = j.get("company", "")
                location = j.get("location") or j.get("location_str", "Remote")
                job_url = j.get("link") or j.get("url") or ""
                description = j.get("snippet") or j.get("description") or ""
                if not title or not job_url:
                    continue
                results.append(
                    {
                        "title": title,
                        "company": company,
                        "location": location,
                        "job_url": job_url,
                        "description": description,
                        "source": self.name,
                        "posted_date": j.get("updated"),
                    }
                )
            LOG.info("Jaoble returned %d jobs", len(results))
            return results
        except Exception as e:  # pragma: no cover
            LOG.error("Jooble API failed: %s", e, exc_info=True)
            return []
        finally:
            if conn:
                conn.close()


class RemotiveAPIScraper(BaseAPIScraper):
    """Remotive Public API scraper."""

    name = "remotive"
    endpoint = "https://remotive.io/api/remote-jobs"

    def _run_sync(self) -> List[Dict[str, Any]]:
        """Scrape jobs from Remotive API."""
        try:
            response = requests.get(self.endpoint, timeout=20)
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as e:
            LOG.error("Remotive API failed: %s", e)
            return []
        except Exception as e:  # pragma: no cover
            LOG.error("Remotive API parse failed: %s", e)
            return []

        jobs = data.get("jobs", [])[: self.jobs_per_site]
        results: List[Dict[str, Any]] = []
        for j in jobs:
            title = j.get("title", "")
            company = j.get("company_name", "")
            location = j.get("candidate_required_location") or "Remote"
            job_url = j.get("url") or ""
            description = j.get("description") or ""
            if not title or not job_url:
                continue
            results.append(
                {
                    "title": title,
                    "company": company,
                    "location": location,
                    "job_url": job_url,
                    "description": description,
                    "source": self.name,
                    "posted_date": j.get("publication_date"),
                    "job_type": j.get("job_type"),
                    "salary": j.get("salary"),
                }
            )
        LOG.info("Remotive returned %d jobs", len(results))
        return results


class SerpAPIGoogleJobsScraper(BaseAPIScraper):
    """Google Jobs via SerpAPI scraper."""

    name = "google_jobs"
    endpoint = "https://serpapi.com/search.json"

    def __init__(
        self,
        jobs_per_site: int,
        resource_manager: ResourceManager,
        query: str = (
            "AI engineer OR machine learning engineer OR data scientist remote"
        ),
    ):
        super().__init__(jobs_per_site)
        self.api_key = os.getenv("SERPAPI_API_KEY_1")
        self.query = query
        self.resource_manager = resource_manager

    def _run_sync(self) -> List[Dict[str, Any]]:
        """Scrape jobs from Google Jobs via SerpAPI."""
        if not self.api_key:
            LOG.warning(
                "SerpAPI key missing (env SERPAPI_API_KEY_1). Skipping Google Jobs."
            )
            return []

        if not self.resource_manager.can_use_credits(1):
            LOG.warning("SerpAPI quota exceeded. Skipping Google Jobs.")
            return []

        params = {
            "engine": "google_jobs",
            "q": self.query,
            "hl": "en",
            "num": self.jobs_per_site,
            "api_key": self.api_key,
        }

        try:
            response = requests.get(self.endpoint, params=params, timeout=25)
            response.raise_for_status()
            data = response.json()
            self.resource_manager.use_credits(1)
            jobs = data.get("jobs_results", [])[: self.jobs_per_site]
            results: List[Dict[str, Any]] = []
            for j in jobs:
                title = j.get("title", "")
                company = j.get("company_name", "")
                location = j.get("location") or "Remote"
                job_url = j.get("link") or ""
                description = j.get("description") or ""
                if not title or not job_url:
                    continue
                detect_extensions = j.get("detected_extensions", {})
                posted_date = detect_extensions.get("posted_at")
                results.append(
                    {
                        "title": title,
                        "company": company,
                        "location": location,
                        "job_url": job_url,
                        "description": description,
                        "source": self.name,
                        "posted_date": posted_date,
                        "job_type": j.get("schedule_type"),
                    }
                )
            LOG.info("SerpAPI Google Jobs returned %d jobs", len(results))
            return results
        except requests.RequestException as e:
            LOG.error("SerpAPI Google Jobs failed: %s", e)
            return []
        except Exception as e:  # pragma: no cover
            LOG.error("SerpAPI Google Jobs parse failed: %s", e)
            return []


# ================================================================================
# SCRAPER ENGINE
# ================================================================================


class ScraperEngine:
    """Master orchestrator for job scraping."""

    def __init__(self, filters_path: Path = FILTERS_PATH):
        load_dotenv("narad.env")

        self.results: List[Dict[str, Any]] = []
        self.seen_job_ids: Set[str] = set()
        self.metrics = ScrapeMetrics()
        self.scrapers: List[Any] = []

        # Initialize components
        self.resource_manager = ResourceManager(monthly_quota=250)
        self.proxy_pool = ProxyPool()

        # Load filters once; pass data to FilterEngine + ScoringEngine
        self.filter_engine = FilterEngine(filters_path)
        self.normalizer = Normalizer()
        self.scoring_engine = ScoringEngine(self.filter_engine.filters_data)

        # Initialize scrapers
        self._init_scrapers()

        LOG.info("ScraperEngine initialized with %d scrapers", len(self.scrapers))

    # ------------------------------------------------------------------ #
    # SCRAPER REGISTRATION
    # ------------------------------------------------------------------ #

    def _init_scrapers(self) -> None:
        """Initialize all enabled scrapers."""
        scrapers: List[Any] = []

        # Extract allowed countries from YAML to feed JobSpy for Glassdoor.
        allowed_countries = self.filter_engine.allowed_countries

        # JobSpy adapter (LinkedIn, Indeed, Glassdoor, ZipRecruiter)
        try:
            scrapers.append(
                JobSpyAdapter(
                    jobs_per_site=20,
                    concurrency=4,
                    hours_old=168,  # 7 days
                    allowed_countries=allowed_countries,
                )
            )
            LOG.info("✓ JobSpy adapter initialized")
        except Exception as e:  # pragma: no cover
            LOG.warning("JobSpyAdapter not available: %s", e)

        # Jooble API
        if os.getenv("JOOBLE_API_KEY"):
            scrapers.append(JoobleAPIScraper(jobs_per_site=20))
            LOG.info("✓ Jooble API scraper initialized")
        else:
            LOG.warning("Jooble API key not found. Skipping Jooble scraper.")

        # Remotive API (no key required)
        scrapers.append(RemotiveAPIScraper(jobs_per_site=20))
        LOG.info("✓ Remotive API scraper initialized")

        # SerpAPI Google Jobs (optional)
        if os.getenv("SERPAPI_API_KEY_2"):
            scrapers.append(
                SerpAPIGoogleJobsScraper(
                    jobs_per_site=20,
                    resource_manager=self.resource_manager,
                )
            )
            LOG.info("✓ SerpAPI Google Jobs initialized")
        else:
            LOG.info("SerpAPI key not found. Skipping Google Jobs scraper.")

        self.scrapers = scrapers
        self.metrics.sites_scraped = {
            getattr(s, "name", "unknown"): {"count": 0, "runtime_ms": 0.0, "errors": 0}
            for s in scrapers
        }

    # ------------------------------------------------------------------ #
    # MAIN EXECUTION
    # ------------------------------------------------------------------ #

    async def run(self) -> Tuple[List[Dict[str, Any]], ScrapeMetrics]:
        """Run all scrapers and return normalized, filtered, scored jobs."""
        start_time = time.time()
        LOG.info("🚀 Starting scrape run with %d scrapers", len(self.scrapers))

        semaphore = asyncio.Semaphore(10)  # Global concurrency limit

        async def _execute(scraper: Any) -> List[Dict[str, Any]]:
            name = getattr(scraper, "name", "unknown")
            metrics_entry = self.metrics.sites_scraped.get(
                name, {"count": 0, "runtime_ms": 0.0, "errors": 0}
            )

            async with semaphore:
                t0 = time.time()
                try:
                    LOG.info("Running %s...", name)

                    # Browser-based scrapers MAY accept proxy_url; JobSpy & API scrapers ignore it.
                    proxy_url = self.proxy_pool.get_next_proxy()
                    run_sig = getattr(scraper, "run")
                    if run_sig.__code__.co_argcount >= 2:
                        raw_jobs = await run_sig(proxy_url)  # type: ignore[arg-type]
                    else:
                        raw_jobs = await run_sig()

                    runtime_ms = (time.time() - t0) * 1000.0
                    metrics_entry["runtime_ms"] = runtime_ms
                    metrics_entry["count"] = len(raw_jobs)
                    self.metrics.sites_scraped[name] = metrics_entry

                    self.metrics.total_jobs_raw += len(raw_jobs)
                    LOG.info("✅ %s: %d jobs (%.0f ms)", name, len(raw_jobs), runtime_ms)

                    # Proxy success feedback (if applicable)
                    if proxy_url:
                        self.proxy_pool.report_success(proxy_url)

                    return raw_jobs
                except Exception as e:  # pragma: no cover
                    runtime_ms = (time.time() - t0) * 1000.0
                    metrics_entry["runtime_ms"] = runtime_ms
                    metrics_entry["errors"] = metrics_entry.get("errors", 0) + 1
                    self.metrics.sites_scraped[name] = metrics_entry

                    LOG.error("❌ %s failed: %s", name, e, exc_info=True)
                    self.metrics.scrapers_failed += 1

                    # Proxy failure feedback
                    proxy_url = locals().get("proxy_url")
                    if proxy_url:
                        self.proxy_pool.report_failure(proxy_url)

                    return []

        batches = await asyncio.gather(*[_execute(s) for s in self.scrapers])
        self.metrics.scrapers_succeeded = len(self.scrapers) - self.metrics.scrapers_failed

        # Process all raw jobs
        all_raw_jobs: List[Dict[str, Any]] = []
        for batch in batches:
            all_raw_jobs.extend(batch)

        for raw_job in all_raw_jobs:
            normalized = self.normalizer.normalize(raw_job)
            if not normalized:
                continue

            job_id = normalized["job_id"]
            if job_id in self.seen_job_ids:
                self.metrics.deduped_jobs += 1
                continue
            self.seen_job_ids.add(job_id)

            # Hard filters
            if self.filter_engine.should_exclude(normalized):
                continue

            # YAML-driven scoring
            score = self.scoring_engine.calculate_score(normalized)
            decision = self.scoring_engine.classify_decision(score)

            # Only keep jobs above minimum_application_score
            if decision == "reject":
                continue

            normalized["relevance_score"] = round(score, 2)
            normalized["decision"] = decision

            self.results.append(normalized)

        # Metrics
        self.metrics.total_jobs_unique = len(self.seen_job_ids)
        self.metrics.total_jobs_filtered = len(self.results)
        self.metrics.execution_time_ms = (time.time() - start_time) * 1000.0

        elapsed_minutes = max(1e-6, (time.time() - start_time) / 60.0)
        self.metrics.jobs_per_minute = self.metrics.total_jobs_filtered / elapsed_minutes

        score_ranges = {
            "90-100": 0,
            "80-89": 0,
            "70-79": 0,
            "60-69": 0,
            "50-59": 0,
            "0-49": 0,
        }
        for job in self.results:
            s = job.get("relevance_score", 0)
            if s >= 90:
                score_ranges["90-100"] += 1
            elif s >= 80:
                score_ranges["80-89"] += 1
            elif s >= 70:
                score_ranges["70-79"] += 1
            elif s >= 60:
                score_ranges["60-69"] += 1
            elif s >= 50:
                score_ranges["50-59"] += 1
            else:
                score_ranges["0-49"] += 1
        self.metrics.score_distribution = score_ranges

        self.metrics.resource_usage = self.resource_manager.get_status()

        LOG.info(
            "🎉 Scrape completed: %d filtered jobs (from %d raw, %d deduped) in %.0f ms (%.1f jobs/min)",
            self.metrics.total_jobs_filtered,
            self.metrics.total_jobs_raw,
            self.metrics.deduped_jobs,
            self.metrics.execution_time_ms,
            self.metrics.jobs_per_minute,
        )

        self._save_results()
        self._save_metrics()

        return self.results, self.metrics

    # ------------------------------------------------------------------ #
    # PERSISTENCE
    # ------------------------------------------------------------------ #

    def _save_results(self) -> None:
        """Save results to JSON file."""
        try:
            with LATEST_RUN_PATH.open("w", encoding="utf-8") as f:
                json.dump(self.results, f, indent=2)
            LOG.info("Results saved to %s", LATEST_RUN_PATH)
        except Exception as e:  # pragma: no cover
            LOG.error("Failed to save results: %s", e)

    def _save_metrics(self) -> None:
        """Save metrics to JSON file."""
        try:
            with LATEST_METRICS_PATH.open("w", encoding="utf-8") as f:
                json.dump(self.metrics.to_dict(), f, indent=2)
            LOG.info("Metrics saved to %s", LATEST_METRICS_PATH)
        except Exception as e:  # pragma: no cover
            LOG.error("Failed to save metrics: %s", e)

    # ------------------------------------------------------------------ #
    # PUBLIC ACCESSORS
    # ------------------------------------------------------------------ #

    def get_dataframe(self) -> pd.DataFrame:
        """Get results as Pandas DataFrame."""
        if not self.results:
            return pd.DataFrame()
        return pd.DataFrame(self.results)

    def get_json(self) -> List[Dict[str, Any]]:
        """Get results as JSON-serializable list."""
        return self.results

    def get_ingestion_payload(self) -> Dict[str, Any]:
        """Get results as ingestion-ready payload."""
        return {
            "jobs": self.results,
            "metadata": {
                "total_jobs": len(self.results),
                "scraped_at": datetime.now(timezone.utc).isoformat(),
                "metrics": self.metrics.to_dict(),
            },
        }

    def run_sync(self) -> Tuple[List[Dict[str, Any]], ScrapeMetrics]:
        """Synchronous wrapper for run()."""
        return asyncio.run(self.run())


# ================================================================================
# CLI ENTRY
# ================================================================================


async def main() -> None:
    """Main entry point."""
    engine = ScraperEngine()
    jobs, metrics = await engine.run()
    print("\n" + "=" * 60)
    print("SCRAPER COMPLETED")
    print(f"📊 {metrics.total_jobs_filtered} FILTERED JOBS")
    print(f"📥 {metrics.total_jobs_raw} RAW JOBS")
    print(f"🔄 {metrics.deduped_jobs} DEDUPED")
    print(f"⏱️ {metrics.execution_time_ms:.0f}ms")
    print("=" * 60)
    print(f"Results saved to: {LATEST_RUN_PATH}")
    print(f"Metrics saved to: {LATEST_METRICS_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
