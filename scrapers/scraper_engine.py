"""
scrapers/scraper_engine.py

ENTERPRISE-GRADE JOB SCRAPER ENGINE
===================================

Purpose:
    Scrape and normalize job listings from multiple platforms into the
    unified job schema, returning them as plain Python dicts for AI agents
    to consume.

    This layer does NOT perform database writes and does NOT make auto-apply
    vs. manual-review routing decisions. All routing is handled exclusively
    by downstream agents.

Supported Sources:

  Phase 1 (active):
  - JobSpy (LinkedIn, Indeed) — via scrapers/jobspy_adapter.py
  - RemoteOK API
  - Himalayas API
  - Arbeitnow API
  - Jobicy API
  - Google Jobs via SerpAPI
  - Playwright-based scrapers (managed in scraper_service.py) with optional
    proxy support via WEBSHARE_PROXY_* env vars

Key Features:
- Comprehensive normalization to unified job schema
- Deterministic filtering based on platform_settings.json and user_profile.json
- Rule-based static pre-filter scoring (0.0–1.0) driven from JSON config files
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
import requests
import pandas as pd
from dateutil import parser as date_parser
from fuzzywuzzy import fuzz

from .jobspy_adapter import JobSpyAdapter
from utils.db_utils import get_db_conn
from utils.proxy_rate_limit import get_proxy_dict, get_next_proxy, reset_cycle
from config.config_loader import ConfigLoader, config_loader

__all__ = ["ScraperEngine", "ScoringEngine", "FilterEngine", "Normalizer"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler()],
)

logger = logging.getLogger(__name__)


# ================================================================================
# PROXY HELPER
# ================================================================================


def _make_proxied_request(
    url: str,
    method: str = "GET",
    **kwargs: Any,
) -> requests.Response:
    """Make an HTTP request routed through the Webshare proxy pool with rotation.

    Uses round-robin proxy selection from ``utils.proxy_ratelimit``. On
    ``ProxyError`` or ``ConnectionError`` the proxy is rotated and the request
    is retried (max 3 attempts). If all proxies fail the request is attempted
    directly as a last resort. Falls back to direct if no proxy is configured.

    Args:
        url: Target URL string.
        method: HTTP method ``"GET"`` or ``"POST"``.
        **kwargs: Additional kwargs forwarded to ``requests.get`` / ``requests.post``.

    Returns:
        ``requests.Response`` object.

    Raises:
        requests.RequestException: If all proxy attempts AND the direct fallback fail.
    """
    proxies = get_proxy_dict()
    if proxies:
        kwargs["proxies"] = proxies
    else:
        logger.debug("No proxy configured — direct request to %s", url)

    last_exc: Optional[Exception] = None
    for attempt in range(1, 4):  # max 3 attempts
        try:
            if method.upper() == "POST":
                return requests.post(url, **kwargs)
            return requests.get(url, **kwargs)
        except (requests.exceptions.ProxyError,
                requests.exceptions.ConnectionError) as e:
            last_exc = e
            logger.warning(
                "Proxy request attempt %d/3 failed for %s: %s — rotating proxy",
                attempt, url, str(e),
            )
            new_proxies = get_proxy_dict()
            if new_proxies:
                kwargs["proxies"] = new_proxies

    # All proxies failed — try direct as last resort
    logger.warning(
        "All proxy attempts failed for %s — attempting direct request", url
    )
    kwargs.pop("proxies", None)
    if method.upper() == "POST":
        return requests.post(url, **kwargs)
    return requests.get(url, **kwargs)


# ================================================================================
# SYSTEM CONFIG HELPERS
# ================================================================================


# ================================================================================
# PATHS
# ================================================================================

BASE_DIR = Path(__file__).resolve().parent.parent

OUTPUT_DIR = BASE_DIR / "logs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LATEST_RUN_PATH = OUTPUT_DIR / "latest_run.json"
LATEST_METRICS_PATH = OUTPUT_DIR / "latest_metrics.json"



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
            logger.info(
                "ProxyPool initialized with %d proxies | failure_threshold=%d",
                len(self.proxies),
                self.failure_threshold,
            )
        else:
            logger.info("ProxyPool initialized with no proxies. Running direct-only.")

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
            logger.warning(
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
        logger.warning(
            "Proxy failure recorded | proxy=%s | failures=%d",
            proxy,
            self.failures[proxy],
        )
        # If all proxies are now unhealthy, switch to direct mode.
        if all(count >= self.failure_threshold for count in self.failures.values()):
            logger.error(
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
            logger.info("Proxy success | resetting failure counter | proxy=%s", proxy)


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
#
# UNIFIED NORMALIZED JOB SCHEMA
# ==============================
# Every job dict returned by ScraperEngine.run() MUST contain these keys.
# Unknown / unavailable values should be None (or [] for list fields).
#
#   job_id            str   — SHA-256 hash of title|company|url
#   title             str
#   company           str
#   location          str
#   remote_type       str   — "remote" | "hybrid" | "onsite" | "unknown"
#   job_url           str
#   application_url   str
#   source            str   — scraper name (e.g. "linkedin", "remoteok")
#   platform          str   — defaults to source; may differ for aggregators
#   description       str
#   job_type          str   — e.g. "full-time", "contract"
#   employment_type   str   — same as job_type (alias kept for compatibility)
#   salary_min        float | None
#   salary_max        float | None
#   salary_currency   str | None
#   experience_min    int | None
#   experience_max    int | None
#   experience_level  str | None — e.g. "entry", "mid", "senior"
#   posted_date       str | None — ISO-8601
#   scraped_at        str        — ISO-8601 UTC timestamp
#   company_size      str | None
#   company_url       str | None
#   industry          str | None
#   required_skills   list[str]
#   preferred_skills  list[str]
#   benefits          list[str]
#   tags              list[str]     — tags/skills from API (e.g. RemoteOK)
#   visa_sponsorship  bool | None
#   education_required str | None
#   application_method str | None
#   normalisation_status str   — "ok" on successful normalization
#   static_score      float   — pre-filter relevance score in [0.0, 1.0]
#


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
            if min_sal < 1_000:
                min_sal *= 1000
            if max_sal < 1_000:
                max_sal *= 1000
            return min_sal, max_sal, currency or "USD"
        if "k" in match.group(0):
            sal = float(groups[0].replace(",", "")) * 1000
            return sal, None, currency or "USD"
        sal = float(groups[0].replace(",", ""))
        if sal < 1_000:
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
            title = str(
                raw.get("title") or
                raw.get("position") or
                raw.get("job_title") or
                ""
            ).strip()
            company = str(
                raw.get("company") or raw.get("company_name") or ""
            ).strip()
            url = str(raw.get("job_url") or raw.get("url") or "").strip()
            location = str(raw.get("location") or "Remote").strip()
            description = str(
                raw.get("description") or
                raw.get("body") or
                raw.get("snippet") or
                raw.get("excerpt") or
                raw.get("text") or
                ""
            ).strip()
            source = raw.get("source", "unknown")

            raw_tags = raw.get("tags") or raw.get("skills") or raw.get("required_skills") or []
            tags: list[str] = raw_tags if isinstance(raw_tags, list) else []

            # Hard requirement: title and URL must be present
            if not title or not url:
                return None

            # Drop jobs with no meaningful description
            raw_tags_check = (
                raw.get("tags") or raw.get("required_skills") or raw.get("skills") or []
            )
            if len(description) < 20 and not raw_tags_check:
                logger.debug(
                    "Dropping job '%s' — description too short (%d chars) and no tags",
                    title,
                    len(description),
                )
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
            if not isinstance(required_skills, list):
                required_skills = []

            preferred_skills = raw.get("preferred_skills") or []
            if not isinstance(preferred_skills, list):
                preferred_skills = []

            benefits = raw.get("benefits") or []
            if not isinstance(benefits, list):
                benefits = []

            normalized = {
                # Identity
                "job_id": job_id,
                "title": title[:200],
                "company": company[:100],
                "location": location[:100],
                "remote_type": remote_type,
                "job_url": url,
                "application_url": raw.get("application_url") or url,
                "source": source,
                "platform": raw.get("platform") or source,
                # Content
                "description": description[:2500],
                # Job classification
                "job_type": job_type,
                "employment_type": job_type,
                # Compensation
                "salary_min": salary_min,
                "salary_max": salary_max,
                "salary_currency": salary_currency,
                # Experience
                "experience_min": exp_min,
                "experience_max": exp_max,
                "experience_level": raw.get("experience_level"),
                # Dates
                "posted_date": posted_date.isoformat() if posted_date else None,
                "scraped_at": datetime.now(timezone.utc).isoformat(),
                # Company info
                "company_size": raw.get("company_size"),
                "company_url": raw.get("company_url"),
                "industry": raw.get("industry"),
                # Skills
                "required_skills": required_skills,
                "preferred_skills": preferred_skills,
                # Additional
                "benefits": benefits,
                "tags": tags,
                "visa_sponsorship": raw.get("visa_sponsorship"),
                "education_required": raw.get("education_required"),
                "application_method": raw.get("application_method"),
                # Schema metadata
                "normalisation_status": "ok",
                "static_score": None,
            }

            return normalized
        except Exception as e:  # pragma: no cover
            logger.warning("Failed to normalize job: %s", e)
            return None


# ================================================================================
# FILTER ENGINE
# ================================================================================


class FilterEngine:
    """
    Deterministic filtering based on platform_settings and user_profile.
    """

    def __init__(
        self,
        cfg: ConfigLoader,
    ) -> None:
        filters: dict = cfg.settings.get("job_filters", {})
        self.required_keywords: list[str] = [
            kw.lower() for kw in filters.get("required_keywords_any", [])
        ]
        self.exclude_seniority: list[str] = list(set(
            [kw.lower() for kw in filters.get("exclude_seniority_keywords", [])] +
            [kw.lower() for kw in cfg.user.get("job_preferences", {})
                                           .get("hard_filters", {})
                                           .get("exclude_seniority_keywords", [])]
        ))
        self.exclude_job_types: list[str] = [
            jt.lower() for jt in filters.get("exclude_job_types", [])
        ]
        
        logger.info(
            "FilterEngine initialized | required_keywords=%d | exclude_seniority=%d | exclude_job_types=%d",
            len(self.required_keywords),
            len(self.exclude_seniority),
            len(self.exclude_job_types),
        )

    def passes(self, job: Dict[str, Any]) -> tuple[bool, str]:
        """
        Check if job should be excluded by hard filters.
        """
        title = job.get("title", "").lower()
        description = job.get("description", "").lower()
        tags = job.get("tags", [])
        tags_text = " ".join(tags).lower() if isinstance(tags, list) else ""
        full_text = f"{title} {description} {tags_text}"

        # BUG-1: description length gate
        desc: str = job.get("description", "")
        has_tags: bool = bool(job.get("tags") or job.get("required_skills"))
        title_lower: str = job.get("title", "").lower()
        title_has_keyword: bool = any(
            kw.lower() in title_lower for kw in self.required_keywords
        )
        if len(desc) < 100 and not has_tags and not title_has_keyword:
            return False, "description_too_short"

        # 1. Required keywords check (PASSES if contains ANY)
        if self.required_keywords:
            desc_lower: str = desc.lower()
            combined_text: str = f"{title_lower} {desc_lower}"
            if not any(kw.lower() in combined_text for kw in self.required_keywords):
                return False, "missing_required_keywords"

        # 2. Seniority keyword exclusion
        if any(kw in title for kw in self.exclude_seniority):
            return False, "seniority_exclusion"

        # 3. Job type exclusion
        job_type_raw = job.get("job_type") or ""
        if isinstance(job_type_raw, list):
            job_type_val = " ".join(str(x) for x in job_type_raw).lower()
        else:
            job_type_val = str(job_type_raw).lower()
        
        if any(t in job_type_val for t in self.exclude_job_types):
            return False, "job_type_exclusion"

        return True, "ok"


# ================================================================================
# SCORING ENGINE
# ================================================================================


class ScoringEngine:
    """
    Scoring engine using JSON thresholds and user profile skills.
    """

    def __init__(
        self,
        cfg: ConfigLoader,
    ) -> None:
        prefs: dict = cfg.user.get("job_preferences", {})
        # Read search_queries from user_profile job_preferences
        self.search_queries: list[str] = prefs.get("search_queries", prefs.get("search_queries", []))
        self.seniority_keywords: list[str] = [kw.lower() for kw in prefs.get("hard_filters", {}).get("exclude_seniority_keywords", ["senior", "lead", "staff", "principal"])]
        
        # Abbreviation Expansion (F2a)
        _abbrev_map = {"ML": "Machine Learning", "AI": "Artificial Intelligence",
                       "NLP": "Natural Language Processing", "LLM": "Large Language Model"}
        _expanded: list[str] = list(self.search_queries)
        for title in self.search_queries:
            for abbr, full in _abbrev_map.items():
                if abbr in title:
                    _expanded.append(title.replace(abbr, full))
                if full in title:
                    _expanded.append(title.replace(full, abbr))
        self.search_queries = list(set(_expanded))
        
        raw_skills: dict = cfg.user.get("skills", {})
        self.all_skills: set[str] = {
            s.lower()
            for category_skills in raw_skills.values()
            if isinstance(category_skills, list)
            for s in category_skills
            if isinstance(s, str)
        }
        # user_profile.json skill values can be descriptive phrases (not atomic
        # keywords). Tokenise them so tags like "pytorch" / "llm" can match.
        _tokenised: set[str] = set()
        for phrase in self.all_skills:
            tokens = re.split(r"[\s,/()&+]+", phrase.lower())
            _tokenised.update(t.strip() for t in tokens if len(t.strip()) > 2)
        self.all_skills = _tokenised
        score_cfg: dict = cfg.settings.get("scoring_thresholds", {})
        self.min_score: float = score_cfg.get("auto_eligible", 0.65) * 100
        self.title_weight: int = int(score_cfg.get("title_weight", 40))
        self.skills_weight: int = int(score_cfg.get("skills_weight", 30))
        self.remote_bonus: int = int(score_cfg.get("remote_bonus", 10))
        self.seniority_penalty: int = int(score_cfg.get("seniority_penalty", 20))
        self.fuzzy_threshold: int = int(score_cfg.get("fuzzy_title_threshold", 80))
        self.skills_min_count: int = int(score_cfg.get("skills_min_count", 3))

        logger.info(
            "ScoringEngine initialized | target_titles=%d | skills=%d | min_score=%.1f",
            len(self.search_queries),
            len(self.all_skills),
            self.min_score,
        )

    def calculate_score(self, job: Dict[str, Any]) -> float:
        """
        Calculate relevance score (0-100).
        """
        score = 0.0

        # BUG-2 (variable shadow in fuzzy match):
        title_lower: str = job.get("title", "").lower()
        title_matched: bool = any(
            fuzz.partial_ratio(t.lower(), title_lower) >= self.fuzzy_threshold
            for t in self.search_queries
        )
        score += self.title_weight if title_matched else 0

        # BUG-3 (skills must check tags + required_skills + preferred_skills):
        job_skill_pool: set[str] = {
            s.lower() for s in (
                job.get("required_skills", []) +
                job.get("tags", []) +
                job.get("preferred_skills", [])
            ) if isinstance(s, str)
        }
        matched: set[str] = self.all_skills & job_skill_pool
        ratio: float = min(len(matched) / max(self.skills_min_count, 1), 1.0)
        score += int(self.skills_weight * ratio)

        # BUG-5 (remote check must cover remote_type field):
        is_remote: bool = (
            job.get("remote_type", "").lower() == "remote" or
            any(kw in job.get("location", "").lower()
                for kw in ("remote", "worldwide", "anywhere"))
        )
        score += self.remote_bonus if is_remote else 0

        # Seniority penalty
        title = job.get("title", "").lower()
        if any(kw in title for kw in self.seniority_keywords):
            score -= self.seniority_penalty

        return max(0.0, min(100.0, score))

    # ------------------------------------------------------------------ #
    # DECISION
    # ------------------------------------------------------------------ #

    def classify_decision(self, score: float) -> str:
        """
        Classify score into a tier string.
        """
        if score >= self.min_score:
            return "above_threshold"
        return "below_threshold"


# ================================================================================
# SCRAPER ENGINE
# ================================================================================


class ScraperEngine:
    """Master orchestrator for job scraping."""

    def __init__(
        self,
        cfg: Optional[ConfigLoader] = None,
        min_jobs_target: int = 5,
        safety_net: bool = False,
        config_path: Optional[str] = None,
    ) -> None:
        # Back-compat: older call sites may have passed a config_path positionally.
        if cfg is not None and not isinstance(cfg, ConfigLoader):
            config_path = str(cfg)
            cfg = None

        # NOTE: `min_jobs_target` and `safety_net` are accepted for compatibility
        # but intentionally do not override config-driven behavior yet.
        self._init_min_jobs_target = int(min_jobs_target)
        self._init_safety_net = bool(safety_net)
        self._init_config_path = config_path

        self.cfg = cfg or config_loader
        self.min_jobs_target = self.cfg.settings.get("run_config", {}).get("jobs_per_run_target", 100)
        self.enable_safety_net = self.cfg.settings.get("run_config", {}).get("enable_serpapi_safety_net", True)

        self.results: List[Dict[str, Any]] = []
        self.seen_job_ids: Set[str] = set()
        self.metrics = ScrapeMetrics()
        self.scrapers: List[Any] = []

        # Initialize components
        self.proxy_pool = ProxyPool()

        self.filter_engine = FilterEngine(self.cfg)
        self.normalizer = Normalizer()
        self.scoring_engine = ScoringEngine(self.cfg)

        # Initialize scrapers
        self._init_scrapers()

        logger.info(
            "ScraperEngine initialized | scrapers=%d | min_jobs_target=%d | safety_net=%s",
            len(self.scrapers),
            self.min_jobs_target,
            self.enable_safety_net,
        )

    # ------------------------------------------------------------------ #
    # SCRAPER REGISTRATION
    # ------------------------------------------------------------------ #

    def _init_scrapers(self) -> None:
        """Initialize all enabled scrapers from ConfigLoader."""
        scrapers: List[Any] = []

        # 1. JobSpy adapter (LinkedIn, Indeed, etc.)
        try:
            adapter = JobSpyAdapter(self.cfg)
            if adapter.enabled_sites:
                scrapers.append(adapter)
                logger.info("✓ JobSpy adapter initialized with %d sites", len(adapter.enabled_sites))
        except Exception as e:
            logger.warning("JobSpyAdapter initialization failed: %s", e)

        # 2. Site-specific scrapers
        from scrapers.scraper_service import (
            RemoteOKAPIScraper, HimalayasScraper, RemotiveScraper,
            WeWorkRemotelyScraper, ArbeitnowScraper, JobicyScraper,
        )

        _rest_scrapers = [
            ("remoteok",       RemoteOKAPIScraper),
            ("himalayas",      HimalayasScraper),
            ("remotive",       RemotiveScraper),
            ("weworkremotely", WeWorkRemotelyScraper),
            ("arbeitnow",      ArbeitnowScraper),
            ("jobicy",         JobicyScraper),
        ]
        _platform_cfg = self.cfg.settings.get("platform_settings", {}).get("platforms", {})
        for _name, _cls in _rest_scrapers:
            if _platform_cfg.get(_name, {}).get("active", False):
                try:
                    scrapers.append(_cls())
                    logger.info("✓ %s scraper registered", _name)
                except Exception as exc:
                    logger.error("✗ %s scraper failed to register: %s", _name, exc)

        self.scrapers = scrapers
        logger.info("Scraper registration complete | total=%d", len(self.scrapers))

    # ------------------------------------------------------------------ #
    # HTTP HELPERS
    # ------------------------------------------------------------------ #

    def _fetch_with_proxy_and_retry(
        self,
        url: str,
        method: str = "GET",
        **kwargs: Any,
    ) -> Any:
        """Fetch a URL through the shared proxy pool with automatic retry.

        Delegates entirely to the module-level :func:`_make_proxied_request`
        helper which handles round-robin proxy rotation (up to 3 attempts)
        and falls back to a direct request if all proxies fail.

        Args:
            url: Target URL string.
            method: HTTP method ``"GET"`` or ``"POST"``.
            **kwargs: Additional kwargs forwarded to
                ``requests.get`` / ``requests.post``.

        Returns:
            ``requests.Response`` object.
        """
        return _make_proxied_request(url, method=method, **kwargs)

    # ------------------------------------------------------------------ #
    # MAIN RUN
    # ------------------------------------------------------------------ #

    async def run(self) -> Tuple[List[Dict[str, Any]], ScrapeMetrics]:
        """Orchestrate all scrapers, apply safety-net, normalise and score.

        Flow:
          1. Run all primary scrapers concurrently -> collect all_raw_jobs.
          2. Safety-net: if primary scrapers return fewer than the target,
             call search_google_jobs exactly once to supplement results.
          3. Normalisation loop: deduplicate, hard-filter, score every job.
          4. Persist results + metrics.
        """
        start_time = time.time()
        all_raw_jobs: List[Dict[str, Any]] = []

        # ---- Step 1: Run all primary scrapers ---------------------------
        tasks = [scraper.run() for scraper in self.scrapers]
        scraper_results = await asyncio.gather(*tasks, return_exceptions=True)

        for scraper, result in zip(self.scrapers, scraper_results):
            if isinstance(result, Exception):
                logger.error("Scraper '%s' raised: %s", scraper.name, result)
                self.metrics.scrapers_failed += 1
                self.metrics.sites_scraped[scraper.name] = {
                    "count": 0,
                    "runtime_ms": 0.0,
                    "errors": 1,
                }
            else:
                jobs_from_scraper = result or []
                all_raw_jobs.extend(jobs_from_scraper)
                self.metrics.scrapers_succeeded += 1
                self.metrics.sites_scraped[scraper.name] = {
                    "count": len(jobs_from_scraper),
                    "runtime_ms": 0.0,
                    "errors": 0,
                }

        self.metrics.total_jobs_raw = len(all_raw_jobs)
        logger.info(
            "Primary scrapers done: %d raw jobs from %d scrapers",
            len(all_raw_jobs),
            self.metrics.scrapers_succeeded,
        )

        # ---- Step 2: SerpAPI safety-net (called ONCE, OUTSIDE loop) -----
        # Safety-net: only trigger SerpAPI if primary scrapers
        # returned fewer than the minimum job threshold
        SAFETY_NET_THRESHOLD = config_loader.get_run_config().get(
            "jobs_per_run_target", 100
        )
        if self.enable_safety_net and len(all_raw_jobs) < SAFETY_NET_THRESHOLD:
            logger.info(
                "Safety-net triggered: %d jobs from primary scrapers "
                "(threshold: %d) — activating SerpAPI",
                len(all_raw_jobs),
                SAFETY_NET_THRESHOLD,
            )

            # Primary source: explicit search_queries from user_profile
            search_queries: list[str] = (
                self.cfg.user.get("job_preferences", {}).get("search_queries", [])
            )

            if not search_queries:
                # Fallback to search_queries from user_profile
                search_queries = (
                    self.cfg.user.get("job_preferences", {}).get("search_queries", [])
                )[:5]

            # Use up to first 3 queries to build a generic search term
            self.search_term = " ".join(search_queries[:3])

            serp_query = (search_queries[0] if search_queries else "").strip()
            if not serp_query:
                logger.warning(
                    "SerpAPI safety-net: no search_queries configured in "
                    "user_profile — using empty query"
                )

            try:
                try:
                    # local import (optional dependency)
                    from tools.serpapi_tool import search_google_jobs  # type: ignore[import]
                except Exception as import_exc:  # noqa: BLE001
                    logger.warning(
                        "SerpAPI safety-net: could not import search_google_jobs — "
                        "skipping: %s",
                        import_exc,
                    )
                    search_google_jobs = None  # type: ignore[assignment]

                if not search_google_jobs:
                    raise RuntimeError("serpapi_tool_unavailable")

                try:
                    serp_json_str = search_google_jobs(
                        query=serp_query,
                        location="Remote",
                        num_results=20,
                    )
                except TypeError:
                    serp_json_str = search_google_jobs(
                        serp_query,
                        "Remote",
                        20,
                    )
                serp_jobs = json.loads(serp_json_str)
                if isinstance(serp_jobs, list):
                    logger.info(
                        "SerpAPI safety-net returned %d additional jobs",
                        len(serp_jobs),
                    )
                    all_raw_jobs.extend(serp_jobs)
                    self.metrics.sites_scraped["serpapi_safety_net"] = {
                        "count": len(serp_jobs),
                        "runtime_ms": 0.0,
                        "errors": 0,
                    }
                elif isinstance(serp_jobs, dict) and "error" in serp_jobs:
                    logger.warning(
                        "SerpAPI safety-net failed: %s", serp_jobs["error"]
                    )
            except Exception as e:
                logger.warning(
                    "SerpAPI safety-net exception: %s", str(e)
                )
        else:
            logger.info(
                "Safety-net not needed: %d jobs collected from "
                "primary scrapers", len(all_raw_jobs),
            )

        # ---- Step 3: Normalisation loop — iterate all_raw_jobs ONCE -----
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
            passed, reason = self.filter_engine.passes(normalized)
            if not passed:
                logger.debug("Job excluded by FilterEngine: %s | reason=%s", normalized.get('title'), reason)
                continue

            # YAML-driven static pre-filter scoring (0-100 internal, 0.0-1.0 output)
            score = self.scoring_engine.calculate_score(normalized)
            if score < self.scoring_engine.min_score:
                logger.debug("Job excluded by scoring (score=%.1f < %.1f): %s", 
                          score, self.scoring_engine.min_score, normalized.get('title'))
                continue

            static_score = max(0.0, min(1.0, score / 100.0))
            normalized["static_score"] = round(static_score, 2)

            self.results.append(normalized)

        # ---- Enforce unified schema completeness ------------------------
        # Every job dict reaching here must expose all keys from the unified
        # schema so downstream agents never encounter a KeyError.
        # List-type fields default to [] ; scalar fields default to None.
        # No routing or DB-related keys are added here.
        _LIST_KEYS = {"required_skills", "preferred_skills", "benefits", "tags"}
        _SCALAR_KEYS = {
            "job_id", "title", "company", "location", "remote_type",
            "job_url", "application_url", "source", "platform", "description",
            "job_type", "employment_type", "salary_min", "salary_max",
            "salary_currency", "experience_min", "experience_max",
            "experience_level", "posted_date", "scraped_at", "company_size",
            "company_url", "industry", "visa_sponsorship", "education_required",
            "application_method", "normalisation_status", "static_score",
        }
        for job in self.results:
            for key in _LIST_KEYS:
                if key not in job:
                    job[key] = []
            for key in _SCALAR_KEYS:
                if key not in job:
                    job[key] = None

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
            s = round(job.get("static_score", 0.0) * 100.0, 2)
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

        logger.info(
            "Scrape completed: %d filtered jobs (from %d raw, %d deduped) in %.0f ms (%.1f jobs/min)",
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
            logger.info("Results saved to %s", LATEST_RUN_PATH)
        except Exception as e:  # pragma: no cover
            logger.error("Failed to save results: %s", e)

    def _save_metrics(self) -> None:
        """Save metrics to JSON file."""
        try:
            with LATEST_METRICS_PATH.open("w", encoding="utf-8") as f:
                json.dump(self.metrics.to_dict(), f, indent=2)
            logger.info("Metrics saved to %s", LATEST_METRICS_PATH)
        except Exception as e:  # pragma: no cover
            logger.error("Failed to save metrics: %s", e)

    # ------------------------------------------------------------------ #
    # PUBLIC ACCESSORS
    # ------------------------------------------------------------------ #

    def get_dataframe(self):
        """Get results as a pandas DataFrame."""
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
            "total_jobs": len(self.results),
            "scraped_at": datetime.now(timezone.utc).isoformat(),
            "metrics": self.metrics.to_dict(),
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
    logger.info("=" * 60)
    logger.info("SCRAPER COMPLETED")
    logger.info("FILTERED JOBS: %d", metrics.total_jobs_filtered)
    logger.info("RAW JOBS: %d", metrics.total_jobs_raw)
    logger.info("DEDUPED: %d", metrics.deduped_jobs)
    logger.info("EXECUTION TIME: %.0fms", metrics.execution_time_ms)
    logger.info("=" * 60)
    logger.info("Results saved to: %s", LATEST_RUN_PATH)
    logger.info("Metrics saved to: %s", LATEST_METRICS_PATH)


if __name__ == "__main__":
    asyncio.run(main())
