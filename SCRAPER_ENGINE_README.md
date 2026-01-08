# Enterprise Scraper Engine - Specifications & Documentation

## Overview
Production-ready, deterministic job scraper engine that discovers jobs from approved sources, normalizes data, applies rule-based filters, and outputs structured job records. **No browser automation, no AI/LLM dependencies, purely API and library-based scraping.**

## Architecture

### Core Components
```
scraper_engine.py          → Master orchestrator, normalization, deduplication, filtering, scoring
jobspy_adapter.py          → JobSpy wrapper (LinkedIn, Indeed, Glassdoor, ZipRecruiter)
job_filters.yaml           → Profile, preferences, filtering rules, keywords
```

**Note:** `scraper_service.py` exists but is not used by this scraper engine (Playwright-based scrapers excluded per spec).

### Site Coverage (7 Total - API & Library Based Only)
- **JobSpy** (4): LinkedIn, Indeed, Glassdoor, ZipRecruiter
- **SerpAPI** (1): Google Jobs (optional)
- **Official APIs** (2): Jooble, Remotive

**Excluded:** All browser-based scrapers (Wellfound, WeWorkRemotely, RemoteOK, SimplyHired, StackOverflow, YCStartups, HiringCafe)

## Configuration Files

### 1. `config/job_filters.yaml`
Profile, preferences, and filtering rules.

**Key Sections:**
- `search_criteria`: Job titles, required/preferred keywords (ML, DS, Backend, Automation)
- `locations`: Remote-first preference, allowed countries
- `experience`: 0-5 years range (preference weighting)
- `companies`: Startup and foreign tech company preference
- `exclusions`: Jobs to exclude
- `priority_rules`: High/medium/low priority conditions

**Filter Logic (Deterministic Rule-Based):**
1. **Keyword Matching**: Must contain role keywords (ML, DS, Backend, Automation, AI)
2. **Date**: Hard filter - exclude jobs >7 days old (UTC)
3. **Experience**: Prefer 0-3 years, lower priority for >3 years (scoring only)
4. **Remote**: Remote-first preference (scored higher, not hard filtered)
5. **Salary Floor**: Enforce minimum salary if specified in config
6. **Exclusions**: Exclude based on excluded_keywords and exclusion rules

### 2. `narad.env` (Environment Variables)
All API keys and credentials stored here (loaded via `python-dotenv`).

**Required Variables:**
```
# SerpAPI (optional, for Google Jobs)
SERPAPI_API_KEY_1=your_serpapi_key

# Jooble API
JOOBLE_API_KEY=your_jooble_key
```

**Note:** No proxy configuration needed (API-based scraping only).

## Resource Management

### SerpAPI Credits (Optional)
- **Monthly Quota**: 250 credits (if using Google Jobs via SerpAPI)
- **Per Run Allocation**: ~62 credits (250 / 4 runs per month)
- **Tracking**: `logs/serpapi_usage.json`
- **Reset**: Monthly (1st of month)
- **Behavior**: Skip Google Jobs scraping if quota exceeded

**Note:** Proxy bandwidth management not required (API-based scraping only).

## Normalization Schema

All jobs normalized to unified schema (deterministic, consistent across all sources):

```python
{
    # Core Fields
    "job_id": str,              # hash(title|company|url)
    "title": str,               # Job title (max 200 chars)
    "company": str,             # Company name (max 100 chars)
    "location": str,            # Location (max 100 chars)
    "job_url": str,             # Original job posting URL
    "description": str,         # Full job description (max 2500 chars)
    "source": str,              # Site name (e.g., "linkedin", "indeed")
    "scraped_at": str,          # ISO 8601 timestamp (UTC)
    
    # Compensation
    "salary_min": Optional[float],
    "salary_max": Optional[float],
    "salary_currency": Optional[str],
    
    # Experience & Level
    "experience_min": Optional[int],        # Years
    "experience_max": Optional[int],        # Years
    "experience_level": Optional[str],      # "entry", "mid", "senior"
    
    # Job Details
    "job_type": Optional[str],              # "full-time", "contract", "part-time"
    "posted_date": Optional[str],           # ISO 8601 date (UTC)
    "application_deadline": Optional[str],  # ISO 8601 date
    "remote_type": Optional[str],           # "remote", "hybrid", "onsite"
    "employment_type": Optional[str],
    
    # Company Information
    "company_size": Optional[str],
    "company_url": Optional[str],
    "industry": Optional[str],
    
    # Application
    "application_url": Optional[str],
    "application_method": Optional[str],
    
    # Skills & Requirements
    "required_skills": List[str],
    "preferred_skills": List[str],
    "education_required": Optional[str],
    
    # Additional Metadata
    "benefits": List[str],
    "visa_sponsorship": Optional[bool]
}
```

## Deduplication Logic

**Algorithm:** `hash(title|company|url)`

```python
def generate_job_hash(title: str, company: str, url: str) -> str:
    # Normalize: lowercase, strip whitespace, remove special chars
    normalized_title = normalize_string(title)
    normalized_company = normalize_string(company)
    normalized_url = normalize_url(url)  # Remove query params, fragments
    
    key = f"{normalized_title}|{normalized_company}|{normalized_url}"
    return hashlib.sha256(key.lower().encode()).hexdigest()
```

**Result**: Duplicate jobs (same title+company from different sites) are identified and only the first occurrence is kept.

## Filtering & Scoring Logic

### Processing Pipeline
1. **Scrape** → Raw jobs from all enabled sources (JobSpy, APIs)
2. **Normalize** → Convert to unified schema (deterministic)
3. **Deduplicate** → Remove duplicates using hash (title|company|url)
4. **Hard Filter** → Apply deterministic exclusion rules
5. **Score** → Calculate relevance score (0-100) using rule-based weighting
6. **Output** → Export as DataFrame, JSON, or ingestion-ready payload

### Hard Filters (Deterministic Exclusions)
Jobs are **excluded** if they fail any of these:
- Missing required fields (title, url)
- Posted date > 7 days old (UTC comparison)
- Contains exclusion keywords (sales, marketing, support, business-only roles)
- No role keywords in title/description (ML, DS, Backend, Automation, AI)
- Salary below configured floor (if salary_floor specified in filters)

### Relevance Scoring (0-100 Scale)
Rule-based scoring with deterministic weights:

**Base Score**: 50 points

**Freshness Weighting**:
- Posted < 3 days: +20 points
- Posted 3-5 days: +10 points
- Posted 5-7 days: +5 points

**Remote Preference**:
- Remote: +15 points
- Hybrid: +7 points
- Onsite: 0 points

**Experience Alignment**:
- 0-1 years: +10 points
- 2-3 years: +5 points
- 4-5 years: 0 points
- >5 years: -5 points

**Source Reliability** (configurable per source):
- High reliability (LinkedIn, Indeed): +5 points
- Medium reliability (Glassdoor, ZipRecruiter): +3 points
- Other sources: 0 points

**Company Type**:
- Startup: +5 points
- Foreign tech company: +3 points
- Other: 0 points

**Final Score**: Clamped between 0-100

## Error Handling & Fail-Soft

### Strategy (No Silent Failures)
- **Site Failure**: Log structured error with site name, continue with other sites
- **API Quota Exceeded**: Skip site, log warning, continue
- **API Error**: Retry 2x with exponential backoff, then skip site
- **Parse Error**: Log warning with job source, skip malformed job, continue
- **Network Timeout**: Retry 2x, then skip site
- **Rate Limit Hit**: Back off, wait, retry once more

### Logging (Structured Only)
- Structured logging: `timestamp | component | level | message | context`
- All errors include: site name, error type, error message, retry attempts
- Metrics tracked: success/failure counts per site, jobs scraped per site
- Output: `logs/latest_metrics.json` with comprehensive run statistics
- No bare `except:` clauses - explicit exception handling only

## Output Formats

### 1. Pandas DataFrame
```python
df = engine.get_dataframe()  # Returns DataFrame with all normalized jobs
```

### 2. JSON Export
```python
jobs_json = engine.get_json()  # Returns list of job dicts
# Or save to file: logs/latest_run.json
```

### 3. Ingestion-Ready Payload
```python
payload = engine.get_ingestion_payload()  # Returns structured payload for downstream systems
```

### Output Schema
Each job includes all normalized fields plus:
- `relevance_score`: float (0-100)
- `scraped_at`: ISO 8601 timestamp (UTC)
- `source`: string (site name)

### `logs/latest_metrics.json`
```json
{
    "total_jobs_raw": 320,
    "total_jobs_unique": 285,
    "total_jobs_filtered": 245,
    "deduped_jobs": 35,
    "scrapers_succeeded": 6,
    "scrapers_failed": 1,
    "execution_time_ms": 25000,
    "sites_scraped": {
        "linkedin": 85,
        "indeed": 72,
        "glassdoor": 45,
        "ziprecruiter": 38,
        "jooble": 52,
        "remotive": 28,
        "google_jobs": 0
    },
    "resource_usage": {
        "serpapi_credits_used": 0,
        "serpapi_credits_remaining": 250
    },
    "score_distribution": {
        "90-100": 15,
        "80-89": 42,
        "70-79": 88,
        "60-69": 75,
        "50-59": 25
    }
}
```

### `logs/serpapi_usage.json` (if using SerpAPI)
```json
{
    "current_month": "2025-01",
    "credits_used": 25,
    "credits_remaining": 225,
    "last_reset": "2025-01-01T00:00:00Z",
    "next_reset": "2025-02-01T00:00:00Z",
    "runs_this_month": 1
}
```

## Usage

### Run Scraper (Async)
```python
from core.scraper_engine import ScraperEngine

async def main():
    engine = ScraperEngine()
    jobs, metrics = await engine.run()
    
    # Access results
    print(f"Scraped {len(jobs)} unique jobs")
    df = engine.get_dataframe()
    json_data = engine.get_json()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

### Run Scraper (Sync Wrapper)
```python
from core.scraper_engine import ScraperEngine

engine = ScraperEngine()
jobs, metrics = engine.run_sync()  # Synchronous wrapper
df = engine.get_dataframe()
```

### Command Line
```bash
python core/scraper_engine.py
```

## Run Frequency
- **Recommended**: 4 times per month (weekly)
- **Resource Allocation**: 25% of SerpAPI quota per run (if using)
- **Reset**: Automatic monthly reset (1st of month)

## Dependencies
- `jobspy` (for LinkedIn, Indeed, Glassdoor, ZipRecruiter)
- `requests` (for API scrapers: Jooble, Remotive)
- `python-dotenv` (for narad.env loading)
- `pyyaml` (for job_filters.yaml parsing)
- `pandas` (for DataFrame output)
- `python-dateutil` (for date parsing/comparison)

**Excluded Dependencies:**
- `playwright` (not used - API-based scraping only)
- Any AI/LLM libraries (OpenAI, Anthropic, etc.)
- Any embedding/reasoning models

## Key Principles

### Deterministic Behavior
- Same input → Same output (idempotent runs)
- No randomness in filtering or scoring
- Consistent normalization across runs

### Production-Ready Standards
- Type hints everywhere
- Structured logging only (no print statements)
- Explicit exception handling (no bare `except:`)
- Graceful failure (site failures don't crash engine)
- Rate limit awareness (respects API limits)

### No Silent Failures
- All errors logged with context
- Metrics track success/failure per site
- Clear error messages for debugging

### Configuration-Driven
- All API keys from environment (narad.env)
- Filtering rules from job_filters.yaml
- Site limits configurable via code/config

## Notes
- All timestamps in UTC
- Date comparisons use UTC timezone
- Job descriptions truncated to 2500 chars to manage size
- Fail-soft: Individual site failures don't stop entire scrape run
- Rate limiting: Built-in retry logic with exponential backoff
- Idempotent: Running multiple times produces consistent results (deduplication handles duplicates)

