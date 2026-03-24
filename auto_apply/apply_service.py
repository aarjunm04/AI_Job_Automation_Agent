"""
Auto Apply FastAPI Microservice for AI Job Application Agent.

Production-grade HTTP service that receives job application requests,
routes them to the correct platform-specific Playwright script via
PLATFORM_REGISTRY, and returns structured results with proof capture.

Runs on port 8003 in the Docker Compose stack.
Endpoints:
- POST /apply: Execute job application
- GET /health: Service health check
- GET /queue/pending: Get pending manual queue jobs
- POST /queue/retry: Retry a queued job
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from dataclasses import asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional

import agentops
from agentops.sdk.decorators import operation
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import psycopg2
import psycopg2.extras
from playwright.async_api import async_playwright, Page

from auto_apply.ats_detector import ATSDetector, ATSType, ATSProfile
from auto_apply.form_filler import FormFiller, FillResult
from auto_apply.platforms.greenhouse import GreenhouseApply
from auto_apply.platforms.lever import LeverApply
from auto_apply.platforms.base_platform import ApplyResult
from config.settings import db_config, run_config, budget_config
from config.config_loader import config_loader
from tools.budget_tools import check_xai_run_cap, record_llm_cost
from utils.proxy_rate_limit import get_playwright_proxy, get_next_proxy, mark_proxy_dead, mark_proxy_success

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
LOG = logging.getLogger(__name__)
LOG.setLevel(os.getenv("LOG_LEVEL", "INFO"))

__all__ = ["app", "ApplyRequest", "ApplyResponse", "PLATFORM_REGISTRY"]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DRY_RUN: bool = os.getenv("DRY_RUN", "false").lower() == "true"
RESUME_DIR: Path = Path(os.getenv("RESUME_DIR", "app/resumes"))
SCREENSHOT_DIR: Path = Path(os.getenv("SCREENSHOT_DIR", "/tmp/apply_screenshots"))
SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Platform Registry - Maps platform names to apply module classes
# ---------------------------------------------------------------------------
class PlatformType(str, Enum):
    """Supported ATS platforms."""
    GREENHOUSE = "greenhouse"
    LEVER = "lever"
    WORKDAY = "workday"
    LINKEDIN = "linkedin"
    NATIVE = "native"
    INDEED = "indeed"
    WELLFOUND = "wellfound"
    ARC_DEV = "arc_dev"


PLATFORM_REGISTRY: Dict[str, type] = {
    "greenhouse": GreenhouseApply,
    "lever": LeverApply,
    # Future platforms - stubs for now, route to manual queue
    # "workday": WorkdayApply,
    # "linkedin": LinkedInApply,
    # "native": NativeFormApply,
}


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------
class ApplyRequest(BaseModel):
    """Request schema for /apply endpoint."""
    job_id: str = Field(..., description="UUID of the job post in database")
    resume_path: str = Field(..., description="Filename of resume PDF in RESUME_DIR")
    platform: str = Field(..., description="ATS platform name (greenhouse, lever, etc)")
    job_url: str = Field(..., description="Full URL of the job application page")
    fit_score: float = Field(default=0.0, description="Fit score from analyser (0.0-1.0)")
    run_batch_id: Optional[str] = Field(default=None, description="UUID of current run batch")
    user_id: Optional[str] = Field(default=None, description="UUID of the user")
    job_title: Optional[str] = Field(default="", description="Job title for context")
    company: Optional[str] = Field(default="", description="Company name for context")
    dry_run: Optional[bool] = Field(default=None, description="Override global DRY_RUN")


class ApplyResponse(BaseModel):
    """Response schema for /apply endpoint."""
    status: str = Field(..., description="applied|queued|failed")
    job_id: str = Field(..., description="Job post UUID")
    screenshot_path: Optional[str] = Field(default=None, description="Path to proof screenshot")
    error_code: Optional[str] = Field(default=None, description="Error code if failed")
    error_message: Optional[str] = Field(default=None, description="Human-readable error")
    proof_type: Optional[str] = Field(default=None, description="Type of submission proof")
    proof_value: Optional[str] = Field(default=None, description="Proof value (URL, confirmation)")
    proof_confidence: Optional[float] = Field(default=None, description="Confidence 0.0-1.0")
    platform: str = Field(..., description="ATS platform used")
    cost_usd: float = Field(default=0.0, description="LLM cost incurred")
    duration_seconds: float = Field(default=0.0, description="Total apply duration")


class QueuedJobResponse(BaseModel):
    """Response schema for queued job info."""
    job_id: str
    job_url: str
    company: str
    title: str
    priority: int
    queued_at: str
    reason: str


# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Auto Apply Service",
    description="AI Job Application Agent - Playwright Automation Microservice",
    version="1.0.0",
)


# ---------------------------------------------------------------------------
# Database Helpers
# ---------------------------------------------------------------------------
def _get_db_conn():
    """Get database connection with retry logic."""
    for attempt in range(3):
        try:
            if os.getenv("ACTIVE_DB", "local") == "local":
                conn = psycopg2.connect(
                    host=os.getenv("LOCAL_POSTGRES_HOST", "ai_postgres"),
                    port=int(os.getenv("LOCAL_POSTGRES_PORT", "5432")),
                    user=os.getenv("LOCAL_POSTGRES_USER", "aarjunm04"),
                    password=os.getenv("LOCAL_POSTGRES_PASSWORD"),
                    dbname=os.getenv("LOCAL_POSTGRES_DB", "ai_job_db"),
                    connect_timeout=10,
                )
            else:
                conn = psycopg2.connect(os.getenv("SUPABASE_URL"))
            conn.autocommit = False
            return conn
        except Exception as e:
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                LOG.error("Database connection failed after 3 attempts: %s", e)
                raise


def _get_user_profile() -> Dict[str, Any]:
    """Load user profile from config/user_profile.json via config_loader."""
    try:
        meta = config_loader.get_user_metadata()
        name = meta.get("full_name", "")
        parts = name.split()
        return {
            "first_name": parts[0] if parts else "",
            "last_name": parts[-1] if len(parts) > 1 else "",
            "full_name": name,
            "email": meta.get("email", ""),
            "phone": meta.get("phone", ""),
            "linkedin_url": meta.get("linkedin_url", ""),
            "portfolio_url": meta.get("portfolio_url", ""),
            "location": meta.get("location_city", ""),
            "years_experience": str(meta.get("years_experience_total", "0")),
        }
    except Exception as exc:  # noqa: BLE001
        LOG.error("_get_user_profile: failed to load from config_loader: %s", exc)
        return {}


def _insert_application(
    job_id: str,
    user_id: str,
    status: str,
    platform: str,
    error_code: Optional[str] = None,
) -> Optional[str]:
    """Insert application record into database."""
    conn = None
    try:
        conn = _get_db_conn()
        cursor = conn.cursor()
        app_id = str(uuid.uuid4())
        
        # Get resume_id for user (first active resume)
        cursor.execute(
            "SELECT id FROM resumes WHERE user_id = %s AND is_active = true LIMIT 1",
            (user_id,)
        )
        resume_row = cursor.fetchone()
        resume_id = resume_row[0] if resume_row else None
        
        cursor.execute(
            """
            INSERT INTO applications (id, job_post_id, user_id, resume_id, mode, status, platform, error_code)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (app_id, job_id, user_id, resume_id, "auto", status, platform, error_code)
        )
        conn.commit()
        return app_id
    except Exception as e:
        if conn:
            conn.rollback()
        LOG.error("Failed to insert application: %s", e)
        return None
    finally:
        if conn:
            conn.close()


def _insert_queued_job(
    application_id: str,
    job_id: str,
    priority: int,
    reason: str,
) -> bool:
    """Insert job into manual queue."""
    conn = None
    try:
        conn = _get_db_conn()
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO queued_jobs (application_id, job_post_id, priority, notes)
            VALUES (%s, %s, %s, %s)
            """,
            (application_id, job_id, priority, reason)
        )
        conn.commit()
        return True
    except Exception as e:
        if conn:
            conn.rollback()
        LOG.error("Failed to insert queued job: %s", e)
        return False
    finally:
        if conn:
            conn.close()


def _log_audit_event(
    run_batch_id: str,
    level: str,
    event_type: str,
    message: str,
    job_id: Optional[str] = None,
    application_id: Optional[str] = None,
) -> None:
    """Log event to audit_logs table."""
    conn = None
    try:
        conn = _get_db_conn()
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO audit_logs (run_batch_id, job_post_id, application_id, level, event_type, message)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (run_batch_id, job_id, application_id, level, event_type, message)
        )
        conn.commit()
    except Exception as e:
        LOG.warning("Failed to log audit event: %s", e)
    finally:
        if conn:
            conn.close()


# ---------------------------------------------------------------------------
# Core Apply Logic
# ---------------------------------------------------------------------------
async def _execute_apply(
    request: ApplyRequest,
    user_profile: Dict[str, Any],
) -> ApplyResponse:
    """Execute the full apply flow using platform-specific module.
    
    Args:
        request: The apply request with all job details.
        user_profile: User profile data from environment.
        
    Returns:
        ApplyResponse with outcome, proof, and status.
    """
    start_time = time.time()
    dry_run_effective = request.dry_run if request.dry_run is not None else DRY_RUN
    
    # Resolve resume path
    resume_path = RESUME_DIR / request.resume_path
    if not resume_path.exists():
        default_resume = "Aarjun_Gen.pdf"
        resume_path = RESUME_DIR / default_resume
        LOG.warning("Resume %s not found, using default: %s", request.resume_path, default_resume)
    
    # Check budget before proceeding
    if request.run_batch_id:
        try:
            budget_result = json.loads(check_xai_run_cap(request.run_batch_id))
            if budget_result.get("abort"):
                LOG.critical("Budget cap hit - aborting apply for job %s", request.job_id)
                return ApplyResponse(
                    status="failed",
                    job_id=request.job_id,
                    error_code="BUDGET_CAP_HIT",
                    error_message="xAI run cap exceeded",
                    platform=request.platform,
                    duration_seconds=time.time() - start_time,
                )
        except Exception as e:
            LOG.warning("Budget check failed (proceeding): %s", e)
    
    # Check if platform is supported
    platform_lower = request.platform.lower()
    if platform_lower not in PLATFORM_REGISTRY:
        LOG.warning("Unsupported platform %s - routing to manual queue", platform_lower)
        return ApplyResponse(
            status="queued",
            job_id=request.job_id,
            error_code="UNSUPPORTED_PLATFORM",
            error_message=f"Platform {platform_lower} not yet supported for auto-apply",
            platform=request.platform,
            duration_seconds=time.time() - start_time,
        )
    
    # Launch Playwright session
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=[
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--disable-blink-features=AutomationControlled",
            ],
        )
        
        # Proxy configuration — use shared proxy pool via utils.proxy_rate_limit
        proxy = get_playwright_proxy()
        
        context = await browser.new_context(
            proxy=proxy,
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            ),
        )
        
        # Anti-detection script
        await context.add_init_script(
            "Object.defineProperty(navigator,'webdriver',{get:()=>undefined})"
        )
        
        page: Page = await context.new_page()
        screenshot_path = None
        llm_cost = 0.0
        
        try:
            # Build job metadata
            job_meta = {
                "job_url": request.job_url,
                "title": request.job_title or "",
                "company": request.company or "",
                "platform": request.platform,
                "fit_score": request.fit_score,
                "resume_suggested": request.resume_path,
                "run_id": request.run_batch_id or "",
            }
            
            # Get platform apply module
            ApplyModule = PLATFORM_REGISTRY[platform_lower]
            
            # Instantiate and execute
            applier = ApplyModule(
                page=page,
                job_meta=job_meta,
                user_profile=user_profile,
                dry_run=dry_run_effective,
            )
            
            result: ApplyResult = await applier.apply()
            
            # Take screenshot on completion/failure
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            screenshot_filename = f"apply_{request.job_id}_{timestamp}.png"
            screenshot_path = str(SCREENSHOT_DIR / screenshot_filename)
            try:
                await page.screenshot(path=screenshot_path, full_page=True)
            except Exception as ss_err:
                LOG.warning("Screenshot failed: %s", ss_err)
                screenshot_path = None
            
            # Determine status
            if result.success:
                status = "applied"
            elif result.reroute_to_manual:
                status = "queued"
            else:
                status = "failed"
            
            duration = time.time() - start_time
            
            return ApplyResponse(
                status=status,
                job_id=request.job_id,
                screenshot_path=screenshot_path,
                error_code=result.error_code,
                error_message=result.reroute_reason,
                proof_type=result.proof_type,
                proof_value=result.proof_value,
                proof_confidence=result.proof_confidence,
                platform=request.platform,
                cost_usd=llm_cost,
                duration_seconds=duration,
            )
            
        except Exception as e:
            LOG.error("Apply execution failed for job %s: %s", request.job_id, e, exc_info=True)
            
            # Screenshot on error
            try:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                screenshot_filename = f"error_{request.job_id}_{timestamp}.png"
                screenshot_path = str(SCREENSHOT_DIR / screenshot_filename)
                await page.screenshot(path=screenshot_path, full_page=True)
            except:
                pass
            
            return ApplyResponse(
                status="failed",
                job_id=request.job_id,
                screenshot_path=screenshot_path,
                error_code="EXECUTION_ERROR",
                error_message=str(e),
                platform=request.platform,
                duration_seconds=time.time() - start_time,
            )
        finally:
            try:
                await context.close()
            except:
                pass
            try:
                await browser.close()
            except:
                pass


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
@operation
def health_check() -> Dict[str, Any]:
    """Health check endpoint for Docker/Kubernetes probes."""
    return {
        "status": "ok",
        "service": "auto_apply",
        "version": "1.0.0",
        "dry_run": DRY_RUN,
        "supported_platforms": list(PLATFORM_REGISTRY.keys()),
    }


@app.post("/apply", response_model=ApplyResponse)
@operation
async def apply_to_job(request: ApplyRequest) -> ApplyResponse:
    """Execute a job application via Playwright automation.
    
    Routes the request to the appropriate platform-specific apply module
    based on the platform field. Supports Greenhouse, Lever, and falls
    back to manual queue for unsupported platforms.
    
    Args:
        request: ApplyRequest with job details and resume info.
        
    Returns:
        ApplyResponse with status (applied/queued/failed), proof, and metadata.
    """
    LOG.info(
        "Received apply request: job_id=%s platform=%s url=%s",
        request.job_id,
        request.platform,
        request.job_url,
    )
    
    # Load user profile
    user_profile = _get_user_profile()
    
    # Execute apply
    response = await _execute_apply(request, user_profile)
    
    # Insert application record
    user_id = request.user_id or os.getenv("DEFAULT_USER_ID", "")
    if user_id:
        app_status = "applied" if response.status == "applied" else (
            "manual_queued" if response.status == "queued" else "failed"
        )
        app_id = _insert_application(
            job_id=request.job_id,
            user_id=user_id,
            status=app_status,
            platform=request.platform,
            error_code=response.error_code,
        )
        
        # If queued, insert into queued_jobs table
        if response.status == "queued" and app_id:
            priority = int(request.fit_score * 10)
            _insert_queued_job(
                application_id=app_id,
                job_id=request.job_id,
                priority=priority,
                reason=response.error_message or "Auto-routed to manual queue",
            )
        
        # Log audit event
        if request.run_batch_id:
            _log_audit_event(
                run_batch_id=request.run_batch_id,
                level="INFO" if response.status == "applied" else "WARNING",
                event_type=f"apply_{response.status}",
                message=f"{request.company} - {request.job_title} | {response.status} | {response.error_code or 'success'}",
                job_id=request.job_id,
                application_id=app_id,
            )
    
    LOG.info(
        "Apply completed: job_id=%s status=%s duration=%.2fs",
        request.job_id,
        response.status,
        response.duration_seconds,
    )
    
    return response


@app.get("/queue/pending")
@operation
def get_pending_queue() -> Dict[str, Any]:
    """Get all jobs pending in the manual queue."""
    conn = None
    try:
        conn = _get_db_conn()
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cursor.execute(
            """
            SELECT 
                qj.job_post_id as job_id,
                j.url as job_url,
                j.company,
                j.title,
                qj.priority,
                qj.queued_at,
                qj.notes as reason
            FROM queued_jobs qj
            JOIN jobs j ON j.id = qj.job_post_id
            ORDER BY qj.priority DESC, qj.queued_at ASC
            LIMIT 100
            """
        )
        rows = cursor.fetchall()
        return {
            "count": len(rows),
            "jobs": [dict(row) for row in rows],
        }
    except Exception as e:
        LOG.error("Failed to get pending queue: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if conn:
            conn.close()


@app.post("/queue/retry/{job_id}")
@operation
async def retry_queued_job(job_id: str) -> ApplyResponse:
    """Retry applying to a job that was previously queued.
    
    Args:
        job_id: UUID of the job to retry.
        
    Returns:
        ApplyResponse from the retry attempt.
    """
    conn = None
    try:
        conn = _get_db_conn()
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        # Get job details
        cursor.execute(
            """
            SELECT j.url, j.title, j.company, j.source_platform, 
                   js.fit_score, js.resume_id,
                   r.storage_path as resume_path,
                   j.run_batch_id
            FROM jobs j
            LEFT JOIN job_scores js ON js.job_post_id = j.id
            LEFT JOIN resumes r ON r.id = js.resume_id
            WHERE j.id = %s
            """,
            (job_id,)
        )
        row = cursor.fetchone()
        
        if not row:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
        
        # Detect platform from URL if needed
        platform = row.get("source_platform", "native")
        
        request = ApplyRequest(
            job_id=job_id,
            resume_path=row.get("resume_path") or "Aarjun_Gen.pdf",
            platform=platform,
            job_url=row["url"],
            fit_score=float(row.get("fit_score", 0.0)),
            run_batch_id=str(row.get("run_batch_id", "")),
            job_title=row.get("title", ""),
            company=row.get("company", ""),
        )
        
        # Delete from queue first
        cursor.execute(
            "DELETE FROM queued_jobs WHERE job_post_id = %s",
            (job_id,)
        )
        conn.commit()
        
        # Execute apply
        user_profile = _get_user_profile()
        return await _execute_apply(request, user_profile)
        
    except HTTPException:
        raise
    except Exception as e:
        LOG.error("Failed to retry queued job %s: %s", job_id, e)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if conn:
            conn.close()


# ---------------------------------------------------------------------------
# Startup/Shutdown Events
# ---------------------------------------------------------------------------
@app.on_event("startup")
async def startup_event():
    """Initialize service on startup."""
    LOG.info("Auto Apply Service starting on port 8003...")
    LOG.info("DRY_RUN mode: %s", DRY_RUN)
    LOG.info("Supported platforms: %s", list(PLATFORM_REGISTRY.keys()))
    
    # Verify resume directory exists
    if not RESUME_DIR.exists():
        LOG.warning("Resume directory %s does not exist", RESUME_DIR)


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    LOG.info("Auto Apply Service shutting down...")
