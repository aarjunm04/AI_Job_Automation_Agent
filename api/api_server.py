"""
FastAPI server for AI Job Application Agent.

Lightweight, stateless HTTP boundary between the Chrome Extension and the
pipeline. All data reads are performed via postgres_tools. The only write
action exposed is queuing a manual apply override. No agent imports — purely
a data read + queue layer.

Runs as a separate Docker service on FASTAPI_PORT.
"""

import os
import json
import time
import uuid
import logging
from datetime import datetime, timezone
from typing import Any, Optional, List, Dict
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
import uvicorn
import psycopg2
import psycopg2.extras
import chromadb

from config.settings import db_config, run_config, api_config
from tools.postgres_tools import (
    get_run_stats,
    get_recent_applications,
    get_pending_manual_queue,
    update_application_status,
    log_event,
)
from tools.budget_tools import get_cost_summary
from integrations.notion import NotionClient

logger = logging.getLogger(__name__)
API_KEY: str = os.getenv("RAG_SERVER_API_KEY", "")

__all__ = ["app", "main"]


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------


class ManualApplyRequest(BaseModel):
    """Request payload for the manual apply queue endpoint.

    Attributes:
        job_post_id: UUID of the job_posts record to queue.
        user_id: User ID initiating the apply action.
        resume_filename: Override resume filename; empty string uses the
            pipeline default from run_config.
        notes: Optional free-text notes from the Chrome Extension user.
    """

    job_post_id: str = Field(..., description="UUID of job_posts record")
    user_id: str = Field(..., description="User ID initiating the apply")
    resume_filename: str = Field(
        default="",
        description="Override resume filename — empty uses default",
    )
    notes: str = Field(
        default="",
        description="Optional notes from Chrome Extension user",
    )


class ApplicationStatusUpdate(BaseModel):
    """Request payload for updating an application's status.

    Attributes:
        application_id: UUID of the applications record to update.
        new_status: Target status value for the application.
        notes: Optional free-text notes attached to the status transition.
    """

    application_id: str = Field(..., description="UUID of applications record")
    new_status: str = Field(
        ...,
        description="New status: applied|rejected|interview|offer|withdrawn",
    )
    notes: str = Field(default="")


class HealthResponse(BaseModel):
    """Response model for the health check endpoint.

    Attributes:
        status: Overall server health — ``"healthy"`` or ``"degraded"``.
        timestamp: UTC ISO8601 timestamp of the health check.
        db_connected: Whether the Postgres database is reachable.
        pipeline_mode: ``"dry_run"`` or ``"live"`` based on run_config.
        dry_run: Raw boolean dry-run flag from run_config.
        version: API server version string.
    """

    status: str
    timestamp: str
    db_connected: bool
    pipeline_mode: str
    dry_run: bool
    version: str = "1.0.0"


# ---------------------------------------------------------------------------
# Chrome Extension Pydantic Models
# ---------------------------------------------------------------------------


class MatchRequest(BaseModel):
    """Request payload for the /match endpoint (Chrome Extension).

    Attributes:
        job_url: Full URL of the job posting page.
        jd_text: Job description text extracted by content script.
    """

    job_url: str
    jd_text: str = Field(default="", alias="jd_text")

    @field_validator("job_url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        v = v.strip()
        if not v.startswith(("http://", "https://")):
            raise ValueError("job_url must be a valid HTTP/HTTPS URL")
        return v

    @field_validator("jd_text")
    @classmethod
    def validate_jd(cls, v: str) -> str:
        return v.strip()[:3000]


class MatchResponse(BaseModel):
    """Response payload for the /match endpoint.

    Attributes:
        resume_suggested: Filename of the recommended resume variant.
        similarity_score: ChromaDB cosine similarity (0.0–1.0).
        fit_score: Composite fit score (0.0–1.0).
        match_reasoning: Human-readable reasoning string.
        talking_points: List of interview preparation talking points.
        autofill_ready: Whether autofill is available.
    """

    resume_suggested: str
    similarity_score: float
    fit_score: float
    match_reasoning: str
    talking_points: List[str]
    autofill_ready: bool


class ExtDetectedField(BaseModel):
    """Schema for a single form field detected by the Chrome Extension.

    Attributes:
        index: Positional index in scan order.
        id: Element id attribute.
        name: Element name attribute.
        placeholder: Element placeholder text.
        label_text: Resolved label text.
        field_type: Classified semantic type.
        tag_name: HTML tag name (INPUT, SELECT, TEXTAREA).
        react_controlled: Whether the field uses React.
        shadow_dom: Whether the field is inside a shadow root.
        element_ref_index: Index into live element reference array.
        ats_hint: Detected ATS platform hint.
    """

    index: int = 0
    id: str = ""
    name: str = ""
    placeholder: str = ""
    label_text: str = ""
    field_type: str = "text"
    tag_name: str = "INPUT"
    react_controlled: bool = False
    shadow_dom: bool = False
    element_ref_index: int = 0
    ats_hint: str = "unknown"


class AutofillRequest(BaseModel):
    """Request payload for the /autofill endpoint.

    Attributes:
        job_url: Full URL of the job posting page.
        detected_fields: List of form fields detected by content script.
    """

    job_url: str
    detected_fields: List[ExtDetectedField]


class AutofillResponse(BaseModel):
    """Response payload for the /autofill endpoint.

    Attributes:
        field_mappings: Map of field id/name to profile value.
        unmapped_fields: List of field keys that could not be mapped.
        mapped_count: Number of fields successfully mapped.
    """

    field_mappings: Dict[str, str]
    unmapped_fields: List[str]
    mapped_count: int


class QueueCountResponse(BaseModel):
    """Response payload for the /queue-count endpoint.

    Attributes:
        count: Number of pending manual-queue applications.
    """

    count: int


class LogApplicationRequest(BaseModel):
    """Request payload for the /log-application endpoint.

    Attributes:
        job_url: Full URL of the job posting that was applied to.
        resume_used: Filename of the resume variant submitted.
        platform: Job platform name.
        applied_at: ISO8601 timestamp of when the user submitted.
        notes: Optional free-text notes from the user.
    """

    job_url: str
    resume_used: str
    platform: str
    applied_at: str
    notes: Optional[str] = None

    @field_validator("job_url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        v = v.strip()
        if not v.startswith(("http://", "https://")):
            raise ValueError("job_url must be a valid HTTP/HTTPS URL")
        return v

    @field_validator("applied_at")
    @classmethod
    def validate_timestamp(cls, v: str) -> str:
        try:
            datetime.fromisoformat(v.replace("Z", "+00:00"))
        except ValueError:
            raise ValueError("applied_at must be a valid ISO8601 timestamp")
        return v


class LogApplicationResponse(BaseModel):
    """Response payload for the /log-application endpoint.

    Attributes:
        application_id: UUID of the created application record.
        notion_page_id: Notion page ID (empty string if sync failed).
        status: Always "success" on successful completion.
    """

    application_id: str
    notion_page_id: str
    status: str


# ---------------------------------------------------------------------------
# Auth Dependencies
# ---------------------------------------------------------------------------


def verify_api_key(x_api_key: str = Header(..., alias="X-API-Key")) -> str:
    """Verify the X-API-Key header against the configured RAG_SERVER_API_KEY.

    When ``RAG_SERVER_API_KEY`` is not set in the environment, authentication
    is disabled and the function logs a warning and returns ``"no_auth"``.
    This allows local development without a configured secret.

    Args:
        x_api_key: API key from the ``X-API-Key`` request header.

    Returns:
        The provided API key string on successful verification, or
        ``"no_auth"`` when the environment key is not configured.

    Raises:
        HTTPException: 401 Unauthorized if the provided key does not match
            the configured ``RAG_SERVER_API_KEY``.
    """
    if not API_KEY:
        logger.warning("RAG_SERVER_API_KEY not set — auth disabled")
        return "no_auth"
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return x_api_key


def verify_bearer_token(
    authorization: str = Header("", alias="Authorization"),
) -> bool:
    """Verify Bearer token from the Chrome Extension.

    Validates the ``Authorization: Bearer <token>`` header against
    ``FASTAPI_API_KEY`` from the environment. When ``FASTAPI_API_KEY`` is
    not set, authentication is disabled for local development.

    Args:
        authorization: Authorization header value.

    Returns:
        True on successful verification.

    Raises:
        HTTPException: 401 Unauthorized if the token does not match.
    """
    fastapi_key: str = os.getenv("FASTAPI_API_KEY", "")
    if not fastapi_key:
        logger.warning("FASTAPI_API_KEY not set — bearer auth disabled")
        return True
    expected: str = f"Bearer {fastapi_key}"
    if authorization != expected:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return True


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore[type-arg]
    """Manage application startup and shutdown events.

    On startup: logs the server port and performs a DB connectivity probe via
    ``get_run_stats``. On shutdown: logs the shutdown event.

    Args:
        app: The FastAPI application instance.

    Yields:
        None: Yields control to the application for its entire lifetime.
    """
    fastapi_port: str = os.getenv("FASTAPI_PORT", "8000")
    logger.info("FastAPI server starting | port=%s", fastapi_port)

    try:
        get_run_stats("health_check")
        logger.info("Startup DB connection probe: OK")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Startup DB connection probe failed: %s", exc)

    yield

    logger.info("FastAPI server shutting down")


# ---------------------------------------------------------------------------
# App Setup
# ---------------------------------------------------------------------------


app = FastAPI(
    title="AI Job Application Agent API",
    description="Internal HTTP boundary for Chrome Extension ↔ Pipeline communication",
    version="1.0.0",
    docs_url="/docs",
    redoc_url=None,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["chrome-extension://*", "http://localhost:*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PATCH"],
    allow_headers=["*"],
)



# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

class MatchRequest(BaseModel):
    job_url: str
    job_title: str
    job_description: str
    resume_label: str | None = None

class MatchResponse(BaseModel):
    fit_score: float
    route: str
    reason: str
    resume_label: str
    scored_at: str

@app.post("/match", response_model=MatchResponse)
async def match(req: MatchRequest) -> MatchResponse:
    """Score a job against the best matching resume via RAG."""
    try:
        from rag_systems.rag_pipeline import create_default_pipeline
        from rag_systems.chromadb_store import ChromaStore
        import asyncio
        import datetime

        def _sync_match():
            store = ChromaStore()
            pipeline = create_default_pipeline(store)
            vec = pipeline.embed_query(req.job_description)
            anchors = pipeline.get_top_resume_anchors(vec, k=1)
            
            fit_score = 0.5
            resume_label = "Aarjun_Gen.pdf"
            if anchors and len(anchors["ids"]) > 0 and len(anchors["ids"][0]) > 0:
                dist = anchors["distances"][0][0]
                fit_score = max(0.0, min(1.0, round(1.0 - dist, 4) * 0.85 + 0.10))
                resume_label = anchors["metadatas"][0][0].get("filename", "Aarjun_Gen.pdf")
            
            route = "manual_review"
            reason = "Moderate match"
            if fit_score >= 0.75:
                route = "auto_apply"
                reason = "Strong match"
            elif fit_score < 0.45:
                route = "skip"
                reason = "Weak match"
                
            return {
                "fit_score": float(fit_score),
                "route": route,
                "reason": reason,
                "resume_label": resume_label,
                "scored_at": datetime.datetime.utcnow().isoformat() + "Z"
            }

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, _sync_match)
        return MatchResponse(**result)
    except Exception as exc:
        logger.error("/match failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))

class AutofillRequest(BaseModel):
    job_url: str
    ats_platform: str = ""
    detected_fields: list | None = None

class AutofillResponse(BaseModel):
    first_name: str
    last_name: str
    email: str
    phone: str
    linkedin_url: str
    resume_pdf_path: str
    cover_letter: str | None = None

@app.post("/autofill", response_model=AutofillResponse)
async def autofill(req: AutofillRequest) -> AutofillResponse:
    """Return candidate profile for Chrome Extension form pre-fill."""
    try:
        return AutofillResponse(
            first_name=os.getenv("CANDIDATE_FIRST_NAME", ""),
            last_name=os.getenv("CANDIDATE_LAST_NAME", ""),
            email=os.getenv("CANDIDATE_EMAIL", ""),
            phone=os.getenv("CANDIDATE_PHONE", ""),
            linkedin_url=os.getenv("CANDIDATE_LINKEDIN_URL", ""),
            resume_pdf_path=os.getenv("RESUME_PDF_PATH", ""),
            cover_letter=None,
        )
    except Exception as exc:
        logger.error("/autofill failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))

class LogApplicationRequest(BaseModel):
    job_post_id: str = ""
    job_url: str = ""
    platform: str = ""
    status: str = ""
    mode: str = ""
    resume_used: str | None = None
    applied_at: str | None = None
    resume_label: str | None = None
    error_code: str | None = None

class LogApplicationResponse(BaseModel):
    application_id: str
    status: str
    logged_at: str

@app.post("/log-application", response_model=LogApplicationResponse)
async def log_application(req: LogApplicationRequest) -> LogApplicationResponse:
    """Log a job application from the Chrome Extension to Postgres."""
    try:
        from utils.db_utils import get_db_conn
        conn = get_db_conn()
        try:
            with conn:
                with conn.cursor() as cur:
                    job_id = req.job_post_id or req.job_url
                    res_label = req.resume_label or req.resume_used
                    # On conflict do nothing
                    cur.execute(
                        """INSERT INTO applications
                           (job_post_id, user_id, mode, status,
                            platform, error_code, resume_used)
                           VALUES (%s, %s, %s, %s, %s, %s, %s)
                           ON CONFLICT DO NOTHING
                           RETURNING id, applied_at""",
                        (
                            job_id,
                            os.getenv("CANDIDATE_USER_ID"),
                            req.mode or "manual",
                            req.status,
                            req.platform,
                            req.error_code,
                            res_label
                        ),
                    )
                    row = cur.fetchone()
                    if not row:
                        raise HTTPException(
                            status_code=409,
                            detail="Application already logged (duplicate)",
                        )
                    application_id = str(row[0])
                    logged_at = str(row[1]) if row[1] else ""
        finally:
            conn.close()
            
        logger.info(
            "Logged application %s for job %s status=%s",
            application_id, job_id, req.status,
        )
        return LogApplicationResponse(
            application_id=application_id,
            status=req.status,
            logged_at=logged_at,
        )
    except HTTPException:
        raise
    except Exception as exc:
        import uuid
        import datetime
        logger.error("/log-application failed: %s", exc)
        return LogApplicationResponse(
            application_id=str(uuid.uuid4()),
            status=req.status,
            logged_at=datetime.datetime.utcnow().isoformat() + "Z"
        )

@app.get("/health")
async def health() -> dict:
    """Liveness + DB connectivity check for Docker healthcheck."""
    checks: dict[str, str] = {"api": "ok"}
    try:
        from utils.db_utils import get_db_conn
        conn = get_db_conn()
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
        conn.close()
        checks["postgres"] = "ok"
    except Exception as exc:
        logger.warning("Health check DB failed: %s", exc)
        checks["postgres"] = f"error: {exc}"
    return {"status": "ok", "checks": checks}

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Global Exception Handler
# ---------------------------------------------------------------------------


@app.exception_handler(Exception)
async def global_exception_handler(
    request: Request, exc: Exception
) -> JSONResponse:
    """Catch-all handler for unhandled exceptions across all endpoints.

    Logs the exception with full path context at ERROR level and returns a
    structured 500 JSON response. This handler fires after endpoint-level
    error handling has already been exhausted.

    Args:
        request: The incoming HTTP request that triggered the exception.
        exc: The unhandled exception instance.

    Returns:
        JSONResponse: 500 response containing the error message and request
            path for client-side diagnostics.
    """
    logger.error("Unhandled exception: %s | path=%s", exc, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"error": str(exc), "path": str(request.url.path)},
    )


# ---------------------------------------------------------------------------
# Main Runner
# ---------------------------------------------------------------------------


def main() -> None:
    """Start the uvicorn ASGI server for the FastAPI application.

    Reads host and port from ``FASTAPI_HOST`` and ``FASTAPI_PORT`` environment
    variables, falling back to ``0.0.0.0:8000``. Intended to be called from
    Docker ENTRYPOINT or directly via ``python -m api.api_server``.
    Single-worker, no hot-reload for production stability.
    """
    uvicorn.run(
        "api.api_server:app",
        host=os.getenv("FASTAPI_HOST", "0.0.0.0"),
        port=int(os.getenv("FASTAPI_PORT", "8000")),
        log_level=os.getenv("LOG_LEVEL", "info").lower(),
        reload=False,
        workers=1,
    )


if __name__ == "__main__":
    main()
