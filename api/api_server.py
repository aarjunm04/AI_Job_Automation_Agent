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
import logging
from datetime import datetime
from typing import Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

from config.settings import db_config, run_config, api_config
from tools.postgres_tools import (
    get_run_stats,
    get_recent_applications,
    get_pending_manual_queue,
    update_application_status,
    log_event,
)
from tools.budget_tools import get_cost_summary

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
# Auth Dependency
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
# Endpoint 1: GET /status
# ---------------------------------------------------------------------------


@app.get("/status", response_model=HealthResponse, tags=["health"])
async def get_status() -> HealthResponse:
    """Health check endpoint. No authentication required.

    Probes the Postgres database by calling ``get_run_stats``. Returns
    ``"healthy"`` when the database is reachable, ``"degraded"`` otherwise.
    Safe to call at high frequency — no write side-effects.

    Returns:
        HealthResponse: Structured health payload including DB connectivity,
            pipeline mode, and server version.
    """
    db_connected: bool = True
    try:
        get_run_stats("health_check")
    except Exception:  # noqa: BLE001
        db_connected = False

    return HealthResponse(
        status="healthy" if db_connected else "degraded",
        timestamp=datetime.utcnow().isoformat() + "Z",
        db_connected=db_connected,
        pipeline_mode="dry_run" if run_config.dry_run else "live",
        dry_run=run_config.dry_run,
    )


# ---------------------------------------------------------------------------
# Endpoint 2: GET /dashboard
# ---------------------------------------------------------------------------


@app.get("/dashboard", tags=["data"])
async def get_dashboard(
    api_key: str = Depends(verify_api_key),
) -> JSONResponse:
    """Aggregate pipeline state for the Chrome Extension dashboard panel.

    Fetches the latest run stats, cost summary, 20 most recent applications,
    and up to 50 pending manual-queue jobs. Each data source is wrapped in an
    independent try/except — a single failing tool returns an empty
    dict/list without preventing the rest of the payload from being returned.

    Args:
        api_key: Validated API key from the ``X-API-Key`` header.

    Returns:
        JSONResponse: Dashboard payload containing pipeline run metrics,
            cost summary, recent applications, and manual queue with config.
    """
    # Run stats — most recent run batch
    run_stats: dict[str, Any] = {}
    try:
        raw: Any = get_run_stats("latest")
        run_stats = json.loads(raw) if isinstance(raw, str) else {}
    except Exception:  # noqa: BLE001
        run_stats = {}

    # Cost summary for the current or most recent run
    cost_summary: dict[str, Any] = {}
    try:
        raw = get_cost_summary("latest")
        cost_summary = json.loads(raw) if isinstance(raw, str) else {}
    except Exception:  # noqa: BLE001
        cost_summary = {}

    # Recent applications list
    recent_apps: list[Any] = []
    try:
        raw = get_recent_applications(limit=20)
        parsed: Any = json.loads(raw) if isinstance(raw, str) else raw
        recent_apps = parsed if isinstance(parsed, list) else []
    except Exception:  # noqa: BLE001
        recent_apps = []

    # Pending manual-queue jobs
    manual_queue: list[Any] = []
    try:
        raw = get_pending_manual_queue(limit=50)
        parsed = json.loads(raw) if isinstance(raw, str) else raw
        manual_queue = parsed if isinstance(parsed, list) else []
    except Exception:  # noqa: BLE001
        manual_queue = []

    return JSONResponse(
        content={
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "latest_run": run_stats,
            "cost": cost_summary,
            "recent_applications": recent_apps,
            "manual_queue": {
                "count": len(manual_queue),
                "jobs": manual_queue,
            },
            "config": {
                "dry_run": run_config.dry_run,
                "auto_apply_enabled": run_config.auto_apply_enabled,
                "jobs_per_run_target": run_config.jobs_per_run_target,
                "default_resume": run_config.default_resume,
            },
        }
    )


# ---------------------------------------------------------------------------
# Endpoint 3: POST /apply/manual
# ---------------------------------------------------------------------------


@app.post("/apply/manual", tags=["actions"])
async def trigger_manual_apply(
    request: ManualApplyRequest,
    api_key: str = Depends(verify_api_key),
) -> JSONResponse:
    """Queue a job post for manual apply. The Chrome Extension 'Apply Now' button.

    Does NOT invoke Playwright directly. Updates the application record to
    ``manual_queued`` status and logs the event so the next pipeline run picks
    it up. The job will appear in the Notion Applications DB after the next
    Tracker Agent sync.

    Args:
        request: Manual apply payload containing job_post_id, user_id, and
            optional resume override and notes.
        api_key: Validated API key from the ``X-API-Key`` header.

    Returns:
        JSONResponse: Queue confirmation with job_post_id, resolved resume
            filename, success message, and UTC timestamp.

    Raises:
        HTTPException: 500 on any unexpected DB or tool error.
    """
    try:
        # Step 1: Validate DB is reachable before making any writes
        get_run_stats("health_check")

        # Step 2: Resolve resume — use override if provided, else pipeline default
        resume: str = request.resume_filename or run_config.default_resume

        # Step 3: Update application record to manual_queued
        update_application_status(
            application_id=request.job_post_id,
            new_status="manual_queued",
            resume_used=resume,
            notes=request.notes,
        )

        # Step 4: Emit audit log event
        log_event(
            run_batch_id="manual_trigger",
            level="INFO",
            event_type="chrome_extension_manual_queue",
            message=(
                f"Manual apply queued | job={request.job_post_id} "
                f"| resume={resume} | user={request.user_id}"
            ),
        )

        return JSONResponse(
            content={
                "queued": True,
                "job_post_id": request.job_post_id,
                "resume": resume,
                "message": (
                    "Job queued for manual apply. Will appear in Notion "
                    "Applications DB within next sync."
                ),
                "timestamp": datetime.utcnow().isoformat() + "Z",
            }
        )

    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# Bonus Endpoint 4: PATCH /application/{application_id}/status
# ---------------------------------------------------------------------------


_VALID_STATUSES: list[str] = [
    "applied",
    "rejected",
    "interview",
    "offer",
    "withdrawn",
    "manual_queued",
]


@app.patch("/application/{application_id}/status", tags=["actions"])
async def update_status(
    application_id: str,
    body: ApplicationStatusUpdate,
    api_key: str = Depends(verify_api_key),
) -> JSONResponse:
    """Update the lifecycle stage of an existing application record.

    Allows the Chrome Extension to move an application through its pipeline
    stages (e.g. Applied → Interview → Offer). Validates the status value
    before writing and logs every status transition to the audit log.

    Args:
        application_id: UUID of the applications record (path parameter).
        body: Status update payload with ``new_status`` and optional ``notes``.
        api_key: Validated API key from the ``X-API-Key`` header.

    Returns:
        JSONResponse: Confirmation payload with ``application_id`` and
            ``new_status``.

    Raises:
        HTTPException: 422 if ``body.new_status`` is not in the valid set.
        HTTPException: 500 on any unexpected DB or tool error.
    """
    if body.new_status not in _VALID_STATUSES:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Invalid status '{body.new_status}'. "
                f"Valid values: {_VALID_STATUSES}"
            ),
        )

    update_application_status(
        application_id,
        body.new_status,
        notes=body.notes,
    )

    log_event(
        run_batch_id="manual_trigger",
        level="INFO",
        event_type="status_update",
        message=f"application={application_id} → {body.new_status}",
    )

    return JSONResponse(
        content={
            "updated": True,
            "application_id": application_id,
            "new_status": body.new_status,
        }
    )


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
