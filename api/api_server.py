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
import asyncio
import psycopg2
import psycopg2.extras
import chromadb

from config.settings import db_config, run_config, api_config
from utils.db_utils import get_db_conn
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
API_KEY: str = os.getenv("SCRAPER_SERVICE_API_KEY", "")

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
    status: str
    timestamp: str
    version: str
    db: str


# ---------------------------------------------------------------------------
# Chrome Extension Pydantic Models
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


# Removed Bearer token auth logic

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
    title: str
    company: str
    description: str
    url: str

class MatchResponse(BaseModel):
    fit_score: float
    resume_id: str
    talking_points: list[str]
    route: str

@app.post("/match", response_model=MatchResponse)
async def match(req: MatchRequest, _: str = Depends(verify_api_key)) -> MatchResponse:
    """Score a job against the best matching resume via RAG."""
    try:
        from rag_systems.rag_pipeline import create_default_pipeline
        from rag_systems.chromadb_store import ChromaStore
        import asyncio

        def _sync_match():
            store = ChromaStore()
            pipeline = create_default_pipeline(store)
            vec = pipeline.embed_query(req.description)
            anchors = pipeline.get_top_resume_anchors(vec, k=1)
            
            fit_score = 0.5
            resume_label = "Aarjun_Gen.pdf"
            if anchors and len(anchors["ids"]) > 0 and len(anchors["ids"][0]) > 0:
                dist = anchors["distances"][0][0]
                fit_score = max(0.0, min(1.0, round(1.0 - dist, 4) * 0.85 + 0.10))
                resume_label = anchors["metadatas"][0][0].get("filename", "Aarjun_Gen.pdf")
            
            route = "manual"
            if fit_score >= 0.75:
                route = "auto"
            elif fit_score < 0.45:
                route = "skip"
                
            return {
                "fit_score": float(fit_score),
                "resume_id": resume_label,
                "talking_points": [],
                "route": route
            }

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, _sync_match)
        return MatchResponse(**result)
    except Exception as exc:
        logger.error("/match failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))

class AutofillRequest(BaseModel):
    job_id: str
    platform: str
    url: str

class AutofillResponse(BaseModel):
    fields: dict[str, str]

@app.post("/autofill", response_model=AutofillResponse)
async def autofill(req: AutofillRequest, _: str = Depends(verify_api_key)) -> AutofillResponse:
    """Return candidate profile for Chrome Extension form pre-fill."""
    try:
        return AutofillResponse(
            fields={
                "first_name": os.getenv("CANDIDATE_FIRST_NAME", ""),
                "last_name": os.getenv("CANDIDATE_LAST_NAME", ""),
                "email": os.getenv("CANDIDATE_EMAIL", ""),
                "phone": os.getenv("CANDIDATE_PHONE", ""),
                "linkedin_url": os.getenv("CANDIDATE_LINKEDIN_URL", ""),
                "resume_pdf_path": os.getenv("RESUME_PDF_PATH", ""),
            }
        )
    except Exception as exc:
        logger.error("/autofill failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))

class LogApplicationRequest(BaseModel):
    job_id: str
    platform: str
    url: str
    status: str
    resume_id: str
    user_id: str

class LogApplicationResponse(BaseModel):
    id: str
    status: str

@app.post("/log_application", response_model=LogApplicationResponse)
async def log_application(req: LogApplicationRequest, _: str = Depends(verify_api_key)) -> LogApplicationResponse:
    """Log a job application from the Chrome Extension to Postgres."""
    try:
        conn = await asyncio.to_thread(get_db_conn)
        try:
            with conn:
                with conn.cursor() as cur:
                    # On conflict do nothing
                    cur.execute(
                        """INSERT INTO applications
                           (job_post_id, user_id, mode, status,
                            platform, resume_used)
                           VALUES (%s, %s, %s, %s, %s, %s)
                           ON CONFLICT DO NOTHING
                           RETURNING id""",
                        (
                            req.job_id,
                            req.user_id,
                            "manual",
                            req.status,
                            req.platform,
                            req.resume_id
                        ),
                    )
                    row = cur.fetchone()
                    if not row:
                        raise HTTPException(
                            status_code=409,
                            detail="Application already logged (duplicate)",
                        )
                    application_id = str(row[0])
        finally:
            conn.close()
            
        logger.info(
            "Logged application %s for job %s status=%s",
            application_id, req.job_id, req.status,
        )
        return LogApplicationResponse(
            id=application_id,
            status=req.status,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("/log_application failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))

@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Liveness + DB connectivity check for Docker healthcheck."""
    db_ok = False
    try:
        def _ping_db():
            conn = get_db_conn()
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
            conn.close()
        await asyncio.to_thread(_ping_db)
        db_ok = True
    except Exception as exc:
        logger.warning("Health check DB failed: %s", exc)
        
    return HealthResponse(
        status="ok",
        timestamp=datetime.utcnow().isoformat(),
        version=os.getenv("APP_VERSION", "1.0.0"),
        db="connected" if db_ok else "unreachable",
    )

class QueueCountResponse(BaseModel):
    count: int

@app.get("/queue/count", response_model=QueueCountResponse)
async def queue_count(_: str = Depends(verify_api_key)) -> QueueCountResponse:
    """Return current depth of the manual apply queue."""
    try:
        def _get_count():
            conn = get_db_conn()
            try:
                with conn.cursor() as cur:
                    cur.execute("SELECT COUNT(*) FROM jobs WHERE route = 'manual' AND status = 'pending'")
                    row = cur.fetchone()
                    return row[0] if row else 0
            finally:
                conn.close()
        
        count = await asyncio.to_thread(_get_count)
        return QueueCountResponse(count=count)
    except Exception as exc:
        logger.error("/queue/count failed: %s", exc)
        return QueueCountResponse(count=0)

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
