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
# Chrome Extension Helpers
# ---------------------------------------------------------------------------


def get_db_connection() -> psycopg2.extensions.connection:
    """Create a Postgres connection using ACTIVE_DB to select the URL.

    Returns:
        psycopg2 connection with RealDictCursor factory.
    """
    active: str = os.getenv("ACTIVE_DB", "local")
    url: str = (
        os.getenv("LOCAL_POSTGRES_URL", "")
        if active == "local"
        else os.getenv("SUPABASE_URL", "")
    )
    return psycopg2.connect(url, cursor_factory=psycopg2.extras.RealDictCursor)


def get_chroma_client() -> chromadb.ClientAPI:
    """Create a ChromaDB persistent client.

    Returns:
        ChromaDB PersistentClient pointed at CHROMADB_PATH.
    """
    return chromadb.PersistentClient(
        path=os.getenv("CHROMADB_PATH", "app/chromadb")
    )


# Keyword map for autofill field matching — order matters, first match wins
_AUTOFILL_KEYWORD_MAP: list[tuple[list[str], str]] = [
    (["first name", "firstname", "given name"], "first_name"),
    (["last name", "lastname", "surname", "family name"], "last_name"),
    (["full name", "your name", "candidate name", "name"], "name"),
    (["email", "e-mail", "email address"], "email"),
    (["phone", "mobile", "tel", "telephone", "contact number", "cell"], "phone"),
    (["linkedin", "linkedin url", "linkedin profile"], "linkedin"),
    (
        ["portfolio", "github", "website", "personal url", "personal site", "url"],
        "portfolio",
    ),
    (
        ["location", "city", "country", "current location", "where are you", "residence"],
        "location",
    ),
    (
        [
            "years of experience",
            "years experience",
            "experience years",
            "how many years",
            "total experience",
        ],
        "experience",
    ),
]

# Talking-point keyword triggers
_TP_TRIGGERS: list[tuple[list[str], str]] = [
    (["python", "Python"], "Highlight Python experience and projects"),
    (
        ["machine learning", "ML", "deep learning"],
        "Discuss ML project outcomes",
    ),
    (
        ["sql", "SQL", "postgres", "database"],
        "Mention database design experience",
    ),
    (["api", "API", "REST", "FastAPI"], "Reference API development work"),
    (
        ["docker", "Docker", "kubernetes", "k8s"],
        "Highlight containerisation skills",
    ),
    (["remote", "distributed"], "Emphasise remote work readiness"),
    (
        ["leadership", "team lead", "manage"],
        "Discuss any leadership experience",
    ),
    (
        ["startup", "fast-paced"],
        "Demonstrate adaptability and ownership mindset",
    ),
]


# ---------------------------------------------------------------------------
# Chrome Extension Endpoint 5: GET /health
# ---------------------------------------------------------------------------


@app.get("/health", tags=["health"])
async def health_check() -> JSONResponse:
    """Lightweight health endpoint for Chrome Extension test-connection button.

    No authentication required. Returns a simple status payload. Separate from
    the more detailed ``/status`` endpoint.

    Returns:
        JSONResponse with ``{"status": "ok"}``.
    """
    return JSONResponse(content={"status": "ok"})


# ---------------------------------------------------------------------------
# Chrome Extension Endpoint 6: POST /match
# ---------------------------------------------------------------------------


@app.post("/match", response_model=MatchResponse, tags=["chrome-extension"])
async def match_resume(
    request: MatchRequest,
    _: bool = Depends(verify_bearer_token),
) -> MatchResponse:
    """RAG resume match for the current job page.

    Queries ChromaDB ``resumes`` collection with the job description text to
    find the best-matching resume variant. Computes a composite fit score and
    generates rule-based talking points from JD keyword detection.

    Args:
        request: Match request with job_url and jd_text.

    Returns:
        MatchResponse with resume suggestion, scores, reasoning, and
        talking points.

    Raises:
        HTTPException: 503 if ChromaDB is unreachable or has no resumes.
        HTTPException: 500 on any unexpected error.
    """
    try:
        # Step 1 — ChromaDB query
        try:
            client = get_chroma_client()
            collection = client.get_collection("resumes")
            results = collection.query(
                query_texts=[request.jd_text],
                n_results=1,
                include=["documents", "metadatas", "distances"],
            )
            if not results["ids"] or not results["ids"][0]:
                raise HTTPException(
                    status_code=503, detail="No resumes in ChromaDB"
                )

            distance: float = float(results["distances"][0][0])
            metadata: dict[str, Any] = results["metadatas"][0][0]

            similarity_score: float = round(1.0 - distance, 4)
            resume_suggested: str = metadata.get(
                "filename",
                os.getenv("DEFAULT_RESUME", "AarjunGen.pdf"),
            )
        except HTTPException:
            raise
        except Exception as e:
            logger.error("ChromaDB query failed: %s", str(e))
            raise HTTPException(
                status_code=503, detail="ChromaDB unavailable"
            ) from e

        # Step 2 — Fit score (Phase 1: similarity-based, Phase 2 adds LLM)
        fit_score: float = round(similarity_score * 0.85 + 0.10, 4)
        fit_score = max(0.0, min(1.0, fit_score))

        # Step 3 — Match reasoning (rule-based Phase 1)
        if fit_score >= 0.75:
            match_reasoning = (
                "Strong match — your profile closely aligns with "
                "this role's requirements."
            )
        elif fit_score >= 0.50:
            match_reasoning = (
                "Moderate match — core skills align but some gaps "
                "may exist."
            )
        elif fit_score >= 0.40:
            match_reasoning = (
                "Low confidence match — review carefully before applying."
            )
        else:
            match_reasoning = "Weak match — significant skill gaps detected."

        # Step 4 — Talking points (rule-based keyword detection)
        jd_text: str = request.jd_text
        talking_points: list[str] = []
        seen_points: set[str] = set()
        for triggers, point in _TP_TRIGGERS:
            for kw in triggers:
                if kw in jd_text:
                    if point not in seen_points:
                        talking_points.append(point)
                        seen_points.add(point)
                    break
        # Always-add points
        talking_points.append(
            "Align answers with company mission and the specific "
            "job requirements"
        )
        talking_points.append(
            f"Suggested resume: {resume_suggested} — refer to this "
            f"variant's highlights"
        )
        talking_points = talking_points[:8]  # max 8

        # Step 5 — Return
        return MatchResponse(
            resume_suggested=resume_suggested,
            similarity_score=similarity_score,
            fit_score=fit_score,
            match_reasoning=match_reasoning,
            talking_points=talking_points,
            autofill_ready=True,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Unexpected error in /match: %s", str(e))
        raise HTTPException(
            status_code=500, detail="Internal server error"
        ) from e


# ---------------------------------------------------------------------------
# Chrome Extension Endpoint 7: POST /autofill
# ---------------------------------------------------------------------------


@app.post("/autofill", response_model=AutofillResponse, tags=["chrome-extension"])
async def autofill_fields(
    request: AutofillRequest,
    _: bool = Depends(verify_bearer_token),
) -> AutofillResponse:
    """Return field-value mappings for detected form fields.

    Reads the user's profile from environment variables and matches detected
    form fields by keyword analysis of their label, placeholder, name, and id.

    Args:
        request: Autofill request with job_url and detected_fields list.

    Returns:
        AutofillResponse with field_mappings dict, unmapped_fields list,
        and mapped_count.

    Raises:
        HTTPException: 500 on any unexpected error.
    """
    try:
        # Load user profile from environment
        full_name: str = os.getenv("USERNAME", "")
        profile: dict[str, str] = {
            "name": full_name,
            "first_name": full_name.split()[0] if full_name.strip() else "",
            "last_name": (
                full_name.split()[-1]
                if len(full_name.strip().split()) > 1
                else full_name
            ),
            "email": os.getenv("USER_EMAIL", ""),
            "phone": os.getenv("USER_PHONE", ""),
            "linkedin": os.getenv("USER_LINKEDIN_URL", ""),
            "portfolio": os.getenv("USER_PORTFOLIO_URL", ""),
            "location": os.getenv("USER_LOCATION", ""),
            "experience": os.getenv("USER_YEARS_EXPERIENCE", ""),
        }

        field_mappings: dict[str, str] = {}
        unmapped_fields: list[str] = []

        for field in request.detected_fields:
            combined: str = (
                f"{field.label_text} {field.placeholder} "
                f"{field.name} {field.id}"
            ).lower().strip()

            key_to_use: str = field.id if field.id else field.name
            if not key_to_use:
                key_to_use = f"field_{field.index}"

            matched: bool = False
            for keywords, profile_key in _AUTOFILL_KEYWORD_MAP:
                for kw in keywords:
                    if kw in combined:
                        value: str = profile.get(profile_key, "")
                        if not value:
                            break  # profile value empty — treat as unmapped
                        field_mappings[key_to_use] = value
                        matched = True
                        break
                if matched:
                    break

            if not matched:
                unmapped_fields.append(key_to_use)

        return AutofillResponse(
            field_mappings=field_mappings,
            unmapped_fields=unmapped_fields,
            mapped_count=len(field_mappings),
        )

    except Exception as e:
        logger.error("Unexpected error in /autofill: %s", str(e))
        raise HTTPException(
            status_code=500, detail="Internal server error"
        ) from e


# ---------------------------------------------------------------------------
# Chrome Extension Endpoint 8: GET /queue-count
# ---------------------------------------------------------------------------


@app.get(
    "/queue-count", response_model=QueueCountResponse, tags=["chrome-extension"]
)
async def get_queue_count(
    _: bool = Depends(verify_bearer_token),
) -> QueueCountResponse:
    """Return the current manual-queue depth from Postgres.

    Queries the ``applications`` table for rows with
    ``status = 'manual_queued'``.

    Returns:
        QueueCountResponse with the pending count.

    Raises:
        HTTPException: 503 if the database is unreachable.
        HTTPException: 500 on any unexpected error.
    """
    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) AS count FROM applications "
                "WHERE status = %s",
                ("manual_queued",),
            )
            row = cur.fetchone()
            count: int = int(row["count"]) if row else 0
        return QueueCountResponse(count=count)
    except psycopg2.Error as e:
        logger.error("DB error in /queue-count: %s", str(e))
        raise HTTPException(
            status_code=503, detail="Database unavailable"
        ) from e
    except Exception as e:
        logger.error("Unexpected error in /queue-count: %s", str(e))
        raise HTTPException(
            status_code=500, detail="Internal server error"
        ) from e
    finally:
        if conn:
            conn.close()


# ---------------------------------------------------------------------------
# Chrome Extension Endpoint 9: POST /log-application
# ---------------------------------------------------------------------------


@app.post(
    "/log-application",
    response_model=LogApplicationResponse,
    tags=["chrome-extension"],
)
async def log_application(
    request: LogApplicationRequest,
    _: bool = Depends(verify_bearer_token),
) -> LogApplicationResponse:
    """Log a manual application to Postgres and sync to Notion.

    All Postgres writes execute in a single atomic transaction. Notion sync
    is non-fatal — a Notion failure does not roll back the Postgres commit.

    Args:
        request: Log application payload with job_url, resume_used, platform,
            applied_at, and optional notes.

    Returns:
        LogApplicationResponse with application_id, notion_page_id, and
        status.

    Raises:
        HTTPException: 503 if the database write fails.
        HTTPException: 500 on any unexpected error.
    """
    conn = None
    notion_page_id: str = ""
    application_id: str = ""
    job_id: str = ""

    try:
        conn = get_db_connection()
        conn.autocommit = False

        with conn.cursor() as cur:
            # Check if job exists by job_url
            cur.execute(
                "SELECT id FROM jobs WHERE job_url = %s",
                (request.job_url,),
            )
            job_row = cur.fetchone()

            if job_row:
                job_id = str(job_row["id"])
            else:
                # Create minimal job record for manually logged applications
                job_id = str(uuid.uuid4())
                cur.execute(
                    """
                    INSERT INTO jobs
                        (id, job_url, title, company, platform, fit_score,
                         route, resume_used, status, scraped_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        job_id,
                        request.job_url,
                        "Unknown Title",
                        "Unknown Company",
                        request.platform,
                        0.0,
                        "manual",
                        request.resume_used,
                        "applied_manual",
                        datetime.now(timezone.utc),
                    ),
                )

            # Update job status and resume
            cur.execute(
                "UPDATE jobs SET status = %s, resume_used = %s WHERE id = %s",
                ("applied_manual", request.resume_used, job_id),
            )

            # Insert application record
            application_id = str(uuid.uuid4())
            applied_at_dt = datetime.fromisoformat(
                request.applied_at.replace("Z", "+00:00")
            )
            cur.execute(
                """
                INSERT INTO applications
                    (id, job_id, status, proof_json, proof_confidence,
                     notion_synced, applied_at, resume_used, notes)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    application_id,
                    job_id,
                    "applied_manual",
                    None,
                    None,
                    False,
                    applied_at_dt,
                    request.resume_used,
                    request.notes,
                ),
            )

            # Write audit log entry
            audit_id: str = str(uuid.uuid4())
            cur.execute(
                """
                INSERT INTO audit_logs
                    (id, run_id, agent, event_type, error_code,
                     metadata_json, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    audit_id,
                    "manual_apply",
                    "chrome_extension",
                    "manual_application_logged",
                    None,
                    psycopg2.extras.Json(
                        {
                            "job_url": request.job_url,
                            "platform": request.platform,
                            "resume_used": request.resume_used,
                            "application_id": application_id,
                        }
                    ),
                    datetime.now(timezone.utc),
                ),
            )

        conn.commit()
        logger.info(
            "Application logged: %s for %s", application_id, request.job_url
        )

        # Notion sync — non-fatal, separate try/except
        try:
            notion_client = NotionClient()
            # NotionClient.create_job_tracker_page takes individual args
            result = notion_client.create_job_tracker_page(
                title="Unknown Title",
                company="Unknown Company",
                job_url=request.job_url,
                stage="Applied",
                date_applied=applied_at_dt.strftime("%Y-%m-%d"),
                platform=request.platform,
                applied_via="Manual",
                ctc="",
                notes=request.notes or "",
                job_type="",
                location="",
                resume_used=request.resume_used,
            )
            notion_page_id = result.get("id", "") if isinstance(result, dict) else ""

            # Update notion_synced flag (non-fatal separate transaction)
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE applications SET notion_synced = %s WHERE id = %s",
                    (True, application_id),
                )
            conn.commit()
            logger.info("Notion sync success: page_id=%s", notion_page_id)

        except Exception as notion_err:
            logger.warning(
                "Notion sync failed (non-fatal) for application %s: %s",
                application_id,
                str(notion_err),
            )
            notion_page_id = ""

        return LogApplicationResponse(
            application_id=application_id,
            notion_page_id=notion_page_id,
            status="success",
        )

    except HTTPException:
        raise
    except psycopg2.Error as e:
        if conn:
            conn.rollback()
        logger.error("DB error in /log-application: %s", str(e))
        raise HTTPException(
            status_code=503, detail="Database write failed"
        ) from e
    except Exception as e:
        if conn:
            conn.rollback()
        logger.error("Unexpected error in /log-application: %s", str(e))
        raise HTTPException(
            status_code=500, detail="Internal server error"
        ) from e
    finally:
        if conn:
            conn.close()


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
