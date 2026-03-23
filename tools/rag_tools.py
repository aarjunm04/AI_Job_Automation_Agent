from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

import httpx
from crewai.tools import tool
import agentops
from agentops.sdk.decorators import agent, operation

from tools.postgres_tools import _fetch_user_config

# RAG server connection — set in docker-compose environment block
RAG_SERVER_URL: str = os.getenv("RAG_SERVER_URL", "http://ai_rag_server:8090")
RAG_API_KEY: str = os.getenv("SCRAPER_SERVICE_API_KEY", "")


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _retry_call(fn, *args, max_retries: int = 3, **kwargs):
    """Execute fn with exponential backoff retry."""
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            last_exc = exc
            wait = 2.0 ** attempt
            logger.warning(
                "Attempt %d/%d failed: %s — retrying in %.1fs",
                attempt + 1, max_retries, exc, wait,
            )
            time.sleep(wait)
    raise RuntimeError(
        f"All {max_retries} attempts failed. Last error: {last_exc}"
    )


__all__ = [
    "query_resume_match",
    "get_resume_context",
    "embed_job_description",
    "get_resume_pdf_path",
]


def _safe_json_dumps(payload: Dict[str, Any]) -> str:
    """Safe JSON serialization helper."""
    try:
        return json.dumps(payload, default=str)
    except Exception as exc:  # noqa: BLE001
        logger.error("JSON serialization failed: %s", exc, exc_info=True)
        # Fallback very minimal payload
        return json.dumps({"error": "serialization_failed"})


@tool
@operation
@agentops.track_tool
def query_resume_match(job_description: str, job_title: str, required_skills: str) -> str:
    """Suggest the best resume for a given job description.

    Calls ``POST {RAG_SERVER_URL}/match`` and retries up to 3 times with
    exponential back-off.  Falls back to the default resume stored in
    ``users.user_settings`` (Postgres) when the RAG server is unreachable.

    Args:
        job_description: Full text of the job description.
        job_title: Title of the role.
        required_skills: Comma-separated list of required skills.

    Returns:
        JSON string with keys: resume_suggested, similarity_score, fit_score,
        match_reasoning, talking_points.
    """
    for attempt in range(3):
        try:
            with httpx.Client(timeout=30.0) as client:
                resp = client.post(
                    f"{RAG_SERVER_URL}/match",
                    headers={"X-API-Key": RAG_API_KEY, "Content-Type": "application/json"},
                    json={
                        "job_description": job_description,
                        "job_title": job_title,
                        "required_skills": required_skills,
                    },
                )
            if resp.status_code == 200:
                data: Dict[str, Any] = resp.json()
                return _safe_json_dumps({
                    "resume_suggested": data.get("resume_suggested", ""),
                    "similarity_score": float(data.get("similarity_score", 0.0)),
                    "fit_score": float(data.get("fit_score", 0.0)),
                    "match_reasoning": data.get("match_reasoning", ""),
                    "talking_points": data.get("talking_points", []),
                })
            logger.warning(
                "query_resume_match: RAG server returned HTTP %d (attempt %d/3)",
                resp.status_code,
                attempt + 1,
            )
        except httpx.RequestError as exc:
            logger.warning(
                "query_resume_match: connection error (attempt %d/3): %s",
                attempt + 1,
                exc,
            )
        if attempt < 2:
            time.sleep(2 ** attempt)

    # All retries exhausted — fall back to DB default_resume
    logger.error("query_resume_match: RAG server unreachable after 3 attempts, using DB fallback")
    fallback_resume: str = "AarjunGen.pdf"
    try:
        _cfg: Dict[str, Any] = _fetch_user_config()
        _db_resume: Optional[str] = _cfg.get("user_settings", {}).get("default_resume")
        if _db_resume:
            fallback_resume = str(_db_resume)
        else:
            logger.warning(
                "query_resume_match: default_resume NULL in users.user_settings "
                "— using hardcoded fallback"
            )
    except Exception as db_exc:  # noqa: BLE001
        logger.warning(
            "query_resume_match: DB fallback for default_resume also failed (%s) "
            "— using hardcoded AarjunGen.pdf",
            db_exc,
        )
    return _safe_json_dumps({
        "resume_suggested": fallback_resume,
        "similarity_score": 0.0,
        "fit_score": 0.0,
        "match_reasoning": "rag_unavailable_db_fallback",
        "talking_points": [],
    })


@tool
@operation
@agentops.track_tool
def get_resume_context(resume_filename: str, job_description: str) -> str:
    """Return formatted resume text chunks relevant to the given job description.

    Calls ``POST {RAG_SERVER_URL}/autofill`` and returns chunks joined as a
    plain string for direct LLM context injection.  Returns an empty string
    and logs a WARNING on any HTTP failure.

    Args:
        resume_filename: Filename of the selected resume (forwarded to server).
        job_description: Full text of the job description to match against.

    Returns:
        Newline-separated ``[CHUNK N] ...`` string, or empty string on failure.
    """
    def _do_post():
        with httpx.Client(timeout=30.0) as client:
            return client.post(
                f"{RAG_SERVER_URL}/autofill",
                headers={"X-API-Key": RAG_API_KEY, "Content-Type": "application/json"},
                json={"resume_filename": resume_filename, "job_description": job_description},
            )

    try:
        resp = _retry_call(_do_post)
        if resp.status_code != 200:
            logger.warning(
                "get_resume_context: rag_autofill_unavailable — HTTP %d", resp.status_code
            )
            return ""
        data: Dict[str, Any] = resp.json()
        chunks: List[str] = data.get("context_chunks", [])
        if not isinstance(chunks, list):
            chunks = []
        formatted: List[str] = [
            f"[CHUNK {i + 1}] {c.strip()}"
            for i, c in enumerate(chunks)
            if isinstance(c, str) and c.strip()
        ]
        return "\n\n".join(formatted)
    except Exception as exc:  # noqa: BLE001
        logger.warning("get_resume_context: rag_autofill_unavailable — %s", exc)
        return ""


@tool
@operation
@agentops.track_tool
def embed_job_description(job_url: str, job_description: str) -> str:
    """Trigger embedding of a job description via the RAG server.

    Reuses ``POST {RAG_SERVER_URL}/match`` with the job description as input.
    The match payload is discarded; the call ensures the RAG server indexes
    the description for future similarity lookups.

    Args:
        job_url: Full URL of the job posting (used as identifier).
        job_description: Full text of the job description to embed.

    Returns:
        JSON string with keys: embedded (bool), job_url (str),
        and error (str, only present on failure).
    """
    def _do_post():
        with httpx.Client(timeout=30.0) as client:
            return client.post(
                f"{RAG_SERVER_URL}/match",
                headers={"X-API-Key": RAG_API_KEY, "Content-Type": "application/json"},
                json={
                    "job_description": job_description,
                    "job_title": "",
                    "required_skills": "",
                },
            )

    try:
        resp = _retry_call(_do_post)
        if resp.status_code == 200:
            return _safe_json_dumps({"embedded": True, "job_url": job_url})
        logger.warning(
            "embed_job_description: RAG server returned HTTP %d", resp.status_code
        )
        return _safe_json_dumps(
            {"embedded": False, "job_url": job_url, "error": "rag_server_unavailable"}
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("embed_job_description failed: %s", exc, exc_info=True)
        return _safe_json_dumps(
            {"embedded": False, "job_url": job_url, "error": "rag_server_unavailable"}
        )


@tool
@operation
@agentops.track_tool
def get_resume_pdf_path(resume_filename: str) -> str:
    """
    Return the absolute path and existence flag for a given resume PDF.

    Uses RESUME_DIR (default: 'resumes') as the base directory.
    Returns a JSON string:
    {
        "path": str,
        "exists": bool,
        "filename": str
    }
    """
    try:
        base_dir = os.getenv("RESUME_DIR", "resumes")
        path = os.path.abspath(os.path.join(base_dir, resume_filename))
        exists = os.path.isfile(path)

        response = {
            "path": path,
            "exists": exists,
            "filename": resume_filename,
        }
        return _safe_json_dumps(response)
    except Exception as exc:  # noqa: BLE001
        logger.error("get_resume_pdf_path failed: %s", exc, exc_info=True)
        base_dir = os.getenv("RESUME_DIR", "resumes")
        fallback_path = os.path.abspath(os.path.join(base_dir, resume_filename))
        response = {
            "path": fallback_path,
            "exists": False,
            "filename": resume_filename,
        }
        return _safe_json_dumps(response)
