from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

from crewai.tools import tool
import agentops
from agentops.sdk.decorators import agent, operation


# Ensure project root is on sys.path so that rag_systems is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag_systems import rag_api  # type: ignore  # Imported via adjusted sys.path
from rag_systems.rag_pipeline import EmbeddingService  # type: ignore
from rag_systems.chromadb_store import ChromaStore, ChromaStoreConfig  # type: ignore


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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
def query_resume_match(job_description: str, job_title: str, required_skills: str) -> str:
    """
    Suggest the best resume for a given job description.

    Returns a JSON string with:
    {
        "resume_suggested": str,
        "similarity_score": float,
        "fit_score": float,
        "match_reasoning": str,
        "talking_points": list
    }
    """
    try:
        parts: List[str] = []
        if job_title and job_title.strip():
            parts.append(job_title.strip())
        if job_description and job_description.strip():
            parts.append(job_description.strip())
        if required_skills and required_skills.strip():
            parts.append(required_skills.strip())

        combined_text = "\n\n".join(parts)

        job_payload = {"job_text": combined_text}
        result = rag_api.select_resume(job_payload)  # type: ignore[no-untyped-call]

        # Normalise result to a dict-like structure
        if isinstance(result, dict):
            data = result
        else:
            data = getattr(result, "__dict__", {}) or {}

        resume_suggested = data.get("top_resume_id")
        similarity_score = float(data.get("top_score", 0.0) or 0.0)
        fit_score = similarity_score

        match_reasoning = ""
        talking_points: List[str] = []

        try:
            candidates = data.get("candidates") or []
            if isinstance(candidates, list) and candidates:
                top_cand = candidates[0]
                if isinstance(top_cand, dict):
                    final_score = float(top_cand.get("final_score", similarity_score) or 0.0)
                    match_reasoning = (
                        f"Selected resume '{resume_suggested}' with final_score={final_score:.3f} "
                        f"based on anchor and chunk similarity."
                    )

                    chunks = top_cand.get("recommended_chunks") or []
                    if isinstance(chunks, list):
                        for chunk in chunks:
                            if isinstance(chunk, dict):
                                text = chunk.get("document") or chunk.get("text") or ""
                                if isinstance(text, str) and text.strip():
                                    talking_points.append(text.strip())
                    # Limit the number of talking points for brevity
                    talking_points = talking_points[:5]
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to derive detailed reasoning from RAG result: %s", exc)

        response = {
            "resume_suggested": resume_suggested,
            "similarity_score": similarity_score,
            "fit_score": fit_score,
            "match_reasoning": match_reasoning,
            "talking_points": talking_points,
        }
        return _safe_json_dumps(response)
    except Exception as exc:  # noqa: BLE001
        logger.error("query_resume_match failed: %s", exc, exc_info=True)
        fallback_resume = os.getenv("DEFAULT_RESUME", "AarjunGen.pdf")
        error_response = {
            "error": "rag_unavailable",
            "fallback_resume": fallback_resume,
        }
        return _safe_json_dumps(error_response)


@tool
@operation
def get_resume_context(resume_filename: str, job_description: str) -> str:
    """
    Return formatted resume text chunks relevant to the given job description.

    The return value is a plain string suitable for direct LLM context injection.
    """
    del resume_filename  # Currently unused, selection is handled inside the RAG engine.

    try:
        job_payload = {"job_text": job_description}
        result = rag_api.get_rag_context(  # type: ignore[no-untyped-call]
            job_payload=job_payload,
            top_k_chunks=10,
        )

        if isinstance(result, dict):
            chunks = result.get("top_chunks") or result.get("chunks") or []
        else:
            chunks = getattr(result, "top_chunks", []) or getattr(result, "chunks", [])

        if not isinstance(chunks, list):
            chunks = []

        formatted_chunks: List[str] = []
        for idx, chunk in enumerate(chunks, start=1):
            text = ""
            if isinstance(chunk, dict):
                text = chunk.get("document") or chunk.get("text") or ""
            if isinstance(text, str):
                text = text.strip()
            else:
                text = ""
            if not text:
                continue
            formatted_chunks.append(f"[CHUNK {idx}] {text}")

        return "\n\n".join(formatted_chunks)
    except Exception as exc:  # noqa: BLE001
        logger.error("get_resume_context failed: %s", exc, exc_info=True)
        return "RAG context unavailable for this job description."


@tool
@operation
def embed_job_description(job_url: str, job_description: str) -> str:
    """
    Embed a job description into the ChromaDB 'job_descriptions' collection.

    Returns a JSON string:
    {
        "embedded": bool,
        "job_url": str,
        "collection": "job_descriptions"
    }
    """
    collection_name = os.getenv("JOB_DESCRIPTIONS_COLLECTION", "job_descriptions")

    try:
        embedding_service = EmbeddingService()
        vector = embedding_service.embed_text(job_description or "")

        store = ChromaStore(
            config=ChromaStoreConfig(collection_name=collection_name)
        )

        from uuid import uuid4

        doc_id = f"job::{uuid4()}"
        metadata: Dict[str, Any] = {
            "job_url": job_url,
            "source": "job_description",
        }

        store.upsert_chunks(
            resume_id=job_url or doc_id,
            chunk_ids=[doc_id],
            embeddings=[vector],
            documents=[job_description],
            metadatas=[metadata],
        )

        response = {
            "embedded": True,
            "job_url": job_url,
            "collection": collection_name,
        }
        return _safe_json_dumps(response)
    except Exception as exc:  # noqa: BLE001
        logger.error("embed_job_description failed: %s", exc, exc_info=True)
        response = {
            "embedded": False,
            "job_url": job_url,
            "collection": collection_name,
        }
        return _safe_json_dumps(response)


@tool
@operation
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

