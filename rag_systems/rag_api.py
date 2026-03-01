# rag_systems/rag_api.py
"""
RAG API FACADE — MCP READY

This module exposes a simple, stable function-level API that the
MCP server, Chrome Extension, Apply Agent, or any other subsystem
can call directly to access the RAG/resume engine features.

This file contains NO Chroma logic, NO ingestion logic —
it simply forwards calls to the singleton ResumeEngine instance.

Public API Surface:
- select_resume(job_payload)
- get_resume_pdf_path(resume_id)
- get_rag_context(job_payload, top_k_chunks=10)
- reindex_resume(resume_id)
- list_resumes()
- healthcheck()

"""

from __future__ import annotations
from typing import Dict, Any, List
import logging

from rag_systems.resume_engine import get_default_engine
from rag_systems.rag_pipeline import RAGPipeline

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ------------------------------------------------------------
# INTERNAL ENGINE SINGLETON
# ------------------------------------------------------------

def _engine():
    """Get global ResumeEngine instance."""
    return get_default_engine()


# ------------------------------------------------------------
# PUBLIC API FUNCTIONS (MCP-INTEGRATION FRIENDLY)
# ------------------------------------------------------------

def select_resume(job_payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Select the best resume for a job posting.

    Input:
    {
        "job_text": "<combined JD and Requirements>"
    }

    Returns:
    {
        "top_resume_id": "resume_ml",
        "top_score": 0.91,
        "candidates": [...],
        "selected_resume_path": "/Users/.../Resume/Aarjun_ML.pdf"
    }
    """
    engine = _engine()
    return engine.select_resume(job_payload)


def get_resume_pdf_path(resume_id: str) -> str:
    """
    Returns absolute local path to a resume PDF.
    Used by Playwright Apply Engine & Chrome Extension.
    """
    engine = _engine()
    return engine.get_resume_path(resume_id)


def get_rag_context(job_payload: Dict[str, Any], top_k_chunks: int = 10) -> Dict[str, Any]:
    """
    Provide semantic context (top chunks) for Job Analyzer Agent / Apply Agent.
    """
    engine = _engine()
    rag: RAGPipeline = engine.rag

    job_text = job_payload.get("job_text", "")
    return rag.build_grounding_context(job_text, top_k_chunks=top_k_chunks)


def reindex_resume(resume_id: str) -> Dict[str, Any]:
    """
    Manually reindex any resume.

    For example:
    - When you update the PDF content
    - When embeddings model changes
    """
    engine = _engine()
    return engine.reindex_resume(resume_id)


def list_resumes() -> List[Dict[str, Any]]:
    """
    Returns metadata for all resumes defined in resume_config.json.
    """
    engine = _engine()
    return engine.list_resumes()


def healthcheck() -> Dict[str, Any]:
    """
    Returns a system healthcheck.
    Useful for MCP readiness checks and debugging.

    Output example:
    {
        "status": "ok",
        "resume_count": 8,
        "chroma_dir": "/Users/.../.chroma",
        "embedding_provider": "GeminiEmbedder"
    }
    """
    try:
        engine = _engine()
        return {
            "status": "ok",
            "resume_count": len(engine.list_resumes()),
            "chroma_dir": engine.chroma.config.persist_dir,
            "embedding_provider": engine.embedder.__class__.__name__,
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
        }


# ------------------------------------------------------------
# OPTIONAL CLI DEBUG MODE
# ------------------------------------------------------------

if __name__ == "__main__":
    print("Healthcheck:", healthcheck())
