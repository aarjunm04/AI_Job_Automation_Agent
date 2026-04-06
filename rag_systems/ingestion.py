from __future__ import annotations
import logging
import os
import sys
from pathlib import Path
from typing import Optional

from resume_engine import get_default_engine

__all__ = ["ingest_all_resumes", "ingest_single_resume"]

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# SINGLE RESUME INGESTION
# ─────────────────────────────────────────────────────────────
def ingest_single_resume(pdf_path: str) -> bool:
    """Ingest one resume by path — delegates to ResumeEngine."""
    import os
    from pathlib import Path
    if os.getenv("DRY_RUN", "false").lower() == "true":
        logger.info("ingest_single_resume: DRY_RUN=true — skipping %s", pdf_path)
        return False
    engine = get_default_engine()
    resume_id = Path(pdf_path).name       # filename = resume_id
    try:
        engine.ingest_resume(resume_id)
        return True
    except Exception as exc:
        logger.error("ingest_single_resume: failed %s — %s", resume_id, exc)
        return False

# ─────────────────────────────────────────────────────────────
# BATCH INGESTION
# ─────────────────────────────────────────────────────────────
def ingest_all_resumes() -> dict[str, int]:
    """Single entry point — delegates to ResumeEngine.
    
    Reads resume list from config/resume_config.json via ResumeEngine.
    Writes both chunk vectors and anchor vectors to ChromaDB.
    Respects DRY_RUN env var.
    
    Returns:
        Dict with keys: total, success, failed, skipped.
    """
    import os
    if os.getenv("DRY_RUN", "false").lower() == "true":
        logger.info("ingest_all_resumes: DRY_RUN=true — skipping")
        return {"total": 0, "success": 0, "failed": 0, "skipped": 0}
    
    engine = get_default_engine()           # ResumeEngine singleton
    resume_list = engine.list_resumes()     # reads config/resume_config.json
    
    total   = len(resume_list)
    success = 0
    failed  = 0
    
    for r in resume_list:
        resume_id = r.get("resume_id") or r.get("filename") or r.get("id")
        if not resume_id:
            failed += 1
            continue
        try:
            result = engine.ingest_resume(resume_id)
            logger.info(
                "ingest_all_resumes: ✅ %s — chunks=%s anchor=%s",
                resume_id,
                result.get("num_chunks", 0),
                result.get("anchor_id", "")[:30],
            )
            success += 1
        except Exception as exc:
            logger.error("ingest_all_resumes: ❌ %s — %s", resume_id, exc)
            failed += 1
    
    logger.info(
        "ingest_all_resumes: COMPLETE total=%d success=%d failed=%d",
        total, success, failed,
    )
    return {"total": total, "success": success, "failed": failed, "skipped": 0}

# ─────────────────────────────────────────────────────────────
# STANDALONE ENTRYPOINT
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(
        level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO),
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger.info("=" * 64)
    logger.info("AI Job Agent — Resume Ingestion Pipeline")
    logger.info("RESUME_DIR : %s", os.getenv("RESUME_DIR", "/app/resumes"))
    logger.info("=" * 64)

    results = ingest_all_resumes()

    if results["total"] == 0:
        logger.error("No resumes found — nothing to ingest")
        sys.exit(1)
    if results["failed"] > 0:
        logger.error(
            "Ingestion completed with %d failure(s) — "
            "review logs above for details", results["failed"]
        )
        sys.exit(1)

    logger.info(
        "All %d resumes ingested successfully", results["success"]
    )
    sys.exit(0)
