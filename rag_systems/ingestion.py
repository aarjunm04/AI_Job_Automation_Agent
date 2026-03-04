from __future__ import annotations
import logging
import os
import sys
from pathlib import Path
from typing import Optional

__all__ = ["ingest_all_resumes", "ingest_single_resume"]

logger = logging.getLogger(__name__)

EXPECTED_EMBEDDING_DIM = 1024


# ─────────────────────────────────────────────────────────────
# PDF TEXT EXTRACTION
# ─────────────────────────────────────────────────────────────
def _extract_pdf_text(pdf_path: str) -> str:
    """Extract plain text from PDF using pdfplumber.
    Args:
        pdf_path: Absolute path to the PDF file.
    Returns:
        Extracted text. Empty string on failure.
    """
    try:
        import pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            text = " ".join(page.extract_text() or "" for page in pdf.pages)
        return text.strip()
    except Exception as exc:
        logger.error("PDF extraction failed for %s: %s", pdf_path, exc)
        return ""


# ─────────────────────────────────────────────────────────────
# EMBEDDING RESOLVER
# Resolves correct embed method at runtime — confirmed chain first
# ─────────────────────────────────────────────────────────────
def _get_embedding(engine: object, text: str) -> Optional[list[float]]:
    """Resolve and call embed method on ResumeEngine at runtime.

    ResumeEngine exposes:
      - engine.embedding_service (EmbeddingService — primary+fallback with retry)
      - engine.embedder (NVIDIANIMEmbedder — direct primary)
    This helper tries the confirmed chain first, then fallbacks.

    Args:
        engine: ResumeEngine instance from get_default_engine().
        text: Plain text string to generate embedding for.

    Returns:
        List of floats (embedding vector) or None on any failure.
    """
    # Pattern 1 — CONFIRMED by deep audit: engine.embedder.embed_text()
    if hasattr(engine, "embedder") and hasattr(engine.embedder, "embed_text"):
        try:
            result = engine.embedder.embed_text(text)
            if result and len(result) == EXPECTED_EMBEDDING_DIM:
                return result
            elif result:
                logger.error(
                    "engine.embedder.embed_text() returned %d dims, "
                    "expected %d \u2014 discarding embedding, file will be skipped",
                    len(result),
                    EXPECTED_EMBEDDING_DIM,
                )
                return None
        except Exception as exc:
            logger.error("engine.embedder.embed_text() failed: %s", exc)

    # Pattern 2 — direct primary embedder (confirmed: NVIDIANIMEmbedder)
    embedder = getattr(engine, "embedder", None)
    if embedder is not None:
        for method_name in ("embed_text", "embed", "get_embedding",
                            "encode", "create_embedding"):
            if hasattr(embedder, method_name):
                try:
                    result = getattr(embedder, method_name)(text)
                    if result:
                        return result
                except Exception as exc:
                    logger.debug(
                        "engine.embedder.%s() failed: %s", method_name, exc
                    )

    # Pattern 3 — direct embed() on engine
    if hasattr(engine, "embed"):
        try:
            result = engine.embed(text)
            if result:
                return result
        except Exception as exc:
            logger.debug("engine.embed() failed: %s", exc)

    # Pattern 4 — direct embed_text() on engine
    if hasattr(engine, "embed_text"):
        try:
            result = engine.embed_text(text)
            if result:
                return result
        except Exception as exc:
            logger.debug("engine.embed_text() failed: %s", exc)

    # Pattern 5 — via rag sub-object
    rag = getattr(engine, "rag", None)
    if rag is not None:
        if hasattr(rag, "embed_query"):
            try:
                result = rag.embed_query(text)
                if result:
                    return result
            except Exception as exc:
                logger.debug("engine.rag.embed_query() failed: %s", exc)

    # All patterns exhausted — log full introspection for debugging
    engine_attrs = [m for m in dir(engine) if not m.startswith("_")]
    embedder_attrs = []  # type: list[str]
    if getattr(engine, "embedder", None):
        embedder_attrs = [
            m for m in dir(engine.embedder) if not m.startswith("_")
        ]
    logger.error(
        "EMBED_RESOLUTION_FAILED — no known embed method found.\n"
        "  engine attrs: %s\n  engine.embedder attrs: %s",
        engine_attrs, embedder_attrs
    )
    return None


# ─────────────────────────────────────────────────────────────
# SINGLE RESUME INGESTION
# ─────────────────────────────────────────────────────────────
def ingest_single_resume(pdf_path: str) -> bool:
    """Ingest one resume PDF into ChromaDB resumes collection.

    Extracts text, generates embedding via ResumeEngine (NVIDIA NIM
    primary, Gemini fallback), upserts to ChromaDB with metadata.
    Fails soft — returns False on any error, never raises.

    Args:
        pdf_path: Absolute path to the resume PDF file.

    Returns:
        True on successful ingest, False on any failure.
    """
    from rag_systems.resume_engine import get_default_engine

    filename = Path(pdf_path).name
    logger.info("Ingesting: %s", filename)

    # Step 1 — Extract text
    text = _extract_pdf_text(pdf_path)
    if not text:
        logger.warning("No text extracted from %s — skipping", filename)
        return False

    logger.debug("Extracted %d chars from %s", len(text), filename)

    # Step 2 — Get engine and resolve embedding
    try:
        engine = get_default_engine()
    except Exception as exc:
        logger.error("Failed to initialise ResumeEngine: %s", exc)
        return False

    embedding = _get_embedding(engine, text)
    if not embedding:
        logger.error("Embedding returned None for %s — skipping", filename)
        return False

    logger.debug("Embedding dim=%d for %s", len(embedding), filename)

    # Step 3 — Upsert to ChromaDB via engine.chroma.upsert_chunks()
    try:
        engine.chroma.upsert_chunks(
            resume_id=filename,
            chunk_ids=[filename],
            embeddings=[embedding],
            documents=[text],
            metadatas=[{
                "filename": filename,
                "path": pdf_path,
                "char_count": len(text),
                "source": "ingestion",
            }]
        )
        logger.info("Ingested: %s", filename)
        return True
    except Exception as exc:
        logger.error("ChromaDB upsert failed for %s: %s", filename, exc)
        return False


# ─────────────────────────────────────────────────────────────
# BATCH INGESTION
# ─────────────────────────────────────────────────────────────
def ingest_all_resumes() -> dict[str, int]:
    """Ingest all PDF resumes from RESUME_DIR into ChromaDB.

    Reads RESUME_DIR env var (default: app/resumes). One bad file
    never stops the rest — fail soft per file.

    Returns:
        Dict: {total: int, success: int, failed: int, skipped: int}
    """
    dry_run: bool = os.getenv("DRY_RUN", "false").strip().lower() == "true"
    if dry_run:
        logger.warning("INGESTION SKIPPED — DRY_RUN=true")
        return {"total": 0, "success": 0, "failed": 0, "skipped": 0}

    resume_dir = os.getenv("RESUME_DIR", "app/resumes")
    resume_path = Path(resume_dir)
    results = {
        "total": 0, "success": 0, "failed": 0, "skipped": 0
    }  # type: dict[str, int]

    if not resume_path.exists():
        logger.error(
            "RESUME_DIR not found: %s — set RESUME_DIR in java.env",
            str(resume_path)
        )
        return results

    pdf_files = sorted(resume_path.glob("*.pdf"))
    if not pdf_files:
        logger.warning("No PDF files in %s", str(resume_path))
        return results

    results["total"] = len(pdf_files)
    logger.info(
        "Found %d PDFs in %s — starting ingestion",
        len(pdf_files), str(resume_path)
    )

    for pdf_path in pdf_files:
        try:
            ok = ingest_single_resume(str(pdf_path))
            if ok:
                results["success"] += 1
            else:
                results["failed"] += 1
        except Exception as exc:
            logger.error(
                "Unexpected error on %s: %s", pdf_path.name, exc
            )
            results["failed"] += 1

    logger.info(
        "Ingestion done — total:%d success:%d failed:%d skipped:%d",
        results["total"], results["success"],
        results["failed"], results["skipped"]
    )
    return results


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
    logger.info("RESUME_DIR : %s", os.getenv("RESUME_DIR", "app/resumes"))
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
