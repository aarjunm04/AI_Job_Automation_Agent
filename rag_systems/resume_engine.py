# rag_systems/resume_engine.py
"""
Resume Engine (ChromaDB v1.x Compatible)

Handles:
- Loading resume_config.json
- PDF → text extraction
- Chunking + embedding
- Upserting resume chunks into Chroma
- Creating anchor vectors
- Deterministic resume selection pipeline (anchor + chunk reranking)
- Serving MCP-facing APIs (through rag_api.py)

This file assumes:
- chromadb_store.ChromaStore (v1.x PersistentClient)
- rag_pipeline.RAGPipeline + GeminiEmbedder
"""

from __future__ import annotations
import os
import json
import time
import logging
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple
from dotenv import load_dotenv
load_dotenv()


import uuid

# LOCAL MODULES
from chromadb_store import ChromaStore, ChromaStoreConfig
from rag_systems.rag_pipeline import (
    RAGPipeline,
    EmbeddingService,
    chunk_text,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


RESUME_DIR = os.getenv("RESUME_DIR", "resumes")


def _resolve_resume_path(filename: str) -> str:
    """Resolve a resume filename against RESUME_DIR to an absolute path."""
    return os.path.abspath(os.path.join(RESUME_DIR, filename))


# ============================================================
# DATA CLASS
# ============================================================

@dataclass
class ResumeEntry:
    resume_id: str
    label: str
    role_focus: str
    languages: List[str]
    local_path: str
    embedding_version: str
    vector_anchor_id: str

    def to_dict(self):
        return asdict(self)


# ============================================================
# RESUME ENGINE
# ============================================================

class ResumeEngine:
    """
    Core resume ingestion + selection module for RAG.
    """

    def __init__(
        self,
        resume_config_path: Optional[str] = None,
        chroma_config: Optional[ChromaStoreConfig] = None,
        embedder: Optional[Any] = None,
    ):
        self.resume_config_path = resume_config_path or os.path.join(
            os.path.dirname(__file__), "resume_config.json"
        )

        # CHROMA STORE (PersistentClient version)
        self.chroma = ChromaStore(config=chroma_config or ChromaStoreConfig())

        # EMBEDDING SERVICE (NVIDIA NIM primary, Gemini fallback)
        if isinstance(embedder, EmbeddingService):
            self.embedding_service = embedder
        else:
            self.embedding_service = EmbeddingService()

        # Expose primary embedder for compatibility with existing callers
        self.embedder = self.embedding_service.primary

        # RAG PIPELINE
        self.rag = RAGPipeline(
            store=self.chroma,
            embedder=self.embedder,
            embedding_service=self.embedding_service,
        )

        # LOAD RESUME CONFIG
        self.resumes: Dict[str, ResumeEntry] = {}
        self._load_resume_config()

    # ============================================================
    # LOAD CONFIG
    # ============================================================
    def _load_resume_config(self):
        if not os.path.exists(self.resume_config_path):
            raise FileNotFoundError(f"resume_config.json missing at: {self.resume_config_path}")

        with open(self.resume_config_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        for r in raw.get("resumes", []):
            resume_id = r["resume_id"]
            filename = r["local_path"]
            path = _resolve_resume_path(filename)

            anchor_id = f"resume_vector::{resume_id}::{r['embedding_version']}"

            self.resumes[resume_id] = ResumeEntry(
                resume_id=resume_id,
                label=r["label"],
                role_focus=r["role_focus"],
                languages=r.get("languages", ["en"]),
                local_path=path,
                embedding_version=r["embedding_version"],
                vector_anchor_id=anchor_id,
            )

        logger.info("Loaded %d resumes from config.", len(self.resumes))

    # ============================================================
    # PDF → TEXT EXTRACTION
    # ============================================================
    def _extract_pdf_text(self, path: str) -> str:
        try:
            import PyPDF2
            text_list = []
            with open(path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    try:
                        text = page.extract_text() or ""
                    except Exception:
                        text = ""
                    if text.strip():
                        text_list.append(text)
            full = "\n".join(text_list).strip()
            if not full:
                raise RuntimeError(f"No text extracted from PDF: {path}")
            return full

        except Exception as e:
            logger.exception("PDF extract failed: %s", e)
            raise RuntimeError("Install PyPDF2 or ensure resume PDF is valid") from e

    # ============================================================
    # BUILD ANCHOR TEXT
    # ============================================================
    def _anchor_text(self, raw_text: str, max_sentences: int = 8) -> str:
        import re
        sents = re.split(r"(?<=[.!?])\s+", raw_text.strip())
        anchor = " ".join(sents[:max_sentences]).strip()
        if not anchor:
            anchor = raw_text[:1000]
        return anchor

    # ============================================================
    # INGEST RESUME
    # ============================================================
    def ingest_resume(self, resume_id: str) -> Dict[str, Any]:
        if resume_id not in self.resumes:
            raise KeyError(f"Unknown resume_id: {resume_id}")

        entry = self.resumes[resume_id]

        if not os.path.exists(entry.local_path):
            raise FileNotFoundError(f"Missing PDF at: {entry.local_path}")

        logger.info("Ingesting resume %s", resume_id)

        raw_text = self._extract_pdf_text(entry.local_path)

        # 1. CHUNK
        chunks = chunk_text(raw_text)

        # 2. EMBED CHUNKS
        chunk_embeddings = [self.embedding_service.embed_text(c["text"]) for c in chunks]

        # 3. UPSERT CHUNKS
        self.chroma.upsert_resume(resume_id, chunks, chunk_embeddings)

        # 4. ANCHOR VECTOR (blend PDF + config keywords)
        pdf_anchor = self._anchor_text(raw_text)
        config_keywords = entry.role_focus  # Rich keywords from config
        anchor_text = f"{config_keywords}\n\n{pdf_anchor}"  # Config first, then PDF
        anchor_embedding = self.embedding_service.embed_text(anchor_text)


        self.chroma.add_anchor(
            anchor_id=entry.vector_anchor_id,
            anchor_embedding=anchor_embedding,
            anchor_text=anchor_text,
            metadata={
                "resume_id": resume_id,
                "anchor": True,
                "role_focus": entry.role_focus,
                "embedding_version": entry.embedding_version,
                "label": entry.label,
            },
        )

        ingest_result = {
            "resume_id": resume_id,
            "num_chunks": len(chunks),
            "anchor_id": entry.vector_anchor_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

        return ingest_result

    # ============================================================
    # REINDEX RESUME
    # ============================================================
    def reindex_resume(self, resume_id: str) -> Dict[str, Any]:
        logger.info("Reindex requested for %s", resume_id)
        self.chroma.delete_resume(resume_id)
        return self.ingest_resume(resume_id)

    # ============================================================
    # RESUME SELECTION
    # ============================================================
    def select_resume(
        self,
        job_payload: Dict[str, Any],
        top_k_anchors: int = 7,
        top_k_chunks: int = 3
    ) -> Dict[str, Any]:

        job_text = job_payload.get("job_text", "").strip()
        if not job_text:
            raise ValueError("job_text missing in payload")

        query_vec = self.rag.embed_query(job_text)

        anchors = self.rag.get_top_resume_anchors(query_vec, k=top_k_anchors)

        def dist_to_sim(d: float) -> float:
            try:
                return 1.0 / (1.0 + d)
            except:
                return 0.0

        candidates = []

        for a in anchors:
            meta = a.get("metadata", {})
            rid = meta.get("resume_id")
            if not rid:
                continue

            anchor_sim = dist_to_sim(a["distance"])

            # chunk-level tie-break
            chunks = self.rag.get_relevant_chunks(query_vec, filters={"resume_id": rid}, k=top_k_chunks)

            chunk_score = 0.0
            weights = [0.6, 0.3, 0.1]
            for idx, c in enumerate(chunks):
                sim = dist_to_sim(c["distance"])
                chunk_score += weights[idx] * sim

            # metadata nudge
            bonus = 0.1 if (self.resumes[rid].role_focus.lower() in job_text.lower()) else 0.0

            final = (0.45 * anchor_sim) + (0.50 * chunk_score) + (0.05 * bonus)

            candidates.append({
                "resume_id": rid,
                "anchor_similarity": anchor_sim,
                "chunk_score": chunk_score,
                "metadata_bonus": bonus,
                "final_score": final,
                "recommended_chunks": chunks,
            })

        if not candidates:
            raise RuntimeError("No resume candidates. Did you ingest resumes?")

        candidates.sort(key=lambda x: x["final_score"], reverse=True)
        top = candidates[0]

        return {
            "top_resume_id": top["resume_id"],
            "top_score": top["final_score"],
            "candidates": candidates,
            "selected_resume_path": self.get_resume_path(top["resume_id"]),
        }

    # ============================================================
    # UTILITIES
    # ============================================================
    def get_resume_path(self, resume_id: str) -> str:
        if resume_id not in self.resumes:
            raise KeyError(f"Unknown resume_id: {resume_id}")
        return self.resumes[resume_id].local_path

    def list_resumes(self) -> List[Dict[str, Any]]:
        return [r.to_dict() for r in self.resumes.values()]


# ============================================================
# SINGLETON ACCESSOR (for MCP / rag_api)
# ============================================================

_default_engine: Optional[ResumeEngine] = None

def get_default_engine(
    resume_config_path: Optional[str] = None,
    chroma_config: Optional[ChromaStoreConfig] = None,
    embedder: Optional[Any] = None,
) -> ResumeEngine:
    global _default_engine
    if _default_engine is None:
        _default_engine = ResumeEngine(
            resume_config_path=resume_config_path,
            chroma_config=chroma_config,
            embedder=embedder,
        )
    return _default_engine
