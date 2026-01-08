# rag_systems/rag_pipeline.py
"""
Temporary DEBUG version of RAG Pipeline to inspect Gemini embedding API response.
"""

from __future__ import annotations
import os
import re
import httpx
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np

from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# ============================================================
# EMBEDDING PROVIDERS (DEBUG)
# ============================================================

class EmbeddingProvider:
    def embed_text(self, text: str) -> List[float]:
        raise NotImplementedError


@dataclass
class GeminiEmbedder(EmbeddingProvider):
    api_key: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    model: str = "models/text-embedding-004"
    timeout_seconds: float = 20

    def embed_text(self, text: str) -> List[float]:
        if not self.api_key:
            raise RuntimeError("Missing GEMINI_API_KEY")

        url = (
            f"https://generativelanguage.googleapis.com/v1beta/"
            f"{self.model}:embedContent?key={self.api_key}"
        )

        payload = {
            "model": self.model,
            "content": {
                "parts": [
                    {"text": text}
                ]
            }
        }

        try:
            resp = httpx.post(url, json=payload, timeout=self.timeout_seconds)
            raw = resp.json()

            print("\n\n===== GEMINI RAW RESPONSE =====")
            print(raw)
            print("================================\n\n")

            resp.raise_for_status()

            # TEMPORARY: return dummy vector so pipeline continues
            return [0.0] * 768

        except Exception as e:
            logger.exception("Gemini embedding error: %s", e)
            raise RuntimeError("Gemini embedding request failed") from e


@dataclass
class LocalDeterministicEmbedder(EmbeddingProvider):
    dim: int = 384
    def embed_text(self, text: str) -> List[float]:
        import hashlib
        h = hashlib.sha256(text.encode()).digest()
        vec = np.frombuffer(h * 3, dtype=np.uint8)[:self.dim].astype(float)
        return (vec / (np.linalg.norm(vec) or 1)).tolist()


# ============================================================
# CHUNKING
# ============================================================

def _sentence_split(text: str) -> List[str]:
    return re.split(r"(?<=[.!?])\s+", text.strip())


def chunk_text(text: str, max_tokens: int = 450, overlap: int = 80):
    import uuid
    sents = _sentence_split(text)
    chunks = []
    buf, count, offset = [], 0, 0

    for s in sents:
        wc = len(s.split())
        if count + wc > max_tokens and buf:
            chunks.append({
                "chunk_id": str(uuid.uuid4()),
                "text": " ".join(buf),
                "offset": offset,
                "token_count": count,
                "section_heading": None
            })

            # overlap
            if overlap > 0:
                keep, kept = [], 0
                for prev in reversed(buf):
                    w = len(prev.split())
                    if kept >= overlap: break
                    keep.insert(0, prev)
                    kept += w
                buf = keep
                count = sum(len(x.split()) for x in buf)
            else:
                buf = []
                count = 0

        buf.append(s)
        count += wc
        offset += 1

    if buf:
        chunks.append({
            "chunk_id": str(uuid.uuid4()),
            "text": " ".join(buf),
            "offset": offset,
            "token_count": count,
            "section_heading": None
        })

    return chunks


# ============================================================
# RAG PIPELINE
# ============================================================

@dataclass
class RAGPipeline:
    store: Any
    embedder: EmbeddingProvider

    def clean_job_text(self, t: str):
        return re.sub(r"\s+", " ", t or "").strip()

    def embed_query(self, text: str):
        cleaned = self.clean_job_text(text)
        return self.embedder.embed_text(cleaned)

    def get_top_resume_anchors(self, embedding, k=7):
        return self.store.query_anchor(query_embedding=embedding, n_results=k)

    def get_relevant_chunks(self, embedding, filters=None, k=10):
        return self.store.query(query_embeddings=[embedding], n_results=k, where=filters or {})

    def build_grounding_context(self, text, top_k_chunks=10, filters=None):
        vec = self.embed_query(text)
        return {
            "query_text": text,
            "query_embedding": vec,
            "top_chunks": self.get_relevant_chunks(vec, filters, top_k_chunks)
        }

