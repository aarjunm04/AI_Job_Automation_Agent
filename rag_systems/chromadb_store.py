# rag_systems/chromadb_store.py
"""
ChromaDB v1.x compatible store wrapper for the RAG subsystem.

Changes from prior:
- Uses HttpClient to connect to the shared ChromaDB Docker container
- Reads CHROMADB_HOST / CHROMADB_PORT from environment
- Retry logic with exponential backoff on init/add/query/delete
- Robust metadata sanitization: convert None -> "", ensure only primitive types
- Fixed empty where={} handling for ChromaDB v1.x
- Added query_anchor() method for resume-level embeddings
"""

from __future__ import annotations
import os
import time
import logging
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
import math

__all__ = [
    "ChromaStore",
    "ChromaStoreConfig",
    "CHROMADB_AVAILABLE",
]

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# HttpClient for connecting to ChromaDB container over HTTP
try:
    import chromadb
    CHROMADB_AVAILABLE = True
except Exception:  # pragma: no cover - env dependent
    chromadb = None  # type: ignore
    CHROMADB_AVAILABLE = False


@dataclass
class ChromaStoreConfig:
    host: str = field(default_factory=lambda: os.getenv("CHROMA_HOST", "ai_chromadb"))
    port: int = field(default_factory=lambda: int(os.getenv("CHROMA_PORT", "8001")))
    collection_name: str = field(default_factory=lambda: os.getenv("CHROMA_COLLECTION", "resumes"))
    retry_attempts: int = 3
    retry_delay_seconds: float = 0.5


class ChromaStore:
    """
    Production-ready Chroma wrapper (v1.x HttpClient API).
    Connects to a shared ChromaDB container over HTTP.
    """

    def __init__(self, config: Optional[ChromaStoreConfig] = None):
        self.config = config or ChromaStoreConfig()
        self._client = None
        self._collection = None
        self._fallback_store: Dict[str, Dict[str, Any]] = {"documents": {}}
        # Logger kept on self so dim-guard helpers can reference it
        self._logger = logging.getLogger(self.__class__.__name__)

        if CHROMADB_AVAILABLE:
            self._init_chromadb()
        else:
            logger.warning("chromadb package not available. Using deterministic in-memory fallback store.")

    def _init_chromadb(self) -> None:
        attempts = 0
        while True:
            try:
                self._client = chromadb.HttpClient(
                    host=os.getenv("CHROMA_HOST", "ai_chromadb"),
                    port=int(os.getenv("CHROMA_PORT", "8001")),
                )
                self._collection = self._client.get_or_create_collection(
                    name=self.config.collection_name,
                    embedding_function=None,  # We supply our own vectors — no default ef needed
                )
                logger.info(
                    "Chroma HttpClient connected to %s:%d, collection=%s",
                    self.config.host, self.config.port, self.config.collection_name,
                )
                return
            except Exception as e:
                attempts += 1
                logger.exception(
                    "Failed to connect to ChromaDB at %s:%d (attempt %d): %s",
                    self.config.host, self.config.port, attempts, e,
                )
                if attempts >= self.config.retry_attempts:
                    raise RuntimeError(
                        f"ChromaDB connection to {self.config.host}:{self.config.port} "
                        f"failed after {attempts} attempts. See logs."
                    ) from e
                time.sleep(self.config.retry_delay_seconds * (2 ** (attempts - 1)))

    # -------------------------
    # Metadata sanitization helpers
    # -------------------------
    @staticmethod
    def _sanitize_value(v: Any) -> Any:
        """
        Convert a metadata value into a Chroma-compatible primitive:
        - None -> ""
        - bool, int, float, str -> kept
        - list/tuple -> converted to string (comma-joined)
        - dict -> JSON string
        - other types -> string representation
        """
        if v is None:
            return ""
        if isinstance(v, (bool, int, float, str)):
            return v
        if isinstance(v, (list, tuple)):
            sanitized = []
            for item in v:
                if item is None:
                    sanitized.append("")
                elif isinstance(item, (bool, int, float, str)):
                    sanitized.append(item)
                else:
                    sanitized.append(str(item))
            return ",".join(map(str, sanitized))
        if isinstance(v, dict):
            try:
                import json
                return json.dumps({k: (v2 if isinstance(v2, (bool, int, float, str)) else str(v2)) for k, v2 in v.items()})
            except Exception:
                return str(v)
        return str(v)

    @classmethod
    def _sanitize_metadata_list(cls, metadatas: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        sanitized_list: List[Dict[str, Any]] = []
        for md in metadatas:
            if md is None:
                sanitized_list.append({})
                continue
            sanitized: Dict[str, Any] = {}
            for k, v in md.items():
                key = str(k)
                sanitized[key] = cls._sanitize_value(v)
            sanitized_list.append(sanitized)
        return sanitized_list

    # -------------------------
    # Upsert / Add methods
    # -------------------------
    def upsert_chunks(
        self,
        resume_id: str,
        chunk_ids: List[str],
        embeddings: List[List[float]],
        documents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """
        Upsert chunk vectors. Ensures metadatas sanitized.
        """
        if metadatas is None:
            metadatas = [{} for _ in chunk_ids]

        for md in metadatas:
            if md is None:
                continue
            md.setdefault("resume_id", resume_id)

        sanitized = self._sanitize_metadata_list(metadatas)

        if CHROMADB_AVAILABLE and self._collection is not None:
            # ---- DIM GUARD — skip embeddings with wrong dimension --------
            expected_dim = int(os.getenv("EMBEDDING_DIM", "1024"))
            valid_ids: List[str] = []
            valid_docs: List[str] = []
            valid_embs: List[List[float]] = []
            valid_metas: List[Dict[str, Any]] = []
            for i, emb in enumerate(embeddings):
                if len(emb) != expected_dim:
                    self._logger.error(
                        "Embedding dim mismatch: got %d expected %d, skipping doc %s",
                        len(emb), expected_dim, chunk_ids[i],
                    )
                    continue
                valid_ids.append(chunk_ids[i])
                valid_docs.append(documents[i])
                valid_embs.append(emb)
                valid_metas.append(sanitized[i] if sanitized else {})
            # --------------------------------------------------------------
            if not valid_ids:
                logger.warning(
                    "upsert_chunks: all %d embeddings failed dim guard — nothing upserted",
                    len(chunk_ids),
                )
                return
            attempts = 0
            while True:
                try:
                    self._collection.upsert(
                        ids=valid_ids,
                        embeddings=valid_embs,
                        documents=valid_docs,
                        metadatas=valid_metas,
                    )
                    logger.info("Upserted %d chunks for resume_id=%s", len(valid_ids), resume_id)
                    return
                except Exception as e:
                    attempts += 1
                    logger.exception("Chroma upsert failed (attempt %d): %s", attempts, e)
                    if attempts >= self.config.retry_attempts:
                        raise
                    time.sleep(self.config.retry_delay_seconds)
        else:
            for cid, emb, doc, md in zip(chunk_ids, embeddings, documents, sanitized):
                self._fallback_store["documents"][cid] = {"embedding": emb, "document": doc, "metadata": md}
            logger.info("Fallback store: stored %d chunks for resume_id=%s", len(chunk_ids), resume_id)

    def add_anchor(
        self,
        anchor_id: str,
        anchor_embedding: List[float],
        anchor_text: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        metadata = metadata or {}
        metadata.setdefault("anchor", True)
        metadata.setdefault("resume_id", metadata.get("resume_id", anchor_id))
        sanitized = self._sanitize_metadata_list([metadata])[0]

        documents = [anchor_text or ""]

        if CHROMADB_AVAILABLE and self._collection is not None:
            # ---- DIM GUARD — validate anchor embedding -------------------
            expected_dim = int(os.getenv("EMBEDDING_DIM", "1024"))
            if not anchor_embedding or len(anchor_embedding) != expected_dim:
                self._logger.error(
                    "Embedding dim mismatch: got %d expected %d, skipping doc %s",
                    len(anchor_embedding) if anchor_embedding else 0,
                    expected_dim,
                    anchor_id,
                )
                return
            # --------------------------------------------------------------
            attempts = 0
            while True:
                try:
                    self._collection.upsert(
                        ids=[anchor_id],
                        embeddings=[anchor_embedding],
                        documents=documents,
                        metadatas=[sanitized],
                    )
                    logger.info("Anchor upserted: %s", anchor_id)
                    return
                except Exception as e:
                    attempts += 1
                    logger.exception("Chroma anchor upsert failed (attempt %d): %s", attempts, e)
                    if attempts >= self.config.retry_attempts:
                        raise
                    time.sleep(self.config.retry_delay_seconds)
        else:
            self._fallback_store["documents"][anchor_id] = {"embedding": anchor_embedding, "document": documents[0], "metadata": sanitized}
            logger.info("Fallback anchor stored: %s", anchor_id)

    # -------------------------
    # Query methods
    # -------------------------
    def query(
        self,
        query_embeddings: List[List[float]],
        n_results: int = 8,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Query ChromaDB with optional metadata filters.

        Args:
            query_embeddings: List of query vectors
            n_results: Number of results to return
            where: Optional metadata filter dict (must have content or be None)

        Returns:
            List of matching documents with metadata and distances
        """
        # ChromaDB v1.x doesn't accept empty where={}, must be None
        if where is not None and len(where) == 0:
            where = None

        if CHROMADB_AVAILABLE and self._collection is not None:
            attempts = 0
            while True:
                try:
                    if where:
                        resp = self._collection.query(
                            query_embeddings=query_embeddings,
                            n_results=n_results,
                            where=where
                        )
                    else:
                        resp = self._collection.query(
                            query_embeddings=query_embeddings,
                            n_results=n_results
                        )

                    hits: List[Dict[str, Any]] = []
                    if resp and "ids" in resp and resp["ids"]:
                        ids = resp["ids"][0]
                        dists = resp["distances"][0]
                        metas = resp["metadatas"][0]
                        docs = resp["documents"][0]
                        for idx, _id in enumerate(ids):
                            hits.append({
                                "id": _id,
                                "distance": dists[idx],
                                "metadata": metas[idx],
                                "document": docs[idx]
                            })
                    return hits

                except Exception as e:
                    attempts += 1
                    logger.exception("Chroma query failed (attempt %d): %s", attempts, e)
                    if attempts >= self.config.retry_attempts:
                        raise
                    time.sleep(self.config.retry_delay_seconds)
        else:
            q = query_embeddings[0]
            hits = []
            for _id, rec in self._fallback_store["documents"].items():
                emb = rec["embedding"]
                sim = self._cosine_sim(q, emb)
                distance = 1.0 - sim
                hits.append({
                    "id": _id,
                    "distance": distance,
                    "metadata": rec["metadata"],
                    "document": rec["document"]
                })

            if where:
                def match(m: Dict[str, Any], w: Dict[str, Any]) -> bool:
                    for k, v in w.items():
                        if str(m.get(k, "")) != str(v):
                            return False
                    return True
                hits = [h for h in hits if match(h["metadata"], where)]

            hits.sort(key=lambda x: x["distance"])
            return hits[:n_results]

    def query_anchor(self, query_embedding: List[float], n_results: int = 7) -> List[Dict[str, Any]]:
        """
        Query for anchor vectors (resume-level embeddings).
        Anchors have metadata key 'anchor'=True

        Args:
            query_embedding: Single query vector
            n_results: Number of anchor results to return

        Returns:
            List of anchor matches with metadata
        """
        return self.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where={"anchor": True}
        )

    # -------------------------
    # Resume-level utilities
    # -------------------------
    def delete_resume(self, resume_id: str) -> None:
        if CHROMADB_AVAILABLE and self._collection is not None:
            attempts = 0
            while True:
                try:
                    self._collection.delete(where={"resume_id": resume_id})
                    logger.info("Deleted resume_id=%s from Chroma", resume_id)
                    return
                except Exception as e:
                    attempts += 1
                    logger.exception("Chroma delete failed (attempt %d): %s", attempts, e)
                    if attempts >= self.config.retry_attempts:
                        raise
                    time.sleep(self.config.retry_delay_seconds)
        else:
            to_delete = [doc_id for doc_id, rec in self._fallback_store["documents"].items() if rec.get("metadata", {}).get("resume_id") == resume_id]
            for doc_id in to_delete:
                del self._fallback_store["documents"][doc_id]
            logger.info("Fallback delete removed %d docs for resume_id=%s", len(to_delete), resume_id)

    def list_resume_ids(self, sample_limit: int = 1000) -> List[str]:
        if CHROMADB_AVAILABLE and self._collection is not None:
            try:
                data = self._collection.get(include=["metadatas"], limit=sample_limit)
                resume_ids = set()
                for block in data.get("metadatas", []):
                    for md in block:
                        if isinstance(md, dict) and "resume_id" in md:
                            resume_ids.add(md["resume_id"])
                return list(resume_ids)
            except Exception as e:
                logger.exception("Chroma get metadatas failed: %s", e)
                raise
        else:
            resume_ids = set()
            for rec in self._fallback_store["documents"].values():
                md = rec.get("metadata", {})
                if "resume_id" in md:
                    resume_ids.add(md["resume_id"])
            return list(resume_ids)

    # -------------------------
    # Convenience wrapper for chunk-format
    # -------------------------
    def upsert_resume(self, resume_id: str, chunks: List[Dict[str, Any]], embeddings: List[List[float]]) -> None:
        chunk_ids = [c["chunk_id"] for c in chunks]
        docs = [c.get("text", "") for c in chunks]
        metadatas = []
        for c in chunks:
            md = {
                "resume_id": resume_id,
                "chunk_id": c.get("chunk_id", ""),
                "offset": c.get("offset", 0) if c.get("offset", 0) is not None else 0,
                "token_count": c.get("token_count", 0) if c.get("token_count", 0) is not None else 0,
                "section_heading": c.get("section_heading") if c.get("section_heading") is not None else ""
            }
            metadatas.append(md)
        metadatas = self._sanitize_metadata_list(metadatas)
        self.upsert_chunks(resume_id=resume_id, chunk_ids=chunk_ids, embeddings=embeddings, documents=docs, metadatas=metadatas)

    # -------------------------
    # Helpers
    # -------------------------
    @staticmethod
    def _cosine_sim(a: List[float], b: List[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a)) or 1.0
        norm_b = math.sqrt(sum(x * x for x in b)) or 1.0
        return dot / (norm_a * norm_b)

    def close(self) -> None:
        """Close the ChromaDB connection (no-op for HttpClient)."""
        pass
