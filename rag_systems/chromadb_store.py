# rag_systems/chromadb_store.py
"""
ChromaDB v1.x compatible store wrapper for the RAG subsystem.

Changes from prior:
- Robust metadata sanitization: convert None -> "", ensure only primitive types allowed
- Uses PersistentClient (chroma >=1.x)
- Auto-creates CHROMA_DIR if missing
- Retry logic on add/query/delete
- Clear logging for production
"""

from __future__ import annotations
import os
import time
import logging
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
import math

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# try chromadb import for v1.x (PersistentClient)
try:
    from chromadb import PersistentClient
    CHROMADB_AVAILABLE = True
except Exception:  # pragma: no cover - env dependent
    PersistentClient = None  # type: ignore
    CHROMADB_AVAILABLE = False

@dataclass
class ChromaStoreConfig:
    persist_dir: str = field(default_factory=lambda: os.getenv("CHROMA_DIR", "./.chroma"))
    collection_name: str = field(default_factory=lambda: os.getenv("CHROMA_COLLECTION", "resumes"))
    retry_attempts: int = 3
    retry_delay_seconds: float = 0.5

class ChromaStore:
    """
    Production-ready Chroma wrapper (v1.x API).
    """

    def __init__(self, config: Optional[ChromaStoreConfig] = None):
        self.config = config or ChromaStoreConfig()
        self._client = None
        self._collection = None
        self._fallback_store: Dict[str, Dict[str, Any]] = {"documents": {}}

        # Ensure persist_dir exists
        persist_dir = os.path.abspath(self.config.persist_dir)
        if not os.path.exists(persist_dir):
            try:
                os.makedirs(persist_dir, exist_ok=True)
                logger.info("Created Chroma persist directory at: %s", persist_dir)
            except Exception as e:
                logger.exception("Failed to create CHROMA_DIR (%s): %s", persist_dir, e)
                raise

        if CHROMADB_AVAILABLE:
            self._init_chromadb(persist_dir)
        else:
            logger.warning("chromadb package not available. Using deterministic in-memory fallback store.")

    def _init_chromadb(self, persist_dir: str) -> None:
        attempts = 0
        while True:
            try:
                self._client = PersistentClient(path=persist_dir)
                self._collection = self._client.get_or_create_collection(name=self.config.collection_name)
                logger.info("Chroma PersistentClient initialized at %s, collection=%s", persist_dir, self.config.collection_name)
                return
            except Exception as e:
                attempts += 1
                logger.exception("Failed to initialize Chroma PersistentClient (attempt %d): %s", attempts, e)
                if attempts >= self.config.retry_attempts:
                    raise RuntimeError("ChromaDB (v1.x) initialization failed. See logs.") from e
                time.sleep(self.config.retry_delay_seconds)

    # -------------------------
    # Metadata sanitization helpers
    # -------------------------
    @staticmethod
    def _sanitize_value(v: Any) -> Any:
        """
        Convert a metadata value into a Chroma-compatible primitive:
        - None -> ""
        - bool, int, float, str -> kept
        - list/tuple -> converted to string (comma-joined) or left as list of primitives if safe
        - dict -> JSON-ish string
        - other types -> string representation
        """
        if v is None:
            return ""
        if isinstance(v, (bool, int, float, str)):
            return v
        if isinstance(v, (list, tuple)):
            # convert nested primitives; if non-primitive found, stringify element
            sanitized = []
            for item in v:
                if item is None:
                    sanitized.append("")
                elif isinstance(item, (bool, int, float, str)):
                    sanitized.append(item)
                else:
                    sanitized.append(str(item))
            # return as string to avoid nested non-primitive issues
            return ",".join(map(str, sanitized))
        if isinstance(v, dict):
            # shallow stringify to avoid nested structures
            try:
                import json
                return json.dumps({k: (v2 if isinstance(v2, (bool, int, float, str)) else str(v2)) for k, v2 in v.items()})
            except Exception:
                return str(v)
        # fallback
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
                # keys must be strings
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

        # ensure resume_id present and sanitize
        for md in metadatas:
            if md is None:
                continue
            md.setdefault("resume_id", resume_id)

        sanitized = self._sanitize_metadata_list(metadatas)

        if CHROMADB_AVAILABLE and self._collection is not None:
            attempts = 0
            while True:
                try:
                    self._collection.add(ids=chunk_ids, embeddings=embeddings, documents=documents, metadatas=sanitized)
                    logger.info("Upserted %d chunks for resume_id=%s", len(chunk_ids), resume_id)
                    return
                except Exception as e:
                    attempts += 1
                    logger.exception("Chroma add failed (attempt %d): %s", attempts, e)
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
        # sanitize metadata
        sanitized = self._sanitize_metadata_list([metadata])[0]

        documents = [anchor_text or ""]

        if CHROMADB_AVAILABLE and self._collection is not None:
            attempts = 0
            while True:
                try:
                    self._collection.add(ids=[anchor_id], embeddings=[anchor_embedding], documents=documents, metadatas=[sanitized])
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
        where = where or {}

        if CHROMADB_AVAILABLE and self._collection is not None:
            attempts = 0
            while True:
                try:
                    resp = self._collection.query(query_embeddings=query_embeddings, n_results=n_results, where=where)
                    hits: List[Dict[str, Any]] = []
                    if resp and "ids" in resp and resp["ids"]:
                        ids = resp["ids"][0]
                        dists = resp["distances"][0]
                        metas = resp["metadatas"][0]
                        docs = resp["documents"][0]
                        for idx, _id in enumerate(ids):
                            hits.append({"id": _id, "distance": dists[idx], "metadata": metas[idx], "document": docs[idx]})
                    return hits
                except Exception as e:
                    attempts += 1
                    logger.exception("Chroma query failed (attempt %d): %s", attempts, e)
                    if attempts >= self.config.retry_attempts:
                        raise
                    time.sleep(self.config.retry_delay_seconds)
        else:
            # fallback simple similarity
            q = query_embeddings[0]
            hits = []
            for _id, rec in self._fallback_store["documents"].items():
                emb = rec["embedding"]
                sim = self._cosine_sim(q, emb)
                distance = 1.0 - sim
                hits.append({"id": _id, "distance": distance, "metadata": rec["metadata"], "document": rec["document"]})
            # apply where filter (equality for keys)
            if where:
                def match(m: Dict[str, Any], where: Dict[str, Any]) -> bool:
                    for k, v in where.items():
                        if str(m.get(k, "")) != str(v):
                            return False
                    return True
                hits = [h for h in hits if match(h["metadata"], where)]
            hits.sort(key=lambda x: x["distance"])
            return hits[:n_results]

    def query_anchor(self, query_embedding: List[float], n_results: int = 7) -> List[Dict[str, Any]]:
        return self.query(query_embeddings=[query_embedding], n_results=n_results, where={"anchor": True})

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
        # sanitize prior to adding
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

    def persist(self) -> None:
        if CHROMADB_AVAILABLE and self._client is not None:
            try:
                if hasattr(self._client, "persist"):
                    self._client.persist()
                    logger.debug("Chroma client persist() called.")
            except Exception:
                logger.exception("Chroma persist failed.")

    def close(self) -> None:
        try:
            self.persist()
        except Exception:
            pass
