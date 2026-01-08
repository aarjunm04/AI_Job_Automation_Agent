# mcp/mcp_client.py
"""
Model Context Protocol (MCP) microservice - single-file production-ready implementation.

Features:
- FastAPI async service with Pydantic request/response models
- Async SQLAlchemy (SQLite default) persistence with migrate() helper
- Session lifecycle (create/read/delete) with TTL and background cleanup
- Context items store with append / replace-last-n / paginated retrieval / filter-by-role/source
- Snapshot/summarization (local extractive summarizer) + optional LLM integration hook
- RAG hook interface for vector DBs (no builtin vector DB)
- Evidence store
- API-key auth and simple in-memory token-bucket rate limiter (dev)
- Metrics endpoint, health endpoint
- Safe `meta_json` DB column name (no SQLAlchemy metadata collision)
- Graceful shutdown, retries, logging

How to run locally:
1) Install dependencies:
   pip install fastapi uvicorn sqlalchemy aiosqlite httpx pydantic rich

2) Configure env vars (minimum):
   export MCP_API_KEY="super-secret"
   # optional dev:
   export MCP_DEV_MODE=1

3) Run:
   uvicorn mcp.mcp_client:app --host 0.0.0.0 --port 8080 --reload

Notes:
- By default uses SQLite at mcp/mcp_context.db (created automatically).
- To use Postgres: set MCP_DATABASE_URL to e.g. "postgresql+asyncpg://user:pass@host:5432/mcp"
"""

from __future__ import annotations
import os
import json
import uuid
import time
import logging
import asyncio
import re
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta

from fastapi import FastAPI, Header, HTTPException, Depends, Request, BackgroundTasks, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator

from sqlalchemy import Column, String, Integer, DateTime, Text, Boolean, func, select, Index
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.exc import IntegrityError

import httpx
from rich.logging import RichHandler

from dotenv import load_dotenv
load_dotenv()

# -----------------------
# Logging
# -----------------------
LOG_LEVEL = os.getenv("MCP_LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, handlers=[RichHandler()])
logger = logging.getLogger("mcp")
logger.setLevel(LOG_LEVEL)

# -----------------------
# Configuration (env-driven)
# -----------------------
MCP_API_KEY = os.getenv("MCP_API_KEY", "")
MCP_DEV_MODE = os.getenv("MCP_DEV_MODE", "0") == "1"
MCP_DATABASE_URL = os.getenv("MCP_DATABASE_URL", f"sqlite+aiosqlite:///{os.path.join('mcp','mcp_context.db')}")
MCP_SESSION_TTL_HOURS = int(os.getenv("MCP_SESSION_TTL_HOURS", "72"))
MCP_CLEANUP_INTERVAL_SECONDS = int(os.getenv("MCP_CLEANUP_INTERVAL_SECONDS", "1800"))
MCP_MAX_PAGE_SIZE = int(os.getenv("MCP_MAX_PAGE_SIZE", "200"))

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_BASE = os.getenv("GEMINI_BASE", "https://api.generative.googleapis.com/v1")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE = os.getenv("OPENROUTER_BASE", "https://api.openrouter.ai")

RATE_BUCKET_CAPACITY = int(os.getenv("MCP_RATE_CAPACITY", "60"))
RATE_BUCKET_REFILL_SEC = int(os.getenv("MCP_RATE_REFILL_SEC", "60"))

# -----------------------
# DB setup
# -----------------------
os.makedirs("mcp", exist_ok=True)
DATABASE_URL = MCP_DATABASE_URL

Base = declarative_base()
engine = create_async_engine(DATABASE_URL, future=True, echo=False)
AsyncSessionLocal = sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

# -----------------------
# DB Models
# -----------------------
class SessionModel(Base):
    __tablename__ = "sessions"

    session_id = Column(String(36), primary_key=True)
    owner = Column(String(128), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    last_active_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    meta_json = Column(Text, nullable=True)
    ttl_hours = Column(Integer, default=MCP_SESSION_TTL_HOURS, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    version = Column(Integer, default=1, nullable=False)

    __table_args__ = (
        Index("ix_sessions_last_active", "last_active_at"),
    )

class ContextItemModel(Base):
    __tablename__ = "context_items"
    item_id = Column(String(48), primary_key=True)
    session_id = Column(String(36), nullable=False, index=True)
    role = Column(String(32), nullable=False)  # system/user/assistant/tool
    content = Column(Text, nullable=False)
    vector_id = Column(String(128), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    meta_json = Column(Text, nullable=True)    # stores JSON as text
    trusted = Column(Boolean, default=False)
    deprecated = Column(Boolean, default=False)
    sequence = Column(Integer, nullable=False, default=0)
    __table_args__ = (Index("ix_ctx_session_sequence", "session_id", "sequence"),
                      Index("ix_ctx_session_created_at", "session_id", "created_at"))

class SnapshotModel(Base):
    __tablename__ = "snapshots"
    snapshot_id = Column(String(48), primary_key=True)
    session_id = Column(String(36), nullable=False, index=True)
    summary_text = Column(Text, nullable=False)
    method = Column(String(32), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    meta_json = Column(Text, nullable=True)

class EvidenceModel(Base):
    __tablename__ = "evidence"
    evidence_id = Column(String(48), primary_key=True)
    session_id = Column(String(36), nullable=False, index=True)
    attached_to = Column(String(48), nullable=True)
    data = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    meta_json = Column(Text, nullable=True)

# -----------------------
# Pydantic Schemas
# -----------------------
class MetadataSchema(BaseModel):
    source: Optional[str] = Field(None, description="scraper, playwright_scraper, playwright_apply, n8n, chrome_extension, agent")
    job_id: Optional[str] = None
    url: Optional[str] = None
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    tags: Optional[List[str]] = []

class SessionCreateSchema(BaseModel):
    owner: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    ttl_hours: Optional[int] = None

class SessionReadSchema(BaseModel):
    session_id: str
    owner: Optional[str]
    created_at: datetime
    last_active_at: datetime
    metadata: Optional[Dict[str, Any]]
    ttl_hours: int
    is_active: bool

class ContextItemCreateSchema(BaseModel):
    role: str
    content: str
    vector_id: Optional[str] = None
    metadata: Optional[MetadataSchema] = None
    trusted: Optional[bool] = False

    @validator("role")
    def validate_role(cls, v):
        if v not in ("system", "user", "assistant", "tool"):
            raise ValueError("role must be one of: system,user,assistant,tool")
        return v

class ContextItemReadSchema(BaseModel):
    item_id: str
    session_id: str
    role: str
    content: str
    vector_id: Optional[str]
    created_at: datetime
    metadata: Optional[Dict[str, Any]]
    trusted: bool
    deprecated: bool
    sequence: int

class SnapshotCreateSchema(BaseModel):
    strategy: str = "rolling"
    max_sentences: Optional[int] = 8

class SnapshotReadSchema(BaseModel):
    snapshot_id: str
    session_id: str
    summary_text: str
    method: str
    created_at: datetime
    metadata: Optional[Dict[str, Any]]

class EvidenceCreateSchema(BaseModel):
    attached_to: Optional[str] = None
    data: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None

# -----------------------
# Utilities
# -----------------------
def make_uuid() -> str:
    return str(uuid.uuid4())

def now() -> datetime:
    return datetime.utcnow()

def safe_json_load(text: Optional[str]) -> Optional[dict]:
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        return None

# Simple async retry decorator
def async_retry(attempts: int = 3, delay: float = 0.5, backoff: float = 2.0):
    def deco(fn):
        async def wrapper(*args, **kwargs):
            cur = delay
            for i in range(1, attempts + 1):
                try:
                    return await fn(*args, **kwargs)
                except Exception as e:
                    if i == attempts:
                        raise
                    logger.debug("Retry %s/%s for %s due to %s", i, attempts, fn.__name__, e)
                    await asyncio.sleep(cur)
                    cur *= backoff
        return wrapper
    return deco

# -----------------------
# Simple extractive summarizer
# -----------------------
def tokenize_text(s: str) -> List[str]:
    return re.findall(r"\w+", s.lower())

def score_sentences(sentences: List[str], top_terms: List[str]) -> List[Tuple[int, float]]:
    termset = set(top_terms)
    scored = []
    for i, s in enumerate(sentences):
        toks = tokenize_text(s)
        if not toks:
            scored.append((i, 0.0)); continue
        tf = sum(1 for t in toks if t in termset)
        score = tf / max(1, len(toks))
        score *= (1 + min(1.0, len(toks) / 20.0))
        scored.append((i, score))
    return scored

def local_extractive_summary(texts: List[str], max_sentences: int = 6) -> str:
    full = "\n".join(texts)
    if not full.strip():
        return ""
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', full) if s.strip()]
    if not sentences:
        return ""
    tokens = tokenize_text(full)
    freq = {}
    for t in tokens:
        freq[t] = freq.get(t, 0) + 1
    stopwords = set(["the","and","to","a","of","in","for","is","on","with","that","as","are","it","by","an"])
    top_terms = [t for t,_ in sorted(freq.items(), key=lambda x: -x[1]) if t not in stopwords][:50]
    scored = score_sentences(sentences, top_terms)
    scored_sorted = sorted(scored, key=lambda x: -x[1])
    selected_idxs = sorted([i for i,_ in scored_sorted[:max_sentences]])
    return " ".join(sentences[i] for i in selected_idxs)

# -----------------------
# RAG interface
# -----------------------
class RagClientInterface:
    async def add_context(self, session_id: str, text: str, metadata: dict) -> str:
        raise NotImplementedError()
    async def search(self, session_id: str, top_k: int=10) -> List[dict]:
        raise NotImplementedError()

# -----------------------
# Simple in-memory token bucket (dev)
# -----------------------
@dataclass
class TokenBucket:
    capacity: int
    tokens: float
    refill_interval_sec: int
    last_refill: float

    def consume(self, amount: float = 1.0) -> bool:
        now_ts = time.time()
        elapsed = now_ts - self.last_refill
        if elapsed >= self.refill_interval_sec:
            refill_amount = (elapsed / self.refill_interval_sec) * self.capacity
            self.tokens = min(self.capacity, self.tokens + refill_amount)
            self.last_refill = now_ts
        if self.tokens >= amount:
            self.tokens -= amount
            return True
        return False

_rate_buckets: Dict[str, TokenBucket] = {}
def get_bucket_for_key(api_key: str) -> TokenBucket:
    if api_key not in _rate_buckets:
        _rate_buckets[api_key] = TokenBucket(capacity=RATE_BUCKET_CAPACITY, tokens=RATE_BUCKET_CAPACITY, refill_interval_sec=RATE_BUCKET_REFILL_SEC, last_refill=time.time())
    return _rate_buckets[api_key]

# -----------------------
# Core MCP service class
# -----------------------
class MCPService:
    def __init__(self):
        self.rag_client: Optional[RagClientInterface] = None
        self.metrics = {"sessions_created": 0, "items_appended": 0, "snapshots_created": 0, "auth_failures": 0}
        self._sequence_map: Dict[str, int] = {}
        self._cleanup_task: Optional[asyncio.Task] = None

    async def migrate(self):
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("DB migrated (tables ensured)")

    # Session lifecycle
    async def create_session(self, owner: Optional[str]=None, metadata: Optional[dict]=None, ttl_hours: Optional[int]=None) -> dict:
        sid = str(uuid.uuid4())
        now_dt = now()
        ttl = ttl_hours if ttl_hours is not None else MCP_SESSION_TTL_HOURS
        meta_text = json.dumps(metadata) if metadata is not None else None
        async with AsyncSessionLocal() as db:
            s = SessionModel(session_id=sid, owner=owner, created_at=now_dt, last_active_at=now_dt, meta_json=meta_text, ttl_hours=ttl, is_active=True)
            db.add(s)
            try:
                await db.commit()
            except IntegrityError:
                await db.rollback()
                raise HTTPException(status_code=500, detail="DB integrity error")
        self._sequence_map[sid] = 0
        self.metrics["sessions_created"] += 1
        logger.info("created session %s owner=%s", sid, owner)
        return {"session_id": sid, "owner": owner, "created_at": now_dt.isoformat(), "last_active_at": now_dt.isoformat(), "ttl_hours": ttl}

    async def get_session(self, session_id: str) -> Optional[dict]:
        async with AsyncSessionLocal() as db:
            res = await db.execute(select(SessionModel).where(SessionModel.session_id == session_id))
            s = res.scalar_one_or_none()
            if not s:
                return None
            return {"session_id": s.session_id, "owner": s.owner, "created_at": s.created_at, "last_active_at": s.last_active_at, "metadata": safe_json_load(s.meta_json), "ttl_hours": s.ttl_hours, "is_active": s.is_active}

    async def delete_session(self, session_id: str) -> bool:
        async with AsyncSessionLocal() as db:
            res = await db.execute(select(SessionModel).where(SessionModel.session_id == session_id))
            s = res.scalar_one_or_none()
            if not s:
                return False
            await db.delete(s)
            await db.commit()
            # best-effort cleanup related items
            await db.execute(ContextItemModel.__table__.delete().where(ContextItemModel.session_id == session_id))
            await db.execute(SnapshotModel.__table__.delete().where(SnapshotModel.session_id == session_id))
            await db.execute(EvidenceModel.__table__.delete().where(EvidenceModel.session_id == session_id))
            await db.commit()
        logger.info("deleted session %s", session_id)
        return True

    # Context store
    async def append_context_item(self, session_id: str, item: ContextItemCreateSchema) -> ContextItemReadSchema:
        async with AsyncSessionLocal() as db:
            s = (await db.execute(select(SessionModel).where(SessionModel.session_id == session_id))).scalar_one_or_none()
            if not s:
                raise HTTPException(status_code=404, detail="Session not found")
            seq = self._sequence_map.get(session_id, 0) + 1
            self._sequence_map[session_id] = seq
            item_id = make_uuid()
            meta_text = json.dumps(item.metadata.dict()) if item.metadata is not None else None
            ct = ContextItemModel(item_id=item_id, session_id=session_id, role=item.role, content=item.content, vector_id=item.vector_id, created_at=now(), meta_json=meta_text, trusted=bool(item.trusted), deprecated=False, sequence=seq)
            db.add(ct)
            s.last_active_at = now()
            s.version = (s.version or 0) + 1
            db.add(s)
            await db.commit()
            self.metrics["items_appended"] += 1
            logger.debug("appended item %s to session %s", item_id, session_id)
            return ContextItemReadSchema(item_id=item_id, session_id=session_id, role=item.role, content=item.content, vector_id=item.vector_id, created_at=ct.created_at, metadata=item.metadata.dict() if item.metadata else None, trusted=ct.trusted, deprecated=False, sequence=seq)

    async def retrieve_context_items(self, session_id: str, last_n: Optional[int]=20, since: Optional[datetime]=None, role: Optional[str]=None, source: Optional[str]=None, page: int=1, page_size: int=50) -> dict:
        page_size = min(page_size, MCP_MAX_PAGE_SIZE)
        offset = (page - 1) * page_size
        async with AsyncSessionLocal() as db:
            query = select(ContextItemModel).where(ContextItemModel.session_id == session_id, ContextItemModel.deprecated == False)
            if since:
                query = query.where(ContextItemModel.created_at >= since)
            if role:
                query = query.where(ContextItemModel.role == role)
            if source:
                like_val = f'%{source}%'
                query = query.where(ContextItemModel.meta_json.like(like_val))
            if last_n:
                subq = query.order_by(ContextItemModel.sequence.desc()).limit(last_n)
                res = await db.execute(subq)
                items = res.scalars().all()
                items_sorted = sorted(items, key=lambda x: x.sequence)
                total = len(items_sorted)
                return {"total": total, "page": 1, "page_size": total, "items": [self._ctx_to_schema(i) for i in items_sorted]}
            count_q = await db.execute(select(func.count()).select_from(query.subquery()))
            total = count_q.scalar_one()
            q = query.order_by(ContextItemModel.sequence.asc()).offset(offset).limit(page_size)
            res = await db.execute(q)
            items = res.scalars().all()
            return {"total": total, "page": page, "page_size": page_size, "items": [self._ctx_to_schema(i) for i in items]}

    def _ctx_to_schema(self, row: ContextItemModel) -> ContextItemReadSchema:
        return ContextItemReadSchema(item_id=row.item_id, session_id=row.session_id, role=row.role, content=row.content, vector_id=row.vector_id, created_at=row.created_at, metadata=safe_json_load(row.meta_json), trusted=bool(row.trusted), deprecated=bool(row.deprecated), sequence=int(row.sequence))

    async def replace_last_n(self, session_id: str, n: int, new_item: ContextItemCreateSchema) -> List[ContextItemReadSchema]:
        async with AsyncSessionLocal() as db:
            res = await db.execute(select(ContextItemModel).where(ContextItemModel.session_id == session_id).order_by(ContextItemModel.sequence.desc()).limit(n))
            items = res.scalars().all()
            if not items:
                raise HTTPException(status_code=404, detail="No items to replace")
            for it in items:
                it.deprecated = True
                db.add(it)
            seq = self._sequence_map.get(session_id, 0) + 1
            self._sequence_map[session_id] = seq
            item_id = make_uuid()
            meta_text = json.dumps(new_item.metadata.dict()) if new_item.metadata else None
            ct = ContextItemModel(item_id=item_id, session_id=session_id, role=new_item.role, content=new_item.content, vector_id=new_item.vector_id, created_at=now(), meta_json=meta_text, trusted=bool(new_item.trusted), deprecated=False, sequence=seq)
            db.add(ct)
            await db.commit()
            return [self._ctx_to_schema(ct)]

    async def mark_item_trusted(self, item_id: str, trusted: bool = True) -> bool:
        async with AsyncSessionLocal() as db:
            res = await db.execute(select(ContextItemModel).where(ContextItemModel.item_id == item_id))
            it = res.scalar_one_or_none()
            if not it:
                return False
            it.trusted = trusted
            db.add(it)
            await db.commit()
            return True

    async def mark_item_deprecated(self, item_id: str, deprecated: bool = True) -> bool:
        async with AsyncSessionLocal() as db:
            res = await db.execute(select(ContextItemModel).where(ContextItemModel.item_id == item_id))
            it = res.scalar_one_or_none()
            if not it:
                return False
            it.deprecated = deprecated
            db.add(it)
            await db.commit()
            return True

    # Snapshots / summarization
    async def summarize_session(self, session_id: str, strategy: str = "rolling", max_sentences: int = 8) -> SnapshotReadSchema:
        if strategy not in ("rolling", "snapshot", "topic_based"):
            raise HTTPException(status_code=400, detail="Invalid strategy")
        async with AsyncSessionLocal() as db:
            res = await db.execute(select(ContextItemModel).where(ContextItemModel.session_id == session_id, ContextItemModel.deprecated == False).order_by(ContextItemModel.sequence.desc()).limit(500))
            items = res.scalars().all()
            if not items:
                raise HTTPException(status_code=404, detail="No context to summarize")
            texts = [i.content for i in list(reversed(items))]
        summary_text = local_extractive_summary(texts, max_sentences=max_sentences)
        method = f"local_{strategy}"
        # Hook into project LLM integration if available (non-blocking)
        try:
            # integrations.llm_interface is optional; when present, should expose async abstractive_summary
            import integrations.llm_interface as llmi  # type: ignore
            if hasattr(llmi, "abstractive_summary"):
                abstr = await llmi.abstractive_summary("\n\n".join(texts), strategy=strategy, max_sentences=max_sentences)
                if abstr:
                    summary_text = abstr
                    method = f"llm_abstractive_{strategy}"
        except Exception:
            # absent or failed - stick with local summary
            pass
        snapshot_id = make_uuid()
        meta_text = json.dumps({"strategy": strategy})
        async with AsyncSessionLocal() as db:
            snap = SnapshotModel(snapshot_id=snapshot_id, session_id=session_id, summary_text=summary_text, method=method, created_at=now(), meta_json=meta_text)
            db.add(snap)
            await db.commit()
        self.metrics["snapshots_created"] += 1
        logger.info("snapshot %s created for session %s method=%s", snapshot_id, session_id, method)
        return SnapshotReadSchema(snapshot_id=snapshot_id, session_id=session_id, summary_text=summary_text, method=method, created_at=now(), metadata={"strategy": strategy})

    async def list_snapshots(self, session_id: str) -> List[SnapshotReadSchema]:
        async with AsyncSessionLocal() as db:
            res = await db.execute(select(SnapshotModel).where(SnapshotModel.session_id == session_id).order_by(SnapshotModel.created_at.desc()))
            snaps = res.scalars().all()
            return [SnapshotReadSchema(snapshot_id=s.snapshot_id, session_id=s.session_id, summary_text=s.summary_text, method=s.method, created_at=s.created_at, metadata=safe_json_load(s.meta_json)) for s in snaps]

    # Evidence
    async def attach_evidence(self, session_id: str, ev: EvidenceCreateSchema) -> dict:
        eid = make_uuid()
        async with AsyncSessionLocal() as db:
            s = (await db.execute(select(SessionModel).where(SessionModel.session_id == session_id))).scalar_one_or_none()
            if not s:
                raise HTTPException(status_code=404, detail="Session not found")
            ev_entry = EvidenceModel(evidence_id=eid, session_id=session_id, attached_to=ev.attached_to, data=json.dumps(ev.data), created_at=now(), meta_json=json.dumps(ev.metadata) if ev.metadata else None)
            db.add(ev_entry)
            await db.commit()
        logger.info("evidence %s attached for session %s", eid, session_id)
        return {"evidence_id": eid}

    # RAG-ready payload
    async def get_relevant_context_for_rag(self, session_id: str, top_k: int = 10) -> List[dict]:
        if self.rag_client:
            try:
                results = await self.rag_client.search(session_id=session_id, top_k=top_k)
                return results
            except Exception as e:
                logger.warning("rag_client.search failed: %s - falling back to local context", e)
        async with AsyncSessionLocal() as db:
            res = await db.execute(select(ContextItemModel).where(ContextItemModel.session_id == session_id, ContextItemModel.deprecated == False).order_by(ContextItemModel.sequence.desc()).limit(top_k))
            items = res.scalars().all()
            out = []
            for it in reversed(items):
                out.append({"id": it.item_id, "text": it.content, "metadata": safe_json_load(it.meta_json), "created_at": it.created_at.isoformat(), "vector_id": it.vector_id})
            return out

    # Cleanup TTL enforcement
    async def cleanup_expired_sessions(self):
        cutoff = now() - timedelta(hours=MCP_SESSION_TTL_HOURS)
        async with AsyncSessionLocal() as db:
            res = await db.execute(select(SessionModel).where(SessionModel.last_active_at < cutoff, SessionModel.is_active == True))
            sessions = res.scalars().all()
            for s in sessions:
                s.is_active = False
                db.add(s)
            await db.commit()
        logger.info("cleanup expired sessions done; marked %d inactive", len(sessions))

    def start_background_cleanup(self):
        if self._cleanup_task and not self._cleanup_task.done():
            return
        loop = asyncio.get_event_loop()
        self._cleanup_task = loop.create_task(self._cleanup_loop())
        logger.info("started background cleanup task")

    async def _cleanup_loop(self):
        try:
            while True:
                try:
                    await self.cleanup_expired_sessions()
                except Exception as e:
                    logger.exception("cleanup loop error: %s", e)
                await asyncio.sleep(MCP_CLEANUP_INTERVAL_SECONDS)
        except asyncio.CancelledError:
            return

    async def shutdown(self):
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except Exception:
                pass

mcp_service = MCPService()

# -----------------------
# FastAPI App and routes
# -----------------------
app = FastAPI(title="MCP Service - AI Job Automation Agent", version="2.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"] if MCP_DEV_MODE else [], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# Startup / shutdown events
@app.on_event("startup")
async def startup():
    await mcp_service.migrate()
    mcp_service.start_background_cleanup()
    logger.info("MCP service started - DB=%s dev_mode=%s", DATABASE_URL, MCP_DEV_MODE)

@app.on_event("shutdown")
async def shutdown_event():
    await mcp_service.shutdown()
    await engine.dispose()
    logger.info("MCP service shutdown complete")

# Auth & rate limit dependency
async def auth_and_rate_limit(x_mcp_api_key: Optional[str] = Header(None)):
    if MCP_DEV_MODE:
        return True
    if not x_mcp_api_key or x_mcp_api_key != MCP_API_KEY:
        mcp_service.metrics["auth_failures"] += 1
        raise HTTPException(status_code=401, detail="Invalid API key")
    bucket = get_bucket_for_key(x_mcp_api_key)
    if not bucket.consume(1.0):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    return True

@app.post("/v1/auth/validate")
async def api_auth_validate(ok: bool = Depends(auth_and_rate_limit)):
    return {"ok": True}

@app.post("/v1/sessions", dependencies=[Depends(auth_and_rate_limit)])
async def api_create_session(payload: SessionCreateSchema):
    s = await mcp_service.create_session(owner=payload.owner, metadata=payload.metadata, ttl_hours=payload.ttl_hours)
    return s

@app.get("/v1/sessions/{session_id}", dependencies=[Depends(auth_and_rate_limit)])
async def api_get_session(session_id: str):
    s = await mcp_service.get_session(session_id)
    if not s:
        raise HTTPException(status_code=404, detail="Session not found")
    return s

@app.delete("/v1/sessions/{session_id}", dependencies=[Depends(auth_and_rate_limit)])
async def api_delete_session(session_id: str):
    ok = await mcp_service.delete_session(session_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"deleted": True}

@app.post("/v1/sessions/{session_id}/items", dependencies=[Depends(auth_and_rate_limit)])
async def api_append_item(session_id: str, item: ContextItemCreateSchema):
    i = await mcp_service.append_context_item(session_id, item)
    return i

@app.get("/v1/sessions/{session_id}/items", dependencies=[Depends(auth_and_rate_limit)])
async def api_get_items(session_id: str, last_n: Optional[int] = None, since: Optional[str] = None, role: Optional[str] = None, source: Optional[str] = None, page: int = 1, page_size: int = 50):
    since_dt = None
    if since:
        try:
            since_dt = datetime.fromisoformat(since)
        except Exception:
            raise HTTPException(status_code=400, detail="since must be ISO datetime")
    res = await mcp_service.retrieve_context_items(session_id, last_n=last_n, since=since_dt, role=role, source=source, page=page, page_size=page_size)
    return res

@app.post("/v1/sessions/{session_id}/replace_last", dependencies=[Depends(auth_and_rate_limit)])
async def api_replace_last(session_id: str, n: int = 1, new_item: ContextItemCreateSchema = None):
    if n < 1:
        raise HTTPException(status_code=400, detail="n must be >=1")
    if not new_item:
        raise HTTPException(status_code=400, detail="new_item required")
    res = await mcp_service.replace_last_n(session_id, n, new_item)
    return {"replaced": res}

@app.post("/v1/sessions/{session_id}/snapshot", dependencies=[Depends(auth_and_rate_limit)])
async def api_create_snapshot(session_id: str, payload: SnapshotCreateSchema):
    snap = await mcp_service.summarize_session(session_id, strategy=payload.strategy, max_sentences=payload.max_sentences)
    return snap

@app.get("/v1/sessions/{session_id}/snapshots", dependencies=[Depends(auth_and_rate_limit)])
async def api_list_snapshots(session_id: str):
    snaps = await mcp_service.list_snapshots(session_id)
    return snaps

@app.post("/v1/sessions/{session_id}/evidence", dependencies=[Depends(auth_and_rate_limit)])
async def api_attach_evidence(session_id: str, ev: EvidenceCreateSchema):
    res = await mcp_service.attach_evidence(session_id, ev)
    return res

@app.post("/v1/sessions/{session_id}/items/{item_id}/trust", dependencies=[Depends(auth_and_rate_limit)])
async def api_mark_trusted(session_id: str, item_id: str, trusted: bool = True):
    ok = await mcp_service.mark_item_trusted(item_id, trusted=trusted)
    if not ok:
        raise HTTPException(status_code=404, detail="Item not found")
    return {"ok": True}

@app.post("/v1/sessions/{session_id}/items/{item_id}/deprecated", dependencies=[Depends(auth_and_rate_limit)])
async def api_mark_deprecated(session_id: str, item_id: str, deprecated: bool = True):
    ok = await mcp_service.mark_item_deprecated(item_id, deprecated=deprecated)
    if not ok:
        raise HTTPException(status_code=404, detail="Item not found")
    return {"ok": True}

@app.get("/v1/relevant/{session_id}", dependencies=[Depends(auth_and_rate_limit)])
async def api_rag(session_id: str, top_k: int = 10):
    out = await mcp_service.get_relevant_context_for_rag(session_id, top_k=top_k)
    return {"items": out}

@app.get("/metrics")
async def api_metrics():
    return mcp_service.metrics

@app.get("/health")
async def api_health():
    return {"status": "ok", "time": now().isoformat()}

# -----------------------
# Migration SQL (file output)
# -----------------------
MIGRATION_SQL = """
CREATE TABLE IF NOT EXISTS sessions (
  session_id TEXT PRIMARY KEY,
  owner TEXT,
  created_at TIMESTAMP,
  last_active_at TIMESTAMP,
  meta_json TEXT,
  ttl_hours INTEGER,
  is_active BOOLEAN DEFAULT 1,
  version INTEGER DEFAULT 1
);
CREATE INDEX IF NOT EXISTS ix_sessions_last_active ON sessions(last_active_at);

CREATE TABLE IF NOT EXISTS context_items (
  item_id TEXT PRIMARY KEY,
  session_id TEXT,
  role TEXT,
  content TEXT,
  vector_id TEXT,
  created_at TIMESTAMP,
  meta_json TEXT,
  trusted BOOLEAN DEFAULT 0,
  deprecated BOOLEAN DEFAULT 0,
  sequence INTEGER
);
CREATE INDEX IF NOT EXISTS ix_ctx_session_sequence ON context_items(session_id, sequence);
CREATE INDEX IF NOT EXISTS ix_ctx_session_created_at ON context_items(session_id, created_at);

CREATE TABLE IF NOT EXISTS snapshots (
  snapshot_id TEXT PRIMARY KEY,
  session_id TEXT,
  summary_text TEXT,
  method TEXT,
  created_at TIMESTAMP,
  meta_json TEXT
);

CREATE TABLE IF NOT EXISTS evidence (
  evidence_id TEXT PRIMARY KEY,
  session_id TEXT,
  attached_to TEXT,
  data TEXT,
  created_at TIMESTAMP,
  meta_json TEXT
);
"""

# write SQL file for reference
try:
    with open(os.path.join("mcp", "mcp_schema.sql"), "w", encoding="utf-8") as f:
        f.write(MIGRATION_SQL.strip())
except Exception:
    pass

# -----------------------
# Example snippets (docstrings)
# -----------------------
EXAMPLE_SCRAPER_SNIPPET = """
Example scraper -> post item then retrieve last 5 items:
curl -X POST "http://localhost:8080/v1/sessions" -H "X-MCP-API-KEY: <key>" -d '{"owner":"scraper"}'
curl -X POST "http://localhost:8080/v1/sessions/{session_id}/items" -H "X-MCP-API-KEY: <key>" -H "Content-Type: application/json" -d '{"role":"tool","content":"Job scraped...","metadata":{"source":"scraper","job_id":"id123","url":"https://...","confidence":0.9}}'
curl -X GET "http://localhost:8080/v1/sessions/{session_id}/items?last_n=5" -H "X-MCP-API-KEY: <key>"
"""

EXAMPLE_AGENT_SNIPPET = """
Example agent -> snapshot & RAG payload:
curl -X POST "http://localhost:8080/v1/sessions/{session_id}/snapshot" -H "X-MCP-API-KEY: <key>" -d '{"strategy":"rolling","max_sentences":6}'
curl -X GET "http://localhost:8080/v1/relevant/{session_id}?top_k=10" -H "X-MCP-API-KEY: <key>"
"""

# End of file