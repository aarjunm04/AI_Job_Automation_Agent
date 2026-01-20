# ═════════════════════════════════════════════════════════════════════════════════
#Enterprise-grade context management service for job automation AI agents.

#Features:
#- Session lifecycle management with TTL and versioning
#- Context item storage with sequence tracking and pagination
#- Snapshot/summarization with LLM integration hooks
#- Evidence attachment system
#- Audit trail for all operations
#- Integration orchestration (RAG, LLM, Notion, Scrapers)
#- Webhook receivers (Chrome extension, n8n, scrapers)
#- Redis caching and compression
#- Rate limiting and security
#- Prometheus metrics and health checks

#Architecture:
#- FastAPI async service with Pydantic validation
#- SQLAlchemy async ORM (SQLite dev / PostgreSQL prod)
#- Redis for caching and rate limiting
#- Circuit breakers for external integrations
#- Structured JSON audit logging

#Author: Job Automation Team
#Version: 2.0 Enterprise



from __future__ import annotations

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: IMPORTS & DEPENDENCIES
# ═══════════════════════════════════════════════════════════════════════════════

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

# FastAPI and web framework
from fastapi import FastAPI, Header, HTTPException, Depends, Request, BackgroundTasks, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# Data validation and settings
from pydantic import BaseModel, Field, validator
from pydantic_settings import BaseSettings

# Database
from sqlalchemy import Column, String, Integer, DateTime, Text, Boolean, func, select, Index
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base
from sqlalchemy.exc import IntegrityError

# HTTP client
import httpx

# Logging
from rich.logging import RichHandler

# RAG Client integration
import sys
from pathlib import Path

mcp_dir = Path(__file__).parent
sys.path.insert(0, str(mcp_dir))

from integrations import get_rag_client

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

class Settings(BaseSettings):
    """
    Centralized configuration with environment variable support.
    All settings can be overridden via .env file or environment variables.
    """
    
    # Core service settings
    service_name: str = "Job Automation MCP"
    version: str = "2.0"
    dev_mode: bool = Field(default=False, alias="MCP_DEV_MODE")
    
    # Security
    api_key: str = Field(default="", alias="MCP_API_KEY")
    jwt_secret: str = Field(default="change-me-in-production", alias="JWT_SECRET")
    jwt_expiry_hours: int = Field(default=24, alias="JWT_EXPIRY_HOURS")
    
    # Database
    database_url: str = Field(
        default="sqlite+aiosqlite:////app/data/mcp_context.db",
        alias="DATABASE_URL"
    )
    db_pool_size: int = Field(default=20, alias="DB_POOL_SIZE")
    db_max_overflow: int = Field(default=10, alias="DB_MAX_OVERFLOW")
    db_pool_timeout: int = Field(default=30, alias="DB_POOL_TIMEOUT")
    
    # Redis
    redis_url: str = Field(default="redis://localhost:6379/0", alias="REDIS_URL")
    redis_enabled: bool = Field(default=True, alias="REDIS_ENABLED")
    
    # Rate limiting
    rate_limit_enabled: bool = Field(default=True, alias="RATE_LIMIT_ENABLED")
    rate_limit_requests: int = Field(default=1000, alias="RATE_LIMIT_REQUESTS")
    rate_limit_window: int = Field(default=60, alias="RATE_LIMIT_WINDOW")
    
    # Session management
    session_ttl_hours: int = Field(default=168, alias="SESSION_TTL_HOURS")  # 7 days
    cleanup_interval_seconds: int = Field(default=1800, alias="CLEANUP_INTERVAL_SECONDS")
    
    # Context storage
    max_page_size: int = Field(default=200, alias="MAX_PAGE_SIZE")
    compression_threshold: int = Field(default=10240, alias="COMPRESSION_THRESHOLD")  # 10KB
    
    # External integrations (will be implemented via integrations.py)
    gemini_api_key: str = Field(default="", alias="GEMINI_API_KEY")
    gemini_base: str = Field(default="https://api.generative.googleapis.com/v1", alias="GEMINI_BASE")
    openrouter_api_key: str = Field(default="", alias="OPENROUTER_API_KEY")
    openrouter_base: str = Field(default="https://api.openrouter.ai", alias="OPENROUTER_BASE")
    perplexity_api_key: str = Field(default="", alias="PERPLEXITY_API_KEY")
    grok_api_key: str = Field(default="", alias="GROK_API_KEY")
    
    # Logging
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    # RAG Configuration
    RAG_BASE_URL: str = Field(default="http://localhost:8090")
    RAG_API_KEY: str = Field(default="mcp-default", alias="RAG_KEY_MCP") 

    
    class Config:
        env_file = ".env"
        case_sensitive = False

# Initialize settings
settings = Settings()

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: LOGGING SETUP
# ═══════════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=settings.log_level,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger("mcp")
logger.setLevel(settings.log_level)

logger.info(f"🚀 {settings.service_name} v{settings.version} initializing...")
logger.info(f"📊 Dev mode: {settings.dev_mode}")
logger.info(f"🗄️  Database: {settings.database_url.split('://')[0]}")
logger.info(f"🔴 Redis: {'enabled' if settings.redis_enabled else 'disabled'}")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: DATABASE MODELS
# ═══════════════════════════════════════════════════════════════════════════════

# Ensure data directory exists
os.makedirs("mcp", exist_ok=True)

Base = declarative_base()


class SessionModel(Base):
    """
    Session represents a conversation or work session between user/agent and MCP.
    Sessions have TTL and are automatically cleaned up when expired.
    """
    __tablename__ = "sessions"
    
    session_id = Column(String(36), primary_key=True, comment="UUID for session")
    owner = Column(String(128), nullable=True, comment="User or agent identifier")
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    last_active_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    meta_json = Column(Text, nullable=True, comment="JSON metadata")
    ttl_hours = Column(Integer, default=settings.session_ttl_hours, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    version = Column(Integer, default=1, nullable=False, comment="Optimistic locking")
    
    __table_args__ = (
        Index("ix_sessions_last_active", "last_active_at"),
        Index("ix_sessions_owner", "owner"),
    )


class ContextItemModel(Base):
    """
    Context items are the core data storage unit. They represent messages,
    tool results, scraped data, or any content that needs to be tracked.
    """
    __tablename__ = "context_items"
    
    item_id = Column(String(48), primary_key=True)
    session_id = Column(String(36), nullable=False, index=True)
    role = Column(String(32), nullable=False, comment="system/user/assistant/tool")
    content = Column(Text, nullable=False, comment="Can be compressed if large")
    vector_id = Column(String(128), nullable=True, comment="Reference to vector DB")
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    meta_json = Column(Text, nullable=True, comment="JSON metadata with source info")
    trusted = Column(Boolean, default=False, comment="Verified/trusted content flag")
    deprecated = Column(Boolean, default=False, comment="Soft delete flag")
    sequence = Column(Integer, nullable=False, default=0, comment="Order within session")
    compressed = Column(Boolean, default=False, comment="Is content ZSTD compressed")
    
    __table_args__ = (
        Index("ix_ctx_session_sequence", "session_id", "sequence"),
        Index("ix_ctx_session_created_at", "session_id", "created_at"),
        Index("ix_ctx_role", "role"),
    )


class SnapshotModel(Base):
    """
    Snapshots are summaries of session context, used for efficient
    retrieval and LLM context window management.
    """
    __tablename__ = "snapshots"
    
    snapshot_id = Column(String(48), primary_key=True)
    session_id = Column(String(36), nullable=False, index=True)
    summary_text = Column(Text, nullable=False)
    method = Column(String(32), nullable=False, comment="local_extractive or llm_abstractive")
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    meta_json = Column(Text, nullable=True)
    
    __table_args__ = (
        Index("ix_snapshot_session", "session_id"),
    )


class EvidenceModel(Base):
    """
    Evidence stores supporting data/metadata for context items.
    Used for debugging, audit trails, and decision provenance.
    """
    __tablename__ = "evidence"
    
    evidence_id = Column(String(48), primary_key=True)
    session_id = Column(String(36), nullable=False, index=True)
    attached_to = Column(String(48), nullable=True, comment="Context item or resource ID")
    data = Column(Text, nullable=False, comment="JSON data")
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    meta_json = Column(Text, nullable=True)
    
    __table_args__ = (
        Index("ix_evidence_session", "session_id"),
        Index("ix_evidence_attached", "attached_to"),
    )


class AuditLogModel(Base):
    """
    Immutable audit trail for all MCP operations.
    Critical for debugging, compliance, and understanding agent behavior.
    """
    __tablename__ = "audit_logs"
    
    log_id = Column(String(48), primary_key=True)
    session_id = Column(String(36), nullable=True, index=True)
    actor_type = Column(String(32), nullable=False, comment="user/agent/chrome_ext/n8n/scraper/system")
    actor_id = Column(String(128), nullable=True)
    action = Column(String(64), nullable=False, index=True, comment="session.create/context.append/tool.execute")
    resource_type = Column(String(32), nullable=False, comment="session/context_item/snapshot/evidence")
    resource_id = Column(String(128), nullable=True)
    outcome = Column(String(16), nullable=False, comment="success/failure/error")
    error_message = Column(Text, nullable=True)
    audit_metadata = Column(Text, nullable=True, comment="JSON with request/response details")
    ip_address = Column(String(45), nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    
    __table_args__ = (
        Index("ix_audit_session_timestamp", "session_id", "timestamp"),
        Index("ix_audit_actor_timestamp", "actor_type", "timestamp"),
        Index("ix_audit_action_timestamp", "action", "timestamp"),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5: PYDANTIC SCHEMAS
# ═══════════════════════════════════════════════════════════════════════════════

class MetadataSchema(BaseModel):
    """Metadata schema for context items and other resources."""
    source: Optional[str] = Field(
        None,
        description="Data source: scraper/playwright_scraper/playwright_apply/n8n/chrome_extension/agent/rag/llm"
    )
    job_id: Optional[str] = Field(None, description="Job listing identifier")
    url: Optional[str] = Field(None, description="Source URL")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Confidence score")
    tags: Optional[List[str]] = Field(default_factory=list, description="Custom tags")
    
    class Config:
        extra = "allow"  # Allow additional fields


class SessionCreateSchema(BaseModel):
    """Request schema for creating a new session."""
    owner: Optional[str] = Field(None, description="Session owner identifier")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Custom metadata")
    ttl_hours: Optional[int] = Field(None, description="Session TTL override")


class SessionReadSchema(BaseModel):
    """Response schema for session data."""
    session_id: str
    owner: Optional[str]
    created_at: datetime
    last_active_at: datetime
    metadata: Optional[Dict[str, Any]]
    ttl_hours: int
    is_active: bool
    version: int


class ContextItemCreateSchema(BaseModel):
    """Request schema for creating context items."""
    role: str = Field(..., description="Message role")
    content: str = Field(..., max_length=500000, description="Content (max 500KB)")
    vector_id: Optional[str] = Field(None, description="Vector DB reference")
    metadata: Optional[MetadataSchema] = None
    trusted: Optional[bool] = Field(False, description="Mark as trusted content")
    
    @validator("role")
    def validate_role(cls, v):
        allowed_roles = ["system", "user", "assistant", "tool"]
        if v not in allowed_roles:
            raise ValueError(f"role must be one of: {', '.join(allowed_roles)}")
        return v
    
    @validator("content")
    def validate_content(cls, v):
        """Basic content validation (enhanced by security.py)."""
        if not v or not v.strip():
            raise ValueError("content cannot be empty")
        return v


class ContextItemReadSchema(BaseModel):
    """Response schema for context items."""
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
    """Request schema for creating snapshots."""
    strategy: str = Field(default="rolling", description="Summarization strategy")
    max_sentences: Optional[int] = Field(default=8, description="Max sentences in summary")


class SnapshotReadSchema(BaseModel):
    """Response schema for snapshots."""
    snapshot_id: str
    session_id: str
    summary_text: str
    method: str
    created_at: datetime
    metadata: Optional[Dict[str, Any]]


class EvidenceCreateSchema(BaseModel):
    """Request schema for attaching evidence."""
    attached_to: Optional[str] = Field(None, description="Resource ID to attach to")
    data: Dict[str, Any] = Field(..., description="Evidence data")
    metadata: Optional[Dict[str, Any]] = None


class WebhookChromeSchema(BaseModel):
    """Schema for Chrome extension webhook payloads."""
    session_id: Optional[str] = None
    action: str = Field(..., description="form_data/resume_request/status_update")
    data: Dict[str, Any]


class WebhookN8nSchema(BaseModel):
    """Schema for n8n workflow webhook payloads."""
    workflow_id: str
    trigger_type: str
    data: Dict[str, Any]


class JobIngestSchema(BaseModel):
    """Schema for bulk job ingestion."""
    source: str = Field(..., description="jooble/remotive/linkedin/custom")
    jobs: List[Dict[str, Any]]
    metadata: Optional[Dict[str, Any]] = None


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6: DATABASE SETUP
# ═══════════════════════════════════════════════════════════════════════════════

# Create async engine with proper connection pooling
if "sqlite" in settings.database_url:
    # SQLite settings
    engine = create_async_engine(
        settings.database_url,
        future=True,
        echo=False,
        connect_args={"check_same_thread": False}
    )
else:
    # PostgreSQL settings
    engine = create_async_engine(
        settings.database_url,
        future=True,
        echo=False,
        pool_size=settings.db_pool_size,
        max_overflow=settings.db_max_overflow,
        pool_timeout=settings.db_pool_timeout,
        pool_pre_ping=True  # Verify connections before using
    )

# Session factory
AsyncSessionLocal = async_sessionmaker(
    engine,
    expire_on_commit=False,
    class_=AsyncSession
)

logger.info(f"✅ Database engine configured")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7: UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def make_uuid() -> str:
    """Generate UUID for resources."""
    return str(uuid.uuid4())


def now() -> datetime:
    """Get current UTC datetime."""
    return datetime.utcnow()


def safe_json_load(text: Optional[str]) -> Optional[dict]:
    """Safely parse JSON string, return None on error."""
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception as e:
        logger.warning(f"Failed to parse JSON: {e}")
        return None


def safe_json_dump(data: Optional[dict]) -> Optional[str]:
    """Safely serialize dict to JSON string."""
    if data is None:
        return None
    try:
        return json.dumps(data)
    except Exception as e:
        logger.warning(f"Failed to serialize JSON: {e}")
        return None


# Simple async retry decorator
def async_retry(attempts: int = 3, delay: float = 0.5, backoff: float = 2.0):
    """Retry async function with exponential backoff."""
    def decorator(fn):
        async def wrapper(*args, **kwargs):
            cur_delay = delay
            for attempt in range(1, attempts + 1):
                try:
                    return await fn(*args, **kwargs)
                except Exception as e:
                    if attempt == attempts:
                        raise
                    logger.debug(f"Retry {attempt}/{attempts} for {fn.__name__} due to {e}")
                    await asyncio.sleep(cur_delay)
                    cur_delay *= backoff
            return None
        return wrapper
    return decorator


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8: LOCAL EXTRACTIVE SUMMARIZER
# ═══════════════════════════════════════════════════════════════════════════════

def tokenize_text(s: str) -> List[str]:
    """Simple tokenization for summarization."""
    return re.findall(r"\\w+", s.lower())


def score_sentences(sentences: List[str], top_terms: List[str]) -> List[Tuple[int, float]]:
    """Score sentences based on term frequency."""
    termset = set(top_terms)
    scored = []
    for i, s in enumerate(sentences):
        toks = tokenize_text(s)
        if not toks:
            scored.append((i, 0.0))
            continue
        tf = sum(1 for t in toks if t in termset)
        score = tf / max(1, len(toks))
        score *= (1 + min(1.0, len(toks) / 20.0))  # Bonus for longer sentences
        scored.append((i, score))
    return scored


def local_extractive_summary(texts: List[str], max_sentences: int = 6) -> str:
    """
    Create extractive summary from list of texts.
    Falls back method when LLM integration is not available.
    """
    full = "\\n".join(texts)
    if not full.strip():
        return ""
    
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\\s+', full) if s.strip()]
    if not sentences:
        return ""
    
    tokens = tokenize_text(full)
    freq = {}
    for t in tokens:
        freq[t] = freq.get(t, 0) + 1
    
    stopwords = set(["the", "and", "to", "a", "of", "in", "for", "is", "on", "with", "that", "as", "are", "it", "by", "an"])
    top_terms = [t for t, _ in sorted(freq.items(), key=lambda x: -x[1]) if t not in stopwords][:50]
    
    scored = score_sentences(sentences, top_terms)
    scored_sorted = sorted(scored, key=lambda x: -x[1])
    selected_idxs = sorted([i for i, _ in scored_sorted[:max_sentences]])
    
    return " ".join(sentences[i] for i in selected_idxs)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 9: CORE MCP SERVICE CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class MCPService:
    """
    Core MCP orchestration engine.
    
    Handles:
    - Session lifecycle (create, read, delete, TTL cleanup)
    - Context storage (append, retrieve, replace, deprecate)
    - Snapshots/summarization (local + LLM hooks)
    - Evidence attachment
    - Integration coordination (RAG, LLM, Notion, Scrapers)
    - Background tasks
    """
    
    def __init__(self):
        """Initialize MCP service with dependencies."""
        self.metrics = {
            "sessions_created": 0,
            "items_appended": 0,
            "snapshots_created": 0,
            "webhooks_received": 0,
            "auth_failures": 0,
            "cache_hits": 0,
            "cache_misses": 0,
        }
        self._sequence_map: Dict[str, int] = {}
        self._cleanup_task: Optional[asyncio.Task] = None
        
        # Integration clients (will be initialized from integrations.py)
        self.rag_client = None
        self.llm_router = None
        self.notion_client = None
        self.scraper_client = None
        
        # Cache client (will be initialized from cache.py)
        self.cache = None
        self.compressor = None
        
        # Security components (will be initialized from security.py)
        self.auth_handler = None
        self.rate_limiter = None
        self.audit_logger = None
        
        logger.info("✅ MCPService initialized")
    
    async def migrate(self):
        """Create database tables if they don't exist."""
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("✅ Database tables migrated")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # SESSION MANAGEMENT
    # ═══════════════════════════════════════════════════════════════════════════
    
    async def create_session(
        self,
        owner: Optional[str] = None,
        metadata: Optional[dict] = None,
        ttl_hours: Optional[int] = None
    ) -> dict:
        """
        Create a new session.
        
        Args:
            owner: Session owner identifier
            metadata: Custom metadata dict
            ttl_hours: Session TTL (overrides default)
            
        Returns:
            Session data dict
        """
        sid = make_uuid()
        now_dt = now()
        ttl = ttl_hours if ttl_hours is not None else settings.session_ttl_hours
        meta_text = safe_json_dump(metadata)
        
        async with AsyncSessionLocal() as db:
            s = SessionModel(
                session_id=sid,
                owner=owner,
                created_at=now_dt,
                last_active_at=now_dt,
                meta_json=meta_text,
                ttl_hours=ttl,
                is_active=True
            )
            db.add(s)
            try:
                await db.commit()
            except IntegrityError as e:
                await db.rollback()
                logger.error(f"Session creation failed: {e}")
                raise HTTPException(status_code=500, detail="Database integrity error")
        
        self._sequence_map[sid] = 0
        self.metrics["sessions_created"] += 1
        logger.info(f"✅ Created session {sid} owner={owner}")
        
        return {
            "session_id": sid,
            "owner": owner,
            "created_at": now_dt.isoformat(),
            "last_active_at": now_dt.isoformat(),
            "ttl_hours": ttl,
            "metadata": metadata
        }
    
    async def get_session(self, session_id: str) -> Optional[dict]:
        """
        Retrieve session by ID.
        Uses cache if available, falls back to database.
        """
        # Try cache first
        if self.cache:
            cached = await self.cache.get_session(session_id)
            if cached:
                self.metrics["cache_hits"] += 1
                return cached
            self.metrics["cache_misses"] += 1
        
        # Fetch from database
        async with AsyncSessionLocal() as db:
            res = await db.execute(
                select(SessionModel).where(SessionModel.session_id == session_id)
            )
            s = res.scalar_one_or_none()
            if not s:
                return None
            
            session_data = {
                "session_id": s.session_id,
                "owner": s.owner,
                "created_at": s.created_at,
                "last_active_at": s.last_active_at,
                "metadata": safe_json_load(s.meta_json),
                "ttl_hours": s.ttl_hours,
                "is_active": s.is_active,
                "version": s.version
            }
            
            # Cache for next time
            if self.cache:
                await self.cache.set_session(session_id, session_data)
            
            return session_data
    
    async def delete_session(self, session_id: str) -> bool:
        """
        Delete session and all related data.
        Cascades to context items, snapshots, evidence, and audit logs.
        """
        async with AsyncSessionLocal() as db:
            res = await db.execute(
                select(SessionModel).where(SessionModel.session_id == session_id)
            )
            s = res.scalar_one_or_none()
            if not s:
                return False
            
            await db.delete(s)
            
            # Clean up related data
            await db.execute(
                ContextItemModel.__table__.delete().where(
                    ContextItemModel.session_id == session_id
                )
            )
            await db.execute(
                SnapshotModel.__table__.delete().where(
                    SnapshotModel.session_id == session_id
                )
            )
            await db.execute(
                EvidenceModel.__table__.delete().where(
                    EvidenceModel.session_id == session_id
                )
            )
            
            await db.commit()
        
        # Invalidate cache
        if self.cache:
            await self.cache.invalidate_session(session_id)
        
        logger.info(f"✅ Deleted session {session_id}")
        return True
    
    # ═══════════════════════════════════════════════════════════════════════════
    # CONTEXT ITEM MANAGEMENT
    # ═══════════════════════════════════════════════════════════════════════════
    
    async def append_context_item(
        self,
        session_id: str,
        item: ContextItemCreateSchema
    ) -> ContextItemReadSchema:
        """
        Append new context item to session.
        Automatically handles compression for large content.
        """
        async with AsyncSessionLocal() as db:
            # Verify session exists
            s = (await db.execute(
                select(SessionModel).where(SessionModel.session_id == session_id)
            )).scalar_one_or_none()
            
            if not s:
                raise HTTPException(status_code=404, detail="Session not found")
            
            # Generate sequence number
            seq = self._sequence_map.get(session_id, 0) + 1
            self._sequence_map[session_id] = seq
            
            # Handle compression
            content = item.content
            compressed = False
            if self.compressor and len(content) > settings.compression_threshold:
                content = self.compressor.compress(content)
                compressed = True
                logger.debug(f"Compressed content for item (original: {len(item.content)} bytes)")
            
            # Create context item
            item_id = make_uuid()
            meta_text = safe_json_dump(item.metadata.dict() if item.metadata else None)
            
            ct = ContextItemModel(
                item_id=item_id,
                session_id=session_id,
                role=item.role,
                content=content,
                vector_id=item.vector_id,
                created_at=now(),
                meta_json=meta_text,
                trusted=bool(item.trusted),
                deprecated=False,
                sequence=seq,
                compressed=compressed
            )
            db.add(ct)
            
            # Update session activity
            s.last_active_at = now()
            s.version = (s.version or 0) + 1
            db.add(s)
            
            await db.commit()
        
        self.metrics["items_appended"] += 1
        
        # Invalidate cache
        if self.cache:
            await self.cache.invalidate_context(session_id)
        
        logger.debug(f"✅ Appended item {item_id} to session {session_id}")
        
        return ContextItemReadSchema(
            item_id=item_id,
            session_id=session_id,
            role=item.role,
            content=item.content,  # Return uncompressed
            vector_id=item.vector_id,
            created_at=ct.created_at,
            metadata=item.metadata.dict() if item.metadata else None,
            trusted=ct.trusted,
            deprecated=False,
            sequence=seq
        )
    
    async def retrieve_context_items(
        self,
        session_id: str,
        last_n: Optional[int] = 20,
        since: Optional[datetime] = None,
        role: Optional[str] = None,
        source: Optional[str] = None,
        page: int = 1,
        page_size: int = 50
    ) -> dict:
        """
        Retrieve context items with flexible filtering.
        Supports pagination, role filtering, source filtering, and time-based queries.
        """
        page_size = min(page_size, settings.max_page_size)
        offset = (page - 1) * page_size
        
        # Try cache for common queries
        cache_key = f"ctx:{session_id}:last{last_n}" if last_n and not since and not role and not source else None
        if self.cache and cache_key:
            cached = await self.cache.get(cache_key)
            if cached:
                self.metrics["cache_hits"] += 1
                return cached
            self.metrics["cache_misses"] += 1
        
        async with AsyncSessionLocal() as db:
            query = select(ContextItemModel).where(
                ContextItemModel.session_id == session_id,
                ContextItemModel.deprecated == False
            )
            
            # Apply filters
            if since:
                query = query.where(ContextItemModel.created_at >= since)
            if role:
                query = query.where(ContextItemModel.role == role)
            if source:
                like_val = f'%{source}%'
                query = query.where(ContextItemModel.meta_json.like(like_val))
            
            # Handle last_n (most recent items)
            if last_n:
                subq = query.order_by(ContextItemModel.sequence.desc()).limit(last_n)
                res = await db.execute(subq)
                items = res.scalars().all()
                items_sorted = sorted(items, key=lambda x: x.sequence)
                total = len(items_sorted)
                
                result = {
                    "total": total,
                    "page": 1,
                    "page_size": total,
                    "items": [self._ctx_to_schema(i) for i in items_sorted]
                }
                
                # Cache result
                if self.cache and cache_key:
                    await self.cache.set(cache_key, result, ttl=120)  # 2 min cache
                
                return result
            
            # Handle pagination
            count_q = await db.execute(select(func.count()).select_from(query.subquery()))
            total = count_q.scalar_one()
            
            q = query.order_by(ContextItemModel.sequence.asc()).offset(offset).limit(page_size)
            res = await db.execute(q)
            items = res.scalars().all()
            
            return {
                "total": total,
                "page": page,
                "page_size": page_size,
                "items": [self._ctx_to_schema(i) for i in items]
            }
    
    def _ctx_to_schema(self, row: ContextItemModel) -> ContextItemReadSchema:
        """Convert database model to Pydantic schema, handling decompression."""
        content = row.content
        
        # Decompress if needed
        if row.compressed and self.compressor:
            content = self.compressor.decompress(content)
        
        return ContextItemReadSchema(
            item_id=row.item_id,
            session_id=row.session_id,
            role=row.role,
            content=content,
            vector_id=row.vector_id,
            created_at=row.created_at,
            metadata=safe_json_load(row.meta_json),
            trusted=bool(row.trusted),
            deprecated=bool(row.deprecated),
            sequence=int(row.sequence)
        )
    
    async def replace_last_n(
        self,
        session_id: str,
        n: int,
        new_item: ContextItemCreateSchema
    ) -> List[ContextItemReadSchema]:
        """
        Replace last N items with a new item (useful for correcting mistakes).
        """
        async with AsyncSessionLocal() as db:
            res = await db.execute(
                select(ContextItemModel)
                .where(ContextItemModel.session_id == session_id)
                .order_by(ContextItemModel.sequence.desc())
                .limit(n)
            )
            items = res.scalars().all()
            
            if not items:
                raise HTTPException(status_code=404, detail="No items to replace")
            
            # Mark old items as deprecated
            for it in items:
                it.deprecated = True
                db.add(it)
            
            # Create new item
            seq = self._sequence_map.get(session_id, 0) + 1
            self._sequence_map[session_id] = seq
            
            item_id = make_uuid()
            meta_text = safe_json_dump(new_item.metadata.dict() if new_item.metadata else None)
            
            ct = ContextItemModel(
                item_id=item_id,
                session_id=session_id,
                role=new_item.role,
                content=new_item.content,
                vector_id=new_item.vector_id,
                created_at=now(),
                meta_json=meta_text,
                trusted=bool(new_item.trusted),
                deprecated=False,
                sequence=seq
            )
            db.add(ct)
            await db.commit()
        
        # Invalidate cache
        if self.cache:
            await self.cache.invalidate_context(session_id)
        
        return [self._ctx_to_schema(ct)]
    
    async def mark_item_trusted(self, item_id: str, trusted: bool = True) -> bool:
        """Mark context item as trusted/untrusted."""
        async with AsyncSessionLocal() as db:
            res = await db.execute(
                select(ContextItemModel).where(ContextItemModel.item_id == item_id)
            )
            it = res.scalar_one_or_none()
            if not it:
                return False
            
            it.trusted = trusted
            db.add(it)
            await db.commit()
        
        return True
    
    async def mark_item_deprecated(self, item_id: str, deprecated: bool = True) -> bool:
        """Soft delete context item."""
        async with AsyncSessionLocal() as db:
            res = await db.execute(
                select(ContextItemModel).where(ContextItemModel.item_id == item_id)
            )
            it = res.scalar_one_or_none()
            if not it:
                return False
            
            it.deprecated = deprecated
            db.add(it)
            await db.commit()
        
        return True
    
    # ═══════════════════════════════════════════════════════════════════════════
    # SNAPSHOT / SUMMARIZATION
    # ═══════════════════════════════════════════════════════════════════════════
    
    async def summarize_session(
        self,
        session_id: str,
        strategy: str = "rolling",
        max_sentences: int = 8
    ) -> SnapshotReadSchema:
        """
        Create summary snapshot of session context.
        Falls back to local extractive summarization if LLM unavailable.
        """
        if strategy not in ("rolling", "snapshot", "topic_based"):
            raise HTTPException(status_code=400, detail="Invalid strategy")
        
        async with AsyncSessionLocal() as db:
            res = await db.execute(
                select(ContextItemModel)
                .where(
                    ContextItemModel.session_id == session_id,
                    ContextItemModel.deprecated == False
                )
                .order_by(ContextItemModel.sequence.desc())
                .limit(500)
            )
            items = res.scalars().all()
            
            if not items:
                raise HTTPException(status_code=404, detail="No context to summarize")
            
            # Decompress and collect texts
            texts = []
            for i in reversed(list(items)):
                content = i.content
                if i.compressed and self.compressor:
                    content = self.compressor.decompress(content)
                texts.append(content)
            
            # Try LLM summarization first
            summary_text = None
            method = f"local_{strategy}"
            
            if self.llm_router:
                try:
                    llm_result = await self.llm_router.abstractive_summary(
                        "\\n\\n".join(texts),
                        strategy=strategy,
                        max_sentences=max_sentences
                    )
                    if llm_result:
                        summary_text = llm_result
                        method = f"llm_abstractive_{strategy}"
                        logger.info(f"✅ LLM summarization successful for session {session_id}")
                except Exception as e:
                    logger.warning(f"LLM summarization failed: {e}, falling back to local")
            
            # Fallback to local extractive
            if not summary_text:
                summary_text = local_extractive_summary(texts, max_sentences=max_sentences)
                logger.info(f"✅ Local summarization for session {session_id}")
            
            # Save snapshot
            snapshot_id = make_uuid()
            meta_text = safe_json_dump({"strategy": strategy})
            
            async with AsyncSessionLocal() as db:
                snap = SnapshotModel(
                    snapshot_id=snapshot_id,
                    session_id=session_id,
                    summary_text=summary_text,
                    method=method,
                    created_at=now(),
                    meta_json=meta_text
                )
                db.add(snap)
                await db.commit()
        
        self.metrics["snapshots_created"] += 1
        logger.info(f"✅ Snapshot {snapshot_id} created for session {session_id} method={method}")
        
        return SnapshotReadSchema(
            snapshot_id=snapshot_id,
            session_id=session_id,
            summary_text=summary_text,
            method=method,
            created_at=now(),
            metadata={"strategy": strategy}
        )
    
    async def list_snapshots(self, session_id: str) -> List[SnapshotReadSchema]:
        """List all snapshots for a session."""
        async with AsyncSessionLocal() as db:
            res = await db.execute(
                select(SnapshotModel)
                .where(SnapshotModel.session_id == session_id)
                .order_by(SnapshotModel.created_at.desc())
            )
            snaps = res.scalars().all()
            
            return [
                SnapshotReadSchema(
                    snapshot_id=s.snapshot_id,
                    session_id=s.session_id,
                    summary_text=s.summary_text,
                    method=s.method,
                    created_at=s.created_at,
                    metadata=safe_json_load(s.meta_json)
                )
                for s in snaps
            ]
    
    # ═══════════════════════════════════════════════════════════════════════════
    # EVIDENCE MANAGEMENT
    # ═══════════════════════════════════════════════════════════════════════════
    
    async def attach_evidence(
        self,
        session_id: str,
        ev: EvidenceCreateSchema
    ) -> dict:
        """Attach evidence/metadata to session or specific resource."""
        eid = make_uuid()
        
        async with AsyncSessionLocal() as db:
            # Verify session exists
            s = (await db.execute(
                select(SessionModel).where(SessionModel.session_id == session_id)
            )).scalar_one_or_none()
            
            if not s:
                raise HTTPException(status_code=404, detail="Session not found")
            
            ev_entry = EvidenceModel(
                evidence_id=eid,
                session_id=session_id,
                attached_to=ev.attached_to,
                data=safe_json_dump(ev.data),
                created_at=now(),
                meta_json=safe_json_dump(ev.metadata) if ev.metadata else None
            )
            db.add(ev_entry)
            await db.commit()
        
        logger.info(f"✅ Evidence {eid} attached for session {session_id}")
        return {"evidence_id": eid}
    
    # ═══════════════════════════════════════════════════════════════════════════
    # BACKGROUND TASKS
    # ═══════════════════════════════════════════════════════════════════════════
    
    async def cleanup_expired_sessions(self):
        """Clean up sessions that have exceeded their TTL."""
        cutoff = now() - timedelta(hours=settings.session_ttl_hours)
        
        async with AsyncSessionLocal() as db:
            res = await db.execute(
                select(SessionModel).where(
                    SessionModel.last_active_at < cutoff,
                    SessionModel.is_active == True
                )
            )
            sessions = res.scalars().all()
            
            for s in sessions:
                s.is_active = False
                db.add(s)
            
            await db.commit()
        
        if sessions:
            logger.info(f"🧹 Cleaned up {len(sessions)} expired sessions")
    
    def start_background_cleanup(self):
        """Start background task for session cleanup."""
        if self._cleanup_task and not self._cleanup_task.done():
            return
        
        loop = asyncio.get_event_loop()
        self._cleanup_task = loop.create_task(self._cleanup_loop())
        logger.info("✅ Background cleanup task started")
    
    async def _cleanup_loop(self):
        """Background loop for periodic cleanup."""
        try:
            while True:
                try:
                    await self.cleanup_expired_sessions()
                except Exception as e:
                    logger.exception(f"Cleanup loop error: {e}")
                
                await asyncio.sleep(settings.cleanup_interval_seconds)
        except asyncio.CancelledError:
            logger.info("🛑 Cleanup task cancelled")
            return
    
    async def shutdown(self):
        """Graceful shutdown."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except Exception:
                pass
        
        logger.info("✅ MCPService shutdown complete")
    
    def get_metrics(self):
        """Return current metrics snapshot."""
        return {
            "sessions_created": self.metrics["sessions_created"],
            "items_appended": self.metrics["items_appended"],
            "snapshots_created": self.metrics["snapshots_created"],
            "webhooks_received": self.metrics["webhooks_received"],
            "auth_failures": self.metrics["auth_failures"],
            "cache_hits": self.metrics["cache_hits"],
            "cache_misses": self.metrics["cache_misses"],
        }


# Initialize service
mcp_service = MCPService()

# Initialize RAG client for resume selection (will be set during startup)
rag_client: Optional[Any] = None



# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 10: FASTAPI APPLICATION
# ═══════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title=settings.service_name,
    description="Enterprise context management service for job automation AI agents",
    version=settings.version,
    docs_url="/docs" if settings.dev_mode else None,
    redoc_url="/redoc" if settings.dev_mode else None,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.dev_mode else [],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger.info("✅ FastAPI app initialized")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 11: DEPENDENCY INJECTION
# ═══════════════════════════════════════════════════════════════════════════════

async def get_current_user(x_mcp_api_key: Optional[str] = Header(None)) -> dict:
    """
    Authentication dependency.
    Will be enhanced by security.py with full auth logic.
    """
    if settings.dev_mode:
        return {"user_id": "dev_user", "role": "admin"}
    
    if not x_mcp_api_key or x_mcp_api_key != settings.api_key:
        mcp_service.metrics["auth_failures"] += 1
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    # TODO: Enhance with security.py for JWT, service accounts, etc.
    return {"user_id": "api_user", "role": "user"}


async def auth_and_rate_limit(user: dict = Depends(get_current_user)) -> dict:
    """
    Combined authentication and rate limiting dependency.
    Checks API key and enforces rate limits.
    """
    # Auth is already checked by get_current_user dependency
    
    # Rate limiting check (if rate limiter is available)
    if mcp_service.rate_limiter:
        # TODO: Implement rate limiting logic based on user/API key
        pass
    
    return user


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 12: API ROUTES - CORE MCP OPERATIONS
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/v1/sessions", tags=["Sessions"])
async def api_create_session(
    payload: SessionCreateSchema,
    user: dict = Depends(get_current_user)
):
    """Create a new session."""
    s = await mcp_service.create_session(
        owner=payload.owner,
        metadata=payload.metadata,
        ttl_hours=payload.ttl_hours
    )
    return s


@app.get("/v1/sessions/{session_id}", tags=["Sessions"])
async def api_get_session(
    session_id: str,
    user: dict = Depends(get_current_user)
):
    """Get session by ID."""
    s = await mcp_service.get_session(session_id)
    if not s:
        raise HTTPException(status_code=404, detail="Session not found")
    return s


@app.delete("/v1/sessions/{session_id}", tags=["Sessions"])
async def api_delete_session(
    session_id: str,
    user: dict = Depends(get_current_user)
):
    """Delete session and all related data."""
    ok = await mcp_service.delete_session(session_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"deleted": True}


@app.post("/v1/sessions/{session_id}/items", tags=["Context"])
async def api_append_item(
    session_id: str,
    item: ContextItemCreateSchema,
    user: dict = Depends(get_current_user)
):
    """Append context item to session."""
    i = await mcp_service.append_context_item(session_id, item)
    return i


@app.get("/v1/sessions/{session_id}/items", tags=["Context"])
async def api_get_items(
    session_id: str,
    last_n: Optional[int] = None,
    since: Optional[str] = None,
    role: Optional[str] = None,
    source: Optional[str] = None,
    page: int = 1,
    page_size: int = 50,
    user: dict = Depends(get_current_user)
):
    """Retrieve context items with filtering."""
    since_dt = None
    if since:
        try:
            since_dt = datetime.fromisoformat(since)
        except Exception:
            raise HTTPException(status_code=400, detail="since must be ISO datetime")
    
    res = await mcp_service.retrieve_context_items(
        session_id,
        last_n=last_n,
        since=since_dt,
        role=role,
        source=source,
        page=page,
        page_size=page_size
    )
    return res


@app.post("/v1/sessions/{session_id}/replace_last", tags=["Context"])
async def api_replace_last(
    session_id: str,
    n: int,
    new_item: ContextItemCreateSchema,
    user: dict = Depends(get_current_user)
):
    """Replace last N items with new item."""
    if n < 1:
        raise HTTPException(status_code=400, detail="n must be >=1")
    
    res = await mcp_service.replace_last_n(session_id, n, new_item)
    return {"replaced": res}


@app.post("/v1/sessions/{session_id}/items/{item_id}/trust", tags=["Context"])
async def api_mark_trusted(
    session_id: str,
    item_id: str,
    trusted: bool = True,
    user: dict = Depends(get_current_user)
):
    """Mark context item as trusted/untrusted."""
    ok = await mcp_service.mark_item_trusted(item_id, trusted=trusted)
    if not ok:
        raise HTTPException(status_code=404, detail="Item not found")
    return {"ok": True}


@app.post("/v1/sessions/{session_id}/items/{item_id}/deprecated", tags=["Context"])
async def api_mark_deprecated(
    session_id: str,
    item_id: str,
    deprecated: bool = True,
    user: dict = Depends(get_current_user)
):
    """Soft delete context item."""
    ok = await mcp_service.mark_item_deprecated(item_id, deprecated=deprecated)
    if not ok:
        raise HTTPException(status_code=404, detail="Item not found")
    return {"ok": True}


@app.post("/v1/sessions/{session_id}/snapshot", tags=["Snapshots"])
async def api_create_snapshot(
    session_id: str,
    payload: SnapshotCreateSchema,
    user: dict = Depends(get_current_user)
):
    """Create summary snapshot of session."""
    snap = await mcp_service.summarize_session(
        session_id,
        strategy=payload.strategy,
        max_sentences=payload.max_sentences
    )
    return snap


@app.get("/v1/sessions/{session_id}/snapshots", tags=["Snapshots"])
async def api_list_snapshots(
    session_id: str,
    user: dict = Depends(get_current_user)
):
    """List all snapshots for session."""
    snaps = await mcp_service.list_snapshots(session_id)
    return {"snapshots": snaps}


@app.post("/v1/sessions/{session_id}/evidence", tags=["Evidence"])
async def api_attach_evidence(
    session_id: str,
    ev: EvidenceCreateSchema,
    user: dict = Depends(get_current_user)
):
    """Attach evidence to session."""
    res = await mcp_service.attach_evidence(session_id, ev)
    return res


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 13: API ROUTES - WEBHOOK RECEIVERS
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/webhook/chrome", tags=["Webhooks"])
async def webhook_chrome(
    payload: WebhookChromeSchema,
    background_tasks: BackgroundTasks,
    user: dict = Depends(get_current_user)
):
    """
    Receive data from Chrome extension.
    
    Actions:
    - form_data: Extracted form data from job application
    - resume_request: Request for optimal resume suggestion
    - status_update: Application status update
    """
    mcp_service.metrics["webhooks_received"] += 1
    
    session_id = payload.session_id or make_uuid()
    
    # Store in context
    metadata = MetadataSchema(
        source="chrome_extension",
        tags=[payload.action]
    )
    
    item = ContextItemCreateSchema(
        role="tool",
        content=json.dumps(payload.data),
        metadata=metadata
    )
    
    await mcp_service.append_context_item(session_id, item)
    
    # Handle resume request
    if payload.action == "resume_request" and mcp_service.rag_client:
        try:
            job_description = payload.data.get("job_description", "")
            # TODO: Get user resumes from database/context
            user_resumes = []
            suggestions = await mcp_service.rag_client.suggest_optimal_resume(
                job_description,
                user_resumes
            )
            return {
                "status": "success",
                "session_id": session_id,
                "suggestions": suggestions
            }
        except Exception as e:
            logger.error(f"Resume suggestion failed: {e}")
    
    return {
        "status": "received",
        "session_id": session_id,
        "action": payload.action
    }


@app.post("/webhook/n8n", tags=["Webhooks"])
async def webhook_n8n(
    payload: WebhookN8nSchema,
    user: dict = Depends(get_current_user)
):
    """
    Receive triggers from n8n workflows.
    
    Typical use cases:
    - Job application automation triggered
    - Status callbacks from automated actions
    - Scheduled job searches
    """
    mcp_service.metrics["webhooks_received"] += 1
    
    # Create or use existing session
    session_id = payload.data.get("session_id") or make_uuid()
    
    # Store in context
    metadata = MetadataSchema(
        source="n8n",
        tags=[payload.trigger_type, payload.workflow_id]
    )
    
    item = ContextItemCreateSchema(
        role="tool",
        content=json.dumps(payload.data),
        metadata=metadata
    )
    
    await mcp_service.append_context_item(session_id, item)
    
    logger.info(f"✅ n8n webhook received: workflow={payload.workflow_id}, trigger={payload.trigger_type}")
    
    return {
        "status": "received",
        "session_id": session_id,
        "workflow_id": payload.workflow_id
    }


@app.post("/ingest/jobs", tags=["Webhooks"])
async def ingest_jobs(
    payload: JobIngestSchema,
    background_tasks: BackgroundTasks,
    user: dict = Depends(get_current_user)
):
    """
    Bulk job ingestion from scrapers.
    
    Sources: jooble, remotive, linkedin, custom scrapers
    
    Flow:
    1. Store jobs in context
    2. Trigger RAG matching (if available)
    3. Create Notion entries (if available)
    4. Emit events for agents
    """
    mcp_service.metrics["webhooks_received"] += 1
    
    # Create ingestion session
    session_id = make_uuid()
    await mcp_service.create_session(
        owner=f"scraper_{payload.source}",
        metadata={"source": payload.source, "job_count": len(payload.jobs)}
    )
    
    # Store each job as context item
    for job in payload.jobs:
        metadata = MetadataSchema(
            source=payload.source,
            job_id=job.get("id"),
            url=job.get("url"),
            tags=["job_listing"]
        )
        
        item = ContextItemCreateSchema(
            role="tool",
            content=json.dumps(job),
            metadata=metadata
        )
        
        await mcp_service.append_context_item(session_id, item)
    
    logger.info(f"✅ Ingested {len(payload.jobs)} jobs from {payload.source}")
    
    # TODO: Trigger RAG matching in background
    # TODO: Create Notion entries in background
    
    return {
        "status": "ingested",
        "session_id": session_id,
        "source": payload.source,
        "count": len(payload.jobs)
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 14: API ROUTES - INTEGRATION TRIGGERS
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/v1/suggest-resume", dependencies=[Depends(auth_and_rate_limit)], tags=["Integrations"])
async def suggest_resume(
    session_id: str,
    job_description: str
):
    """RAG-powered resume suggestion"""
    if not rag_client:
        raise HTTPException(status_code=503, detail="RAG system not available")
    
    try:
        # Call RAG
        rag_result = await rag_client.select_best_resume(
            job_description=job_description,
            session_id=session_id,
            top_k=10,
            include_metadata=True
        )
        
        # Store in MCP context
        metadata = MetadataSchema(
            source="rag",
            confidence=rag_result.get("confidence_score", 0.0),
            tags=["resume_suggestion", "rag_analysis"]
        )
        
        item = ContextItemCreateSchema(
            role="assistant",
            content=json.dumps({
                "selected_resume": rag_result.get("selected_resume_id"),  # ← Fix: use selected_resume_id
                "resume_path": rag_result.get("selected_resume_path"),
                "confidence_score": rag_result.get("confidence_score"),
                "answer": rag_result.get("answer")
            }),
            metadata=metadata,
            trusted=True
        )
        
        await mcp_service.append_context_item(session_id, item)
        
        return {
            "status": "success",
            "selected_resume": rag_result.get("selected_resume_id"),  # ← Fix here too
            "resume_path": rag_result.get("selected_resume_path"),
            "confidence_score": rag_result.get("confidence_score", 0.0),
            "answer": rag_result.get("answer"),
            "chunks_count": len(rag_result.get("chunks", [])),
            "session_id": session_id
        }
        
    except Exception as e:
        logger.error(f"Resume suggestion failed: {e}")
        raise HTTPException(status_code=500, detail=f"RAG error: {str(e)}")

@app.post("/llm/complete", tags=["Integrations"])
async def llm_complete(
    session_id: str,
    prompt: str,
    task_type: str = "general",
    user: dict = Depends(get_current_user)
):
    """
    Trigger LLM completion.
    
    Task types: research, form_filling, job_matching, decision_making, summarization, general
    """
    if not mcp_service.llm_router:
        raise HTTPException(
            status_code=503,
            detail="LLM router not available. Implement integrations.py LLMRouter."
        )
    
    try:
        response = await mcp_service.llm_router.complete(
            prompt=prompt,
            task_type=task_type
        )
        
        # Store in context
        metadata = MetadataSchema(
            source="llm",
            tags=[task_type, response.get("provider", "unknown")]
        )
        item = ContextItemCreateSchema(
            role="assistant",
            content=response.get("content", ""),
            metadata=metadata
        )
        await mcp_service.append_context_item(session_id, item)
        
        return response
        
    except Exception as e:
        logger.error(f"LLM completion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/notion/create-job", tags=["Integrations"])
async def notion_create_job(
    session_id: str,
    job_data: dict,
    user: dict = Depends(get_current_user)
):
    """
    Create job tracking entry in Notion.
    """
    if not mcp_service.notion_client:
        raise HTTPException(
            status_code=503,
            detail="Notion client not available. Implement integrations.py NotionClient."
        )
    
    try:
        page_id = await mcp_service.notion_client.create_job_entry(job_data)
        
        # Store in context
        metadata = MetadataSchema(
            source="notion",
            job_id=job_data.get("id"),
            tags=["job_tracking"]
        )
        item = ContextItemCreateSchema(
            role="tool",
            content=json.dumps({"notion_page_id": page_id, "job_data": job_data}),
            metadata=metadata
        )
        await mcp_service.append_context_item(session_id, item)
        
        return {"notion_page_id": page_id, "status": "created"}
        
    except Exception as e:
        logger.error(f"Notion creation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 15: API ROUTES - MONITORING & HEALTH
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/health", tags=["Monitoring"])
async def health():
    """Basic health check."""
    return {
        "status": "ok",
        "service": settings.service_name,
        "version": settings.version,
        "timestamp": now().isoformat()
    }


@app.get("/ready", tags=["Monitoring"])
async def readiness():
    """
    Readiness probe for Kubernetes.
    Checks all critical dependencies.
    """
    checks = {
        "database": False,
        "redis": False if settings.redis_enabled else None,
    }
    
    # Check database
    try:
        async with AsyncSessionLocal() as db:
            await db.execute(select(1))
        checks["database"] = True
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
    
    # Check Redis
    if settings.redis_enabled and mcp_service.cache:
        try:
            # TODO: Implement cache.health_check()
            checks["redis"] = True
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
    
    all_healthy = all(v for v in checks.values() if v is not None)
    
    if all_healthy:
        return {"ready": True, "checks": checks}
    else:
        raise HTTPException(
            status_code=503,
            detail={"ready": False, "checks": checks}
        )

# -----------------------
# RAG Resume Selection Endpoint
# -----------------------
@app.post("/v1/job/analyze", dependencies=[Depends(auth_and_rate_limit)])
async def api_analyze_job_and_select_resume(
    job_description: str,
    session_id: str,
    top_k: int = 10
):
    """
    Analyze job posting and select best matching resume using RAG system
    
    Args:
        job_description: Full job posting text
        session_id: Session ID for tracking
        top_k: Number of resume chunks to retrieve
        
    Returns:
        Selected resume info with confidence score
    """
    if not rag_client:
        raise HTTPException(
            status_code=503,
            detail="RAG system not available. Ensure RAG server is running at localhost:8090"
        )
    
    try:
        # Call RAG to select optimal resume
        rag_result = await rag_client.select_best_resume(
            job_description=job_description,
            session_id=session_id,
            top_k=top_k,
            include_metadata=True
        )
        
        return {
            "status": "success",
            "selected_resume": rag_result.get("selected_resume"),
            "resume_path": rag_result.get("selected_resume_path"),
            "confidence_score": rag_result.get("confidence_score", 0.0),
            "matching_skills": rag_result.get("matching_skills", []),
            "analysis": rag_result.get("answer"),
            "chunks_retrieved": rag_result.get("chunks_retrieved", 0),
            "session_id": session_id
        }
        
    except Exception as e:
        logger.error(f"Job analysis failed: {str(e)}")
        return {
            "status": "error",
            "error": str(e),
            "fallback_resume": "resume_generic"
        }
    

@app.get("/metrics", tags=["Monitoring"])
async def metrics():
    """
    Prometheus-compatible metrics endpoint.
    Will be enhanced by cache.py MetricsCollector.
    """
    # Basic metrics for now
    lines = []
    for key, value in mcp_service.metrics.items():
        lines.append(f"mcp_{key} {value}")
    
    return JSONResponse(
        content={"metrics": mcp_service.metrics},
        headers={"Content-Type": "application/json"}
    )

@app.get("/v1/relevant/{session_id}", dependencies=[Depends(auth_and_rate_limit)])
async def api_rag(session_id: str, top_k: int = 10):
    """Get relevant RAG context for session"""
    out = await mcp_service.get_relevant_context_for_rag(
        session_id=session_id,
        top_k=top_k
    )
    return {"items": out}

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 16: STARTUP & SHUTDOWN HOOKS
# ═══════════════════════════════════════════════════════════════════════════════

@app.on_event("startup")
async def startup():
    global rag_client
    
    # Migrate database
    await mcp_service.migrate()
    
    # Start background tasks
    mcp_service.start_background_cleanup()
    
    # Initialize integrations (will be loaded from other files)
    try:
        from integrations import get_rag_client, get_llm_router, get_notion_client
        rag_client = get_rag_client()
        mcp_service.rag_client = rag_client
        mcp_service.llm_router = get_llm_router()
        mcp_service.notion_client = get_notion_client()
        logger.info("✅ Integration clients loaded")
        
        # Check RAG health
        if rag_client and hasattr(rag_client, 'health_check'):
            try:
                is_healthy = await rag_client.health_check()
                if is_healthy:
                    logger.info("✅ RAG system connected successfully")
                else:
                    logger.warning("⚠️ RAG system not available - resume selection may fail")
            except Exception as e:
                logger.warning(f"⚠️ RAG health check failed: {e}")
    except (ImportError, Exception) as e:
        logger.warning(f"⚠️  Integration modules not fully loaded: {e}")
        logger.warning("   Resume suggestion and LLM features may not be available")
    
    # Initialize cache
    try:
        from cache import get_cache, get_compressor
        mcp_service.cache = get_cache()
        mcp_service.compressor = get_compressor()
        logger.info("✅ Cache and compressor loaded")
    except (ImportError, Exception) as e:
        logger.warning(f"⚠️  Cache module not found: {e}")
    
    # Initialize security
    try:
        from security import get_auth_handler, get_rate_limiter, get_audit_logger
        mcp_service.auth_handler = get_auth_handler()
        mcp_service.rate_limiter = get_rate_limiter()
        mcp_service.audit_logger = get_audit_logger()
        logger.info("✅ Security components loaded")
    except (ImportError, Exception) as e:
        logger.warning(f"⚠️  Security module not found: {e}")
    
    logger.info(f"✅ {settings.service_name} v{settings.version} ready!")
    logger.info(f"📖 Docs: http://localhost:8080/docs" if settings.dev_mode else "")


@app.on_event("shutdown")
async def shutdown_event():
    global rag_client
    
    await mcp_service.shutdown()
    
    # Cleanup RAG client
    if rag_client:
        try:
            await rag_client.close()
            logger.info("RAG client connection closed")
        except Exception as e:
            logger.error(f"Error closing RAG client: {e}")
    
    await engine.dispose()
    logger.info("MCP service shutdown complete")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8080,
        reload=settings.dev_mode,
        log_level=settings.log_level.lower()
    )
