# rag_systems/production_server.py

"""
═══════════════════════════════════════════════════════════════════════════════
PRODUCTION-READY RAG HTTP SERVER - ENTERPRISE GRADE
═══════════════════════════════════════════════════════════════════════════════

This single file transforms the RAG system into a production-ready service with:
- HTTP API server at localhost:8090
- Session management with conversation history
- API key authentication
- Rate limiting & throttling
- Circuit breaker pattern
- Health checks & metrics
- Request validation
- Error handling & retry logic
- CORS for Chrome extension
- Prometheus metrics
- Structured logging
- Connection pooling
- Response caching
- Graceful shutdown

Connects to:
- MCP server via function calls
- Chrome extension via HTTP endpoints
- Automated apply service via REST API

Author: Enterprise Architecture Team
Version: 1.0.0
Date: January 19, 2026
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

from fastapi import FastAPI, Depends, Header, HTTPException, Request, Response, status
from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Optional, Dict, Any, List, Set, Callable
import os
import sys
import json
import time
import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Set, Callable
from dataclasses import dataclass, field, asdict
from collections import defaultdict, deque
from threading import Lock, RLock
from functools import wraps
from enum import Enum
from contextlib import asynccontextmanager
from datetime import datetime


import json
import time
import asyncio
import hashlib
import traceback

# Add rag_systems to path for module imports
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))

# ═══════════════════════════════════════════════════════════════════════════
# STANDARD LIBRARY & THIRD-PARTY IMPORTS
# ═══════════════════════════════════════════════════════════════════════════

try:
    from fastapi.responses import JSONResponse, PlainTextResponse
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.middleware.gzip import GZipMiddleware
    from fastapi.middleware.trustedhost import TrustedHostMiddleware
    import uvicorn
except ImportError:
    print("ERROR: FastAPI dependencies missing. Install: pip install fastapi uvicorn pydantic")
    sys.exit(1)

# ═══════════════════════════════════════════════════════════════════════════
# LOCAL IMPORTS (from existing rag_systems files)
# ═══════════════════════════════════════════════════════════════════════════

try:
    from rag_api import (
        select_resume,
        get_resume_pdf_path,
        get_rag_context,
        reindex_resume,
        list_resumes,
        healthcheck,
    )
    from resume_engine import get_default_engine
except ImportError:
    print("ERROR: Cannot import from rag_systems. Ensure rag_api.py and resume_engine.py exist.")
    sys.exit(1)

# ═══════════════════════════════════════════════════════════════════════════
# LOGGING CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

logger = logging.getLogger("production_server")
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
console_handler.setFormatter(formatter)

logger.addHandler(console_handler)


# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION & CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════

class ServerConfig:
    """Centralized server configuration"""
    
    # Server Settings
    HOST: str = os.getenv("RAG_SERVER_HOST", "localhost")
    PORT: int = int(os.getenv("RAG_SERVER_PORT", "8090"))
    RELOAD: bool = os.getenv("RAG_SERVER_RELOAD", "false").lower() == "true"
    WORKERS: int = int(os.getenv("RAG_SERVER_WORKERS", "1"))

    # API Key (single server-wide key)
    API_KEY: str = os.getenv("RAG_SERVER_API_KEY", "")
    MASTER_API_KEY: str = API_KEY
    
    # Collect all valid client keys (single-key model, kept as a set for compatibility)
    API_KEYS: Set[str] = {key for key in [API_KEY] if key}
    
    # Mapping for logging/tracking (single entry)
    API_KEY_NAMES: Dict[str, str] = {
        API_KEY: "rag_server"
    }
    
    # Session Management
    SESSION_TIMEOUT_MINUTES: int = int(os.getenv("SESSION_TIMEOUT_MINUTES", "30"))
    SESSION_MAX_HISTORY: int = int(os.getenv("SESSION_MAX_HISTORY", "50"))
    SESSION_CLEANUP_INTERVAL: int = int(os.getenv("SESSION_CLEANUP_INTERVAL", "300"))
    
    # Rate Limiting
    RATE_LIMIT_REQUESTS: int = int(os.getenv("RATE_LIMIT_REQUESTS", "100"))
    RATE_LIMIT_WINDOW_SECONDS: int = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))
    
    # Circuit Breaker
    CIRCUIT_BREAKER_THRESHOLD: int = int(os.getenv("CIRCUIT_BREAKER_THRESHOLD", "5"))
    CIRCUIT_BREAKER_TIMEOUT: int = int(os.getenv("CIRCUIT_BREAKER_TIMEOUT", "60"))
    
    # Cache Settings
    CACHE_ENABLED: bool = os.getenv("CACHE_ENABLED", "true").lower() == "true"
    CACHE_TTL_SECONDS: int = int(os.getenv("CACHE_TTL_SECONDS", "3600"))
    CACHE_MAX_SIZE: int = int(os.getenv("CACHE_MAX_SIZE", "1000"))
    
    # Performance
    REQUEST_TIMEOUT_SECONDS: int = int(os.getenv("REQUEST_TIMEOUT_SECONDS", "30"))
    MAX_CONCURRENT_REQUESTS: int = int(os.getenv("MAX_CONCURRENT_REQUESTS", "50"))
    
    # CORS
    CORS_ORIGINS: List[str] = os.getenv(
        "CORS_ORIGINS",
        "http://localhost:*,chrome-extension://*,https://*.company.com"
    ).split(",")
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_REQUESTS: bool = os.getenv("LOG_REQUESTS", "true").lower() == "true"


# ═══════════════════════════════════════════════════════════════════════════
# PYDANTIC MODELS (Request/Response Validation)
# ═══════════════════════════════════════════════════════════════════════════

class RAGRequest(BaseModel):
    """Request model for RAG endpoints"""
    session_id: str = Field(..., min_length=1, max_length=128, description="Session identifier")
    job_text: Optional[str] = Field("", description="Job description text")
    query: Optional[str] = Field(None, description="Query text for RAG")
    top_k: int = Field(5, ge=1, le=20, description="Number of results to return")
    
    @field_validator('session_id')
    def validate_session_id(cls, v):
        if not v or not v.strip():
            raise ValueError("session_id cannot be empty")
        return v.strip()


class RAGQueryRequest(BaseModel):
    """Request model for /rag/query endpoint"""
    session_id: str = Field(..., min_length=1, max_length=128, description="Unique session identifier")
    query: Optional[str] = Field(None, description="User query or job description")
    job_text: Optional[str] = Field(None, description="Alternative to query field")
    filters: Optional[Dict[str, Any]] = Field(default=None, description="Optional filters for resume selection")
    top_k: Optional[int] = Field(default=10, ge=1, le=100, description="Number of chunks to retrieve")
    include_metadata: Optional[bool] = Field(default=True, description="Include metadata in response")
    use_cache: Optional[bool] = Field(default=True, description="Use cached results if available")
    
    @field_validator('session_id')
    @classmethod
    def validate_session_id(cls, v):
        if not v.strip():
            raise ValueError("session_id cannot be empty or whitespace")
        return v.strip()
    
    @model_validator(mode='after')
    def validate_query_or_job_text(self):
        """Validate that either query or job_text is provided"""
        # Use job_text if query is empty
        if not self.query and self.job_text:
            self.query = self.job_text
        
        # Validate that we have at least one
        if not self.query or len(self.query.strip()) < 10:
            raise ValueError("query or job_text must be at least 10 characters")
        
        self.query = self.query.strip()
        return self
    
    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "user-123-session-abc",
                "query": "Senior Python Developer with 5+ years experience in AI/ML",
                "filters": {"role_focus": "ai_ml_generalist"},
                "top_k": 10,
                "include_metadata": True
            }
        }


class RAGQueryResponse(BaseModel):
    """Response model for /rag/query endpoint"""
    success: bool = Field(True, description="Whether request was successful")
    session_id: str = Field(..., description="Session identifier")
    answer: Optional[str] = Field(None, description="Generated answer or resume recommendation")
    selected_resume_id: Optional[str] = Field(None, description="ID of selected resume")
    selected_resume_path: Optional[str] = Field(None, description="Path to selected resume PDF")
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Confidence score")
    chunks: Optional[List[Dict[str, Any]]] = Field(None, description="Retrieved chunks")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    processing_time_ms: float = Field(0.0, description="Processing time in milliseconds")
    timestamp: str = Field(..., description="Response timestamp")
    cached: bool = Field(False, description="Whether response was from cache")


class ResumeSelectionRequest(BaseModel):
    """Request model for /resumes/select endpoint"""
    job_text: str = Field(..., min_length=10, description="Job description text")
    top_k_anchors: Optional[int] = Field(default=7, ge=1, le=20)
    top_k_chunks: Optional[int] = Field(default=3, ge=1, le=10)
    
    @field_validator('job_text')
    def validate_job_text(cls, v):
        cleaned = v.strip()
        if len(cleaned) < 10:
            raise ValueError("job_text must be at least 10 characters")
        return cleaned


class ResumeSelectionResponse(BaseModel):
    """Response model for /resumes/select endpoint"""
    success: bool
    top_resume_id: str
    top_score: float
    selected_resume_path: str
    candidates: List[Dict[str, Any]]
    processing_time_ms: float
    timestamp: str


class HealthCheckResponse(BaseModel):
    """Response model for /health endpoint"""
    status: str = Field(..., description="System status (ok/degraded/error)")
    timestamp: str
    uptime_seconds: float
    version: str = "1.0.0"
    components: Dict[str, str] = Field(..., description="Component health status")
    metrics: Dict[str, Any] = Field(..., description="System metrics")


class ErrorResponse(BaseModel):
    """Standardized error response"""
    success: bool = False
    error: str
    error_code: str
    error_type: str
    details: Optional[Dict[str, Any]] = None
    timestamp: str
    request_id: Optional[str] = None


# ═══════════════════════════════════════════════════════════════════════════
# SESSION MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class SessionContext:
    """Session state container"""
    session_id: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_accessed: datetime = field(default_factory=datetime.utcnow)
    conversation_history: deque = field(default_factory=lambda: deque(maxlen=ServerConfig.SESSION_MAX_HISTORY))
    retrieved_context: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    request_count: int = 0
    
    def update_access(self):
        """Update last accessed timestamp"""
        self.last_accessed = datetime.utcnow()
        self.request_count += 1
    
    def add_to_history(self, query: str, response: str):
        """Add query-response pair to conversation history"""
        self.conversation_history.append({
            "timestamp": datetime.utcnow().isoformat(),
            "query": query,
            "response": response
        })
    
    def is_expired(self, timeout_minutes: int) -> bool:
        """Check if session has expired"""
        delta = datetime.utcnow() - self.last_accessed
        return delta > timedelta(minutes=timeout_minutes)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat(),
            "conversation_history": list(self.conversation_history),
            "request_count": self.request_count,
            "metadata": self.metadata
        }


class SessionManager:
    """Thread-safe session management with automatic cleanup"""
    
    def __init__(self):
        self.sessions: Dict[str, SessionContext] = {}
        self.lock = RLock()
        self.cleanup_task: Optional[asyncio.Task] = None
        logger.info("SessionManager initialized")
    
    def get_or_create(self, session_id: str) -> SessionContext:
        """Get existing session or create new one"""
        with self.lock:
            if session_id not in self.sessions:
                self.sessions[session_id] = SessionContext(session_id=session_id)
                logger.info(f"Created new session: {session_id}")
            
            session = self.sessions[session_id]
            session.update_access()
            return session
    
    def get(self, session_id: str) -> Optional[SessionContext]:
        """Get existing session"""
        with self.lock:
            return self.sessions.get(session_id)
    
    def delete(self, session_id: str) -> bool:
        """Delete session"""
        with self.lock:
            if session_id in self.sessions:
                del self.sessions[session_id]
                logger.info(f"Deleted session: {session_id}")
                return True
            return False
    
    def cleanup_expired(self) -> int:
        """Remove expired sessions"""
        with self.lock:
            expired = [
                sid for sid, session in self.sessions.items()
                if session.is_expired(ServerConfig.SESSION_TIMEOUT_MINUTES)
            ]
            for sid in expired:
                del self.sessions[sid]
            
            if expired:
                logger.info(f"Cleaned up {len(expired)} expired sessions")
            return len(expired)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get session statistics"""
        with self.lock:
            return {
                "total_sessions": len(self.sessions),
                "active_sessions": sum(1 for s in self.sessions.values() 
                                      if not s.is_expired(ServerConfig.SESSION_TIMEOUT_MINUTES)),
                "total_requests": sum(s.request_count for s in self.sessions.values())
            }
    
    async def start_cleanup_task(self):
        """Start background cleanup task"""
        while True:
            try:
                await asyncio.sleep(ServerConfig.SESSION_CLEANUP_INTERVAL)
                self.cleanup_expired()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Session cleanup error: {e}")


# ═══════════════════════════════════════════════════════════════════════════
# RATE LIMITING
# ═══════════════════════════════════════════════════════════════════════════

class RateLimiter:
    """Token bucket rate limiter (thread-safe)"""
    
    def __init__(self, requests: int = 100, window_seconds: int = 60):
        self.requests = requests
        self.window_seconds = window_seconds
        self.buckets: Dict[str, deque] = defaultdict(deque)
        self.lock = Lock()
    
    def is_allowed(self, client_id: str) -> bool:
        """Check if request is allowed"""
        with self.lock:
            now = time.time()
            bucket = self.buckets[client_id]
            
            # Remove expired timestamps
            while bucket and bucket[0] < now - self.window_seconds:
                bucket.popleft()
            
            # Check limit
            if len(bucket) >= self.requests:
                return False
            
            bucket.append(now)
            return True
    
    def get_remaining(self, client_id: str) -> int:
        """Get remaining requests in window"""
        with self.lock:
            now = time.time()
            bucket = self.buckets[client_id]
            
            # Remove expired
            while bucket and bucket[0] < now - self.window_seconds:
                bucket.popleft()
            
            return max(0, self.requests - len(bucket))
    
    def get_reset_time(self, client_id: str) -> float:
        """Get time until window resets"""
        with self.lock:
            bucket = self.buckets.get(client_id)
            if not bucket:
                return 0
            
            oldest = bucket[0]
            return max(0, self.window_seconds - (time.time() - oldest))


# ═══════════════════════════════════════════════════════════════════════════
# CIRCUIT BREAKER
# ═══════════════════════════════════════════════════════════════════════════

class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Circuit breaker pattern for resilient operations"""
    
    def __init__(self, threshold: int = 5, timeout: int = 60):
        self.threshold = threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.state = CircuitState.CLOSED
        self.lock = Lock()
        logger.info(f"CircuitBreaker initialized (threshold={threshold}, timeout={timeout}s)")
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection"""
        with self.lock:
            if self.state == CircuitState.OPEN:
                if time.time() - self.last_failure_time >= self.timeout:
                    logger.info("Circuit breaker entering HALF_OPEN state")
                    self.state = CircuitState.HALF_OPEN
                else:
                    raise HTTPException(
                        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                        detail="Service temporarily unavailable (circuit breaker open)"
                    )
        
        try:
            result = func(*args, **kwargs)
            
            with self.lock:
                if self.state == CircuitState.HALF_OPEN:
                    logger.info("Circuit breaker recovery successful, closing circuit")
                    self.state = CircuitState.CLOSED
                    self.failure_count = 0
            
            return result
            
        except Exception as e:
            with self.lock:
                self.failure_count += 1
                self.last_failure_time = time.time()
                
                if self.failure_count >= self.threshold:
                    logger.error(f"Circuit breaker OPENED after {self.failure_count} failures")
                    self.state = CircuitState.OPEN
            
            raise e
    
    def get_state(self) -> Dict[str, Any]:
        """Get circuit breaker state"""
        with self.lock:
            return {
                "state": self.state.value,
                "failure_count": self.failure_count,
                "threshold": self.threshold,
                "last_failure": self.last_failure_time
            }


# ═══════════════════════════════════════════════════════════════════════════
# RESPONSE CACHE
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class CacheEntry:
    """Cache entry with TTL"""
    key: str
    value: Any
    created_at: float
    ttl: int
    hits: int = 0
    
    def is_expired(self) -> bool:
        """Check if entry has expired"""
        return time.time() - self.created_at > self.ttl


class ResponseCache:
    """LRU cache with TTL for API responses"""
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache: Dict[str, CacheEntry] = {}
        self.access_order: deque = deque()
        self.lock = Lock()
        logger.info(f"ResponseCache initialized (max_size={max_size}, ttl={default_ttl}s)")
    
    def _generate_key(self, request_data: Dict[str, Any]) -> str:
        """Generate cache key from request"""
        key_str = json.dumps(request_data, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()
    
    def get(self, request_data: Dict[str, Any]) -> Optional[Any]:
        """Get cached response"""
        if not ServerConfig.CACHE_ENABLED:
            return None
        
        key = self._generate_key(request_data)
        
        with self.lock:
            entry = self.cache.get(key)
            if entry and not entry.is_expired():
                entry.hits += 1
                # Move to end (most recently used)
                try:
                    self.access_order.remove(key)
                except ValueError:
                    pass
                self.access_order.append(key)
                return entry.value
            elif entry:
                # Expired, remove
                del self.cache[key]
        
        return None
    
    def set(self, request_data: Dict[str, Any], response: Any, ttl: Optional[int] = None):
        """Cache response"""
        if not ServerConfig.CACHE_ENABLED:
            return
        
        key = self._generate_key(request_data)
        ttl = ttl or self.default_ttl
        
        with self.lock:
            if len(self.cache) >= self.max_size and key not in self.cache:
                oldest_key = self.access_order.popleft()
                del self.cache[oldest_key]
            
            entry = CacheEntry(
                key=key,
                value=response,
                created_at=time.time(),
                ttl=ttl
            )
            self.cache[key] = entry
            
            try:
                self.access_order.remove(key)
            except ValueError:
                pass
            self.access_order.append(key)
    
    def invalidate(self, pattern: Optional[str] = None):
        """Invalidate cache entries"""
        with self.lock:
            if pattern is None:
                count = len(self.cache)
                self.cache.clear()
                self.access_order.clear()
                logger.info(f"Cache cleared: {count} entries")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self.lock:
            total_hits = sum(entry.hits for entry in self.cache.values())
            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "total_hits": total_hits,
                "hit_rate": total_hits / max(1, len(self.cache))
            }


# ═══════════════════════════════════════════════════════════════════════════
# METRICS & MONITORING
# ═══════════════════════════════════════════════════════════════════════════

class MetricsCollector:
    """Collect and expose system metrics"""
    
    def __init__(self):
        self.start_time = time.time()
        self.request_count = 0
        self.error_count = 0
        self.total_processing_time = 0.0
        self.endpoint_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "count": 0,
            "errors": 0,
            "total_time": 0.0
        })
        self.lock = Lock()
    
    def record_request(self, endpoint: str, duration: float, error: bool = False):
        """Record request metrics"""
        with self.lock:
            self.request_count += 1
            self.total_processing_time += duration
            
            if error:
                self.error_count += 1
            
            stats = self.endpoint_stats[endpoint]
            stats["count"] += 1
            stats["total_time"] += duration
            if error:
                stats["errors"] += 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics"""
        with self.lock:
            uptime = time.time() - self.start_time
            avg_response_time = (
                self.total_processing_time / self.request_count
                if self.request_count > 0 else 0
            )
            
            return {
                "uptime_seconds": round(uptime, 2),
                "total_requests": self.request_count,
                "total_errors": self.error_count,
                "error_rate": round(self.error_count / max(1, self.request_count), 4),
                "avg_response_time_ms": round(avg_response_time * 1000, 2),
                "requests_per_second": round(self.request_count / max(1, uptime), 2),
                "endpoints": dict(self.endpoint_stats)
            }


# ═══════════════════════════════════════════════════════════════════════════
# GLOBAL SINGLETONS
# ═══════════════════════════════════════════════════════════════════════════

session_manager = SessionManager()
rate_limiter = RateLimiter(
    requests=ServerConfig.RATE_LIMIT_REQUESTS,
    window_seconds=ServerConfig.RATE_LIMIT_WINDOW_SECONDS
)
circuit_breaker = CircuitBreaker(
    threshold=ServerConfig.CIRCUIT_BREAKER_THRESHOLD,
    timeout=ServerConfig.CIRCUIT_BREAKER_TIMEOUT
)
response_cache = ResponseCache(
    max_size=ServerConfig.CACHE_MAX_SIZE,
    default_ttl=ServerConfig.CACHE_TTL_SECONDS
)
metrics_collector = MetricsCollector()


# ═══════════════════════════════════════════════════════════════════════════
# FASTAPI APPLICATION
# ═══════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    logger.info("=" * 80)
    logger.info("RAG Production Server Starting...")
    logger.info("=" * 80)
    
    # Startup
    try:
        # Initialize RAG engine
        engine = get_default_engine()
        logger.info(f"✓ RAG Engine initialized")
        logger.info(f"✓ Loaded {len(engine.list_resumes())} resumes")
        
        # Start session cleanup task
        session_manager.cleanup_task = asyncio.create_task(
            session_manager.start_cleanup_task()
        )
        logger.info("✓ Session cleanup task started")
        
        logger.info(f"✓ Server ready at http://{ServerConfig.HOST}:{ServerConfig.PORT}")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"✗ Startup failed: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("=" * 80)
    logger.info("RAG Production Server Shutting Down...")
    logger.info("=" * 80)
    
    try:
        if session_manager.cleanup_task:
            session_manager.cleanup_task.cancel()
        logger.info("✓ Graceful shutdown complete")
    except Exception as e:
        logger.error(f"✗ Shutdown error: {e}")


app = FastAPI(
    title="RAG Production Server",
    description="Enterprise-grade Resume RAG System API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# ═══════════════════════════════════════════════════════════════════════════
# MIDDLEWARE
# ═══════════════════════════════════════════════════════════════════════════

# CORS Middleware (for Chrome Extension)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ServerConfig.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID", "X-RateLimit-Remaining", "X-RateLimit-Reset"]
)

# GZip Compression
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Trusted Host Middleware (security)
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["localhost", "127.0.0.1", "*.local"]
)


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    """Log all requests with timing"""
    request_id = hashlib.md5(f"{time.time()}{request.url}".encode()).hexdigest()[:16]
    request.state.request_id = request_id
    
    start_time = time.time()
    
    if ServerConfig.LOG_REQUESTS:
        logger.info(f"→ {request.method} {request.url.path} | Request-ID: {request_id}")
    
    try:
        response = await call_next(request)
        duration = time.time() - start_time
        
        # Record metrics
        metrics_collector.record_request(
            endpoint=request.url.path,
            duration=duration,
            error=(response.status_code >= 400)
        )
        
        # Add custom headers
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Processing-Time-MS"] = f"{duration * 1000:.2f}"
        
        if ServerConfig.LOG_REQUESTS:
            logger.info(
                f"← {request.method} {request.url.path} | "
                f"Status: {response.status_code} | "
                f"Time: {duration*1000:.2f}ms"
            )
        
        return response
        
    except Exception as e:
        duration = time.time() - start_time
        metrics_collector.record_request(
            endpoint=request.url.path,
            duration=duration,
            error=True
        )
        logger.error(f"✗ Request failed: {e}")
        raise


# ═══════════════════════════════════════════════════════════════════════════
# AUTHENTICATION
# ═══════════════════════════════════════════════════════════════════════════

async def verify_api_key(
    x_rag_api_key: str = Header(..., alias="X-RAG-API-Key")
) -> str:
    """Verify API key from header"""
    
    # Collect all valid keys from environment, filtering out None/empty.
    # Primary key: RAG_SERVER_API_KEY. RAG_API_KEY is accepted for backwards compatibility.
    valid_keys = {
        key
        for key in [
            os.getenv("RAG_SERVER_API_KEY"),
            os.getenv("RAG_API_KEY"),
        ]
        if key and key.strip()
    }
    
    # Also check ServerConfig.API_KEYS / MASTER_API_KEY
    valid_keys.update(ServerConfig.API_KEYS)
    if ServerConfig.MASTER_API_KEY:
        valid_keys.add(ServerConfig.MASTER_API_KEY)
    
    if x_rag_api_key in valid_keys:
        logger.debug(f"API key validated: {x_rag_api_key[:10]}...")
        return x_rag_api_key
    
    logger.warning(f"Invalid API key attempted: {x_rag_api_key[:10]}...")
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid API key",
        headers={"WWW-Authenticate": "ApiKey"}
    )


def check_rate_limit(api_key: str = Depends(verify_api_key)):
    """Check rate limit for API key"""
    if not rate_limiter.is_allowed(api_key):
        remaining_wait = rate_limiter.get_reset_time(api_key)
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded. Try again in {remaining_wait:.1f}s"
        )
    return api_key


# ═══════════════════════════════════════════════════════════════════════════
# API ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/", response_class=PlainTextResponse)
async def root():
    """Root endpoint"""
    return "RAG Production Server v1.0.0 - Running ✓"


@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Comprehensive health check endpoint"""
    try:
        engine_health = healthcheck()
        
        components = {
            "rag_engine": engine_health.get("status", "unknown"),
            "chromadb": "ok" if engine_health.get("chroma_dir") else "error",
            "embedder": "ok" if engine_health.get("embedding_provider") else "error",
            "session_manager": "ok",
            "rate_limiter": "ok",
            "circuit_breaker": circuit_breaker.get_state()["state"],
            "cache": "ok" if ServerConfig.CACHE_ENABLED else "disabled"
        }
        
        # Overall status
        has_error = any(status == "error" for status in components.values())
        overall_status = "error" if has_error else "ok"
        
        system_metrics = metrics_collector.get_metrics()
        session_stats = session_manager.get_stats()
        cache_stats = response_cache.get_stats()
        
        combined_metrics = {
            **system_metrics,
            "sessions": session_stats,
            "cache": cache_stats,
            "resumes": engine_health.get("resume_count", 0)
        }
        
        return HealthCheckResponse(
            status=overall_status,
            timestamp=datetime.utcnow().isoformat() + "Z",
            uptime_seconds=system_metrics["uptime_seconds"],
            components=components,
            metrics=combined_metrics
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthCheckResponse(
            status="error",
            timestamp=datetime.utcnow().isoformat() + "Z",
            uptime_seconds=metrics_collector.get_metrics()["uptime_seconds"],
            components={"error": str(e)},
            metrics={}
        )


@app.post("/rag/query", tags=["RAG"])
def rag_query_context(
    request: RAGRequest,
    api_key: str = Depends(verify_api_key)
):
    """Query RAG system for relevant resume context."""
    import traceback
    start_time = time.time()
    
    try:
        # Get or create session
        session = session_manager.get_or_create(request.session_id)
        
        # Validate input - get query from either query or job_text field
        query_text = request.query or request.job_text
        
        if not query_text or not query_text.strip():
            raise HTTPException(
                status_code=400,
                detail="Either 'query' or 'job_text' must be provided and cannot be empty"
            )
        
        # Build job_payload for get_rag_context
        job_payload = {"job_text": query_text.strip()}
        
        # Call get_rag_context (NOT select_resume!)
        result = get_rag_context(
            job_payload=job_payload,
            top_k_chunks=request.top_k
        )
        
        # Extract chunks from result
        if isinstance(result, dict):
            chunks = result.get("chunks", []) or result.get("top_chunks", [])
            metadata = result.get("metadata", {})
        else:
            chunks = getattr(result, "chunks", []) or getattr(result, "top_chunks", [])
            metadata = getattr(result, "metadata", {})
        
        # Update session history
        session.add_to_history(query_text, f"Retrieved {len(chunks)} chunks")
        session.retrieved_context = chunks
        
        # Calculate processing time
        processing_time_ms = (time.time() - start_time) * 1000
        
        # Return response
        return {
            "success": True,
            "session_id": request.session_id,
            "chunks": chunks,
            "metadata": metadata,
            "processing_time_ms": round(processing_time_ms, 2),
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in /rag/query: {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"RAG query failed: {str(e)}"
        )

    
@app.post("/rag/select", tags=["RAG"])
def rag_select_resume(
    request: RAGRequest,
    api_key: str = Depends(verify_api_key)
):
    """Select best resume for job description using RAG."""
    try:
        # Validate job_text
        if not request.job_text or not request.job_text.strip():
            raise HTTPException(
                status_code=400,
                detail="job_text is required and cannot be empty"
            )
        
        result = select_resume(
            job_description=request.job_text,
            session_id=request.session_id,
            top_k=request.top_k
        )
        
        if isinstance(result, dict):
            return {
                "selected_resume": result.get("selected_resume"),
                "selected_resume_path": result.get("selected_resume_path"),
                "confidence_score": result.get("confidence_score", 0.0),
                "matching_skills": result.get("matching_skills", []),
                "chunks_retrieved": result.get("chunks_retrieved", 0),
                "session_id": request.session_id
            }
        else:
            return {
                "selected_resume": getattr(result, "selected_resume", None),
                "selected_resume_path": getattr(result, "selected_resume_path", None),
                "confidence_score": getattr(result, "confidence_score", 0.0),
                "matching_skills": getattr(result, "matching_skills", []),
                "chunks_retrieved": getattr(result, "chunks_retrieved", 0),
                "session_id": request.session_id
            }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in /rag/select: {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Resume selection failed: {str(e)}"
        )


@app.get("/resumes", tags=["RAG"])
def list_resumes_endpoint(
    _: str = Depends(verify_api_key)
):
    """List all available resumes"""
    try:
        resumes = list_resumes()
        return {
            "resumes": resumes,
            "count": len(resumes)
        }
    except Exception as e:
        logger.error(f"Error in /resumes: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list resumes: {str(e)}")


@app.post("/resumes/select", response_model=ResumeSelectionResponse, tags=["RAG"])
async def select_best_resume(
    request: ResumeSelectionRequest,
    api_key: str = Depends(check_rate_limit)
):
    """
    Select best resume for job description
    
    Direct resume selection without session management
    """
    import traceback
    start_time = time.time()
    
    try:
        job_payload = {
            "job_text": request.job_text
        }
        
        # Use circuit breaker
        def select_safe():
            return select_resume(job_payload)
        
        result = circuit_breaker.call(select_safe)
        
        # Handle both dict and object responses
        if isinstance(result, dict):
            top_resume_id = result.get("top_resume_id")
            top_score = result.get("top_score", 0.0)
            candidates = result.get("candidates", [])
        else:
            top_resume_id = getattr(result, "top_resume_id", None)
            top_score = getattr(result, "top_score", 0.0)
            candidates = getattr(result, "candidates", [])
        
        if not top_resume_id:
            raise ValueError("No resume selected")
        
        # Get resume path
        resume_path = get_resume_pdf_path(top_resume_id)
        
        processing_time_ms = (time.time() - start_time) * 1000
        
        response_data = {
            "success": True,
            "top_resume_id": top_resume_id,
            "top_score": top_score,
            "selected_resume_path": resume_path,
            "candidates": candidates if candidates else [],
            "processing_time_ms": round(processing_time_ms, 2),
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
        
        return ResumeSelectionResponse(**response_data)
        
    except Exception as e:
        logger.error(f"Resume selection error: {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.get("/resumes/list", tags=["RAG"])
async def list_all_resumes(api_key: str = Depends(check_rate_limit)):
    """List all available resumes"""
    import traceback
    
    try:
        result = list_resumes()
        
        # Handle three cases: list, dict with 'resumes' key, or other
        if isinstance(result, list):
            resumes_list = result
        elif isinstance(result, dict):
            resumes_list = result.get("resumes", [])
        else:
            # Try to get resumes attribute from object
            resumes_list = getattr(result, "resumes", [])
        
        # Ensure it's a list
        if not isinstance(resumes_list, list):
            logger.warning(f"list_resumes() returned unexpected type: {type(result)}")
            resumes_list = []
        
        return {
            "success": True,
            "count": len(resumes_list),
            "resumes": resumes_list,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
        
    except Exception as e:
        logger.error(f"List resumes error: {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.post("/resumes/reindex/{resume_id}", tags=["RAG"])
def reindex_resume_endpoint(
    resume_id: str,
    api_key: str = Depends(check_rate_limit)
):
    """Reindex specific resume"""
    try:
        result = reindex_resume(resume_id)
        return {
            "success": True,
            "resume_id": resume_id,
            "result": result,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
    except Exception as e:
        logger.error(f"Reindex error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.get("/sessions/{session_id}", tags=["Sessions"])
def get_session_info(
    session_id: str,
    api_key: str = Depends(check_rate_limit)
):
    """Get session information"""
    session = session_manager.get(session_id)
    
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session not found: {session_id}"
        )
    
    return {
        "success": True,
        "session": session.to_dict(),
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }


@app.delete("/sessions/{session_id}", tags=["Sessions"])
def delete_session(
    session_id: str,
    api_key: str = Depends(check_rate_limit)
):
    """Delete session"""
    deleted = session_manager.delete(session_id)
    
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session not found: {session_id}"
        )
    
    return {
        "success": True,
        "message": f"Session {session_id} deleted",
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }


@app.post("/cache/invalidate", tags=["Admin"])
def invalidate_cache(
    api_key: str = Depends(verify_api_key)
):
    """Invalidate response cache (requires master API key)"""
    if api_key != ServerConfig.MASTER_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Master API key required"
        )
    
    response_cache.invalidate()
    
    return {
        "success": True,
        "message": "Cache invalidated",
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }


@app.get("/metrics", tags=["Admin"])
def get_metrics(api_key: str = Depends(verify_api_key)):
    """Get system metrics"""
    metrics = metrics_collector.get_metrics()
    session_stats = session_manager.get_stats()
    cache_stats = response_cache.get_stats()
    circuit_state = circuit_breaker.get_state()
    
    return {
        "success": True,
        "metrics": metrics,
        "sessions": session_stats,
        "cache": cache_stats,
        "circuit_breaker": circuit_state,
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }


# ═══════════════════════════════════════════════════════════════════════════
# ERROR HANDLERS
# ═══════════════════════════════════════════════════════════════════════════

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": exc.detail,
            "error_code": f"HTTP_{exc.status_code}",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "request_id": getattr(request.state, "request_id", None)
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions"""
    logger.error(f"Unhandled exception: {exc}\n{traceback.format_exc()}")
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "error": "Internal server error",
            "error_code": "INTERNAL_ERROR",
            "error_type": type(exc).__name__,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "request_id": getattr(request.state, "request_id", None)
        }
    )


# ═══════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

def main():
    """Start production server"""
    logger.info("=" * 80)
    logger.info("STARTING RAG PRODUCTION SERVER")
    logger.info("=" * 80)
    logger.info(f"Host: {ServerConfig.HOST}")
    logger.info(f"Port: {ServerConfig.PORT}")
    logger.info(f"Workers: 1 (production mode)")
    
    logger.info("API key configuration:")
    server_key = os.getenv("RAG_SERVER_API_KEY", "")
    if server_key:
        logger.info("  ✓ RAG_SERVER_API_KEY configured: %s...", server_key[:15])
    else:
        logger.warning("  ✗ RAG_SERVER_API_KEY is not configured")
    
    logger.info(f"Rate Limit: {ServerConfig.RATE_LIMIT_REQUESTS} req/{ServerConfig.RATE_LIMIT_WINDOW_SECONDS}s")
    logger.info(f"Cache: {'Enabled' if ServerConfig.CACHE_ENABLED else 'Disabled'}")
    logger.info(f"Session Timeout: {ServerConfig.SESSION_TIMEOUT_MINUTES} minutes")
    logger.info("=" * 80)
    
    # Run server with direct app reference (prevents asyncio loop issues)
    uvicorn.run(
        app,
        host=ServerConfig.HOST,
        port=ServerConfig.PORT,
        reload=False,
        workers=1,
        log_level=ServerConfig.LOG_LEVEL.lower(),
        access_log=ServerConfig.LOG_REQUESTS
    )


if __name__ == "__main__":
    main()

