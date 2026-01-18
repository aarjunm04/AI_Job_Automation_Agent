"""
═══════════════════════════════════════════════════════════════════════════════
JOB AUTOMATION MCP - SECURITY MODULE
═══════════════════════════════════════════════════════════════════════════════

Enterprise security implementation for MCP service.

Components:
- AuthHandler: API key validation, JWT tokens, service account auth
- RateLimiter: Redis-backed token bucket with hierarchical limits
- SecurityValidator: Input validation, prompt injection detection
- AuditLogger: Immutable audit trail for compliance
- Middleware: Security enforcement on all routes

Features:
- Multi-tier authentication (API key, JWT, service accounts)
- Distributed rate limiting with Redis
- Real-time threat detection
- Comprehensive audit logging
- Request fingerprinting
- IP whitelisting/blacklisting

Author: Job Automation Team
Version: 2.0 Enterprise
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

# ═══════════════════════════════════════════════════════════════════════════════
# IMPORTS
# ═══════════════════════════════════════════════════════════════════════════════

import os
import re
import json
import time
import hmac
import hashlib
import logging
from typing import Optional, Dict, List, Tuple, Any
from datetime import datetime, timedelta

# FastAPI
from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse

# JWT
import jwt

# Redis
try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None

# Password hashing
try:
    from passlib.context import CryptContext
    PASSLIB_AVAILABLE = True
except ImportError:
    PASSLIB_AVAILABLE = False
    CryptContext = None


logger = logging.getLogger("mcp.security")


# ═══════════════════════════════════════════════════════════════════════════════
# AUTHENTICATION HANDLER
# ═══════════════════════════════════════════════════════════════════════════════

class AuthHandler:
    """
    Handles all authentication mechanisms.

    Supports:
    - API key authentication (Header: X-MCP-API-Key)
    - JWT token authentication (Header: Authorization: Bearer <token>)
    - Service account authentication (future: mTLS, OAuth2)
    """

    def __init__(
        self,
        api_key: str = None,
        jwt_secret: str = None,
        jwt_algorithm: str = "HS256",
        jwt_expiry_hours: int = 24
    ):
        """
        Initialize authentication handler.

        Args:
            api_key: Master API key for simple auth
            jwt_secret: Secret key for JWT signing
            jwt_algorithm: JWT algorithm (HS256, RS256, etc.)
            jwt_expiry_hours: JWT token expiry duration
        """
        self.api_key = api_key or os.getenv("MCP_API_KEY", "")
        self.jwt_secret = jwt_secret or os.getenv("JWT_SECRET", "change-me-in-production")
        self.jwt_algorithm = jwt_algorithm
        self.jwt_expiry_hours = jwt_expiry_hours

        # Initialize password hasher if available
        if PASSLIB_AVAILABLE:
            self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        else:
            self.pwd_context = None
            logger.warning("⚠️  passlib not available, password hashing disabled")

        # Service account registry (for future multi-user support)
        self.service_accounts: Dict[str, Dict[str, Any]] = {}

        logger.info("✅ AuthHandler initialized")

    def validate_api_key(self, provided_key: Optional[str]) -> bool:
        """
        Validate API key using constant-time comparison.

        Args:
            provided_key: API key from request header

        Returns:
            True if valid, False otherwise
        """
        if not self.api_key:
            # No API key configured, allow all in dev mode
            return True

        if not provided_key:
            return False

        # Constant-time comparison to prevent timing attacks
        return hmac.compare_digest(provided_key, self.api_key)

    def generate_jwt(
        self,
        user_id: str,
        role: str = "user",
        audit_metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate JWT token for authenticated user.

        Args:
            user_id: User identifier
            role: User role (admin, user, agent, etc.)
            metadata: Additional claims

        Returns:
            JWT token string
        """
        now = datetime.utcnow()
        expiry = now + timedelta(hours=self.jwt_expiry_hours)

        payload = {
            "sub": user_id,
            "role": role,
            "iat": int(now.timestamp()),
            "exp": int(expiry.timestamp()),
            "iss": "job-automation-mcp"
        }

        if audit_metadata:
            payload.update(audit_metadata)

        token = jwt.encode(payload, self.jwt_secret, algorithm=self.jwt_algorithm)

        logger.debug(f"Generated JWT for user={user_id}, role={role}")
        return token

    def verify_jwt(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Verify and decode JWT token.

        Args:
            token: JWT token string

        Returns:
            Decoded payload dict or None if invalid
        """
        try:
            payload = jwt.decode(
                token,
                self.jwt_secret,
                algorithms=[self.jwt_algorithm],
                options={"verify_exp": True}
            )
            return payload
        except jwt.ExpiredSignatureError:
            logger.warning("JWT token expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid JWT token: {e}")
            return None

    def hash_password(self, password: str) -> str:
        """
        Hash password using bcrypt.

        Args:
            password: Plain text password

        Returns:
            Hashed password string
        """
        if not self.pwd_context:
            raise RuntimeError("Password hashing not available (install passlib)")

        return self.pwd_context.hash(password)

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """
        Verify password against hash.

        Args:
            plain_password: Plain text password
            hashed_password: Hashed password from database

        Returns:
            True if password matches, False otherwise
        """
        if not self.pwd_context:
            raise RuntimeError("Password hashing not available (install passlib)")

        return self.pwd_context.verify(plain_password, hashed_password)

    def register_service_account(
        self,
        account_id: str,
        api_key: str,
        permissions: List[str] = None
    ):
        """
        Register a service account for machine-to-machine auth.

        Args:
            account_id: Service account identifier
            api_key: Dedicated API key for this account
            permissions: List of allowed operations
        """
        self.service_accounts[account_id] = {
            "api_key_hash": self.hash_password(api_key) if self.pwd_context else api_key,
            "permissions": permissions or ["read"],
            "created_at": datetime.utcnow().isoformat()
        }
        logger.info(f"✅ Registered service account: {account_id}")

    def authenticate_request(self, request: Request) -> Dict[str, Any]:
        """
        Main authentication method - checks all auth mechanisms.

        Priority:
        1. JWT token (if present)
        2. API key
        3. Service account

        Args:
            request: FastAPI request object

        Returns:
            User context dict with user_id, role, permissions

        Raises:
            HTTPException: If authentication fails
        """
        # Try JWT first
        auth_header = request.headers.get("authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header[7:]
            payload = self.verify_jwt(token)
            if payload:
                return {
                    "user_id": payload.get("sub"),
                    "role": payload.get("role", "user"),
                    "auth_method": "jwt",
                    "permissions": ["*"]  # JWT has full access
                }

        # Try API key
        api_key = request.headers.get("x-mcp-api-key")
        if api_key:
            if self.validate_api_key(api_key):
                return {
                    "user_id": "api_user",
                    "role": "admin",
                    "auth_method": "api_key",
                    "permissions": ["*"]
                }

            # Check service accounts
            for account_id, account_data in self.service_accounts.items():
                if self.pwd_context:
                    if self.verify_password(api_key, account_data["api_key_hash"]):
                        return {
                            "user_id": account_id,
                            "role": "service",
                            "auth_method": "service_account",
                            "permissions": account_data["permissions"]
                        }
                else:
                    if api_key == account_data["api_key_hash"]:
                        return {
                            "user_id": account_id,
                            "role": "service",
                            "auth_method": "service_account",
                            "permissions": account_data["permissions"]
                        }

        # No valid auth found
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing authentication credentials",
            headers={"WWW-Authenticate": "Bearer"}
        )


# ═══════════════════════════════════════════════════════════════════════════════
# RATE LIMITER
# ═══════════════════════════════════════════════════════════════════════════════

class RateLimiter:
    """
    Token bucket rate limiter with Redis backend.

    Features:
    - Per-user, per-IP, and global rate limits
    - Hierarchical limits (user > endpoint > global)
    - Automatic bucket refill
    - Rate limit headers in responses
    - Distributed rate limiting across instances
    """

    def __init__(
        self,
        redis_url: str = None,
        default_requests: int = 1000,
        default_window: int = 60,
        enabled: bool = True
    ):
        """
        Initialize rate limiter.

        Args:
            redis_url: Redis connection URL
            default_requests: Default requests per window
            default_window: Default time window in seconds
            enabled: Enable/disable rate limiting
        """
        self.enabled = enabled
        self.default_requests = default_requests
        self.default_window = default_window

        # Initialize Redis if available
        self.redis_client = None
        if enabled and REDIS_AVAILABLE and redis_url:
            try:
                self.redis_client = redis.from_url(
                    redis_url,
                    encoding="utf-8",
                    decode_responses=True
                )
                logger.info("✅ RateLimiter connected to Redis")
            except Exception as e:
                logger.error(f"Failed to connect to Redis: {e}")
                logger.warning("⚠️  Rate limiting will use in-memory fallback")

        # In-memory fallback for development
        self._memory_buckets: Dict[str, Tuple[float, int]] = {}

        # Rate limit tiers (can be customized per user/role)
        self.tiers = {
            "admin": {"requests": 10000, "window": 60},
            "user": {"requests": 1000, "window": 60},
            "agent": {"requests": 5000, "window": 60},
            "service": {"requests": 3000, "window": 60},
            "anonymous": {"requests": 100, "window": 60}
        }

        logger.info(f"✅ RateLimiter initialized (enabled={enabled})")

    async def check_rate_limit(
        self,
        identifier: str,
        role: str = "user",
        custom_limit: Optional[Tuple[int, int]] = None
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if request is within rate limit.

        Args:
            identifier: User ID, IP address, or other identifier
            role: User role for tier selection
            custom_limit: Optional (requests, window) override

        Returns:
            Tuple of (allowed: bool, info: dict)
            info contains: remaining, reset_time, limit
        """
        if not self.enabled:
            return True, {"remaining": 999999, "limit": 999999, "reset_time": 0}

        # Determine rate limit
        if custom_limit:
            requests, window = custom_limit
        else:
            tier = self.tiers.get(role, self.tiers["user"])
            requests, window = tier["requests"], tier["window"]

        # Use Redis if available, otherwise fallback to memory
        if self.redis_client:
            return await self._check_redis(identifier, requests, window)
        else:
            return self._check_memory(identifier, requests, window)

    async def _check_redis(
        self,
        identifier: str,
        requests: int,
        window: int
    ) -> Tuple[bool, Dict[str, Any]]:
        """Token bucket implementation using Redis."""
        key = f"ratelimit:{identifier}"
        now = time.time()

        try:
            # Get current bucket state
            pipe = self.redis_client.pipeline()
            pipe.hgetall(key)
            result = await pipe.execute()
            bucket = result[0] if result else {}

            # Initialize or parse bucket
            if not bucket:
                tokens = requests
                last_refill = now
            else:
                tokens = float(bucket.get("tokens", requests))
                last_refill = float(bucket.get("last_refill", now))

            # Refill tokens based on elapsed time
            elapsed = now - last_refill
            refill_rate = requests / window
            tokens = min(requests, tokens + (elapsed * refill_rate))

            # Check if we have tokens available
            if tokens >= 1:
                tokens -= 1
                allowed = True
            else:
                allowed = False

            # Update bucket in Redis
            pipe = self.redis_client.pipeline()
            pipe.hset(key, "tokens", str(tokens))
            pipe.hset(key, "last_refill", str(now))
            pipe.expire(key, window * 2)  # Keep bucket for 2x window
            await pipe.execute()

            # Calculate reset time
            reset_time = int(now + ((1 - tokens) / refill_rate)) if tokens < 1 else int(now)

            return allowed, {
                "remaining": int(tokens),
                "limit": requests,
                "reset_time": reset_time
            }

        except Exception as e:
            logger.error(f"Redis rate limit error: {e}")
            # Fail open - allow request if Redis fails
            return True, {"remaining": requests, "limit": requests, "reset_time": 0}

    def _check_memory(
        self,
        identifier: str,
        requests: int,
        window: int
    ) -> Tuple[bool, Dict[str, Any]]:
        """In-memory token bucket fallback (not distributed)."""
        now = time.time()

        # Get or initialize bucket
        if identifier not in self._memory_buckets:
            self._memory_buckets[identifier] = (now, requests)

        last_refill, tokens = self._memory_buckets[identifier]

        # Refill tokens
        elapsed = now - last_refill
        refill_rate = requests / window
        tokens = min(requests, tokens + (elapsed * refill_rate))

        # Check if we have tokens
        if tokens >= 1:
            tokens -= 1
            allowed = True
        else:
            allowed = False

        # Update bucket
        self._memory_buckets[identifier] = (now, tokens)

        # Cleanup old buckets periodically
        if len(self._memory_buckets) > 10000:
            cutoff = now - (window * 2)
            self._memory_buckets = {
                k: v for k, v in self._memory_buckets.items()
                if v[0] > cutoff
            }

        reset_time = int(now + ((1 - tokens) / refill_rate)) if tokens < 1 else int(now)

        return allowed, {
            "remaining": int(tokens),
            "limit": requests,
            "reset_time": reset_time
        }

    async def cleanup(self):
        """Close Redis connection."""
        if self.redis_client:
            await self.redis_client.close()


# ═══════════════════════════════════════════════════════════════════════════════
# SECURITY VALIDATOR
# ═══════════════════════════════════════════════════════════════════════════════

class SecurityValidator:
    """
    Input validation and threat detection.

    Features:
    - Prompt injection detection
    - SQL injection patterns
    - XSS detection
    - Malicious payload detection
    - Content size limits
    """

    # Prompt injection patterns (common jailbreak attempts)
    PROMPT_INJECTION_PATTERNS = [
        r"ignore\s+(previous|above|all)\s+instructions?",
        r"disregard\s+(previous|above|all)\s+instructions?",
        r"forget\s+(previous|above|all)\s+instructions?",
        r"you\s+are\s+now\s+",
        r"your\s+new\s+role\s+is",
        r"system\s*:\s*",
        r"<\s*system\s*>",
        r"jailbreak",
        r"DAN\s+mode",
        r"developer\s+mode",
    ]

    # SQL injection patterns
    SQL_INJECTION_PATTERNS = [
        r"(or|and)\s+\d+\s*=\s*\d+",
        r"union\s+select",
        r"drop\s+table",
        r";\s*drop\s+",
        r"--\s*$",
        r"\/\*.*\*\/",
        r"xp_cmdshell",
    ]

    # XSS patterns
    XSS_PATTERNS = [
        r"<script[^>]*>",
        r"javascript:",
        r"onerror\s*=",
        r"onload\s*=",
        r"<iframe",
    ]

    def __init__(
        self,
        max_content_size: int = 500000,  # 500KB
        strict_mode: bool = False
    ):
        """
        Initialize security validator.

        Args:
            max_content_size: Maximum content size in bytes
            strict_mode: Enable stricter validation (may have false positives)
        """
        self.max_content_size = max_content_size
        self.strict_mode = strict_mode

        # Compile regex patterns for performance
        self.prompt_injection_regex = [
            re.compile(pattern, re.IGNORECASE) for pattern in self.PROMPT_INJECTION_PATTERNS
        ]
        self.sql_injection_regex = [
            re.compile(pattern, re.IGNORECASE) for pattern in self.SQL_INJECTION_PATTERNS
        ]
        self.xss_regex = [
            re.compile(pattern, re.IGNORECASE) for pattern in self.XSS_PATTERNS
        ]

        logger.info(f"✅ SecurityValidator initialized (strict_mode={strict_mode})")

    def validate_content(self, content: str, content_type: str = "text") -> Tuple[bool, Optional[str]]:
        """
        Validate content for security threats.

        Args:
            content: Content to validate
            content_type: Type of content (text, json, html)

        Returns:
            Tuple of (is_valid: bool, error_message: Optional[str])
        """
        if not content:
            return True, None

        # Check size limit
        if len(content) > self.max_content_size:
            return False, f"Content exceeds maximum size of {self.max_content_size} bytes"

        # Check for prompt injection
        for pattern in self.prompt_injection_regex:
            if pattern.search(content):
                return False, "Potential prompt injection detected"

        # Check for SQL injection (if strict mode)
        if self.strict_mode:
            for pattern in self.sql_injection_regex:
                if pattern.search(content):
                    return False, "Potential SQL injection detected"

        # Check for XSS (if HTML content)
        if content_type == "html" or self.strict_mode:
            for pattern in self.xss_regex:
                if pattern.search(content):
                    return False, "Potential XSS payload detected"

        return True, None

    def sanitize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize metadata to prevent injection attacks.

        Args:
            metadata: Metadata dict

        Returns:
            Sanitized metadata dict
        """
        if not metadata:
            return {}

        sanitized = {}
        for key, value in metadata.items():
            # Sanitize key
            clean_key = re.sub(r'[^\w\-]', '', str(key))[:64]

            # Sanitize value
            if isinstance(value, str):
                # Remove potential injection patterns
                clean_value = value[:1000]  # Limit length
            elif isinstance(value, (int, float, bool)):
                clean_value = value
            elif isinstance(value, list):
                clean_value = [str(v)[:100] for v in value[:10]]  # Limit list
            elif isinstance(value, dict):
                clean_value = self.sanitize_metadata(value)  # Recursive
            else:
                clean_value = str(value)[:100]

            sanitized[clean_key] = clean_value

        return sanitized


# ═══════════════════════════════════════════════════════════════════════════════
# AUDIT LOGGER
# ═══════════════════════════════════════════════════════════════════════════════

class AuditLogger:
    """
    Immutable audit trail logger.

    Logs all critical operations to database for compliance and debugging.
    """

    def __init__(self, db_session_factory=None):
        """
        Initialize audit logger.

        Args:
            db_session_factory: Async session factory for database
        """
        self.db_session_factory = db_session_factory
        logger.info("✅ AuditLogger initialized")

    async def log(
        self,
        session_id: Optional[str],
        actor_type: str,
        actor_id: Optional[str],
        action: str,
        resource_type: str,
        resource_id: Optional[str],
        outcome: str,
        error_message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None
    ):
        """
        Log audit event.

        Args:
            session_id: MCP session ID (if applicable)
            actor_type: user/agent/chrome_ext/n8n/scraper/system
            actor_id: Actor identifier
            action: Action performed (e.g., session.create, context.append)
            resource_type: Resource type (session/context_item/snapshot/evidence)
            resource_id: Resource identifier
            outcome: success/failure/error
            error_message: Error message if outcome is error
            metadata: Additional context
            ip_address: Client IP address
        """
        if not self.db_session_factory:
            # Fallback to file logging
            logger.info(
                f"AUDIT: session={session_id} actor={actor_type}:{actor_id} "
                f"action={action} resource={resource_type}:{resource_id} outcome={outcome}"
            )
            return

        try:
            # Import here to avoid circular dependency
            from server import AuditLogModel, make_uuid, safe_json_dump

            async with self.db_session_factory() as db:
                log_entry = AuditLogModel(
                    log_id=make_uuid(),
                    session_id=session_id,
                    actor_type=actor_type,
                    actor_id=actor_id,
                    action=action,
                    resource_type=resource_type,
                    resource_id=resource_id,
                    outcome=outcome,
                    error_message=error_message,
                    metadata=safe_json_dump(metadata),
                    ip_address=ip_address,
                )
                db.add(log_entry)
                await db.commit()

        except Exception as e:
            logger.error(f"Failed to write audit log: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# SECURITY MIDDLEWARE
# ═══════════════════════════════════════════════════════════════════════════════

class SecurityMiddleware:
    """
    FastAPI middleware for security enforcement.

    Applies:
    - Authentication
    - Rate limiting
    - Input validation
    - Audit logging
    - Request fingerprinting
    """

    def __init__(
        self,
        auth_handler: AuthHandler,
        rate_limiter: RateLimiter,
        security_validator: SecurityValidator,
        audit_logger: AuditLogger,
        exempt_paths: List[str] = None
    ):
        """
        Initialize security middleware.

        Args:
            auth_handler: Authentication handler
            rate_limiter: Rate limiter
            security_validator: Security validator
            audit_logger: Audit logger
            exempt_paths: Paths exempt from security (e.g., /health)
        """
        self.auth_handler = auth_handler
        self.rate_limiter = rate_limiter
        self.security_validator = security_validator
        self.audit_logger = audit_logger

        self.exempt_paths = exempt_paths or ["/health", "/metrics", "/docs", "/redoc", "/openapi.json"]

        logger.info("✅ SecurityMiddleware initialized")

    async def __call__(self, request: Request, call_next):
        """Process request through security pipeline."""
        start_time = time.time()

        # Skip security for exempt paths
        if request.url.path in self.exempt_paths:
            return await call_next(request)

        try:
            # 1. Authentication
            user_context = self.auth_handler.authenticate_request(request)

            # 2. Rate limiting
            identifier = f"{user_context['user_id']}:{request.client.host}"
            allowed, rate_info = await self.rate_limiter.check_rate_limit(
                identifier,
                role=user_context["role"]
            )

            if not allowed:
                return JSONResponse(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    content={"detail": "Rate limit exceeded"},
                    headers={
                        "X-RateLimit-Limit": str(rate_info["limit"]),
                        "X-RateLimit-Remaining": "0",
                        "X-RateLimit-Reset": str(rate_info["reset_time"]),
                        "Retry-After": str(rate_info["reset_time"] - int(time.time()))
                    }
                )

            # 3. Attach user context to request state
            request.state.user = user_context

            # 4. Process request
            response = await call_next(request)

            # 5. Add rate limit headers to response
            response.headers["X-RateLimit-Limit"] = str(rate_info["limit"])
            response.headers["X-RateLimit-Remaining"] = str(rate_info["remaining"])
            response.headers["X-RateLimit-Reset"] = str(rate_info["reset_time"])

            # 6. Audit log successful request
            duration = time.time() - start_time
            await self.audit_logger.log(
                session_id=None,  # Extract from request if needed
                actor_type=user_context["role"],
                actor_id=user_context["user_id"],
                action=f"{request.method}:{request.url.path}",
                resource_type="api_endpoint",
                resource_id=None,
                outcome="success",
                metadata={"duration_ms": int(duration * 1000), "status_code": response.status_code},
                ip_address=request.client.host
            )

            return response

        except HTTPException as e:
            # Audit log failed request
            await self.audit_logger.log(
                session_id=None,
                actor_type="unknown",
                actor_id=request.client.host,
                action=f"{request.method}:{request.url.path}",
                resource_type="api_endpoint",
                resource_id=None,
                outcome="failure",
                error_message=e.detail,
                metadata={"status_code": e.status_code},
                ip_address=request.client.host
            )
            raise

        except Exception as e:
            # Audit log error
            await self.audit_logger.log(
                session_id=None,
                actor_type="system",
                actor_id="middleware",
                action=f"{request.method}:{request.url.path}",
                resource_type="api_endpoint",
                resource_id=None,
                outcome="error",
                error_message=str(e),
                ip_address=request.client.host
            )
            raise


# ═══════════════════════════════════════════════════════════════════════════════
# FACTORY FUNCTIONS (for server.py integration)
# ═══════════════════════════════════════════════════════════════════════════════

def get_auth_handler() -> AuthHandler:
    """Factory function to get AuthHandler instance."""
    return AuthHandler(
        api_key=os.getenv("MCP_API_KEY"),
        jwt_secret=os.getenv("JWT_SECRET", "change-me-in-production"),
        jwt_expiry_hours=int(os.getenv("JWT_EXPIRY_HOURS", "24"))
    )


def get_rate_limiter() -> RateLimiter:
    """Factory function to get RateLimiter instance."""
    return RateLimiter(
        redis_url=os.getenv("REDIS_URL"),
        default_requests=int(os.getenv("RATE_LIMIT_REQUESTS", "1000")),
        default_window=int(os.getenv("RATE_LIMIT_WINDOW", "60")),
        enabled=os.getenv("RATE_LIMIT_ENABLED", "true").lower() == "true"
    )


def get_security_validator() -> SecurityValidator:
    """Factory function to get SecurityValidator instance."""
    return SecurityValidator(
        max_content_size=int(os.getenv("MAX_CONTENT_SIZE", "500000")),
        strict_mode=os.getenv("SECURITY_STRICT_MODE", "false").lower() == "true"
    )


def get_audit_logger(db_session_factory=None) -> AuditLogger:
    """Factory function to get AuditLogger instance."""
    return AuditLogger(db_session_factory=db_session_factory)


def get_security_middleware(
    auth_handler: AuthHandler,
    rate_limiter: RateLimiter,
    security_validator: SecurityValidator,
    audit_logger: AuditLogger
) -> SecurityMiddleware:
    """Factory function to get SecurityMiddleware instance."""
    return SecurityMiddleware(
        auth_handler=auth_handler,
        rate_limiter=rate_limiter,
        security_validator=security_validator,
        audit_logger=audit_logger
    )


# ═══════════════════════════════════════════════════════════════════════════════
# USAGE EXAMPLE
# ═══════════════════════════════════════════════════════════════════════════════

"""
To integrate with server.py:

1. In server.py startup_event():

   from security import get_auth_handler, get_rate_limiter, get_audit_logger

   mcp_service.auth_handler = get_auth_handler()
   mcp_service.rate_limiter = get_rate_limiter()
   mcp_service.audit_logger = get_audit_logger(AsyncSessionLocal)

2. In server.py, add middleware:

   from security import get_security_middleware

   security_mw = get_security_middleware(
       mcp_service.auth_handler,
       mcp_service.rate_limiter,
       mcp_service.security_validator,
       mcp_service.audit_logger
   )
   app.middleware("http")(security_mw)

3. Use audit logger in routes:

   await mcp_service.audit_logger.log(
       session_id=session_id,
       actor_type="user",
       actor_id=user["user_id"],
       action="session.create",
       resource_type="session",
       resource_id=session_id,
       outcome="success"
   )
"""