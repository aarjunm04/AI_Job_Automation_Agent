# =============================================================================
# MCP Server - Multi-stage Dockerfile for Model Context Protocol FastAPI
# =============================================================================
# Purpose: Build optimized container for MCP Server with minimal attack surface
# Base: python:3.11-slim (secure, small footprint)
# Features:
#  - Multi-stage: builder stage installs deps, runtime stage keeps image small
#  - Non-root user: app (UID 1000) for security
#  - Health check endpoint: GET /health
#  - Startup: uvicorn on port 3001
# =============================================================================

FROM python:3.11-slim as builder

# Set working directory for build stage
WORKDIR /tmp

# Install build dependencies (temporary, removed in runtime stage)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies with cache optimization
RUN pip install --no-cache-dir --user -r requirements.txt


# =============================================================================
# Runtime Stage - Minimal production image
# =============================================================================
FROM python:3.11-slim

LABEL maintainer="AI Job Automation Team"
LABEL description="MCP Server - Model Context Protocol FastAPI service"
LABEL version="1.0"

WORKDIR /app

# Install curl for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

# Create non-root user FIRST
RUN useradd -m -u 1000 -s /sbin/nologin app

# Copy dependencies to app user's home directory
COPY --from=builder --chown=app:app /root/.local /home/app/.local

# Copy application code
COPY --chown=app:app mcp /app/mcp
COPY --chown=app:app config /app/config

# Create directories with proper ownership
RUN mkdir -p /app/.mcp_cache /app/logs && chown -R app:app /app

# Configure environment for app user
ENV PATH=/home/app/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

USER app

EXPOSE 3001

HEALTHCHECK --interval=15s --timeout=5s --retries=5 --start-period=20s \
    CMD curl -f http://localhost:3001/health || exit 1

CMD ["uvicorn", "mcp.server:app", "--host", "0.0.0.0", "--port", "3001"]


