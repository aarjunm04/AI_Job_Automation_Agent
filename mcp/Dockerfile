# ═══════════════════════════════════════════════════════════════════════════════
# Job Automation MCP - Production Dockerfile
# ═══════════════════════════════════════════════════════════════════════════════
#
# Multi-stage build for optimized production image
# Base image: Python 3.11 slim
# Final size: ~200-300MB (vs ~1GB for full Python image)
#
# Build:
#   docker build -t job-automation-mcp:latest .
#
# Run:
#   docker run -p 8080:8080 --env-file .env job-automation-mcp:latest
#
# ═══════════════════════════════════════════════════════════════════════════════

# ───────────────────────────────────────────────────────────────────────────────
# Stage 1: Builder - Install dependencies
# ───────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim as builder

# Set working directory
WORKDIR /build

# Install system dependencies for building Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    make \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Create virtual environment and install dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# ───────────────────────────────────────────────────────────────────────────────
# Stage 2: Runtime - Create minimal production image
# ───────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim

# Metadata
LABEL maintainer="Job Automation Team"
LABEL description="Job Automation MCP - Context Management Service"
LABEL version="2.0"

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PATH="/opt/venv/bin:$PATH" \
    MCP_DEV_MODE=false

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd -r mcp && useradd -r -g mcp -u 1000 mcp

# Set working directory
WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Copy application code
COPY service/ /app/service/
COPY server.py /app/
COPY security.py /app/
COPY integrations.py /app/
COPY cache.py /app/

# Create data directory with proper permissions
RUN mkdir -p /app/mcp /app/logs && \
    chown -R mcp:mcp /app

# Switch to non-root user
USER mcp

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Default command
CMD ["python", "server.py"]
