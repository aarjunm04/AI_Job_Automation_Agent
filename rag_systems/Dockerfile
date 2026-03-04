# ============================================================
# Stage 1: builder — install all Python deps into /opt/venv
# ============================================================
FROM python:3.11-slim-bookworm AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt

# ============================================================
# Stage 2: runtime — lean final image, no build tools
# ============================================================
FROM python:3.11-slim-bookworm AS runtime

# Runtime-only system deps — libpq5 for psycopg2, curl for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtualenv from builder stage only
COPY --from=builder /opt/venv /opt/venv

# Non-root user for security
RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

# PYTHONPATH must be set — without this EVERY rag_systems import fails
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONPATH=/app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Copy source — .dockerignore ensures venv/ is excluded
COPY --chown=appuser:appuser . /app

USER appuser

EXPOSE 8090

# Health check — polls every 30s, 3 retries, marks unhealthy after 90s
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -sf http://localhost:8090/health || exit 1

# Production uvicorn — 1 worker (single-core Docker env)
CMD ["uvicorn", "rag_systems.production_server:app", \
     "--host", "0.0.0.0", "--port", "8090", \
     "--workers", "1", "--log-level", "info", \
     "--no-access-log"]
