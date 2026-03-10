# ============================================================
# Stage 1: builder — install Python deps for API server
# ============================================================
FROM python:3.11-slim-bookworm AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    fastapi \
    uvicorn[standard] \
    psycopg2-binary \
    redis \
    python-dotenv \
    agentops \
    httpx \
    pydantic \
    python-multipart \
    chromadb \
    notion-client

# ============================================================
# Stage 2: runtime — slim FastAPI API server
# ============================================================
FROM python:3.11-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv

RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONPATH=/app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV LOG_LEVEL=INFO

COPY --chown=appuser:appuser api/ ./api/
COPY --chown=appuser:appuser tools/postgres_tools.py ./tools/postgres_tools.py
COPY --chown=appuser:appuser tools/agentops_tools.py ./tools/agentops_tools.py
COPY --chown=appuser:appuser tools/budget_tools.py ./tools/budget_tools.py
COPY --chown=appuser:appuser tools/__init__.py ./tools/__init__.py
COPY --chown=appuser:appuser config/ ./config/
COPY --chown=appuser:appuser integrations/ ./integrations/
COPY --chown=appuser:appuser database/ ./database/

USER appuser

EXPOSE 8000

CMD ["uvicorn", "api.api_server:app", "--host", "0.0.0.0", \
     "--port", "8000", "--log-level", "info"]
