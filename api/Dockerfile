# ============================================================
# Stage 1: builder
# ============================================================
FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

# ============================================================
# Stage 2: runtime
# ============================================================
FROM python:3.11-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv

RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONPATH=/app PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

COPY --chown=appuser:appuser requirements.txt /app/requirements.txt
COPY --chown=appuser:appuser api/ /app/api/
COPY --chown=appuser:appuser agents/ /app/agents/
COPY --chown=appuser:appuser tools/ /app/tools/
COPY --chown=appuser:appuser integrations/ /app/integrations/
COPY --chown=appuser:appuser config/ /app/config/
COPY --chown=appuser:appuser database/ /app/database/
COPY --chown=appuser:appuser utils/ /app/utils/

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=5 \
    CMD curl -sf http://localhost:8000/health || exit 1

CMD ["uvicorn", "api.api_server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
