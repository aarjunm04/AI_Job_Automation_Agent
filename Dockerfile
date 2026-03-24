# ============================================================
# Stage 1: builder
# ============================================================
FROM python:3.11-slim-bookworm AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt /app/requirements.txt
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
RUN pip install --no-cache-dir -r /app/requirements.txt

# ============================================================
# Stage 2: runtime
# ============================================================
FROM python:3.11-slim-bookworm AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv

RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app
RUN chown appuser:appuser /app

ENV PATH="/opt/venv/bin:$PATH"
ENV PLAYWRIGHT_BROWSERS_PATH=/ms-playwright
RUN pip install playwright && \
    PLAYWRIGHT_BROWSERS_PATH=/ms-playwright playwright install chromium && \
    PLAYWRIGHT_BROWSERS_PATH=/ms-playwright playwright install-deps chromium

ENV PYTHONPATH=/app PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

COPY --chown=appuser:appuser requirements.txt /app/requirements.txt
COPY --chown=appuser:appuser agents/ /app/agents/
COPY --chown=appuser:appuser tools/ /app/tools/
COPY --chown=appuser:appuser integrations/ /app/integrations/
COPY --chown=appuser:appuser config/ /app/config/
COPY --chown=appuser:appuser database/ /app/database/
COPY --chown=appuser:appuser utils/ /app/utils/
COPY --chown=appuser:appuser main.py /app/main.py
COPY --chown=appuser:appuser rag_systems/ /app/rag_systems/
COPY --chown=appuser:appuser app/resumes/ /app/resumes/
COPY --chown=appuser:appuser api/ /app/api/
COPY --chown=appuser:appuser auto_apply/ /app/auto_apply/
COPY --chown=appuser:appuser scrapers/ /app/scrapers/

USER appuser

CMD ["python", "main.py", "--mode", "full"]
