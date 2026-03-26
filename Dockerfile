# ── agent_runner/Dockerfile ────────────────────────────────────────────────
FROM ai_playwright_base:latest

WORKDIR /app

# ── Install any requirements not in base image ────────────────────────────
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# ── Source packages ────────────────────────────────────────────────────────
COPY --chown=appuser:appuser agents/              /app/agents/
COPY --chown=appuser:appuser tools/               /app/tools/
COPY --chown=appuser:appuser integrations/        /app/integrations/
COPY --chown=appuser:appuser config/              /app/config/
COPY --chown=appuser:appuser utils/               /app/utils/
COPY --chown=appuser:appuser database/            /app/database/
COPY --chown=appuser:appuser scrapers/            /app/scrapers/
COPY --chown=appuser:appuser auto_apply/          /app/auto_apply/
COPY --chown=appuser:appuser rag_systems/         /app/rag_systems/
COPY --chown=appuser:appuser scripts/             /app/scripts/
COPY --chown=appuser:appuser main.py              /app/main.py

# ── Runtime dirs ──────────────────────────────────────────────────────────
RUN mkdir -p /app/logs /app/app/resumes \
    && chown -R appuser:appuser /app/logs /app/app

# ── Runtime env ───────────────────────────────────────────────────────────
ENV PYTHONPATH=/app \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PLAYWRIGHT_BROWSERS_PATH=/ms-playwright \
    LOG_LEVEL=INFO \
    ACTIVE_DB=local \
    DRY_RUN=false

USER appuser

HEALTHCHECK --interval=60s --timeout=10s --start-period=30s --retries=3 \
    CMD test -f /app/logs/latest_run.json || exit 1

CMD ["python", "main.py", "--mode", "full"]
ENV LOCAL_POSTGRES_URL="postgresql://aarjunm04:ajdev123@ai_postgres:5432/ai_job_db"
ENV ACTIVE_DB="local"
