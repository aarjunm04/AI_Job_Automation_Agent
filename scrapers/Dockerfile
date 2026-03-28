# ── scrapers/Dockerfile ────────────────────────────────────────────────────
FROM ai_playwright_base:latest

WORKDIR /app

# ── Source packages ────────────────────────────────────────────────────────
COPY --chown=appuser:appuser scrapers/          /app/scrapers/
COPY --chown=appuser:appuser tools/             /app/tools/
COPY --chown=appuser:appuser config/            /app/config/
COPY --chown=appuser:appuser integrations/      /app/integrations/
COPY --chown=appuser:appuser utils/             /app/utils/
COPY --chown=appuser:appuser database/          /app/database/

# ── Create logs dir writable by appuser ───────────────────────────────────
RUN mkdir -p /app/logs && chown appuser:appuser /app/logs

# ── Runtime env ───────────────────────────────────────────────────────────
ENV PYTHONPATH=/app \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PLAYWRIGHT_BROWSERS_PATH=/ms-playwright

USER appuser

EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=15s --start-period=60s --retries=3 \
    CMD curl -sf http://localhost:8001/health || exit 1

CMD ["uvicorn", "scrapers.scraper_service:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "1", "--log-level", "info"]