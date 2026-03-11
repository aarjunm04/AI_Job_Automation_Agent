FROM ai_playwright_base

WORKDIR /app

COPY --chown=appuser:appuser auto_apply/ /app/auto_apply/
COPY --chown=appuser:appuser tools/apply_tools.py /app/tools/apply_tools.py
COPY --chown=appuser:appuser tools/postgres_tools.py /app/tools/postgres_tools.py
COPY --chown=appuser:appuser tools/agentops_tools.py /app/tools/agentops_tools.py
COPY --chown=appuser:appuser config/ /app/config/
COPY --chown=appuser:appuser integrations/ /app/integrations/
COPY --chown=appuser:appuser utils/ /app/utils/

ENV PYTHONPATH=/app PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

USER appuser

EXPOSE 8003

HEALTHCHECK --interval=30s --timeout=15s --start-period=60s --retries=3 \
    CMD curl -sf http://localhost:8003/health || exit 1

CMD ["uvicorn", "auto_apply.apply_service:app", \
     "--host", "0.0.0.0", "--port", "8003", \
     "--workers", "1", "--log-level", "info"]
