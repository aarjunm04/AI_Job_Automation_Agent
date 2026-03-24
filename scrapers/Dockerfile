FROM ai_playwright_base

WORKDIR /app

COPY --chown=appuser:appuser scrapers/ /app/scrapers/
COPY --chown=appuser:appuser tools/scraper_tools.py /app/tools/scraper_tools.py
COPY --chown=appuser:appuser tools/postgres_tools.py /app/tools/postgres_tools.py
COPY --chown=appuser:appuser tools/agentops_tools.py /app/tools/agentops_tools.py
COPY --chown=appuser:appuser tools/__init__.py /app/tools/__init__.py

COPY --chown=appuser:appuser tools/budget_tools.py /app/tools/budget_tools.py
COPY --chown=appuser:appuser tools/notion_tools.py /app/tools/notion_tools.py
COPY --chown=appuser:appuser tools/rag_tools.py /app/tools/rag_tools.py
COPY --chown=appuser:appuser tools/serpapi_tool.py /app/tools/serpapi_tool.py
COPY --chown=appuser:appuser config/ /app/config/
COPY --chown=appuser:appuser integrations/ /app/integrations/
COPY --chown=appuser:appuser utils/ /app/utils/

ENV PYTHONPATH=/app PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

USER appuser

EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=15s --start-period=60s --retries=3 \
    CMD curl -sf http://localhost:8001/health || exit 1

CMD ["uvicorn", "scrapers.scraper_service:app", \
     "--host", "0.0.0.0", "--port", "8001", \
     "--workers", "1", "--log-level", "info"]
