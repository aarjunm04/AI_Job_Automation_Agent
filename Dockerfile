# =============================================================================
# AI Job Automation Agent — Single Build Image (python:3.11-slim)
# Used by: agentrunner (python main.py) + rag-server (uvicorn)
# Build context: repo root (.)
# =============================================================================

FROM python:3.11-slim

WORKDIR /app

# System packages required for Playwright browsers and psycopg2 headers
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl gcc libpq-dev && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (layer-cached until requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Playwright Chromium browser and its OS-level dependencies
RUN playwright install chromium --with-deps

# Copy every source package
COPY agents/       ./agents/
COPY api/          ./api/
COPY app/          ./app/
COPY auto_apply/   ./auto_apply/
COPY config/       ./config/
COPY database/     ./database/
COPY integrations/ ./integrations/
COPY platforms/    ./platforms/
COPY rag_systems/  ./rag_systems/
COPY scrapers/     ./scrapers/
COPY tools/        ./tools/
COPY utils/        ./utils/

# Root-level entry-point files
COPY main.py        .
COPY pyproject.toml .

# RAG server listens on 8090; agentrunner opens no inbound port
EXPOSE 8090

# Default CMD — overridden per-service in docker-compose.yml
CMD ["python", "main.py"]
