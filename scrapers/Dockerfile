# ============================================================
# Stage 1: builder — install Python deps into /root/.local
# ============================================================
FROM python:3.11-slim AS builder

WORKDIR /build

COPY requirements.txt .

RUN pip install --no-cache-dir --user \
    playwright \
    psycopg2-binary \
    redis \
    requests \
    python-dotenv \
    agentops \
    fastapi \
    uvicorn

# ============================================================
# Stage 2: final — Playwright + Chromium + scraper service
# ============================================================
FROM python:3.11-slim

WORKDIR /app

COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# System deps required by Playwright Chromium
RUN apt-get update && apt-get install -y --no-install-recommends \
    libnss3 libatk1.0-0 libatk-bridge2.0-0 libcups2 libdrm2 \
    libxkbcommon0 libxcomposite1 libxdamage1 libxfixes3 \
    libxrandr2 libgbm1 libasound2 libpangocairo-1.0-0 \
    libgtk-3-0 libx11-xcb1 curl \
    && rm -rf /var/lib/apt/lists/*

RUN playwright install chromium --with-deps

COPY scrapers/ ./scrapers/
COPY tools/postgres_tools.py ./tools/postgres_tools.py
COPY tools/agentops_tools.py ./tools/agentops_tools.py
COPY tools/apply_tools.py ./tools/apply_tools.py
COPY tools/__init__.py ./tools/__init__.py
COPY config/ ./config/
COPY integrations/ ./integrations/
COPY database/ ./database/
COPY utils/ ./utils/

ENV PYTHONPATH=/app
ENV LOG_LEVEL=INFO

EXPOSE 8001

CMD ["uvicorn", "scrapers.scraper_service:app", "--host", "0.0.0.0", \
     "--port", "8001", "--log-level", "info"]
