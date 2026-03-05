# ============================================================
# Stage 1: builder — compile deps, install Playwright browsers
# gcc + libpq-dev are only needed here; they never reach the final image
# ============================================================
FROM python:3.11-slim-bookworm AS builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    curl \
    gnupg \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Isolated venv so we can copy it cleanly to the runtime stage
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt

# Install Playwright browsers into a fixed path so we can copy them too
ENV PLAYWRIGHT_BROWSERS_PATH=/ms-playwright
RUN playwright install chromium --with-deps

# ============================================================
# Stage 2: runtime — lean final image, no build tools
# ============================================================
FROM python:3.11-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Only base system deps here — libpq5 for psycopg2, curl for healthchecks
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy the pre-built virtualenv from builder FIRST — playwright binary lives here
COPY --from=builder /opt/venv /opt/venv

# playwright install-deps detects the platform (amd64 or arm64) automatically.
# This is the canonical approach — no hardcoded arch-specific package names.
RUN /opt/venv/bin/playwright install-deps chromium

# Copy the pre-downloaded Playwright browsers from builder
COPY --from=builder /ms-playwright /ms-playwright

# Non-root user for security
RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app
# /app is created by WORKDIR as root:root — chown so appuser can create files at runtime
RUN chown appuser:appuser /app

ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONPATH=/app
ENV PLAYWRIGHT_BROWSERS_PATH=/ms-playwright

# Copy source — .dockerignore ensures venv/ is excluded
COPY --chown=appuser:appuser . /app

USER appuser

CMD ["python", "main.py"]
