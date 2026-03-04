# ============================================================
# Base: Microsoft official Playwright Python image (jammy/22.04)
# All Chromium browser deps pre-installed and verified
# ============================================================
FROM mcr.microsoft.com/playwright/python:v1.50.0-jammy

# Install Python 3.11 from deadsnakes — base image ships with 3.10
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3.11-distutils \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Make python3.11 the default python3 and python
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Install pip for python3.11
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

WORKDIR /app

# PYTHONPATH — critical, without this all package imports fail at runtime
ENV PYTHONPATH=/app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
# Playwright browsers already installed in base image at this path
ENV PLAYWRIGHT_BROWSERS_PATH=/ms-playwright

# Install Python deps BEFORE copying source — preserves Docker layer cache
# When only code changes, this layer is NOT rebuilt, only the COPY below
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# DO NOT run playwright install — browsers already in base image
# Running it again wastes 300MB and causes permission errors

# Copy full project source — .dockerignore excludes venv, git, caches
COPY . /app

# Default entrypoint — overridden by docker-compose run commands
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

CMD ["python", "main.py"]
