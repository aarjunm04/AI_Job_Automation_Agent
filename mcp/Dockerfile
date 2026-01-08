# -----------------------------------------
# MCP Service Dockerfile
# AI Job Automation Agent
# -----------------------------------------

FROM python:3.11-slim

# Prevent Python from writing .pyc and buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Working directory inside container
WORKDIR /app

# Install system dependencies if required (SQLite is included by default)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy MCP source code into container
COPY mcp_client.py .
COPY mcp_schema.sql .

# Install pip deps
COPY requirements.txt requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

# Environment variables loaded at runtime (via compose or env file)
ENV MCP_API_KEY=""
ENV MCP_DEV_MODE="0"
ENV MCP_DATABASE_URL="sqlite+aiosqlite:///mcp_context.db"
ENV MCP_SESSION_TTL_HOURS=72
ENV MCP_LOG_LEVEL=INFO

# Expose MCP port
EXPOSE 8080

# Run MCP via uvicorn
CMD ["uvicorn", "mcp_client:app", "--host", "0.0.0.0", "--port", "8080"]