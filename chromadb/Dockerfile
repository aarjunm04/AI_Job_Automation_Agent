FROM chromadb/chroma:latest

# Switch to root to install curl for healthcheck
USER root

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Do NOT set a USER — let ChromaDB use its own default entrypoint user
# The base image manages its own user context via dumb-init

EXPOSE 8000

HEALTHCHECK --interval=20s --timeout=10s --start-period=30s --retries=5 \
    CMD curl -sf http://localhost:8000/api/v2/heartbeat || exit 1

