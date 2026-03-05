#!/bin/bash
# =============================================================================
# AI JOB AGENT — FULL AUTOMATED DOCKERISATION SCRIPT
# Run from repo root: bash dockerise.sh
# Plug in laptop before running. Expected total time: 20-30 minutes (first run)
# Subsequent runs: ~2-3 minutes (Docker cache used)
# =============================================================================

set -euo pipefail

ENV_FILE="$HOME/java.env"
LOG_FILE="logs/dockerise_$(date +%Y%m%d_%H%M%S).log"
COMPOSE="docker compose --env-file $ENV_FILE"

# Colours
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

mkdir -p logs
exec > >(tee -a "$LOG_FILE") 2>&1

log()  { echo -e "${BLUE}[$(date '+%H:%M:%S')]${NC} $1"; }
ok()   { echo -e "${GREEN}[$(date '+%H:%M:%S')] ✅ $1${NC}"; }
warn() { echo -e "${YELLOW}[$(date '+%H:%M:%S')] ⚠️  $1${NC}"; }
fail() { echo -e "${RED}[$(date '+%H:%M:%S')] ❌ FAILED: $1${NC}"; echo ""; echo "Check log: $LOG_FILE"; exit 1; }

wait_healthy() {
  local service=$1
  local max_wait=${2:-60}
  local elapsed=0
  log "Waiting for $service to be healthy..."
  while [ $elapsed -lt $max_wait ]; do
    status=$($COMPOSE ps "$service" 2>/dev/null | grep -oE "healthy|running" | head -1 || echo "")
    if [[ "$status" == "healthy" || "$status" == "running" ]]; then
      ok "$service is $status"
      return 0
    fi
    sleep 5
    elapsed=$((elapsed + 5))
    echo "  ... still waiting ($elapsed/${max_wait}s)"
  done
  fail "$service did not become healthy within ${max_wait}s"
}

# =============================================================================
echo ""
echo "======================================================================"
echo "  AI JOB AGENT — AUTOMATED DOCKERISATION"
echo "  Started: $(date)"
echo "  Log: $LOG_FILE"
echo "======================================================================"
echo ""

# =============================================================================
log "PHASE 0 — Pre-flight checks"
# =============================================================================

[ -f "$ENV_FILE" ] || fail "java.env not found at $ENV_FILE"
ok "java.env found"

docker info > /dev/null 2>&1 || fail "Docker is not running — start Docker Desktop first"
ok "Docker is running"

[ -f "docker-compose.yml" ] || fail "Not in repo root — cd to AI_Job_Automation_Agent first"
ok "In repo root: $(pwd)"

[ -f "Dockerfile" ] || fail "Root Dockerfile missing"
[ -f "rag_systems/Dockerfile" ] || fail "rag_systems/Dockerfile missing"
ok "Both Dockerfiles present"

POSTGRES_URL=$(grep "LOCAL_POSTGRES_URL" "$ENV_FILE" | cut -d'=' -f2-)
REDIS_URL=$(grep "^REDIS_URL=" "$ENV_FILE" | cut -d'=' -f2-)
CHROMA_HOST=$(grep "^CHROMADB_HOST=" "$ENV_FILE" | cut -d'=' -f2-)
RAG_URL=$(grep "^RAG_SERVER_URL=" "$ENV_FILE" | cut -d'=' -f2-)

[[ "$POSTGRES_URL" == *"@postgres:"* ]] || warn "LOCAL_POSTGRES_URL may still use localhost — should use @postgres:"
[[ "$REDIS_URL" == *"redis://redis:"* ]] || warn "REDIS_URL may still use localhost — should use redis://redis:"
[[ "$CHROMA_HOST" == "chromadb" ]] || warn "CHROMADB_HOST should be 'chromadb' not '$CHROMA_HOST'"
[[ "$RAG_URL" == *"rag-server"* ]] || warn "RAG_SERVER_URL should use rag-server hostname"

# Only tear down if infra is NOT already healthy — preserve healthy containers on re-run
CHROMA_RUNNING=$(docker ps --filter "name=ai_chromadb" --filter "health=healthy" -q 2>/dev/null || echo "")
if [ -z "$CHROMA_RUNNING" ]; then
  log "Cleaning any leftover containers from previous runs..."
  $COMPOSE down --remove-orphans 2>/dev/null || true
  ok "Cleanup done"
else
  ok "Healthy containers detected — skipping teardown (re-run mode)"
fi

# =============================================================================
log "PHASE 1 — Pulling infra images (postgres, redis, chromadb)"
# =============================================================================

$COMPOSE pull postgres redis chromadb || fail "Failed to pull infra images — check internet connection"
ok "Infra images pulled"

# =============================================================================
log "PHASE 2 — Building rag-server image (~400MB, 3-5 mins first run)"
# =============================================================================

# No --no-cache: Docker layer cache means this is instant on re-runs
$COMPOSE build rag-server || fail "rag-server build failed — check rag_systems/Dockerfile"
ok "rag-server image built"

# =============================================================================
log "PHASE 3 — Building agentrunner image (~1GB, 8-15 mins first run)"
# =============================================================================

$COMPOSE build agentrunner || fail "agentrunner build failed — check root Dockerfile"
ok "agentrunner image built"

# =============================================================================
log "PHASE 4 — Booting infra services (postgres, redis, chromadb)"
# =============================================================================

$COMPOSE up -d postgres redis chromadb || fail "Failed to start infra services"

wait_healthy postgres 90
wait_healthy redis 60
wait_healthy chromadb 120

# =============================================================================
log "PHASE 5 — Verifying Postgres schema"
# =============================================================================

# Read postgres credentials from individual keys (more reliable than parsing URL)
PGUSER_RESOLVED=$(grep '^LOCAL_POSTGRES_USER=' "$ENV_FILE" | cut -d'=' -f2-)
PGDB_RESOLVED=$(grep   '^LOCAL_POSTGRES_DB='   "$ENV_FILE" | cut -d'=' -f2-)

# Fallback to compose defaults if keys are not in java.env
PGUSER_RESOLVED="${PGUSER_RESOLVED:-postgres}"
PGDB_RESOLVED="${PGDB_RESOLVED:-ai_job_db}"

# Integer-safe: tr removes whitespace; defaults to 0 if docker exec returns anything non-numeric
TABLE_COUNT=$(docker exec ai_postgres psql -U "$PGUSER_RESOLVED" -d "$PGDB_RESOLVED" \
  -tAc "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema='public';" 2>/dev/null \
  | tr -d '[:space:]' || echo "0")
TABLE_COUNT=$(echo "$TABLE_COUNT" | grep -Eo '^[0-9]+' || echo "0")

if [ "${TABLE_COUNT:-0}" -ge 5 ]; then
  ok "Schema already loaded ($TABLE_COUNT tables found)"
else
  log "Schema not loaded — loading from database/schema.sql"
  docker exec -i ai_postgres psql -U "$PGUSER_RESOLVED" -d "$PGDB_RESOLVED" < database/schema.sql \
    && ok "Schema loaded successfully" \
    || fail "Schema load failed — check database/schema.sql"
fi

# =============================================================================
log "PHASE 6 — Booting RAG server"
# =============================================================================

$COMPOSE up -d rag-server || fail "Failed to start rag-server"
wait_healthy rag-server 90

RAG_HEALTH=$(curl -s --max-time 10 http://localhost:8090/health 2>/dev/null || echo "unreachable")
if [[ "$RAG_HEALTH" == *"healthy"* ]]; then
  ok "RAG server health check passed"
else
  warn "RAG server health endpoint returned: $RAG_HEALTH — continuing anyway"
fi

# =============================================================================
log "PHASE 7 — Running resume ingestion (PDFs → ChromaDB)"
# =============================================================================

# Safe glob count — 'ls *.pdf' under set -e errors if no files match
PDF_COUNT=$(find app/resumes -maxdepth 1 -name '*.pdf' 2>/dev/null | wc -l | tr -d ' ')
if [ -d "app/resumes" ] && [ "${PDF_COUNT:-0}" -gt 0 ]; then
  log "Found $PDF_COUNT resume PDFs — ingesting into ChromaDB..."
  $COMPOSE run --rm agentrunner python rag_systems/ingestion.py \
    && ok "$PDF_COUNT resumes ingested into ChromaDB" \
    || warn "Ingestion had errors — check logs but continuing"
else
  warn "No PDFs found in app/resumes/ — skipping ingestion"
fi

# =============================================================================
log "PHASE 8 — Booting daemon services (postgres, redis, chromadb, rag-server)"
# =============================================================================
# agentrunner is restart:"no" (one-shot batch job) — it must NOT be started here.
# It is invoked explicitly in Phase 10 via 'compose run --rm'.
# Starting it here via 'compose up -d' would trigger main.py before the dry-run.
$COMPOSE up -d postgres redis chromadb rag-server || fail "Failed to boot daemon services"
wait_healthy rag-server 90
ok "All daemon services booted"

# =============================================================================
log "PHASE 9 — Final health verification"
# =============================================================================

echo ""
echo "======================================================================"
echo "  FINAL SERVICE STATUS"
echo "======================================================================"
$COMPOSE ps
echo ""

log "Running final checks..."

POSTGRES_OK=$(docker exec ai_postgres pg_isready -U "$PGUSER_RESOLVED" 2>/dev/null && echo "ok" || echo "fail")
REDIS_OK=$(docker exec ai_redis redis-cli ping 2>/dev/null || echo "fail")
# grep -c returns exit code 1 (count=0) which fires ||, producing "0\n0" — use grep -q instead
if curl -s --max-time 5 http://localhost:8001/api/v2/heartbeat 2>/dev/null | grep -q "nanosecond_heartbeat"; then CHROMA_OK=1; else CHROMA_OK=0; fi
if curl -s --max-time 5 http://localhost:8090/health 2>/dev/null | grep -q "healthy"; then RAG_OK=1; else RAG_OK=0; fi

echo ""
echo "======================================================================"
echo "  HEALTH CHECK RESULTS"
echo "======================================================================"
# Use if/else to avoid set -e treating || as a failure signal
if [[ "$POSTGRES_OK" == "ok" ]];  then ok "Postgres   — HEALTHY"; else warn "Postgres   — check manually"; fi
if [[ "$REDIS_OK"   == "PONG" ]]; then ok "Redis      — HEALTHY"; else warn "Redis      — check manually"; fi
if [[ "${CHROMA_OK:-0}" -gt 0 ]]; then ok "ChromaDB   — HEALTHY"; else warn "ChromaDB   — check manually"; fi
if [[ "${RAG_OK:-0}"    -gt 0 ]]; then ok "RAG Server — HEALTHY"; else warn "RAG Server — check manually"; fi

# =============================================================================
log "PHASE 10 — Running pipeline dry run"
# =============================================================================

echo ""
log "Starting DRY RUN — full pipeline test, no real submissions..."
echo ""

$COMPOSE run --rm agentrunner \
  python main.py --dry-run 2>&1 | tee "logs/dryrun_$(date +%Y%m%d_%H%M%S).log" | tail -60

# =============================================================================
echo ""
echo "======================================================================"
echo "  DOCKERISATION COMPLETE"
echo "  Finished: $(date)"
echo "  Full log: $LOG_FILE"
echo "======================================================================"
echo ""
echo "USEFUL COMMANDS:"
echo "  View live logs:    docker compose --env-file ~/java.env logs -f agentrunner"
echo "  Shell into agent:  docker compose --env-file ~/java.env exec agentrunner bash"
echo "  Stop everything:   docker compose --env-file ~/java.env down"
echo "  Full reset:        docker compose --env-file ~/java.env down -v"
echo "  Re-run pipeline:   docker compose --env-file ~/java.env run --rm agentrunner python main.py --dry-run"
echo "======================================================================"

