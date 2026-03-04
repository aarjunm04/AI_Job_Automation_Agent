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
COMPOSE="docker-compose --env-file $ENV_FILE"

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

[ -f "$ENV_FILE" ] || fail "narad.env not found at $ENV_FILE"
ok "narad.env found"

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

PGUSER_RESOLVED=$(grep '^LOCAL_POSTGRES_URL=' "$ENV_FILE" | sed 's|.*://\([^:]*\):.*|\1|')
PGDB_RESOLVED=$(grep '^LOCAL_POSTGRES_DB=' "$ENV_FILE" | cut -d'=' -f2)

TABLE_COUNT=$(docker exec ai_postgres psql -U "$PGUSER_RESOLVED" -d "$PGDB_RESOLVED" \
  -tAc "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema='public';" 2>/dev/null || echo "0")

if [ "$TABLE_COUNT" -ge "5" ]; then
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

if [ -d "app/resumes" ] && [ "$(ls app/resumes/*.pdf 2>/dev/null | wc -l)" -gt "0" ]; then
  PDF_COUNT=$(ls app/resumes/*.pdf | wc -l | tr -d ' ')
  log "Found $PDF_COUNT resume PDFs — ingesting into ChromaDB..."
  $COMPOSE run --rm agentrunner python rag_systems/ingestion.py \
    && ok "$PDF_COUNT resumes ingested into ChromaDB" \
    || warn "Ingestion had errors — check logs but continuing"
else
  warn "No PDFs found in app/resumes/ — skipping ingestion"
fi

# =============================================================================
log "PHASE 8 — Booting full stack"
# =============================================================================

$COMPOSE up -d || fail "Failed to boot full stack"
sleep 15
ok "Full stack booted"

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
# FIX: v2 API endpoint
CHROMA_OK=$(curl -s --max-time 5 http://localhost:8001/api/v2/heartbeat 2>/dev/null | grep -c "nanosecond" || echo "0")
RAG_OK=$(curl -s --max-time 5 http://localhost:8090/health 2>/dev/null | grep -c "healthy" || echo "0")

echo ""
echo "======================================================================"
echo "  HEALTH CHECK RESULTS"
echo "======================================================================"
[[ "$POSTGRES_OK" == "ok" ]] && ok "Postgres   — HEALTHY" || warn "Postgres   — check manually"
[[ "$REDIS_OK" == "PONG" ]]  && ok "Redis      — HEALTHY" || warn "Redis      — check manually"
[[ "$CHROMA_OK" -gt "0" ]]   && ok "ChromaDB   — HEALTHY" || warn "ChromaDB   — check manually"
[[ "$RAG_OK" -gt "0" ]]      && ok "RAG Server — HEALTHY" || warn "RAG Server — check manually"

# =============================================================================
log "PHASE 10 — Running pipeline dry run"
# =============================================================================

echo ""
log "Starting DRY RUN — full pipeline test, no real submissions..."
echo ""

$COMPOSE exec agentrunner \
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
echo "  View live logs:    docker-compose --env-file ~/java.env logs -f agentrunner"
echo "  Shell into agent:  docker-compose --env-file ~/java.env exec agentrunner bash"
echo "  Stop everything:   docker-compose --env-file ~/java.env down"
echo "  Full reset:        docker-compose --env-file ~/java.env down -v"
echo "  Re-run pipeline:   docker-compose --env-file ~/java.env exec agentrunner python main.py --dry-run"
echo "======================================================================"

