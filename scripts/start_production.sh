#!/bin/bash
set -e

# Get project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}╔═══════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║   🚀 Job Automation System - Production Mode    ║${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════════════╝${NC}"
echo ""

# Load narad.env
if [ -f ~/narad.env ]; then
    source ~/narad.env
    echo -e "${GREEN}✅ Loaded ~/narad.env${NC}"
else
    echo -e "${RED}❌ ~/narad.env not found${NC}"
    exit 1
fi

# Create directories
mkdir -p logs data

# Check Redis
echo -e "\n${YELLOW}🔍 Checking Redis...${NC}"
if redis-cli ping > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Redis is running${NC}"
else
    echo -e "${RED}❌ Redis not running. Start it with: redis-server${NC}"
    exit 1
fi

# Cleanup old processes
echo -e "\n${YELLOW}🧹 Cleaning up old processes...${NC}"
pkill -f "rag_systems.production_server" || true
pkill -f "gunicorn.*server:app" || true
lsof -ti:8090 | xargs kill -9 2>/dev/null || true
lsof -ti:8080 | xargs kill -9 2>/dev/null || true
sleep 2

# Start RAG Server
echo -e "\n${BLUE}🤖 Starting RAG Server (port 8090)...${NC}"
gunicorn rag_systems.production_server:app \
    --workers 2 \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8090 \
    --timeout 120 \
    --access-logfile logs/rag_access.log \
    --error-logfile logs/rag_error.log \
    --daemon
sleep 5

# Check RAG health
echo -e "${YELLOW}🔍 Checking RAG health...${NC}"
if curl -s http://localhost:8090/health | grep -q "ok"; then
    echo -e "${GREEN}✅ RAG Server is healthy${NC}"
else
    echo -e "${RED}❌ RAG Server failed to start${NC}"
    echo -e "${YELLOW}Last 30 lines of error log:${NC}"
    tail -30 logs/rag_error.log
    exit 1
fi

# Start MCP Server
echo -e "\n${BLUE}🎯 Starting MCP Server (port 8080)...${NC}"
cd mcp
gunicorn server:app \
    --workers 4 \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8080 \
    --timeout 120 \
    --access-logfile ../logs/mcp_access.log \
    --error-logfile ../logs/mcp_error.log \
    --daemon
cd "$PROJECT_ROOT"
sleep 3

# Check MCP health
echo -e "${YELLOW}🔍 Checking MCP health...${NC}"
if curl -s http://localhost:8080/health | grep -q "ok"; then
    echo -e "${GREEN}✅ MCP Server is healthy${NC}"
else
    echo -e "${RED}❌ MCP Server failed to start${NC}"
    echo -e "${YELLOW}Last 30 lines of error log:${NC}"
    tail -30 logs/mcp_error.log
    exit 1
fi

# Show status
echo -e "\n${GREEN}╔═══════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║        ✅ All Services Started Successfully       ║${NC}"
echo -e "${GREEN}╚═══════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${BLUE}📊 Service Status:${NC}"
echo -e "  RAG Server:  ${GREEN}http://localhost:8090${NC}"
echo -e "  MCP Server:  ${GREEN}http://localhost:8080${NC}"
echo -e "  Redis:       ${GREEN}localhost:6379${NC}"
echo ""
echo -e "${BLUE}📁 Logs:${NC}"
echo -e "  RAG Access:  tail -f logs/rag_access.log"
echo -e "  RAG Error:   tail -f logs/rag_error.log"
echo -e "  MCP Access:  tail -f logs/mcp_access.log"
echo -e "  MCP Error:   tail -f logs/mcp_error.log"
echo ""
echo -e "${BLUE}🔧 Quick Commands:${NC}"
echo -e "  Status:      ./scripts/status.sh"
echo -e "  Stop:        ./scripts/stop_production.sh"
echo -e "  Restart:     ./scripts/stop_production.sh && ./scripts/start_production.sh"
echo ""
echo -e "${BLUE}📊 Test Endpoints:${NC}"
echo -e "  curl http://localhost:8090/health"
echo -e "  curl http://localhost:8080/health"
echo ""
