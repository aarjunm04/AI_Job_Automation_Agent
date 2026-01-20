#!/bin/bash

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}╔═══════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║        📊 Job Automation System Status          ║${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════════════╝${NC}"
echo ""

# Check Redis
echo -e "${YELLOW}Redis:${NC}"
if redis-cli ping > /dev/null 2>&1; then
    echo -e "  ${GREEN}✅ Running${NC}"
else
    echo -e "  ${RED}❌ Not running${NC}"
fi

# Check RAG Server
echo -e "\n${YELLOW}RAG Server (port 8090):${NC}"
if curl -s http://localhost:8090/health | grep -q "ok"; then
    echo -e "  ${GREEN}✅ Healthy${NC}"
    RAG_PID=$(lsof -ti:8090 | head -1)
    echo -e "  PID: $RAG_PID"
else
    echo -e "  ${RED}❌ Not responding${NC}"
fi

# Check MCP Server
echo -e "\n${YELLOW}MCP Server (port 8080):${NC}"
if curl -s http://localhost:8080/health | grep -q "ok"; then
    echo -e "  ${GREEN}✅ Healthy${NC}"
    MCP_PID=$(lsof -ti:8080 | head -1)
    echo -e "  PID: $MCP_PID"
else
    echo -e "  ${RED}❌ Not responding${NC}"
fi

# Show running processes
echo -e "\n${YELLOW}Gunicorn Processes:${NC}"
ps aux | grep gunicorn | grep -v grep | awk '{printf "  PID: %-6s CPU: %-4s MEM: %-4s CMD: %s\n", $2, $3"%", $4"%", $11" "$12" "$13}'

echo ""

