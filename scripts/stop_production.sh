#!/bin/bash

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${YELLOW}🛑 Stopping Job Automation Services...${NC}"
echo ""

# Kill processes
echo -e "${BLUE}Stopping RAG Server...${NC}"
pkill -f "rag_systems.production_server" || echo "  No RAG process found"
lsof -ti:8090 | xargs kill -9 2>/dev/null || true

echo -e "${BLUE}Stopping MCP Server...${NC}"
pkill -f "gunicorn.*server:app" || echo "  No MCP process found"
lsof -ti:8080 | xargs kill -9 2>/dev/null || true

sleep 2

# Verify
echo ""
echo -e "${YELLOW}🔍 Verifying shutdown...${NC}"
if lsof -i:8090 > /dev/null 2>&1; then
    echo -e "${RED}⚠️  Port 8090 still in use${NC}"
else
    echo -e "${GREEN}✅ Port 8090 is free${NC}"
fi

if lsof -i:8080 > /dev/null 2>&1; then
    echo -e "${RED}⚠️  Port 8080 still in use${NC}"
else
    echo -e "${GREEN}✅ Port 8080 is free${NC}"
fi

echo ""
echo -e "${GREEN}✅ All services stopped${NC}"
echo ""


