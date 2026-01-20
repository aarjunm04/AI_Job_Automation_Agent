#!/bin/bash

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Load API keys from narad.env
source ~/narad.env

# Use the correct key variable names
RAG_API_KEY="${RAG_KEY_DEV}"
MCP_API_KEY="${RAG_KEY_MCP}"

# Verify keys are loaded
if [ -z "$RAG_API_KEY" ]; then
    echo -e "${RED}❌ ERROR: RAG_KEY_DEV not found in narad.env${NC}"
    exit 1
fi

if [ -z "$MCP_API_KEY" ]; then
    echo -e "${RED}❌ ERROR: RAG_KEY_MCP not found in narad.env${NC}"
    exit 1
fi

# Test counters
PASSED=0
FAILED=0

# Helper functions
test_start() {
    echo -e "\n${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}🧪 TEST: $1${NC}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

test_pass() {
    echo -e "${GREEN}✅ PASS: $1${NC}"
    ((PASSED++))
}

test_fail() {
    echo -e "${RED}❌ FAIL: $1${NC}"
    echo -e "${YELLOW}   Response: $2${NC}"
    ((FAILED++))
}

# ============================================================================
# MAIN TEST SUITE
# ============================================================================

echo -e "${BLUE}╔════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║    🚀 Job Automation E2E Test Suite             ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════╝${NC}"

# ============================================================================
# 1. HEALTH CHECKS
# ============================================================================
test_start "Health Checks"

# Redis
if redis-cli ping | grep -q "PONG"; then
    test_pass "Redis is responsive"
else
    test_fail "Redis not responding" ""
fi

# RAG Health
RESPONSE=$(curl -s http://localhost:8090/health)
if echo "$RESPONSE" | grep -q "ok"; then
    test_pass "RAG Server health check"
else
    test_fail "RAG Server health check" "$RESPONSE"
fi

# MCP Health
RESPONSE=$(curl -s http://localhost:8080/health)
if echo "$RESPONSE" | grep -q "ok"; then
    test_pass "MCP Server health check"
else
    test_fail "MCP Server health check" "$RESPONSE"
fi

# ============================================================================
# 2. RAG SYSTEM TESTS
# ============================================================================
test_start "RAG System - Resume Selection"

# Test job description
JOB_DESC="Senior Machine Learning Engineer with 5+ years experience in Python, TensorFlow, PyTorch, and LLM fine-tuning. Experience with RAG systems, vector databases, and production ML deployment required."

# Select resume
RESPONSE=$(curl -s -X POST http://localhost:8090/resumes/select \
    -H "Content-Type: application/json" \
    -H "X-RAG-API-Key: ${RAG_API_KEY}" \
    -d "{
        \"job_text\": \"$JOB_DESC\"
    }")

if echo "$RESPONSE" | grep -q "top_resume_id"; then
    SELECTED=$(echo "$RESPONSE" | jq -r '.top_resume_id')
    CONFIDENCE=$(echo "$RESPONSE" | jq -r '.top_score')
    test_pass "Resume selection - Selected: $SELECTED (Confidence: $CONFIDENCE)"
    echo -e "${YELLOW}   Full response:${NC}"
    echo "$RESPONSE" | jq .
else
    test_fail "Resume selection" "$RESPONSE"
fi

# ============================================================================
# 3. RAG - List Available Resumes
# ============================================================================
test_start "RAG System - List Resumes"

RESPONSE=$(curl -s http://localhost:8090/resumes/list \
    -H "X-RAG-API-Key: ${RAG_API_KEY}")

if echo "$RESPONSE" | grep -q "resumes"; then
    COUNT=$(echo "$RESPONSE" | jq '.count')
    test_pass "List resumes - Found $COUNT resumes"
    echo -e "${YELLOW}   Available resumes:${NC}"
    echo "$RESPONSE" | jq -r '.resumes[].resume_id' | while read r; do
        echo "     - $r"
    done
else
    test_fail "List resumes" "$RESPONSE"
fi

# ============================================================================
# 4. RAG - Get Context
# ============================================================================
test_start "RAG System - Get Context"

RESPONSE=$(curl -s -X POST http://localhost:8090/rag/query \
    -H "Content-Type: application/json" \
    -H "X-RAG-API-Key: ${RAG_API_KEY}" \
    -d "{
        \"session_id\": \"test-context-$(date +%s)\",
        \"query\": \"What are your machine learning skills?\",
        \"top_k\": 3
    }")

if echo "$RESPONSE" | grep -q "selected_resume_id"; then
    RESUME=$(echo "$RESPONSE" | jq -r '.selected_resume_id')
    test_pass "RAG query - Selected resume: $RESUME"
else
    test_fail "RAG query" "$RESPONSE"
fi

# ============================================================================
# 5. MCP - Session Management
# ============================================================================
test_start "MCP System - Create Session"

SESSION_RESPONSE=$(curl -s -X POST http://localhost:8080/v1/sessions \
    -H "Content-Type: application/json" \
    -H "X-MCP-API-Key: ${MCP_API_KEY}" \
    -d "{
        \"owner\": \"e2e-test\",
        \"metadata\": {\"test\": true, \"timestamp\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"}
    }")

if echo "$SESSION_RESPONSE" | grep -q "session_id"; then
    SESSION_ID=$(echo "$SESSION_RESPONSE" | jq -r '.session_id')
    test_pass "Create session - ID: $SESSION_ID"
else
    test_fail "Create session" "$SESSION_RESPONSE"
    exit 1
fi

# ============================================================================
# 6. MCP - Add Context Items
# ============================================================================
test_start "MCP System - Add Context Items"

# Add job scraping context
RESPONSE=$(curl -s -X POST "http://localhost:8080/v1/sessions/$SESSION_ID/items" \
    -H "Content-Type: application/json" \
    -H "X-MCP-API-Key: ${MCP_API_KEY}" \
    -d "{
        \"role\": \"tool\",
        \"content\": \"Scraped 25 jobs from LinkedIn: ML Engineer positions\",
        \"metadata\": {
            \"source\": \"scraper\",
            \"job_count\": 25,
            \"platform\": \"linkedin\"
        }
    }")

if echo "$RESPONSE" | grep -q "item_id"; then
    ITEM_ID=$(echo "$RESPONSE" | jq -r '.item_id')
    test_pass "Add context item - ID: $ITEM_ID"
else
    test_fail "Add context item" "$RESPONSE"
fi

# Add user query
curl -s -X POST "http://localhost:8080/v1/sessions/$SESSION_ID/items" \
    -H "Content-Type: application/json" \
    -H "X-MCP-API-Key: ${MCP_API_KEY}" \
    -d "{
        \"role\": \"user\",
        \"content\": \"Find me ML Engineer jobs with LLM experience\",
        \"metadata\": {\"source\": \"user\"}
    }" > /dev/null

# Add assistant response
curl -s -X POST "http://localhost:8080/v1/sessions/$SESSION_ID/items" \
    -H "Content-Type: application/json" \
    -H "X-MCP-API-Key: ${MCP_API_KEY}" \
    -d "{
        \"role\": \"assistant\",
        \"content\": \"Found 5 matching positions. Best match: Senior ML Engineer at OpenAI\",
        \"metadata\": {\"source\": \"agent\"}
    }" > /dev/null

# ============================================================================
# 7. MCP - Retrieve Context
# ============================================================================
test_start "MCP System - Retrieve Context"

RESPONSE=$(curl -s "http://localhost:8080/v1/sessions/$SESSION_ID/items?last_n=10" \
    -H "X-MCP-API-Key: ${MCP_API_KEY}")

if echo "$RESPONSE" | grep -q "items"; then
    ITEM_COUNT=$(echo "$RESPONSE" | jq '.items | length')
    test_pass "Retrieve context - Got $ITEM_COUNT items"
    echo -e "${YELLOW}   Context items:${NC}"
    echo "$RESPONSE" | jq -r '.items[] | "     [\(.role)] \(.content[:60])..."'
else
    test_fail "Retrieve context" "$RESPONSE"
fi

# ============================================================================
# 8. MCP - Create Snapshot (Summarization)
# ============================================================================
test_start "MCP System - Create Snapshot"

RESPONSE=$(curl -s -X POST "http://localhost:8080/v1/sessions/$SESSION_ID/snapshot" \
    -H "Content-Type: application/json" \
    -H "X-MCP-API-Key: ${MCP_API_KEY}" \
    -d "{
        \"strategy\": \"rolling\",
        \"max_sentences\": 6
    }")

if echo "$RESPONSE" | grep -q "snapshot_id"; then
    SNAPSHOT_ID=$(echo "$RESPONSE" | jq -r '.snapshot_id')
    SUMMARY=$(echo "$RESPONSE" | jq -r '.summary_text')
    test_pass "Create snapshot - ID: $SNAPSHOT_ID"
    echo -e "${YELLOW}   Summary: $SUMMARY${NC}"
else
    test_fail "Create snapshot" "$RESPONSE"
fi

# ============================================================================
# 9. MCP - RAG Integration
# ============================================================================
test_start "MCP System - RAG Context Retrieval"

RESPONSE=$(curl -s "http://localhost:8080/v1/relevant/$SESSION_ID?top_k=5" \
    -H "X-MCP-API-Key: ${MCP_API_KEY}")

if echo "$RESPONSE" | grep -q "items"; then
    COUNT=$(echo "$RESPONSE" | jq '.items | length')
    test_pass "RAG context retrieval - Got $COUNT relevant items"
else
    test_fail "RAG context retrieval" "$RESPONSE"
fi

# ============================================================================
# 10. MCP - Metrics
# ============================================================================
test_start "MCP System - Metrics"

RESPONSE=$(curl -s http://localhost:8080/metrics)

if echo "$RESPONSE" | grep -q "sessions"; then
    test_pass "Metrics endpoint responding"
else
    test_fail "Metrics" "$RESPONSE"
fi

# ============================================================================
# 11. Integration - Full Workflow
# ============================================================================
test_start "Full Integration - Job Application Workflow"

echo -e "${YELLOW}   Simulating complete job application flow...${NC}"

# 1. Create new session
WORKFLOW_SESSION=$(curl -s -X POST http://localhost:8080/v1/sessions \
    -H "Content-Type: application/json" \
    -H "X-MCP-API-Key: ${MCP_API_KEY}" \
    -d '{"owner": "integration-test"}' | jq -r '.session_id')

# 2. Select best resume via RAG
RAG_RESULT=$(curl -s -X POST http://localhost:8090/rag/query \
    -H "Content-Type: application/json" \
    -H "X-RAG-API-Key: ${RAG_API_KEY}" \
    -d "{
        \"session_id\": \"$WORKFLOW_SESSION\",
        \"job_text\": \"$JOB_DESC\",
        \"top_k\": 3
    }")

SELECTED_RESUME=$(echo "$RAG_RESULT" | jq -r '.selected_resume_id')

# 3. Log the decision in MCP
curl -s -X POST "http://localhost:8080/v1/sessions/$WORKFLOW_SESSION/items" \
    -H "Content-Type: application/json" \
    -H "X-MCP-API-Key: ${MCP_API_KEY}" \
    -d "{
        \"role\": \"assistant\",
        \"content\": \"Selected resume: $SELECTED_RESUME for ML Engineer position\",
        \"metadata\": {\"source\": \"rag\", \"resume\": \"$SELECTED_RESUME\"}
    }" > /dev/null

# 4. Create snapshot of the workflow
WORKFLOW_SNAPSHOT=$(curl -s -X POST "http://localhost:8080/v1/sessions/$WORKFLOW_SESSION/snapshot" \
    -H "Content-Type: application/json" \
    -H "X-MCP-API-Key: ${MCP_API_KEY}" \
    -d '{"strategy": "rolling", "max_sentences": 5}')

if echo "$WORKFLOW_SNAPSHOT" | grep -q "snapshot_id"; then
    test_pass "Full workflow completed successfully"
    echo -e "${YELLOW}   Resume: $SELECTED_RESUME${NC}"
    echo -e "${YELLOW}   Session: $WORKFLOW_SESSION${NC}"
else
    test_fail "Full workflow" "$WORKFLOW_SNAPSHOT"
fi

# ============================================================================
# 12. Cleanup Test Session
# ============================================================================
test_start "Cleanup - Delete Test Session"

RESPONSE=$(curl -s -X DELETE "http://localhost:8080/v1/sessions/$SESSION_ID" \
    -H "X-MCP-API-Key: ${MCP_API_KEY}")

if echo "$RESPONSE" | grep -q "deleted.*true"; then
    test_pass "Session deletion"
else
    test_fail "Session deletion" "$RESPONSE"
fi

# ============================================================================
# SUMMARY
# ============================================================================
echo -e "\n${BLUE}╔════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                  TEST SUMMARY                      ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${GREEN}✅ Passed: $PASSED${NC}"
echo -e "${RED}❌ Failed: $FAILED${NC}"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}🎉 All tests passed!${NC}"
    exit 0
else
    echo -e "${RED}⚠️  Some tests failed. Check logs for details.${NC}"
    exit 1
fi
