# ============================================================
# RAG SYSTEM TEST SUITE — Run after rebuild
# ============================================================

echo "============================================================"
echo " RAG TEST SUITE — $(date)"
echo "============================================================"

# ── Prerequisites ─────────────────────────────────────────────
export RAG_URL="http://localhost:8090"
export H1="Content-Type: application/json"
export H2="X-RAG-API-Key: ${RAG_API_KEY}"
PASS=0; FAIL=0

# ── Helper ────────────────────────────────────────────────────
check() {
  local label="$1" expected="$2" actual="$3"
  if [ "$actual" = "$expected" ]; then
    echo "  ✅ PASS — $label"
    PASS=$((PASS+1))
  else
    echo "  ❌ FAIL — $label | expected: $expected | got: $actual"
    FAIL=$((FAIL+1))
  fi
}

# ============================================================
# TEST 0 — Health check
# ============================================================
echo ""
echo "── TEST 0: Health Check ─────────────────────────────────"
HEALTH=$(curl -s "$RAG_URL/health" | jq -r '.status // .healthy // "unknown"')
echo "  Health response: $HEALTH"
check "server alive" "ok" "$HEALTH"

# ============================================================
# TEST 1 — ML Engineer → resume_ml_data_engineer
# ============================================================
echo ""
echo "── TEST 1: ML Engineer ──────────────────────────────────"
R1=$(curl -s -X POST "$RAG_URL/rag/query" \
  -H "$H1" -H "$H2" \
  -d '{"job_description":"ML Engineer Python TensorFlow PyTorch MLflow Kubernetes model training pipelines remote","session_id":"t1","top_k":1}')
echo "$R1" | jq '.'
check "success=true"             "true"                    "$(echo $R1 | jq -r '.success')"
check "resume=ml_data_engineer"  "resume_ml_data_engineer" "$(echo $R1 | jq -r '.chunks[0].metadata.resume_id')"

# ============================================================
# TEST 2 — Data Scientist → resume_data_science
# ============================================================
echo ""
echo "── TEST 2: Data Scientist ───────────────────────────────"
R2=$(curl -s -X POST "$RAG_URL/rag/query" \
  -H "$H1" -H "$H2" \
  -d '{"job_description":"Data Scientist SQL Tableau Power BI A/B testing hypothesis testing business analytics dashboards","session_id":"t2","top_k":1}')
echo "$R2" | jq '.'
check "success=true"           "true"                  "$(echo $R2 | jq -r '.success')"
check "resume=data_science"    "resume_data_science"   "$(echo $R2 | jq -r '.chunks[0].metadata.resume_id')"

# ============================================================
# TEST 3 — AI Automation → resume_ai_automation
# ============================================================
echo ""
echo "── TEST 3: AI Automation ────────────────────────────────"
R3=$(curl -s -X POST "$RAG_URL/rag/query" \
  -H "$H1" -H "$H2" \
  -d '{"job_description":"AI Automation Engineer n8n Make.com CrewAI LangChain RPA Playwright workflow orchestration API integration","session_id":"t3","top_k":1}')
echo "$R3" | jq '.'
check "success=true"             "true"                    "$(echo $R3 | jq -r '.success')"
check "resume=ai_automation"     "resume_ai_automation"    "$(echo $R3 | jq -r '.chunks[0].metadata.resume_id')"

# ============================================================
# TEST 4 — RAG Engineer → resume_rag_engineer
# ============================================================
echo ""
echo "── TEST 4: RAG Engineer ─────────────────────────────────"
R4=$(curl -s -X POST "$RAG_URL/rag/query" \
  -H "$H1" -H "$H2" \
  -d '{"job_description":"RAG Engineer vector databases ChromaDB Pinecone FAISS semantic search LangChain retrieval pipelines embedding optimization","session_id":"t4","top_k":1}')
echo "$R4" | jq '.'
check "success=true"           "true"                  "$(echo $R4 | jq -r '.success')"
check "resume=rag_engineer"    "resume_rag_engineer"   "$(echo $R4 | jq -r '.chunks[0].metadata.resume_id')"

# ============================================================
# TEST 5 — Data Engineer → resume_data_engineering
# ============================================================
echo ""
echo "── TEST 5: Data Engineer ────────────────────────────────"
R5=$(curl -s -X POST "$RAG_URL/rag/query" \
  -H "$H1" -H "$H2" \
  -d '{"job_description":"Data Engineer Apache Airflow PySpark Kafka Snowflake Redshift ETL pipelines DBT star schema data warehousing","session_id":"t5","top_k":1}')
echo "$R5" | jq '.'
check "success=true"               "true"                      "$(echo $R5 | jq -r '.success')"
check "resume=data_engineering"    "resume_data_engineering"   "$(echo $R5 | jq -r '.chunks[0].metadata.resume_id')"

# ============================================================
# TEST 6 — GenAI Developer → resume_llm_genai
# ============================================================
echo ""
echo "── TEST 6: GenAI Developer ──────────────────────────────"
R6=$(curl -s -X POST "$RAG_URL/rag/query" \
  -H "$H1" -H "$H2" \
  -d '{"job_description":"GenAI Developer GPT-4 Claude API LangChain autonomous agents chatbots LLM integration conversational AI production deployment","session_id":"t6","top_k":1}')
echo "$R6" | jq '.'
check "success=true"          "true"               "$(echo $R6 | jq -r '.success')"
check "resume=llm_genai"      "resume_llm_genai"   "$(echo $R6 | jq -r '.chunks[0].metadata.resume_id')"

# ============================================================
# TEST 7 — Wrong field name (backwards compat — must not 500)
# ============================================================
echo ""
echo "── TEST 7: Legacy field 'query' backwards compat ────────"
R7=$(curl -s -X POST "$RAG_URL/rag/query" \
  -H "$H1" -H "$H2" \
  -d '{"query":"Python Backend Engineer FastAPI REST microservices Docker","session_id":"t7","top_k":1}')
check "no 500 on query field"  "true" "$(echo $R7 | jq -r '.success')"

# ============================================================
# TEST 8 — Empty job_description (must return 400 not 500)
# ============================================================
echo ""
echo "── TEST 8: Empty string guard ───────────────────────────"
R8=$(curl -s -X POST "$RAG_URL/rag/query" \
  -H "$H1" -H "$H2" \
  -d '{"job_description":"","session_id":"t8","top_k":1}')
CODE8=$(echo $R8 | jq -r '.error_code // "none"')
check "empty string = HTTP_400"  "HTTP_400"  "$CODE8"

# ============================================================
# TEST 9 — NIM provider confirm (not Gemini fallback)
# ============================================================
echo ""
echo "── TEST 9: Confirm NIM active (not Gemini fallback) ─────"
NIM_LOG=$(docker logs ai_rag_server 2>&1 | grep "integrate.api.nvidia.com" | tail -3)
GEMINI_ERR=$(docker logs ai_rag_server 2>&1 | grep "Gemini.*404\|v1beta" | tail -3)
echo "  NIM calls found:    $(echo "$NIM_LOG" | grep -c "200 OK") × 200 OK"
echo "  Gemini 404 errors:  $(echo "$GEMINI_ERR" | wc -l | tr -d ' ')"
[ -z "$GEMINI_ERR" ] && { echo "  ✅ PASS — No Gemini 404s in logs"; PASS=$((PASS+1)); } \
  || { echo "  ❌ FAIL — Gemini 404s still present"; FAIL=$((FAIL+1)); }

# ============================================================
# FINAL SCORE
# ============================================================
echo ""
echo "============================================================"
echo " RESULTS: $PASS passed | $FAIL failed"
echo "============================================================"
[ $FAIL -eq 0 ] && echo " 🟢 RAG IS PRODUCTION READY" || echo " 🔴 FAILURES DETECTED — DO NOT PROCEED"
echo "============================================================"