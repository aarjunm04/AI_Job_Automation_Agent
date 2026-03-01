#!/usr/bin/env bash
set -uo pipefail

ENV_FILE="${1:-$HOME/java.env}"
REPO="${2:-aarjunm04/AI_Job_Automation_Agent}"

ALLOWED_KEYS="XAI_API_KEY GROQ_API_KEY CEREBRAS_API_KEY SAMBANOVA_API_KEY NVIDIA_NIM_API_KEY NVIDIA_NIM_API_KEY_RAG PERPLEXITY_API_KEY GEMINI_API_KEY_RAG AGENTOPS_API_KEY SERPAPI_API_KEY_1 SERPAPI_API_KEY_2 SERPAPI_API_KEY_3 SERPAPI_API_KEY_4 JOOBLE_API_KEY SUPABASE_URL SUPABASE_API_URL SUPABASE_ANON_KEY SUPABASE_SERVICE_ROLE_KEY SUPABASE_KEY LOCAL_POSTGRES_URL LOCAL_POSTGRES_USER LOCAL_POSTGRES_PASSWORD LOCAL_POSTGRES_DB REDIS_URL REDIS_PASSWORD NOTION_API_KEY NOTION_APPLICATIONS_DB_ID NOTION_JOB_TRACKER_DB_ID RAG_SERVER_API_KEY JWT_SECRET LINKEDIN_EMAIL LINKEDIN_PASSWORD LINKEDIN_COOKIE_li_at LINKEDIN_COOKIE_JSESSIONID INDEED_EMAIL INDEED_PASSWORD GLASSDOOR_EMAIL GLASSDOOR_PASSWORD WELLFOUND_EMAIL WELLFOUND_PASSWORD ARCDEV_EMAIL ARCDEV_PASSWORD WEBSHARE_PROXY_1_1 WEBSHARE_PROXY_1_2 WEBSHARE_PROXY_1_3 WEBSHARE_PROXY_1_4 WEBSHARE_PROXY_1_5 WEBSHARE_PROXY_1_6 WEBSHARE_PROXY_1_7 WEBSHARE_PROXY_1_8 WEBSHARE_PROXY_1_9 WEBSHARE_PROXY_1_10 WEBSHARE_PROXY_2_1 WEBSHARE_PROXY_2_2 WEBSHARE_PROXY_2_3 WEBSHARE_PROXY_2_4 WEBSHARE_PROXY_2_5 WEBSHARE_PROXY_2_6 WEBSHARE_PROXY_2_7 WEBSHARE_PROXY_2_8 WEBSHARE_PROXY_2_9 WEBSHARE_PROXY_2_10 WENSHARE_PROXY_LIST_1 USERNAME USER_EMAIL USER_PHONE USER_LINKEDIN_URL USER_PORTFOLIO_URL USER_LOCATION"

command -v gh >/dev/null 2>&1 || { echo "gh CLI not found"; exit 1; }
test -f "$ENV_FILE"          || { echo "Env file not found: $ENV_FILE"; exit 1; }
chmod 600 "$ENV_FILE"

SUCCESS=0; SKIPPED=0; FAILED=0; FAILED_KEYS=""

is_allowed() {
    local k="$1"
    for allowed in $ALLOWED_KEYS; do
        [[ "$k" == "$allowed" ]] && return 0
    done
    return 1
}

while IFS= read -r line || [[ -n "$line" ]]; do
    [[ "$line" =~ ^[[:space:]]*# ]] && continue
    [[ -z "${line// }" ]] && continue

    key="${line%%=*}"
    value="${line#*=}"
    key="$(echo "$key" | tr -d '[:space:]')"
    value="${value%\"}" ; value="${value#\"}"
    value="${value%\'}" ; value="${value#\'}"

    [[ -z "$key" ]] && continue
    is_allowed "$key" || continue

    if [[ -z "$value" ]]; then
        echo "⚠️  EMPTY — skipping: $key"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    if printf '%s' "$value" | gh secret set "$key" --repo "$REPO" --body -; then
        echo "✅ SET: $key"
        SUCCESS=$((SUCCESS + 1))
    else
        echo "❌ FAILED: $key"
        FAILED_KEYS="$FAILED_KEYS $key"
        FAILED=$((FAILED + 1))
    fi

done < "$ENV_FILE"

echo ""
echo "============================================"
echo "✅ SUCCESS       : $SUCCESS"
echo "⚠️  SKIPPED(empty): $SKIPPED"
echo "❌ FAILED        : $FAILED"
[[ -n "$FAILED_KEYS" ]] && echo "Failed keys:$FAILED_KEYS"
echo "============================================"

