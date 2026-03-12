# AI Job Application Agent - Autonomous Scheduler

## Setup Checklist

### GitHub Repository Secrets

Add these secrets in your GitHub repository settings (Settings → Secrets and variables → Actions):

| Secret Name | Description | Required |
|-------------|-------------|----------|
| `GROQ_API_KEY` | Groq API key (free tier) | ✅ |
| `CEREBRAS_API_KEY` | Cerebras API key (free tier) | ✅ |
| `XAI_API_KEY` | xAI Grok API key | ✅ |
| `PERPLEXITY_API_KEY` | Perplexity API key | ✅ |
| `SAMBANOVA_API_KEY` | SambaNova API key (free tier) | ⭕ |
| `NVIDIA_NIM_API_KEY` | NVIDIA NIM API key | ⭕ |
| `GEMINI_API_KEY` | Google Gemini API key | ⭕ |
| `AGENTOPS_API_KEY` | AgentOps tracking key | ⭕ |
| `LOCAL_POSTGRES_URL` | PostgreSQL connection URL | ✅ |
| `LOCAL_POSTGRES_PASSWORD` | PostgreSQL password | ✅ |
| `SCRAPER_SERVICE_API_KEY` | Internal API key for scraper | ⭕ |
| `SERPAPI_API_KEY_1` | SerpAPI key #1 (round-robin) | ✅ |
| `SERPAPI_API_KEY_2` | SerpAPI key #2 | ⭕ |
| `SERPAPI_API_KEY_3` | SerpAPI key #3 | ⭕ |
| `SERPAPI_API_KEY_4` | SerpAPI key #4 | ⭕ |
| `WEBSHARE_PROXY_LIST` | Webshare proxy URLs | ⭕ |
| `WEBSHARE_USERNAME_1` | Webshare account 1 username | ⭕ |
| `WEBSHARE_PASSWORD_1` | Webshare account 1 password | ⭕ |
| `WEBSHARE_USERNAME_2` | Webshare account 2 username | ⭕ |
| `WEBSHARE_PASSWORD_2` | Webshare account 2 password | ⭕ |
| `NOTION_API_KEY` | Notion integration token | ✅ |
| `NOTION_APPLICATIONS_DB_ID` | Notion Applications database ID | ✅ |
| `NOTION_JOB_TRACKER_DB_ID` | Notion Job Tracker database ID | ✅ |
| `NOTION_ALERTS_DB_ID` | Notion Alerts database ID | ✅ |
| `AUTO_APPLY_ENABLED` | Enable auto-apply ("true"/"false") | ⭕ |

### Cron Schedule Verification

The scheduler runs 3x/week at 09:00 IST (03:30 UTC):

```bash
# Verify cron schedule
# cron: '30 3 * * 1,3,5'
# = minute 30, hour 3, any day of month, any month, Monday(1)/Wednesday(3)/Friday(5)

# Test with crontab guru:
# https://crontab.guru/#30_3_*_*_1,3,5

# IST = UTC + 5:30
# 03:30 UTC = 09:00 IST ✅
```

### Manual Verification Commands

```bash
# 1. Test workflow syntax
gh workflow view scheduler.yml

# 2. Manually trigger a test run
gh workflow run scheduler.yml -f mode=dry-run -f dry_run=true

# 3. Check workflow status
gh run list --workflow=scheduler.yml

# 4. View workflow logs
gh run view --log
```

### Docker Services Health Check

All 7 services must be healthy before pipeline runs:

| Service | Port | Health Endpoint |
|---------|------|-----------------|
| postgres | 5432 | TCP check |
| redis | 6379 | TCP check |
| chromadb | 8000 | `/api/v1/heartbeat` |
| rag-server | 8090 | `/health` |
| api-server | 8080 | `/health` |
| playwright-scraper | 8001 | `/health` |
| playwright-apply | 8003 | `/health` |

### Budget Configuration

| Setting | Value | Description |
|---------|-------|-------------|
| `XAI_COST_CAP_PER_RUN` | $0.38 | Max xAI spend per run |
| `TOTAL_MONTHLY_BUDGET` | $10.00 | Hard monthly cap |
| **Abort Threshold** | $9.50 | Pipeline won't start if exceeded |

### LLM Fallback Chain (in order)

1. **Perplexity** (llama-3.1-sonar-large) — if budget allows
2. **xAI Grok** (grok-beta) — if 429, fallback
3. **Groq** (llama-3.3-70b-versatile) — **FREE**
4. **Cerebras** (llama3.1-70b) — **FREE**
5. **SambaNova** (Meta-Llama-3.1-70B-Instruct) — **FREE**

Budget < $0.50 remaining → skips paid providers (1, 2) automatically.

---

## Architecture Overview

### Pipeline Flow

```
main.py (entrypoint)
    │
    ├── SessionBootstrapper.boot()
    │   └── Verify 7 Docker services healthy (3x retries, 30s backoff)
    │
    ├── BudgetEnforcer.enforce()
    │   └── Check monthly spend < $9.50
    │
    └── PipelineRunner.run_full()
        │
        ├── 1. create_run_sessions (Postgres)
        ├── 2. ScraperAgent.run() → List[Job]
        ├── 3. AnalyserAgent.run(jobs) → List[JobScore]
        ├── 4. ApplyAgent.run(scores) → ApplyResult (skip if DRY_RUN)
        ├── 5. TrackerAgent.generate_report() → FinalReport
        │   └── NotionClient.post_run_report()
        └── 6. Close run_sessions (closed_at=NOW)
```

### SIGTERM Handling

Docker Compose sends SIGTERM → 15s → SIGKILL on `down`.

Pipeline handles this by:
1. Setting `_shutdown_requested = True`
2. Completing current Postgres write (must finish in <10s)
3. Closing run_session gracefully
4. Posting "interrupted" alert to Notion

### Self-Healing Patterns

1. **Service Health Loop**: Each service gets 3 attempts with 30s exponential backoff
2. **Budget Veto**: If projected cost > remaining budget, skip LLM-heavy steps
3. **LLM Exhaustion**: 5-provider fallback chain with auto-retry
4. **Notion Failures**: Swallowed and logged, never crash pipeline

---

## File Quick Reference

| File | Purpose |
|------|---------|
| [main.py](../main.py) | Pipeline entrypoint with SessionBootstrapper, BudgetEnforcer, SIGTERM |
| [.github/workflows/scheduler.yml](../.github/workflows/scheduler.yml) | 3x/week cron (Mon/Wed/Fri 09:00 IST) |
| [integrations/notion.py](../integrations/notion.py) | Async Notion client with run reports + alerts |
| [integrations/llm_interface.py](../integrations/llm_interface.py) | 5-provider fallback chain with async complete() |
| [agents/tracker_agent.py](../agents/tracker_agent.py) | Final reporter with generate_report() |
| [utils/normalise_dedupe.py](../utils/normalise_dedupe.py) | Levenshtein dedup engine |

---

## Troubleshooting

### Pipeline Won't Start

1. **Check secrets**: All required secrets must be set
2. **Check budget**: `python main.py --budget-check`
3. **Check services**: `python main.py --health-check`

### GitHub Actions Failing

```bash
# View recent runs
gh run list --workflow=scheduler.yml --limit 5

# Get logs from failed run
gh run view <run-id> --log-failed
```

### Notion Alerts Not Working

1. Verify `NOTION_ALERTS_DB_ID` is set
2. Database must have: Title, Level (select), Timestamp (date), Message (rich_text)
3. Check bot has access to the database

### Cost Running High

1. Check `--budget-check` output
2. Free providers (Groq, Cerebras) are used when budget < $0.50
3. Set `DRY_RUN=true` for testing without LLM costs