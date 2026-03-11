# IDE_README — AI Job Application Agent
<!-- PRIMARY IDE COORDINATION DOCUMENT | ALWAYS READ BEFORE ANY CODE CHANGE | ALWAYS UPDATE CHANGE LOG AFTER ANY CHANGE -->
<!-- GENERATED: 2026-03-01 | HEAD_DEV: PERPLEXITY_PRO | REPO: github.com/aarjunm04/AIJobAutomationAgent -->

```PROJECT_META
REPO         : github.com/aarjunm04/AIJobAutomationAgent
PROJECT      : AI Job Application Agent
PHASE        : 1 (Phase 2: Apr 2 – May 14 2026)
SPRINT       : 2W | START: 2026-03-01 | TARGET: 2026-03-14
STACK        : Python3.11 | CrewAI | Playwright | PostgreSQL | ChromaDB | Redis | FastAPI | Docker | AgentOps
SECRETS      : ~/java.env (single source of truth, git-ignored, loaded via --env-file)
DB_DEV       : Local Docker Postgres (switch via ACTIVEDB=local)
DB_PROD      : Supabase PostgreSQL (switch via ACTIVEDB=supabase)
RUN_SCHEDULE : Mon/Thu/Sat 12:00 PM IST | cron: 0 6 1,4,6 * *
BUDGET_HARD  : $10/mo total | xAI: $5/mo | Perplexity: $5/mo | xAI per-run cap: $0.38
HEAD_DEV     : PERPLEXITY_PRO (orchestrator, architect, prompt generator)
PHASE1_LAUNCH: 2026-03-14
PHASE2_DONE  : 2026-05-14
```

## SYSTEM OVERVIEW

- Fully autonomous job application ecosystem that discovers, scores, and applies to 150 jobs per run across 10 curated platforms, 3 times per week (Mon/Thu/Sat 12:00 PM IST), targeting 300 quality applications per month with minimal human intervention
- 5 CrewAI agents in hierarchical process: Master, Scraper, Analyser, Apply, Tracker
- Target: 150 jobs/run, 300 apps/month, 50-70% auto-apply, 30-50% manual queue
- Budget discipline: $10/mo hard cap, $0.38/run xAI cap enforced in code
- Data layer: Postgres (SOR), ChromaDB (vectors), Redis (cache/queue P2), Notion (user UI)
- Scheduled via GitHub Actions cron, runs on local Docker, solo-dev maintained <5hrs/week

## AGENT SYSTEM

| AGENT | ROLE | PRIMARY_LLM | FALLBACK_1 | FALLBACK_2 | TIER |
|-------|------|-------------|------------|------------|------|
| Master Agent | CrewAI crew manager, session boot, run lifecycle, budget gate, dynamic routing, error handling | Groq llama-3.3-70b-versatile | Cerebras llama-3.3-70b | — | Free |
| Scraper Agent | Job discovery all 10 platforms, normalisation, dedup, metadata fill via web search | Perplexity sonar | — | — | Paid ($5/mo) |
| Analyser Agent | Eligibility filter, fit scoring (0.0–1.0), RAG resume match, routing decision | xAI grok-4-fast-reasoning | SambaNova Llama-3.1-70B | Cerebras llama-3.3-70b | Paid ($5/mo shared xAI) |
| Apply Agent | Playwright form reasoning, ATS detection, form fill, submission, proof capture, retry | xAI grok-4-1-fast-reasoning | SambaNova Llama-3.1-70B | Cerebras llama-3.3-70b | Paid ($5/mo shared xAI) |
| Tracker Agent | Postgres audit logging, Notion sync, AgentOps run summary | Groq llama-3.3-70b-versatile | Cerebras llama-3.3-70b | — | Free |
| Developer Agent (Phase 2) | Cross-session AgentOps trace analysis, improvement recommendations to Postgres developernotes (suggest-only, never auto-changes prod) | xAI grok-3-mini-latest | Perplexity sonar | — | Paid (shared xAI/Perplexity) |

```text
AGENT_FLOW   : Master → delegates → Scraper → Master → Analyser → Master → Apply → Master → Tracker → Master
HANDOFF      : All agent-to-agent state via Postgres tables ONLY. No in-memory passing.
BUDGET_GATE  : Master checks xAI spend after EVERY Apply Agent call. Abort run gracefully if > $0.38.

## Chrome Extension (Manual Apply)

- Manifest V3, Chrome only (v1), lives in `extension/` directory
- Communicates EXCLUSIVELY via FastAPI slim server — no direct DB or agent access
- No automated submission — user always clicks Submit

### Extension Files

| FILE | PURPOSE |
|------|---------|
| manifest.json | MV3 manifest — permissions: activeTab, storage, scripting, notifications |
| popup/popup.html | Main panel — fit score card, resume suggestion, talking points accordion, autofill button, mark-as-applied button, queue badge |
| popup/popup.js | Button event handling, background message passing, data rendering |
| popup/popup.css | Dark-mode UI |
| content_scripts/content.js | DOM scanner — detects input[type=text/email], select, textarea, file inputs; extracts field names/IDs/labels; sends detected_fields to popup |
| background/service_worker.js | All fetch() calls to FastAPI (auth header with FASTAPI_API_KEY from storage), message routing between content script and popup |
| utils/api_client.js | Shared callMatch, callAutofill, callLogApplication with retry (max 2) and error handling |
| utils/dom_detector.js | Shared DOM parsing, field type classification, React input detection |

### 5-Step User Flow

1. User opens Notion Applications DB → reviews Queued jobs by Priority → opens job URL in Chrome
2. Extension auto-calls /match on page load → shows fit score, reasoning, resume suggestion, talking points, autofill readiness
3. User clicks Autofill → extension injects name/email/phone/LinkedIn/years_of_experience into standard fields
4. User completes custom questions using talking points, uploads suggested resume, submits
5. User clicks Mark as Applied → extension calls /log-application → FastAPI writes applied_manual to Postgres → Tracker Agent → Notion Job Tracker DB sync

### Config

- FASTAPI_HOST and FASTAPI_API_KEY stored in chrome.storage.sync (set in extension options)
- Default host: http://localhost:8000
```

## PLATFORM CONFIGURATION

| PLATFORM | METHOD | TYPE | PHASE | RATE_LIMIT | NOTES |
|----------|--------|------|-------|------------|-------|
| LinkedIn | JobSpy library | Primary | 1+2 | Respect JobSpy defaults | No direct scrape, library wrapper |
| Indeed | JobSpy library | Primary | 1+2 | Respect JobSpy defaults | No direct scrape, library wrapper |
| Wellfound | Playwright headless | Primary | 1+2 | 1 req/3s | Stealth mode, login session required |
| RemoteOK | REST API (public JSON) | Primary | 1+2 | 1 req/5s | No auth, paginated |
| We Work Remotely (WWR) | Playwright headless | Primary | 1+2 | 1 req/3s | RSS alternative available |
| YC Work at a Startup | Playwright headless | Primary | 1+2 | 1 req/3s | Filter: eng/remote only |
| Himalayas | REST API (public JSON) | Primary | 1+2 | 1 req/3s | No auth required |
| Turing | Playwright headless | Primary | 1+2 | 1 req/5s | Requires stealth |
| Crossover | Playwright headless | Primary | 1+2 | 1 req/5s | High CAPTCHA risk, proxy essential |
| Arc.dev | Playwright headless | Primary | 1+2 | 1 req/3s | Remote-only filter |
| Nodesk | Playwright headless | Safety-net | 1+2 | 1 req/3s | Activates only if jobs < 100 after primary platforms |
| Toptal | Playwright headless | Safety-net | 1+2 | 1 req/5s | Activates only if jobs < 100 after primary platforms |
| Remotive | REST API | Phase 2 | Phase 2 | Per API docs | Free REST feed |
| Jooble | REST API | Phase 2 | Phase 2 | 500 req/day | API key auth, POST with filters |
| SerpAPI Google Jobs | REST API | Supplementary | 1+2 | 1000 credits/mo (4 accounts x 250) | NOT an LLM — Google Jobs discovery only |

```text
PROXY        : Webshare static proxies — 20 total (2 accounts x 10, round-robin rotation)
PROXY_BW     : 1 GB/account/month (2 GB total)
PROXY_FORMAT : http://user:pass@host:port
```

## DATA LAYER

### Postgres Tables

| TABLE | PURPOSE | KEY_COLUMNS |
|-------|---------|------------|
| jobs | All discovered jobs (SOR for all agents) | id(UUID), job_url(unique), title, company, platform, fit_score, route(auto/manual/skip), resume_used, run_id, scraped_at, status |
| applications | All application outcomes | id(UUID), job_id(FK), status(applied/failed/manual_queued), proof_json, proof_confidence, notion_synced, applied_at, resume_used |
| run_sessions | Per-run health snapshot | id(UUID), run_id, jobs_discovered, jobs_auto_applied, jobs_manual_queued, jobs_skipped, jobs_failed, xai_cost_usd, total_cost_usd, duration_minutes, safety_net_triggered, fallback_events, errors, started_at, closed_at |
| audit_logs | Immutable append-only event log | id(UUID), run_id, agent, event_type, error_code, metadata_json, created_at |
| developer_notes | Phase 2 Developer Agent improvement suggestions (read-only for production pipeline) | id(UUID), run_ids_analysed, pattern, recommendation, priority, status(pending_review/reviewed/implemented), created_at |

### ChromaDB

```text
PURPOSE      : Local vector store for RAG resume matching
COLLECTIONS  : resumes (15 variants), job_descriptions (per-run embeddings)
EMBEDDINGS   : NVIDIA NIM nv-embedqa-e5-v5 (1024 dims, primary) | Gemini text-embedding-004 (768 dims, fallback)
VOLUME       : Docker named volume chromadb_data — persistent across container restarts
RESUME_FILES : AarjunGen.pdf (general), AarjunBase.pdf (original/all-round), + 13 domain-specific variants
```

### Redis

```text
PHASE 1      : Response caching only
PHASE 2      : Full job queue (queue:auto_apply, queue:manual), session state checkpoints, embedding cache
URL          : redis://localhost:6379 (Docker container)
```

### Notion

```text
ROLE         : User-facing UI only — NOT system of record
DB1          : Job Tracker DB (applied pipeline) — columns: Job Title, Company, Job URL, Stage, Date Applied, Platform, Applied Via, CTC, Notes, Job Type, Location, Resume Used
DB2          : Applications DB (manual queue) — columns: Job Title, Company, Job URL, Application Deadline, Platform, Status, CTC, Priority, Fit Score, Job Type, Location, Notes, Resume Suggested
ACCESS       : Tracker Agent + FastAPI /log-application endpoint
```

## SCORING AND ROUTING RULES

```text
SCALE        : 0.0 – 1.0 (ChromaDB cosine similarity + LLM eligibility composite)
SKIP         : fit_score < 0.40  → discard, no Notion entry
LOW_CONF     : fit_score 0.40–0.49 → manual queue ONLY, never auto-apply
MID          : fit_score 0.50–0.74 → eligible for routing (auto or manual by form complexity)
HIGH         : fit_score ≥ 0.75 → high priority, routed by form complexity
FORCE_MANUAL : fit_score ≥ 0.90 → ALWAYS manual queue (high stakes, regardless of form complexity)
AUTO_APPLY   : 50–70% of eligible jobs per run (routing based on form complexity + platform ToS, NOT score)
MANUAL_QUEUE : 30–50% of eligible jobs per run
```

## FASTAPI ENDPOINTS

```text
BASE_URL     : http://localhost:8000
ROLE         : Chrome Extension HTTP boundary + RAG proxy + manual apply logger
ENDPOINTS    : 3 only — no others
```

| ENDPOINT | METHOD | CALLED_BY | PURPOSE |
|----------|--------|-----------|---------|
| /match | POST | Chrome Extension | RAG resume match for current job page — returns resume_suggested, similarity_score, fit_score, match_reasoning, talking_points |
| /autofill | POST | Chrome Extension | Returns field-value map for detected form fields from java.env user profile |
| /log-application | POST | Chrome Extension | Logs manual application to Postgres + triggers Notion Job Tracker sync |

## NARAD.ENV KEYS

| KEY | USED_BY | PHASE | NOTES |
|-----|---------|-------|-------|
| GROQ_API_KEY | LLM APIs | 1+2 | Groq llama-3.3-70b-versatile for Master & Tracker agents |
| CEREBRAS_API_KEY | LLM APIs | 1+2 | Cerebras llama-3.3-70b fallback for Master/Tracker/Analyser/Apply |
| XAI_API_KEY | LLM APIs | 1+2 | xAI grok-4-fast-reasoning and grok-4-1-fast-reasoning for Analyser/Apply |
| PERPLEXITY_API_KEY | LLM APIs | 1+2 | Perplexity sonar for Scraper + Developer Agent fallback |
| SAMBANOVA_API_KEY | LLM APIs | 1+2 | SambaNova Llama-3.1-70B fallback for Analyser/Apply |
| NVIDIA_NIM_API_KEY | LLM APIs | 1+2 | NVIDIA NIM nv-embedqa-e5-v5 embeddings for ChromaDB |
| GEMINI_API_KEY | LLM APIs | 1+2 | Gemini text-embedding-004 fallback embeddings |
| AGENTOPS_API_KEY | LLM APIs | 1+2 | AgentOps tracing and monitoring for all agents |
| SUPABASE_URL | Database | 1+2 | Supabase Postgres URL for production DB_PROD |
| SUPABASE_KEY | Database | 1+2 | Supabase service role key |
| LOCAL_POSTGRES_URL | Database | 1+2 | Default: postgresql://user:password@localhost:5432/job_agent |
| ACTIVE_DB | Database | 1+2 | Values: local \| supabase |
| REDIS_URL | Database | 1+2 | Default: redis://localhost:6379 |
| CHROMADB_PATH | Database | 1+2 | Default: app/chromadb |
| SERPAPI_KEY_1 | Scraping | 1+2 | SerpAPI account 1 key |
| SERPAPI_KEY_2 | Scraping | 1+2 | SerpAPI account 2 key |
| SERPAPI_KEY_3 | Scraping | 1+2 | SerpAPI account 3 key |
| SERPAPI_KEY_4 | Scraping | 1+2 | SerpAPI account 4 key |
| JOOBLE_API_KEY | Scraping | 2 | Jooble REST API key (Phase 2) |
| WEBSHARE_PROXY_LIST | Proxies | 1+2 | Comma-separated http://user:pass@host:port list |
| WEBSHARE_USERNAME_1 | Proxies | 1+2 | Webshare account 1 username |
| WEBSHARE_PASSWORD_1 | Proxies | 1+2 | Webshare account 1 password |
| WEBSHARE_USERNAME_2 | Proxies | 1+2 | Webshare account 2 username |
| WEBSHARE_PASSWORD_2 | Proxies | 1+2 | Webshare account 2 password |
| NOTION_API_KEY | Notion | 1+2 | Notion integration token |
| NOTION_APPLICATIONS_DB_ID | Notion | 1+2 | Notion Applications DB ID |
| NOTION_JOB_TRACKER_DB_ID | Notion | 1+2 | Notion Job Tracker DB ID |
| RUN_SCHEDULE | System Config | 1+2 | Default: 0 6 1,4,6 |
| XAI_COST_CAP_PER_RUN | System Config | 1+2 | Default: 0.38 |
| TOTAL_MONTHLY_BUDGET | System Config | 1+2 | Default: 10.00 |
| JOBS_PER_RUN_TARGET | System Config | 1+2 | Default: 150 |
| JOBS_PER_RUN_MINIMUM | System Config | 1+2 | Default: 100 |
| MAX_PLAYWRIGHT_SESSIONS | System Config | 1+2 | Default: 5 |
| AUTO_APPLY_ENABLED | System Config | 1+2 | Default: true |
| RESUME_DIR | System Config | 1+2 | Default: app/resumes |
| DEFAULT_RESUME | System Config | 1+2 | Default: AarjunGen.pdf |
| LOG_LEVEL | System Config | 1+2 | Default: INFO |
| FASTAPI_PORT | System Config | 1+2 | Default: 8000 |
| FASTAPI_HOST | System Config | 1+2 | Default: 0.0.0.0 |
| DRY_RUN | System Config | 1+2 | Default: false — set true for testing, no real submissions |
| USERNAME | User Profile | 1+2 | Candidate full name |
| USER_EMAIL | User Profile | 1+2 | Candidate email address |
| USER_PHONE | User Profile | 1+2 | Candidate phone number |
| USER_LINKEDIN_URL | User Profile | 1+2 | Public LinkedIn profile URL |
| USER_PORTFOLIO_URL | User Profile | 1+2 | Portfolio or GitHub URL |
| USER_LOCATION | User Profile | 1+2 | Primary location string |
| USER_ACCEPTED_JOB_TYPES | User Profile | 1+2 | Default: full-time,contract |
| USER_ACCEPTED_LOCATIONS | User Profile | 1+2 | Default: Remote,India |
| USER_YEARS_EXPERIENCE | User Profile | 1+2 | Numeric years of experience |

## DOCKER SERVICES

| SERVICE | CONTAINER | IMAGE | PORT | VOLUME | DEPENDS_ON |
|---------|-----------|-------|------|--------|------------|
| postgres | job_agent_postgres | postgres:16-alpine | 5432 | postgres_data | — |
| redis | job_agent_redis | redis:7-alpine | 6379 | redis_data | — |
| chromadb | job_agent_chromadb | chromadb/chroma:latest | 8001 | chromadb_data | — |
| fastapi | job_agent_fastapi | custom (.services/fastapi/) | 8000 | resumes_data(ro), chromadb_data | postgres, redis, chromadb |
| agent_runner | job_agent_runner | custom (.services/agents/) | — | resumes_data(ro), chromadb_data | postgres, redis, chromadb, fastapi |

```text
BOOT_CMD     : docker-compose --env-file ~/java.env up -d
RUN_CMD      : docker-compose --env-file ~/java.env run --rm agent_runner python main.py
NETWORK      : Single bridge network job_agent_network — all services internal
```

## REPO STRUCTURE

```text
AIJobAutomationAgent/
├── main.py                      # Entrypoint — boots CrewAI crew, starts run session
├── IDE_README.md                # This file — IDE coordination document
├── docker-compose.yml           # Full stack definition
├── java.env.template           # All keys with empty values — safe to commit
├── .gitignore                   # Includes java.env, __pycache__, .env, *.pyc
├── requirements.txt             # Top-level consolidated deps
├── agents/
│   ├── master_agent.py          # CrewAI crew manager, run lifecycle, budget gate
│   ├── scraper_agent.py         # All platform scraping tools, normalisation
│   ├── analyser_agent.py        # Scoring, RAG match, routing decision
│   ├── apply_agent.py           # Playwright form scripts, ATS detection, retry
│   ├── tracker_agent.py         # Postgres logging, Notion sync, AgentOps report
│   └── developer_agent.py       # Phase 2 only — trace analysis, recommendations
├── tools/
│   ├── scraper_tools.py         # JobSpy, Playwright, REST API, SerpAPI tool functions
│   ├── rag_tools.py             # ChromaDB query, embed, resume match tools
│   ├── apply_tools.py           # Per-platform Playwright apply scripts
│   ├── tracker_tools.py         # Postgres write, Notion sync, AgentOps tools
│   └── budget_tools.py          # xAI cost tracking, cap enforcement
├── db/
│   ├── schema.sql               # Full Postgres schema — all 5 tables
│   ├── migrations/              # Versioned migration scripts
│   └── init.sql                 # Docker Postgres init (dev only)
├── rag/
│   ├── embedder.py              # NVIDIA NIM primary + Gemini fallback embedding service
│   ├── ingestion.py             # Resume PDF ingestion pipeline — reads resumes/, writes ChromaDB
│   └── query.py                 # RAG query service — similarity search, resume selection
├── resumes/                     # Resume PDFs (git-ignored — mount as Docker volume)
│   ├── AarjunGen.pdf
│   ├── AarjunBase.pdf
│   └── [13 domain-specific variants].pdf
├── services/
│   ├── fastapi/
│   │   ├── Dockerfile
│   │   ├── main.py              # FastAPI app — 3 endpoints only
│   │   ├── routes/
│   │   │   ├── match.py
│   │   │   ├── autofill.py
│   │   │   └── log_application.py
│   │   └── requirements.txt
│   └── agents/
│       └── Dockerfile
├── extension/                   # Chrome Extension v1
│   ├── manifest.json
│   ├── background.js
│   ├── content.js
│   └── popup/
├── config/
│   └── platforms.json           # Per-platform rate limits, compliance flags, selectors config
├── tests/
│   ├── unit/
│   └── integration/
└── .github/
    └── workflows/
        ├── run_pipeline.yml     # Cron trigger Mon/Thu/Sat 12:00 PM IST
        ├── ci.yml               # Lint, type check, unit tests on every push
        └── resume_sync.yml      # Manual dispatch — re-embeds all resumes
```

## IDE ECOSYSTEM

### Role Hierarchy

```text
HEAD         : PERPLEXITY_PRO — orchestrator, architect, deep research, base file generation, all IDE prompt authoring
PRIMARY      : GITHUB_COPILOT_PRO — heavy implementation, full method bodies, business logic, test suites, complex Playwright scripts
SECONDARY    : OPENAI_CODEX — bug fixes, single-function deltas, error handling, feature additions, schema patches
GRUNT        : GOOGLE_GEMINI_ASSIST — docstrings, type hints, formatting, config edits, boilerplate, simple refactors
```

### Task Delegation Rules

| TASK_TYPE | ASSIGNED_IDE | NOTES |
|-----------|--------------|-------|
| New file creation (full structure) | PERPLEXITY → CLAUDE | Perplexity generates base scaffold + writes Copilot prompt. Copilot completes all implementations. |
| Complex multi-function implementation | CLAUDE | Full prod-ready code, no stubs, complete error handling |
| Bug fix (single function) | CODEX | Precise scope, preserve function signature |
| Feature delta (modify existing function) | CODEX | Targeted change only, no side effects |
| Docstrings / type hints | GEMINI | Google-style docstrings, PEP8 type hints |
| Config file edits | GEMINI | java.env.template, platforms.json, requirements.txt |
| Architecture decision | PERPLEXITY only | Never delegate arch decisions to IDEs |
| Prompt generation for other IDEs | PERPLEXITY only | All IDE prompts written and issued by Perplexity |

### Session Start Protocol (ALL IDEs)

```text
STEP 1: Read IDE_README.md — full file, not a skim
STEP 2: Read CHANGE_LOG (last 10 rows) — understand what was last done and by whom
STEP 3: State: "Last log: L{NNN} | {AUTHOR} | {FILE} | {CHANGE_TYPE} | {NOTES}"
STEP 4: State current sprint week and active task before writing any code
STEP 5: After completing any change — APPEND row to CHANGE_LOG before closing
```

## CHANGE_LOG

<!-- RULES:
  - APPEND ONLY — never edit existing rows
  - Every code change (create/modify/fix/delete) requires an entry
  - LOG_ID: increment from last row (L001, L002, ...)
  - TIMESTAMP: ISO8601 with IST offset (+05:30)
  - AUTHOR: PERPLEXITY | CLAUDE | CODEX | GEMINI
  - CHANGE_TYPE: INIT | CREATE | MODIFY | BUGFIX | REFACTOR | CONFIG | TEST | DELETE
  - NOTES: snake_case, max 80 chars, format = verb:target:detail
  - STATUS: DONE | IN_PROGRESS | BLOCKED | REVIEW
-->

| LOG_ID | TIMESTAMP | AUTHOR | FILE_PATH | CHANGE_TYPE | NOTES | STATUS |
|--------|-----------|--------|-----------|-------------|-------|--------|
| L001 | 2026-03-01T00:00:00+05:30 | PERPLEXITY | IDE_README.md | INIT | create:ide_readme:full_project_context_and_ide_coordination_doc | DONE |
| L002 | 2026-03-01T00:00:00+05:30 | PERPLEXITY | IDE_README.md | MODIFY | fix:structure:remove_prompt_labels_fix_header_fix_overview_bullet | DONE |
| L003 | 2026-02-28T11:13:07+05:30 | PERPLEXITY | PROJECT_STATUS.md | CREATE | create:project_status:live_build_snapshot_all_sections | DONE |
| L004 | 2026-02-28T10:15:00+05:30 | CODEX | PROJECT_STATUS.md | CREATE | create:project_status:initial_snapshot_from_repo | DONE |
| L003 | 2026-02-28T17:30:00+05:30 | PERPLEXITY | rag_systems/rag_pipeline.py | MODIFY | refactor:embedder:nvidia_nim_primary_gemini_fallback_remove_local | DONE |
| L004 | 2026-02-28T17:30:00+05:30 | PERPLEXITY | rag_systems/resume_engine.py | MODIFY | fix:paths:replace_hardcoded_with_RESUME_DIR_env_var | DONE |
| L005 | 2026-02-28T17:30:00+05:30 | PERPLEXITY | rag_systems/resume_config.json | MODIFY | fix:paths:absolute_to_basename_only | DONE |
| L006 | 2026-02-28T17:30:00+05:30 | PERPLEXITY | rag_systems/production_server.py | MODIFY | fix:env_keys:align_RAG_SERVER_API_KEY_GEMINI_API_KEY | DONE |
| L007 | 2026-02-28T17:30:00+05:30 | PERPLEXITY | rag_systems/Dockerfile | CREATE | create:dockerfile:rag_server_python311_slim_port_8090 | DONE |
| L008 | 2026-02-28T17:30:00+05:30 | PERPLEXITY | tools/rag_tools.py | CREATE | create:rag_tools:4_crewai_tools_query_context_embed_path | DONE |
| L009 | 2026-02-28T17:45:00+05:30 | PERPLEXITY | integrations/llm_interface.py | MODIFY | refactor:llm_interface:full_6_agent_provider_chain_crewai_litellm | DONE |
| L012 | 2026-02-28T20:15:00+05:30 | PERPLEXITY | docker-compose.yml | MODIFY | fix:remove_mcp_add_chromadb_fix_rag_build_context | DONE |
| L013 | 2026-02-28T20:15:00+05:30 | PERPLEXITY | database/schema.sql | CREATE | create:postgres_schema:9_tables_pgvector_indexes_erd_aligned | DONE |
| L014 | 2026-02-28T20:15:00+05:30 | PERPLEXITY | database/init.sql | CREATE | create:init_sql:docker_postgres_bootstrap_pgvector | DONE |
| L015 | 2026-02-28T20:30:00+05:30 | PERPLEXITY | tools/postgres_tools.py | CREATE | create:postgres_tools:10_crewai_tools_full_crud_9_tables | DONE |
| L016 | 2026-02-28T20:30:00+05:30 | PERPLEXITY | tools/budget_tools.py | CREATE | create:budget_tools:5_crewai_tools_xai_cap_monthly_cap_enforcement | DONE |
| L017 | 2026-02-28T20:45:00+05:30 | PERPLEXITY | tools/scraper_tools.py | CREATE | create:scraper_tools:7_crewai_tools_wrap_scraper_engine_all_platforms | DONE |
| L018 | 2026-02-28T20:55:00+05:30 | PERPLEXITY | integrations/notion.py | CREATE | create:notion_client:5_methods_pagination_retry_both_dbs | DONE |
| L019 | 2026-02-28T20:55:00+05:30 | PERPLEXITY | tools/notion_tools.py | CREATE | create:notion_tools:5_crewai_tools_sync_queue_update_health | DONE |
| L020 | 2026-02-28T21:00:00+05:30 | PERPLEXITY | config/platforms.json | CREATE | create:platforms_config:15_platforms_full_config_run_sequence | DONE |
| L021 | 2026-02-28T21:00:00+05:30 | PERPLEXITY | agents/scraper_agent.py | CREATE | create:scraper_agent:crewai_agent_full_run_lifecycle_all_platforms | DONE |
| L022 | 2026-02-28T21:30:00+05:30 | CLAUDE | tools/agentops_tools.py | CREATE | create:agentops_tools:record_agent_error_record_fallback_event_crewai_tools | DONE |
| L023 | 2026-02-28T21:30:00+05:30 | CLAUDE | agents/analyser_agent.py | CREATE | create:analyser_agent:full_scoring_rag_routing_fallback_chain | DONE |
| L025 | 2026-03-01T15:30:00+05:30 | GEMINI | config/scoring_weights.json | CONFIG | config:scoring_weights:thresholds_routing_rag_weights | DONE |
| L026 | 2026-03-01T15:30:00+05:30 | GEMINI | config/search_queries.json | CONFIG | config:search_queries:primary_secondary_filters_excludes | DONE |
| L027 | 2026-03-01T15:30:00+05:30 | GEMINI | config/user_preferences.json | CONFIG | config:user_preferences:candidate_job_application_prefs | DONE |
| L028 | 2026-03-01T15:45:00+05:30 | GEMINI | tools/apply_tools.py | CREATE | create:apply_tools:playwright_ats_llm_form_fill_proof_dry_run | DONE |
| L029 | 2026-03-01T15:47:00+05:30 | GEMINI | tools/apply_tools.py | BUGFIX | fix:apply_tools:playwright_timeout_error_cast_deadvar_return_paths | DONE |
| L029 | 2026-03-01T16:00:00+05:30 | GEMINI | config/settings.py | CREATE | create:settings:4_frozen_dataclasses_singleton_fixes_master_run_import | DONE |
| L030 | 2026-03-01T16:15:00+05:30 | GEMINI | agents/tracker_agent.py | CREATE | create:tracker_agent:crewai_notion_sync_agentops_close_session | DONE |
| L031 | 2026-03-01T16:30:00+05:30 | GEMINI | agents/master_agent.py | CREATE | create:master_agent:full_pipeline_orchestrator_boot_phases_budget_gates_report | DONE |
| L032 | 2026-03-01T16:45:00+05:30 | GEMINI | agents/apply_agent.py | CREATE | create:apply_agent:full_manifest_execution_ats_proof_reroute_fallback_chain | DONE |
| L033 | 2026-03-01T17:00:00+05:30 | CODEX | master_run.py | FIX | fix:broken_imports:remove_core_mcp_wire_MasterAgent_from_cli | DONE |
| L034 | 2026-03-01T17:30:00+05:30 | GEMINI | auto_apply/form_filler.py | CREATE | create:form_filler:ats_agnostic_llm_questions_multi_step_dry_run_aihawk_inspired | DONE |
| L035 | 2026-03-01T17:45:00+05:30 | GEMINI | auto_apply/ats_detector.py | CREATE | create:ats_detector:3_layer_url_dom_llm_registry_8_ats_platforms_production_selectors | DONE |
| L036 | 2026-03-01T18:15:00+05:30 | COPILOT | tools/apply_tools.py | REFACTOR | refactor:apply_tools:wire_ATSDetector_FormFiller_multi_step_captcha_gate_proof | DONE |
| L037 | 2026-03-01T18:30:00+05:30 | COPILOT | main.py | CREATE | create:main:cli_entry_point_argparse_banner_env_check_exit_codes | DONE |
| L038 | 2026-03-01T18:45:00+05:30 | GEMINI | Utils/normalise_dedupe.py | CREATE | create:normalise_dedupe:clean_hash_dedup_schema_mapping | DONE |
| L039 | 2026-03-01T18:45:00+05:30 | GEMINI | Utils/proxy_ratelimit.py | CREATE | create:proxy_ratelimit:thread_safe_round_robin_playwright_requests_compat | DONE |
| L040 | 2026-03-01T18:50:00+05:30 | GEMINI | .github/workflows/run_pipeline.yml | CREATE | create:run_pipeline_yml:cron_mon_thu_sat_secrets_health_check_artifacts | DONE |
| L041 | 2026-03-01T18:50:00+05:30 | GEMINI | .github/workflows/ci.yml | CREATE | create:ci_yml:lint_typecheck_mypy_black_isort_pytest_pgvector | DONE |
| L042 | 2026-03-01T19:00:00+05:30 | COPILOT | api/api_server.py | CREATE | create:api_server:3_endpoints_status_dashboard_manual_apply_chrome_extension_boundary | DONE |
| L045 | 2026-03-01T19:15:00+05:30 | CODEX | agents/*.py tools/*.py auto_apply/*.py | FIX | fix:agentops_v04_decorator_migration_track_agent_to_agent_track_tool_to_operation | DONE |
| L046 | 2026-03-01T19:30:00+05:30 | CODEX | requirements.txt | FIX | fix:requirements:add_5_missing_remove_30_bloat_fix_crewai_version | DONE |
| L047 | 2026-03-01T19:10:00+05:30 | CODEX | rag_systems/rag_api.py | FIX | fix:bare_import_to_fully_qualified_rag_systems_package_path | DONE |
| L048 | 2026-03-01T22:43:08+05:30 | CODEX | docker-compose.yml, .github/workflows/run_pipeline.yml, .github/workflows/ci.yml, .gitignore, IDE_README.md | FEATURE DELTA | renameEnvFile:java.env→java.env across all infra files | DONE |
| L049 | 2026-03-01T22:52:49+05:30 | CODEX | README.md, IDE_README.md | FEATURE DELTA | renameEnvFile:env-file-to-java.env in readmes | DONE |
| L050 | 2026-03-02T00:23:00+05:30 | PERPLEXITY | IDE_README.md | UPDATE | session-start:add-chrome-ext-specs-platform-apply-specs-developer-agent-specs-session-handoff | DONE |
| L051 | 2026-03-02T00:30+05:30 | CLAUDE | extension/manifest.json | CREATE | create:manifest_v3:mv3_permissions_content_script_service_worker_options | DONE |
| L052 | 2026-03-02T00:30+05:30 | CLAUDE | extension/options/options.html | CREATE | create:options_html:self_contained_fastapi_host_apikey_config_test_connection | DONE |
| L053 | 2026-03-02T00:30+05:30 | CLAUDE | extension/popup/popup.css | CREATE | create:popup_css:full_dark_mode_all_components_css_custom_props_animations | DONE |
| L054 | 2026-03-02T00:35+05:30 | CLAUDE | extension/utils/dom_detector.js | CREATE | create:dom_detector:scan_classify_react_shadow_inject_mutation_ats_hint_window_shim | DONE |
| L055 | 2026-03-02T00:35+05:30 | CLAUDE | extension/content_scripts/content.js | CREATE | create:content_js:full_scan_spa_observer_message_bridge_iframe_inject_page_meta | DONE |
| L056 | 2026-03-02T00:42+05:30 | CLAUDE | extension/utils/api_client.js | CREATE | create:api_client:callMatch_callAutofill_callLogApp_callQueueCount_retry_timeout_globalThis | DONE |
| L057 | 2026-03-02T00:42+05:30 | CLAUDE | extension/background/service_worker.js | CREATE | create:service_worker:message_router_badge_session_cache_tab_relay_fresh_config | DONE |
| L058 | 2026-03-02T00:51+05:30 | CLAUDE | extension/popup/popup.html | CREATE | create:popup_html:6_states_all_ids_score_resume_talking_points_autofill_applied | DONE |
| L059 | 2026-03-02T00:51+05:30 | CLAUDE | extension/popup/popup.js | CREATE | create:popup_js:init_match_autofill_inject_log_state_machine_badge_30s_timer | DONE |
| L060 | 2026-03-02T00:55+05:30 | CLAUDE | api/api_server.py | MODIFY | add:4_chrome_ext_endpoints_match_autofill_queue_count_log_application_notion_sync | DONE |
| L061 | 2026-03-02T00:59+05:30 | CLAUDE | auto_apply/platforms/base_platform.py | CREATE | create:base_platform:ApplyResult_BasePlatformApply_shared_fill_upload_captcha_proof | DONE |
| L062 | 2026-03-02T00:59+05:30 | CLAUDE | auto_apply/platforms/greenhouse.py | CREATE | create:greenhouse:3_step_apply_field_fill_upload_proof_custom_questions_dry_run | DONE |
| L063 | 2026-03-02T00:59+05:30 | CLAUDE | auto_apply/platforms/lever.py | CREATE | create:lever:2_step_apply_iframe_detect_card_questions_upload_proof_dry_run | DONE |
| L064 | 2026-03-02T01:04+05:30 | CLAUDE | auto_apply/platforms/workday.py | CREATE | create:workday:7_page_loop_react_3tier_fill_voluntary_decline_proof_capture_dry_run | DONE |
| L065 | 2026-03-02T01:08+05:30 | CLAUDE | auto_apply/platforms/linkedin_easy_apply.py | CREATE | create:linkedin:tos_reroute_external_url_detect_metadata_extract_no_click | DONE |
| L066 | 2026-03-02T01:08+05:30 | CLAUDE | auto_apply/platforms/native_form.py | CREATE | create:native_form:confidence_threshold_keyword_fill_proof_capture_ats_detect | DONE |
| L067 | 2026-03-02T01:13+05:30 | CLAUDE | auto_apply/platforms/indeed.py | CREATE | create:indeed:wizard_iframe_external_url_detect_multi_step_proof_dry_run | DONE |
| L068 | 2026-03-02T01:13+05:30 | CLAUDE | auto_apply/platforms/wellfound.py | CREATE | create:wellfound:login_check_easy_apply_full_apply_modal_upload_proof | DONE |
| L069 | 2026-03-02T01:13+05:30 | CLAUDE | auto_apply/platforms/arc_dev.py | CREATE | create:arc_dev:modal_steps_empty_only_fill_resume_check_proof_dry_run | DONE |
| L070 | 2026-03-02T01:17+05:30 | CLAUDE | agents/developer_agent.py | CREATE | create:developer_agent:6_analysis_methods_llm_fallback_chain_dedup_budget_guard_agentops | DONE |
| L071 | 2026-03-02T01:27+05:30 | CODEX | .github/workflows/ci.yml | BUGFIX | fix:ci_lint_typcheck_job_fail_mypy_strict_no_tests_no_playwright_always_run | DONE |
| L072 | 2026-03-02T01:30:00+05:30 | GITHUB_COPILOT | tools/serpapi_tool.py | CREATE | create:serpapi_tool:thread_safe_4_key_round_robin_fallback_tool | DONE |
| L073 | 2026-03-02T01:30:00+05:30 | GITHUB_COPILOT | scrapers/scraper_engine.py | REFACTOR | refactor:scraper_engine:remove_inline_serpapi_replace_with_tool_call | DONE |
| L074 | 2026-03-03T15:34+05:30 | CLAUDE | scrapers/scraper_engine.py | BUGFIX | fix:remove_serpapi_from_loop_delete_SerpAPIGoogleJobsScraper_class | DONE |
| L075 | 2026-03-03T15:34+05:30 | CLAUDE | tools/serpapi_tool.py | BUGFIX | fix:verify_complete_serpapi_tool_4key_rotation_agentops_failsoft | DONE |
| L076 | 2026-03-03T15:42+05:30 | CLAUDE | auto_apply/platforms/base_platform.py | BUGFIX | fix:resolve_resume_path_remove_hardcode_use_env_vars_only | DONE |
| L077 | 2026-03-03T15:42+05:30 | CLAUDE | integrations/llm_interface.py | BUGFIX | fix:add_retry_backoff_unavailable_tracking_all_6_providers | DONE |
| L078 | 2026-03-03T17:28+05:30 | CLAUDE | agents/master_agent.py | BUGFIX | fix:add_run_batch_dedup_filter_before_analyser_handoff_failsoft | DONE |
| L079 | 2026-03-03T17:28+05:30 | CLAUDE | config/platforms.json,agents/scraper_agent.py,tools/scraper_tools.py,scrapers/scraper_engine.py | BUGFIX | fix:add_jobs_per_scrape_daily_job_cap_fields_all_15_platforms | DONE |
| L080 | 2026-03-03T17:28+05:30 | CLAUDE | scrapers/scraper_engine.py | BUGFIX | fix:wire_proxy_ratelimiter_into_rest_api_calls_via_make_proxied_request | DONE |
| L081 | 2026-03-03T17:34+05:30 | CLAUDE | database/schema.sql,database/migrations/ | BUGFIX | fix:add_schema_versions_table_migration_readme_v001_baseline | DONE |
| L082 | 2026-03-03T17:34+05:30 | CLAUDE | utils/normalise_dedupe.py | BUGFIX | fix:clean_description_bs4_word_boundary_truncate_regex_fallback | DONE |
| L083 | 2026-03-03T17:34+05:30 | CLAUDE | .gitignore,README.md,extension/ | BUGFIX | fix:consolidate_dual_extension_dirs_gitignore_chrome_extension | DONE |
| L084 | 2026-03-03T{HH:MM}+05:30 | GEMINI | FULL CODEBASE | AUDIT | sweep002_postbugfix_javaenv_consistency_predryrun_final_audit | DONE |
| L085 | 2026-03-03T18:10+05:30 | CLAUDE | pyproject.toml, agents/analyser_agent.py, agents/apply_agent.py, agents/developer_agent.py, auto_apply/form_filler.py, config/settings.py, docker-compose.yml, main.py, scrapers/scraper_engine.py, tools/serpapi_tool.py | BUGFIX | fix:rename_env_file_to_java_env_all_10_missed_files | DONE |
| L086 | 2026-03-03T18:16+05:30 | CLAUDE | auto_apply/platforms/base_platform.py | BUGFIX | fix:create_missing_base_platform_abc_applyresult_dryrungating | DONE |
| L087 | 2026-03-03T18:16+05:30 | CLAUDE | main.py | BUGFIX | fix:add_dryrun_argparse_flag_pass_dryrun_to_masteragent | DONE |
| L088 | 2026-03-03T18:16+05:30 | CLAUDE | auto_apply/form_filler.py | BUGFIX | fix:gate_submit_actions_with_dryrun_check_4_methods | DONE |
| L089 | 2026-03-03T18:20+05:30 | CLAUDE | scrapers/scraper_engine.py | BUGFIX | fix:wrap_serpapi_call_in_safety_net_threshold_guard | DONE |
| L090 | 2026-03-03T18:20+05:30 | CLAUDE | scrapers/scraper_engine.py | BUGFIX | fix:replace_httpx_with_make_proxied_request_in_fetch_retry | DONE |
| L091 | 2026-03-03T18:30+05:30 | CLAUDE | agents/master_agent.py | BUGFIX | fix:add_original_removed_tracking_to_run_batch_id_dedup_block | DONE |
| L092 | 2026-03-03T18:30+05:30 | CLAUDE | integrations/llm_interface.py | BUGFIX | fix:rename_env_file_to_java_env_in_get_llm_runtimeerror_message | DONE |
| L093 | 2026-03-03T18:30+05:30 | CLAUDE | agents/analyser_agent.py, agents/apply_agent.py, agents/developer_agent.py | BUGFIX | fix:add_3attempt_retry_exponential_backoff_to_resolve_resume_id_platform_apply_counts_deduplicate_suggestions | DONE |
| L094 | 2026-03-03T18:30+05:30 | CLAUDE | tools/scraper_tools.py | BUGFIX | fix:add_3attempt_retry_exponential_backoff_to_get_scrape_summary | DONE |
| L095 | 2026-03-03T18:45+05:30 | CLAUDE | scrapers/scraper_engine.py, tools/scraper_tools.py, tools/serpapi_tool.py | BUGFIX | fix:replace_all_print_statements_with_structured_logging | DONE |
| L096 | 2026-03-03T18:45+05:30 | CLAUDE | utils/db_utils.py, agents/analyser_agent.py, agents/apply_agent.py | BUGFIX | fix:centralise_get_db_conn_into_utils_db_utils_remove_duplicates | DONE |
| L097 | 2026-03-03T19:00+05:30 | CLAUDE | tools/serpapi_tool.py + all .py files | BUGFIX | fix:serpapi_key_names_SERPAPI_KEY_to_SERPAPI_API_KEY_correct | DONE |
| L098 | 2026-03-03T19:00+05:30 | CLAUDE | utils/proxy_ratelimit.py + all files | BUGFIX | fix:webshare_proxy_keys_fix_wenshare_typo_java_env_final_purge | DONE |
| L099 | 2026-03-03T19:15+05:30 | CLAUDE | utils/__init__.py + all package __init__.py files | BUGFIX | fix:create_missing_package_init_files_utils_agents_tools | DONE |
| L0100 | 2026-03-03T19:15+05:30 | CLAUDE | utils/proxy_ratelimit.py | BUGFIX | fix:add_module_level_get_proxy_dict_get_next_proxy_reset_cycle_functions | DONE |
| L0101 | 2026-03-03T19:30+05:30 | CLAUDE | Utils/ → utils/ (directory rename) | BUGFIX | fix:rename_Utils_dir_to_lowercase_utils_resolve_ModuleNotFoundError_on_case_sensitive_fs | DONE |
| L102 | 2026-03-03T19:33+05:30 | CLAUDE | tools/scraper_tools.py, scrapers/__init__.py | BUGFIX | fix:remove_dead_SerpAPIGoogleJobsScraper_import_wire_serpapi_tool | DONE |
| L103 | 2026-03-03T19:39+05:30 | CLAUDE | rag_systems/chromadb_store.py, rag_systems/resume_engine.py | BUGFIX | fix:chromadb_store_missing_module_fully_qualified_import_path | DONE |
| L104 | 2026-03-03T19:52+05:30 | CLAUDE | FULL CODEBASE | BUGFIX | fix:full_import_chain_audit_all_broken_imports_fixed_one_pass | DONE |
| L105 | 2026-03-03T20:02+05:30 | CLAUDE | Dockerfile, docker-compose.yml, .dockerignore | REBUILD | rebuild:docker_from_scratch_correct_context_all_5_services_playwright | DONE |
| L106 | 2026-03-03T20:20+05:30 | CLAUDE | database/schema.sql, database/migrations/v002_rename_tables.sql | BUGFIX | fix:rename_job_posts_to_jobs_logs_events_to_audit_logs_run_batches_to_run_sessions | DONE |
| L107 | 2026-03-03T20:20+05:30 | CLAUDE | rag_systems/ingestion.py | BUGFIX | fix:create_missing_ingestion_py_pdf_extract_embed_chromadb_upsert | DONE |
| L108 | 2026-03-03T23:00+05:30 | CLAUDE | .dockerignore | CREATE | create_dockerignore_exclude_venv_git_secrets_node_chroma | DONE |
| L109 | 2026-03-03T23:00+05:30 | CLAUDE | rag_systems/Dockerfile | MODIFY | refactor_multistage_build_builder_runtime_no_playwright_pythonpath | DONE |
| L110 | 2026-03-03T23:00+05:30 | CLAUDE | Dockerfile | MODIFY | use_ms_playwright_base_image_python311_overlay_pythonpath_layerorder | DONE |
| L111 | 2026-03-03T23:00+05:30 | CLAUDE | rag_systems/ingestion.py | BUGFIX | fix_embed_resolution_7pattern_hasattr_chain_failsoft_fulllog | DONE |
| L112 | 2026-03-03T23:30+05:30 | CLAUDE | requirements.txt | BUGFIX | fix:pin_playwright_1.50_chromadb_1.x_remove_sentence_transformers_pyPDF2 | DONE |
| L113 | 2026-03-03T23:30+05:30 | CLAUDE | requirements-dev.txt | CREATE | create:requirements_dev_txt_split_dev_tools_from_production | DONE |
| L114 | 2026-03-03T23:30+05:30 | CLAUDE | docker-compose.yml | BUGFIX | fix:remove_version_chromadb_healthcheck_rag_dockerfile_path_pythonpath | DONE |
| L115 | 2026-03-03T23:30+05:30 | CLAUDE | rag_systems/chromadb_store.py | BUGFIX | fix:persistent_client_to_http_client_env_host_port_retry_backoff | DONE |
| L116 | 2026-03-03T23:30+05:30 | CLAUDE | rag_systems/rag_pipeline.py | BUGFIX | fix:gemini_768_to_1024_dim_validation_guard_in_embed_service | DONE |
| L117 | 2026-03-03T23:30+05:30 | CLAUDE | rag_systems/ingestion.py | BUGFIX | fix:remove_agentops_import_fix_store_add_to_chroma_upsert_chunks | DONE |
| L118 | 2026-03-03T23:30+05:30 | CLAUDE | rag_systems/production_server.py | BUGFIX | fix:trusted_hosts_env_api_key_redact_4char_uuid_request_id | DONE |
| L119 | 2026-03-03T23:30+05:30 | CLAUDE | config/settings.py | BUGFIX | fix:chromadb_path_to_chromadb_host_port_for_http_client | DONE |
| L120 | 2026-03-04T13:40:00+0530 | CLAUDE | rag_systems/resume_engine.py | BUGFIX | fix_extract_pdf_replace_pypdf2_with_pdfplumber_failsoft | DONE |
| L121 | 2026-03-04T13:40:00+0530 | CLAUDE | requirements.txt | MODIFY | pin_crewai_0.95.0_agentops_0.4.0_exact_no_range | DONE |
| L122 | 2026-03-04T13:40:00+0530 | CLAUDE | rag_systems/rag_pipeline.py | BUGFIX | fix_dim_guard_return_none_not_wrong_vector_log_error | DONE |
| L123 | 2026-03-04T13:40:00+0530 | CLAUDE | rag_systems/ingestion.py | BUGFIX | fix_pattern1_engine_embedder_embed_text_dim_mismatch_return_none | DONE |
| L124 | 2026-03-04T13:40:00+0530 | CLAUDE | rag_systems/production_server.py | MODIFY | fix_host_default_0.0.0.0_remove_duplicate_imports | DONE |
| L125 | 2026-03-04T13:40:00+0530 | CLAUDE | config/settings.py | MODIFY | align_resume_dir_default_app_resumes | DONE |
| L126 | 2026-03-04T13:40:00+0530 | CLAUDE | agents/master_agent.py | MODIFY | replace_agent_decorator_with_agentops_track_agent | DONE |
| L127 | 2026-03-04T14:15:00+0530 | GEMINI | PROJECT_FILE_STATUS.md | CREATE | create_project_file_status_from_raw_scan | DONE |
| L128 | 2026-03-04T15:00:00+0530 | CLAUDE | database/local_postgres_client.py | CREATE | create_local_postgres_client_threaded_pool_full_crud_health | DONE |
| L129 | 2026-03-04T15:00:00+0530 | CLAUDE | database/supabase_client.py | CREATE | create_supabase_client_psycopg2_ssl_identical_interface | DONE |
| L130 | 2026-03-04T15:00:00+0530 | CLAUDE | database/__init__.py | MODIFY | add_get_db_client_factory_singleton_activedb_switch | DONE |
| L131 | 2026-03-04T15:55:00+0530 | CLAUDE | agents/master_agent.py | MODIFY | add_dry_run_bypass_all_external_boot_checks | DONE |
| L132 | 2026-03-04T15:55:00+0530 | CLAUDE | rag_systems/ingestion.py | MODIFY | move_chromastore_init_inside_function_add_dry_run_bypass | DONE |
| L133 | 2026-03-04T15:55:00+0530 | CLAUDE | main.py | MODIFY | ensure_dry_run_cli_flag_sets_os_environ_before_agents | DONE |
| L134 | 2026-03-04T16:30:00+0530 | GEMINI | Dockerfile | MODIFY | add_nonroot_appuser_before_cmd | DONE |
| L135 | 2026-03-04T16:30:00+0530 | GEMINI | .dockerignore | MODIFY | append_postgres_data_platforms_logs_structure_txt | DONE |
| L136 | 2026-03-04T16:30:00+0530 | GEMINI | requirements.txt | MODIFY | remove_duplicate_psycopg2_binary_line_80 | DONE |
| L137 | 2026-03-04T16:30:00+0530 | GEMINI | docker-compose.yml | MODIFY | mount_init_sql_postgres_add_rag_server_url_agentrunner | DONE |
| L138 | 2026-03-04T17:00:00+0530 | CLAUDE | rag_systems/ingestion.py | BUGFIX | fix:chunk_text400token_NIM_limit_average_vectors_chunked_embed_failsoft | DONE |
| L139 | 2026-03-05T17:30:59+05:30 | ANTIGRAVITY | docker-compose.yml | BUGFIX | fix:add_schema_sql_volume_mount_postgres_init_sql_broken_ref | DONE |
| L140 | 2026-03-05T17:30:59+05:30 | ANTIGRAVITY | docker-compose.yml | CONFIG | fix:pin_chromadb_image_latest_to_0.6.3_prevent_silent_breaking_change | DONE |
| L141 | 2026-03-05T17:30:59+05:30 | ANTIGRAVITY | Dockerfile | REFACTOR | refactor:root_dockerfile_single_to_multistage_builder_runtime_strip_gcc | DONE |
| L142 | 2026-03-05T17:30:59+05:30 | ANTIGRAVITY | docker-compose.yml | CONFIG | fix:add_resource_limits_agentrunner_cpus_2_memory_4g_playwright_oom | DONE |
| L143 | 2026-03-05T17:30:59+05:30 | ANTIGRAVITY | rag_systems/Dockerfile | BUGFIX | fix:remove_dead_healthcheck_compose_overrides_dockerfile_one_source | DONE |
| L144 | 2026-03-05T17:30:59+05:30 | ANTIGRAVITY | dockerise.sh | BUGFIX | fix:phase10_exec_to_run_rm_agentrunner_exits_after_run_no_restart | DONE |
| L145 | 2026-03-05T17:30:59+05:30 | ANTIGRAVITY | dockerise.sh | CONFIG | fix:docker_compose_v1_cli_to_docker_compose_v2_eol_migration | DONE |
| L146 | 2026-03-05T17:30:59+05:30 | ANTIGRAVITY | java.env.template | CREATE | create:java_env_template_all_keys_empty_values_safe_to_commit | DONE |
| L147 | 2026-03-05T17:43:32+05:30 | USER | dockerise.sh | BUGFIX | fix:error_msg_typo_narad_env_to_java_env_phase0_preflight | DONE |
| L148 | 2026-03-05T17:43:32+05:30 | ANTIGRAVITY | dockerise.sh | BUGFIX | fix:pguser_resolved_sed_url_parse_to_local_postgres_user_key_direct | DONE |
| L149 | 2026-03-05T17:43:32+05:30 | ANTIGRAVITY | dockerise.sh | BUGFIX | fix:table_count_integer_safe_tr_grep_eo_guard_empty_string_comparison | DONE |
| L150 | 2026-03-05T17:43:32+05:30 | ANTIGRAVITY | dockerise.sh | BUGFIX | fix:pdf_glob_ls_to_find_prevents_set_e_error_when_no_pdfs_exist | DONE |
| L151 | 2026-03-05T17:43:32+05:30 | ANTIGRAVITY | dockerise.sh | BUGFIX | fix:phase8_compose_up_d_all_to_explicit_daemons_only_exclude_agentrunner | DONE |
| L152 | 2026-03-05T17:43:32+05:30 | ANTIGRAVITY | dockerise.sh | BUGFIX | fix:chromadb_heartbeat_grep_nanosecond_to_nanosecond_heartbeat_key_name | DONE |
| L153 | 2026-03-05T17:43:32+05:30 | ANTIGRAVITY | dockerise.sh | BUGFIX | fix:health_check_conditionals_and_or_chaining_to_if_else_safe_with_set_e | DONE |
| L154 | 2026-03-05T17:48:42+05:30 | ANTIGRAVITY | Dockerfile | BUGFIX | fix:libasound2_to_libasound2t64_debian12_bookworm_package_rename_build_fail | DONE |
| L155 | 2026-03-05T18:02:00+05:30 | ANTIGRAVITY | Dockerfile | BUGFIX | fix:replace_hardcoded_chromium_runtime_pkgs_with_playwright_install_deps_arm64_compat | DONE |
| L156 | 2026-03-05T18:02:00+05:30 | ANTIGRAVITY | Dockerfile | BUGFIX | fix:reorder_runtime_stage_copy_venv_before_playwright_install_deps_run | DONE |
| L157 | 2026-03-05T18:16:55+05:30 | ANTIGRAVITY | requirements.txt | BUGFIX | fix:pin_chromadb_0.6.3_exact_match_server_version_resolve_type_keyerror | DONE |
| L158 | 2026-03-05T18:16:55+05:30 | ANTIGRAVITY | rag_systems/chromadb_store.py | BUGFIX | fix:add_embedding_function_none_get_or_create_collection_suppress_type_serialization | DONE |
| L159 | 2026-03-05T18:39:51+05:30 | ANTIGRAVITY | dockerise.sh | BUGFIX | fix:phase9_grep_c_to_grep_q_if_else_eliminates_0_newline_0_syntax_error | DONE |
| L160 | 2026-03-05T18:39:51+05:30 | ANTIGRAVITY | Dockerfile | BUGFIX | fix:add_chown_appuser_app_after_workdir_allows_log_file_creation_at_runtime | DONE |
| L161 | 2026-03-05T18:47:54+05:30 | ANTIGRAVITY | rag_systems/ingestion.py | BUGFIX | fix:chunk_gate_and_chunk_text_max_tokens_400_to_370_words_482_tokens_30_below_nim_512_limit | DONE |
| L162 | 2026-03-05T18:47:54+05:30 | ANTIGRAVITY | rag_systems/rag_pipeline.py | BUGFIX | fix:gemini_embed_url_v1beta_to_v1_text_embedding_004_not_available_on_v1beta | DONE |
| L163 | 2026-03-05T18:47:54+05:30 | ANTIGRAVITY | rag_systems/rag_pipeline.py | BUGFIX | fix:nvidia_nim_truncate_none_to_end_prevents_400_on_oversize_chunks | DONE |
| L164 | 2026-03-05T18:55:11+05:30 | ANTIGRAVITY | rag_systems/rag_pipeline.py | BUGFIX | fix:revert_gemini_url_v1_back_to_v1beta_confirmed_correct_per_google_docs | DONE |
| L165 | 2026-03-05T18:55:11+05:30 | ANTIGRAVITY | rag_systems/rag_pipeline.py | BUGFIX | fix:gemini_model_text_embedding_004_to_gemini_embedding_001_model_renamed | DONE |
| L166 | 2026-03-05T19:40:00+05:30 | ANTIGRAVITY | agents/* | BUGFIX | fix:typeerror_tool_callable_func_wrapper_bypass_native_calls | DONE |
| L167 | 2026-03-05T19:40:00+05:30 | ANTIGRAVITY | dockerise.sh | UPDATE | update:docker_fixes_complete_agentrunner_verified | DONE |
| L168 | 2026-03-09T16:21:00+05:30 | ANTIGRAVITY | test_scripts/* | CREATE | create:test_scripts:all_agents_tools_api_validation | DONE |
| L169 | 2026-03-09T22:34:00+05:30 | ANTIGRAVITY | agents/scraper_agent.py | FEATURE DELTA | feature:scraper_fallback:hardcoded_scrape_sequence_on_llm_failure | DONE |
| L170 | 2026-03-09T22:34:00+05:30 | ANTIGRAVITY | test_scripts/test_scraper.py | TEST | test:scraper_fallback:validate_budget_and_llm_failure_bypasses | DONE |
| L172 | 2026-03-10T22:55:00+05:30 | GEMINI | config/user_settings.json, config/platform_settings.json, database/schema.sql, .github/workflows/config_sync.yml | CREATE | create:postgres-sso-json-packets:user_settings:platform_settings:schema-migration:config_sync-action | DONE |
| L173 | 2026-03-10T23:05:00+05:30 | CLAUDE SONNET | api/api_server.py | REFACTOR | refactor:api-server:fetch-user-profile-and-config-from-postgres-json-packets:remove-os-getenv | DONE |
| L174 | 2026-03-10T23:15:00+05:30 | CLAUDE SONNET | extension/popup/*, extension/background/service_worker.js, extension/content_scripts/sidebar.js | REFACTOR | refactor:chrome-ext:full-sidebar-rewrite:strip-mcp:fastapi-only-comms | DONE |
| L175 | 2026-03-10T23:25:00+05:30 | CLAUDE SONNET | tools/postgres_tools.py, tools/rag_tools.py, tools/apply_tools.py | REFACTOR | refactor:tools-layer:fetch-platform-config-resume-default-dry-run-from-postgres-json-packets | DONE |
| L176 | 2026-03-10T23:35:00+05:30 | CLAUDE OPUS | agents/analyser_agent.py, agents/apply_agent.py, agents/scraper_agent.py, scrapers/scraper_engine.py, scrapers/scraper_service.py | BUGFIX+REFACTOR | fix:func-typeerror-crash:scraper-engine-db-ssot:proxy-integrity-rotation | DONE |
| L177 | 2026-03-10T23:45:00+05:30 | CLAUDE OPUS | test_scripts/test_postgres.py, test_scripts/test_rag.py | REFACTOR | refactor:test-mocks:migrate-config-limits-to-postgres-json-packets:add-fetch-user-config-tests | DONE |
| L178 | 2026-03-10T23:55:00+05:30 | CLAUDE OPUS | rag_systems/rag_api.py, rag_systems/production_server.py, rag_systems/rag_pipeline.py, rag_systems/ingestion.py | REFACTOR | refactor:rag:unify-api-entry-points:deduplicate-chunking:chunk-size-400:health-endpoint-fix | DONE |
| L179 | 2026-03-11T00:10:00+05:30 | CLAUDE OPUS | Dockerfile, scrapers/Dockerfile, api/Dockerfile, docker-compose.yml, java.env.template | REFACTOR | refactor:docker:split-playwright-scraper:agent-runner:api-server:rewire-compose:7-services | DONE |
| L180 | 2026-03-11T00:10:00+05:30 | GEMINI | integrations/llm_interface.py, docker-compose.yml, Dockerfile | BUGFIX | fix:grok-model-ids:chromadb-host-container-name:remove-scrapers-copy-from-agent-runner | DONE |
| L181 | 2026-03-11T00:15:00+05:30 | GEMINI | config/, scrapers/job_filters.yaml, database/schema.sql, extension/*, config/settings.py | CLEANUP | cleanup:delete-deprecated-config-files:schema-migration-patch:mcp-final-sweep:settings-strip | DONE |
| L182 | 2026-03-11T00:20:00+05:30 | CLAUDE SONNET | tools/rag_tools.py, agents/scraper_agent.py | REFACTOR | refactor:http-wiring:rag-tools-httpx-to-rag-server:scraper-agent-httpx-to-scraper-service | DONE |
| L183 | 2026-03-11T00:30:00+05:30 | CLAUDE SONNET | extension/extension_config.js, extension/sidebar.js | CLEANUP | 
| L184 | 2026-03-11T12:45:00+05:30 | CODEX | config/settings.py, auto_apply/form_filler.py | BUGFIX | bugfix:get_settings-tuple-return:form_filler-dry_run-attr-missing:os.getenv-pattern | DONE |
| L185 | 2026-03-11T12:50:00+05:30 | CODEX | tools/apply_tools.py | BUGFIX | bugfix:apply_tools-dry_run-attr-missing:os.getenv-pattern-same-as-form_filler | DONE |
| L186 | 2026-03-11T13:30:00+05:30 | GEMINI | Dockerfile.playwright_base, scrapers/Dockerfile, auto_apply/Dockerfile, rag_systems/Dockerfile, api/Dockerfile, Dockerfile | CREATE+REFACTOR | docker:9-image-split:playwright-base+scraper+apply:audit-fix-rag-api-agents | DONE |
| L187 | 2026-03-11T13:45:00+05:30 | GEMINI | docker-compose.yml | AUDIT+REBUILD | docker-compose:9-service-rebuild:playwright-split:depends-on-chain:port-fix:volume-wire | DONE |
| L188 | 2026-03-11T13:50:00+05:30 | CODEX | Dockerfile | BUGFIX | fix:SC2046-unquoted-var:DL4006-pipefail-shell-directive | DONE |
| L189 | 2026-03-11T14:00:00+05:30 | GEMINI | Dockerfile.playwright_base | BUGFIX | fix:deadsnakes-ppa-gpg-failure:use-system-python310-in-playwright-base:no-ppa | DONE |
| L190 | 2026-03-11T15:10:00+05:30 | GEMINI | docker-compose.yml | BUGFIX | fix:yaml-indentation-ai_chromadb-nested-under-ai_redis | DONE |
| L191 | 2026-03-11T16:00:00+05:30 | GEMINI | Dockerfile | MODIFY | add rag_systems/ and resumes/ COPY lines to agentrunner image | DONE |
| L192 | 2026-03-11T16:25:00+05:30 | CODEX | config/settings.py | BUGFIX | add dry_run:bool=False to RunConfig dataclass | DONE |
| L192 | 2026-03-11T16:00:00+05:30 | GEMINI | docker-compose.yml | MODIFY | add ai_agentrunner service with profiles:run | DONE |
| L193 | 2026-03-11T16:15:00+05:30 | GEMINI | Dockerfile | MODIFY | add all missing package COPY lines scrapers api auto_apply | DONE |
| L194 | 2026-03-11T16:35:00+05:30 | GEMINI | VERIFY | dry_run fix verified + full docker health check + dry run passed | DONE |
| L195 | 2026-03-11T16:45:00+05:30 | GEMINI | agents/scraper_agent.py, agents/master_agent.py, agents/tracker_agent.py, requirements.txt | BUGFIX | fix:platforms_json_path:agentops_v4_migration:get_run_stats_callable:crewai_0_102_0 | DONE |
| L196 | 2026-03-11T17:35:00+05:30 | GEMINI | scrapers/scraper_engine.py | BUGFIX | FilterEngine reads filters from platform_settings.json not job_filters.yaml | DONE |
| L197 | 2026-03-11T17:35:00+05:30 | GEMINI | config/platform_settings.json | MODIFY | add filters block as single source of truth | DONE |
| L198 | 2026-03-11T17:35:00+05:30 | GEMINI | AUDIT | all 4 prior bug fixes confirmed present in source files | DONE |
| L199 | 2026-03-11T18:35:00+05:30 | GEMINI | docker-compose.yml | BUGFIX | pin chromadb image to 0.6.3 to match client version | DONE |
| L200 | 2026-03-11T19:21:00+05:30 | CODEX | rag_systems/ingestion.py | BUGFIX | fix infinite recursion in _get_embedding — split into _embed_single + _get_embedding with no self-calls | DONE |
| L201 | 2026-03-11T19:35:00+05:30 | GEMINI | tools/postgres_tools.py | BUGFIX | replace raw LOCAL_POSTGRES_URL with explicit psycopg2 kwargs host/port/user/pass/dbname | DONE |
| L202 | 2026-03-11T19:35:00+05:30 | GEMINI | agents/Dockerfile | BUGFIX | add playwright install chromium to build if Chromium was missing | DONE |
| L203 | 2026-03-11T19:35:00+05:30 | GEMINI | tools/scraper_tools.py | BUGFIX | fix SerpAPI tool signature to accept query kwarg | DONE |


## 2-WEEK SPRINT PLAN

SPRINT: Phase 1 Prod v1 | 2026-03-01 → 2026-03-14

### Week 1 (Mar 1–7): Foundation + Scraper + Master

| DAY | DATE | TASK | PERPLEXITY | CLAUDE | CODEX | GEMINI | MILESTONE |
|-----|------|------|------------|---------|-------|--------|-----------|
| 1 | Mar 1 | Repo init, directory structure, java.env.template, .gitignore, docker-compose.yml base | Scaffold all files | Complete Dockerfiles + compose | — | env comments + formatting | — |
| 2 | Mar 2 | Postgres schema.sql, db/init.sql, Supabase setup, migrations v1 | Schema design + SQL base | Full schema + migrations | — | SQL docblock comments | — |
| 3 | Mar 3 | ChromaDB setup, resume ingestion pipeline (rag/ingestion.py, rag/embedder.py) | EmbeddingService base class | Full NVIDIA NIM + Gemini fallback impl | — | Docstrings | — |
| 4 | Mar 4 | RAG query service (rag/query.py), ChromaDB collections setup | RAGQueryService base | Full similarity search + resume selection | Bug fixes if any | Type hints | M1: Infra + RAG ready |
| 5 | Mar 5 | Scraper tools — JobSpy (LinkedIn/Indeed), REST APIs (RemoteOK, Himalayas), SerpAPI | scraper_tools.py base + all tool signatures | Full JobSpy + REST + SerpAPI tool implementations | — | — | — |
| 6 | Mar 6 | Scraper tools — Playwright (Wellfound, WWR, YC, Arc, Turing, Crossover, Nodesk, Toptal) | Playwright base + per-platform script stubs | Full headless scripts all 8 platforms + proxy rotation | Bug fixes | — | — |
| 7 | Mar 7 | Scraper Agent (agents/scraper_agent.py) + normalisation + dedup + Perplexity sonar integration | ScraperAgent class base | Full agent impl with all tools wired | Bug fixes | Docstrings | M2: Scraper live |

### Week 2 (Mar 8–14): Core Agents + Integration + Launch

| DAY | DATE | TASK | PERPLEXITY | CLAUDE | CODEX | GEMINI | MILESTONE |
|-----|------|------|------------|---------|-------|--------|-----------|
| 1 | Mar 8 | Master Agent (agents/master_agent.py) + CrewAI crew setup + AgentOps init + budget gate | MasterAgent base + crew config | Full lifecycle + budget_tools.py impl | — | — | — |
| 2 | Mar 9 | Analyser Agent (agents/analyser_agent.py) — eligibility, scoring, RAG match, routing | AnalyserAgent base + routing logic | Full scoring + RAG query + route assignment | Bug fixes | Docstrings | — |
| 3 | Mar 10 | Apply Agent (agents/apply_agent.py) — ATS detection, per-platform apply scripts, retry, proof capture | ApplyAgent base + apply_tools.py stubs | Full Playwright apply flows all platforms + retry + proof | Bug fixes | — | M3: Analyser + Apply live |
| 4 | Mar 11 | Tracker Agent (agents/tracker_agent.py) — Postgres logging, Notion sync, AgentOps summary | TrackerAgent base + tracker_tools.py base | Full Postgres write + Notion sync + AgentOps report | Bug fixes | Docstrings | — |
| 5 | Mar 12 | FastAPI slim server (services/fastapi/) — all 3 endpoints live + Chrome Extension v1 | FastAPI route bases + extension base | Full endpoint impls + extension form detection + autofill | Fixes | Formatting | — |
| 6 | Mar 13 | GitHub Actions (run_pipeline.yml, ci.yml, resume_sync.yml) + budget enforcement + DRY_RUN E2E test | GHA workflow bases | Full workflow yamls + main.py pipeline wiring | All remaining bug fixes | — | — |
| 7 | Mar 14 | Integration QA, full DRY_RUN=false first live run, monitoring | Monitor + prompt fixes | Final polish | Last bug fixes | Final docstrings | M4: PHASE 1 LIVE 🚀 |

## MILESTONES

| ID | DATE | DELIVERABLE | STATUS |
|----|------|-------------|--------|
| M1 | 2026-03-04 | Infra complete: Docker, Postgres schema live, all 15 resumes in ChromaDB | PENDING |
| M2 | 2026-03-07 | Scraper Agent live: all 10 platforms returning normalised jobs | PENDING |
| M3 | 2026-03-11 | Analyser + Apply Agents live: scoring, routing, Playwright apply working | PENDING |
| M4 | 2026-03-14 | PHASE 1 LIVE: first fully autonomous run executed | PENDING |
| M5 | 2026-04-02 | Redis full queue + session checkpoints live | PENDING |
| M6 | 2026-04-16 | Developer Agent live: first cross-session AgentOps trace analysis | PENDING |
| M7 | 2026-04-30 | Remotive + Jooble integrated, all-round resume variant complete | PENDING |
| M8 | 2026-05-14 | Phase 2 complete: self-improving autonomous system | PENDING |

## CODING STANDARDS

- Language: Python 3.11 — no walrus operator, no match-case for compatibility
- All secrets via os.getenv() from java.env — never hardcode
- Every agent tool function decorated with @agentops.track_tool
- Every agent class decorated with @agentops.track_agent
- Error handling: all external calls (LLM APIs, Playwright, Postgres, Notion) wrapped in try/except with retry logic (max 3 retries, exponential backoff)
- Fail-soft: on tool failure, log to audit_logs with error code, escalate to Master Agent — never crash the pipeline
- All DB writes atomic — use transactions for multi-table operations
- LLM fallback chain: always implement primary → fallback_1 → fallback_2 for Analyser and Apply agents
- No MCP — FastAPI slim server is the ONLY HTTP boundary
- Logging: use Python logging module at LOG_LEVEL from java.env — no print() in production code
- Type hints on all function signatures (mypy strict)
- Google-style docstrings on all public methods and classes
- Module-level __all__ defined in every tool file
- config/platforms.json is the single source of truth for all platform-specific config (selectors, rate limits, compliance flags) — never hardcode platform config in agent code

## PROMPT TEMPLATES FOR IDEs

### CLAUDE — Heavy Implementation

```
[PROJECT] AI Job Application Agent — Python3.11/CrewAI/Playwright/Postgres/ChromaDB/Docker
[FILE] {relative_file_path}
[CONTEXT] {1-2 sentences: what this file/class does in the system pipeline}
[BASE_CODE] Perplexity scaffold already in file — complete all method implementations
[TASK]
  {precise description of what to implement — list each method/class if multiple}
[CONSTRAINTS]
  - Prod-ready: full error handling, retry (max 3, exponential backoff), fail-soft
  - No stubs, no TODOs, no placeholder comments — complete implementations only
  - All secrets via os.getenv() from java.env
  - Preserve existing class/method signatures exactly
  - AgentOps decorators already present — do not remove
  - Follow coding standards in IDE_README.md Section 16
[EXPECTED_OUTPUT] {what the completed code should do when called}
[IDE_LOG] Append to IDE_README.md CHANGE_LOG: ID:L{NNN} | AUTHOR:CLAUDE | FILE:{path} | TYPE:{CHANGE_TYPE} | NOTES:{verb:target:detail} | STATUS:DONE
```

### CODEX — Bug Fix / Feature Delta

```
[CTX] AIJAA | Python3.11 | {filename}
[AGENT] {which agent this belongs to, or "shared" if tools/db}
[TASK_TYPE] BUGFIX | FEATURE_DELTA | REFACTOR
[SCOPE] {single function name or class method — one scope only}
[PROBLEM] {exact description of bug or missing behaviour}
[EXPECTED] {exact expected behaviour after fix}
[CONSTRAINTS]
  - No new imports unless strictly necessary
  - Preserve function signature and return type
  - Fail-soft: no exceptions should propagate to caller uncaught
  - java.env for any new config values needed
[IDE_LOG] Append: L{NNN} | CODEX | {path} | {CHANGE_TYPE} | {notes} | DONE
```

### GEMINI — Grunt Work

```
[CTX] AIJAA | {filename}
[TASK] DOCSTRING | TYPECHECK | FORMAT | CONFIG | BOILERPLATE
[TARGET] {function name / class name / file section}
[STYLE] Google-style docstrings | black formatting | PEP8 | mypy type hints
[IDE_LOG] Append: L{NNN} | GEMINI | {path} | {CHANGE_TYPE} | {notes} | DONE
```
