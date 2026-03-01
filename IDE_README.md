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
| L049 | 2026-03-01T22:52:49+05:30 | CODEX | README.md, IDE_README.md | FEATURE DELTA | renameEnvFile:narad.env→java.env in readmes | DONE |
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
