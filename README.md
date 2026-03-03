# 🤖 AI Job Automation Agent

<div align="center">

**A fully autonomous, multi-agent job application ecosystem — discover, score, and apply to hundreds of remote jobs with minimal human intervention.**

[![Python](https://img.shields.io/badge/Python-3.11+-blue?style=flat-square&logo=python)](https://www.python.org/)
[![CrewAI](https://img.shields.io/badge/Framework-CrewAI-orange?style=flat-square)](https://crewai.com)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![Schedule](https://img.shields.io/badge/Schedule-Mon%2FThu%2FSat%2012%3A00%20IST-purple?style=flat-square)]()
[![Budget](https://img.shields.io/badge/Budget-%2410%2Fmonth-red?style=flat-square)]()
[![Status](https://img.shields.io/badge/Status-In%20Development-yellow?style=flat-square)]()

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [How It Works](#-how-it-works)
- [Agent System](#-agent-system)
- [Platform Coverage](#-platform-coverage)
- [Tech Stack](#-tech-stack)
- [Scoring & Routing](#-scoring--routing)
- [RAG Resume System](#-rag-resume-system)
- [Infrastructure](#-infrastructure)
- [Setup & Installation](#-setup--installation)
- [Environment Variables](#-environment-variables)
- [Project Roadmap](#-project-roadmap)
- [Repository Structure](#-repository-structure)
- [Budget](#-budget)
- [License](#-license)

---

## 🎯 Overview

The **AI Job Automation Agent** is a fully autonomous agentic job application system engineered to run **3 scheduled batch sessions per week** — Monday, Thursday, and Saturday at **12:00 PM IST** — and handle the complete end-to-end job application pipeline with minimal human intervention.

| Metric | Target |
|--------|--------|
| Runs per week | 3 — Mon / Thu / Sat @ 12:00 PM IST |
| Jobs discovered per run | ~150 (min 100 before safety-net activates) |
| Auto-apply rate | 50–70% of eligible jobs per run |
| Manual queue rate | 30–50% of eligible jobs per run |
| Monthly applications | ~300+ across ~13 runs |
| Total API budget | **$10/month hard cap** |
| xAI spend cap per run | **$0.38 — enforced in code** |

**The system executes two parallel application paths:**

- **Automated Path (50–70%)** — Playwright browser automation handles form filling and submission end-to-end with zero user action needed.
- **Manual Queue Path (30–50%)** — High-stakes, high-scoring (>0.90 fit), form-complex, or low-confidence jobs are queued in Notion with full metadata, resume recommendations, and match reasoning, surfaced via Chrome Extension for focused human review.

> ⚠️ **Critical design rule:** Auto-apply routing is determined by **form complexity + platform compliance ONLY**. Fit score is used for ranking and prioritisation — never for routing decisions.

---

## ⚙️ How It Works

Every run follows this exact sequence, orchestrated by **CrewAI in hierarchical mode** with the Master Agent as crew manager:

```
GitHub Actions Cron  →  0 6 * * 1,4,6  (UTC = 12:00 PM IST)
         │
         ▼
┌──────────────────────────────────────────┐
│            Master Agent                  │
│  (CrewAI Crew Manager)                   │
│  Groq llama-3.3-70b-versatile            │
│  → Session boot · budget enforcement     │
│  → Delegates sequentially to all workers │
└────────────────┬─────────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────────┐
│            Scraper Agent                 │
│  Perplexity sonar                        │
│  → JobSpy (LinkedIn + Indeed)            │
│  → Playwright (8 platforms)              │
│  → REST APIs (RemoteOK + Himalayas)      │
│  → SerpAPI (Google Jobs supplementary)  │
│  → Normalise · deduplicate · fill gaps  │
│  WRITES → Postgres jobs table            │
└────────────────┬─────────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────────┐
│           Analyser Agent                 │
│  xAI grok-4-fast-reasoning               │
│  → Eligibility filter                    │
│  → Fit scoring (0.0 – 1.0)              │
│  → RAG resume match (ChromaDB)           │
│  → Routing decision (form complexity)    │
│  WRITES → jobs table · applications      │
│           table · Notion Applications DB │
└────────────────┬─────────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────────┐
│            Apply Agent                   │
│  xAI grok-4-1-fast-reasoning             │
│  → Platform-specific Playwright scripts  │
│  → LLM form reasoning for complex fields │
│  → Multi-layer submission proof capture  │
│  → Retry ×2 on CAPTCHA / crash           │
│  WRITES → applications table             │
└────────────────┬─────────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────────┐
│           Tracker Agent                  │
│  Groq llama-3.3-70b-versatile            │
│  → Postgres audit logging                │
│  → Notion Job Tracker DB sync            │
│  → AgentOps run summary report           │
│  → run_session marked complete           │
└──────────────────────────────────────────┘
```

All agent handoffs persist state exclusively to **Postgres tables** — making every step fully traceable, restartable from checkpoint, and independently debuggable without re-running the full pipeline.

### Manual Apply Path (Chrome Extension)

Triggered independently by the user outside the automated pipeline:

1. User reviews queued jobs in Notion Applications DB
2. Opens job URL in Chrome — Extension detects the application form
3. Extension calls **FastAPI → ChromaDB RAG** → returns best resume + match reasoning + autofill data
4. Extension autofills all detectable fields — user handles custom questions manually
5. On submission: Extension calls FastAPI → logs to Postgres `applications` (status: `applied_manual`) → triggers Notion sync

---

## 🤖 Agent System

The system uses **CrewAI** (open-source, free) as the agent orchestration framework. All agents run in a single Python process per session. Every agent capability — Postgres reads/writes, Playwright calls, RAG queries, Notion sync — is implemented as a `@tool` decorated Python function registered to its agent.

| Agent | Role | Primary LLM | Fallback 1 | Fallback 2 | Phase |
|-------|------|-------------|------------|------------|-------|
| **Master Agent** | Crew Manager — Orchestrator | Groq `llama-3.3-70b-versatile` | Cerebras `llama-3.3-70b` | — | 1 + 2 |
| **Scraper Agent** | Job Discovery Specialist | Perplexity `sonar` | *(no fallback — web search irreplaceable)* | — | 1 + 2 |
| **Analyser Agent** | Evaluation & Scoring Specialist | xAI `grok-4-fast-reasoning` | SambaNova `Llama-3.1-70B` | Cerebras `llama-3.3-70b` | 1 + 2 |
| **Apply Agent** | Application Submission Specialist | xAI `grok-4-1-fast-reasoning` | SambaNova `Llama-3.1-70B` | Cerebras `llama-3.3-70b` | 1 + 2 |
| **Tracker Agent** | Record Keeper & Data Persistence | Groq `llama-3.3-70b-versatile` | Cerebras `llama-3.3-70b` | — | 1 + 2 |
| **Developer Agent** | System Improvement Analyst | xAI `grok-3-mini-latest` | Perplexity `sonar` | — | **Phase 2 only** |

**Fallback behaviour:**
- **Master / Tracker:** 1-level fallback (Cerebras). Failure in these agents degrades logging — does not kill the run.
- **Analyser / Apply:** 2-level fallback chain (SambaNova → Cerebras). These agents are run-critical; chain exhaustion routes remaining jobs to manual queue and continues.
- **Scraper:** No fallback. Perplexity `sonar`'s web-grounded search + normalisation cannot be replicated by a pure inference LLM. Retries ×3, then continues with raw platform data.
- **Developer Agent (Phase 2):** Runs outside the main pipeline. Analyses AgentOps traces across sessions and writes structured improvement suggestions to `developer_notes` Postgres table. Suggestion-only — never autonomously modifies production.

---

## 🌐 Platform Coverage

### Primary Platforms (10) — Phase 1 + 2

| Platform | Discovery Method | Category |
|----------|-----------------|----------|
| LinkedIn | JobSpy | Professional Network |
| Indeed | JobSpy | General Job Board |
| Wellfound | Playwright + GraphQL | Startup Jobs |
| RemoteOK | REST API (JSON) | Remote Job Board |
| WeWorkRemotely | Playwright | Remote-First Jobs |
| YC / Work at a Startup | Playwright | YC Startup Jobs |
| Himalayas | REST API (public) | Remote Aggregator |
| Turing | Playwright | Premium Talent |
| Crossover | Playwright | Premium Remote |
| Arc.dev | Playwright + GraphQL | Developer Marketplace |

### Safety-Net Platforms (2) — auto-activate when run < 100 jobs

| Platform | Discovery Method |
|----------|-----------------|
| Nodesk | Playwright |
| Toptal | Playwright |

### Phase 2 Platforms (2)

| Platform | Discovery Method |
|----------|-----------------|
| Remotive | REST API |
| Jooble | REST API (API key required) |

### Supplementary Discovery

**SerpAPI** — Google Jobs search results API (NOT an LLM).
4 accounts × 250 credits = **1,000 credits/month** available.

### Proxy Configuration

**Webshare static proxies** — 20 total (2 accounts × 10 proxies), 1 GB bandwidth/account/month, round-robin rotation. Budget is separate from the $10/month LLM cap.

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose | Tier |
|-------|-----------|---------|------|
| Language | Python 3.11+ | All agents · orchestration · services | Free |
| Agent Framework | CrewAI (OSS) | Agent roles · task definitions · hierarchical crew execution · `@tool` functions | Free |
| Job Scraping | JobSpy | LinkedIn + Indeed structured discovery | Free |
| Browser Automation | Playwright (Chromium) | 8-platform scraping + form filling + auto-apply | Free |
| Job APIs | requests | RemoteOK + Himalayas REST calls | Free |
| Supplementary Discovery | SerpAPI | Google Jobs search results (4 accounts · 1K credits/month) | Free tier |
| Primary DB (Prod) | PostgreSQL via Supabase | System of record — jobs · applications · run_sessions · audit_logs | Free tier |
| Primary DB (Dev) | PostgreSQL (local Docker) | Dev mirror — switched via `ACTIVE_DB` env var | Free |
| Vector Store | ChromaDB (local) | Resume + job embeddings for RAG matching | Free |
| Primary Embeddings | NVIDIA NIM `nv-embedqa-e5-v5` | 1024-dim embeddings | Free tier |
| Fallback Embeddings | Gemini `text-embedding-004` | 768-dim fallback if NVIDIA NIM unavailable | Free tier |
| Cache / Queue | Redis (local Docker) | Phase 1: response caching · Phase 2: full job queue | Free |
| User Tracking UI | Notion API v1 | Applications DB + Job Tracker DB (UI only — not source of truth) | Free tier |
| API Layer | FastAPI (slim) | Chrome Extension HTTP bridge · RAG proxy · manual apply logging | Free |
| Manual Assist | Chrome Extension v1 | Autofill · match analysis · resume suggestion · tracking | Free |
| Containerisation | Docker + Docker Compose | Full local dev stack | Free |
| CI/CD + Cron Scheduler | GitHub Actions | `0 6 * * 1,4,6` UTC · CI/CD pipelines | Free tier |
| Observability | AgentOps | Agent traces · LLM cost tracking · error patterns | Free tier |
| Secrets Management | `~/java.env` | Single git-ignored env file — passed to Docker via `--env-file` | Free |

### LLM Provider Summary

| Provider | Models Used | Tier | Monthly Budget |
|----------|------------|------|----------------|
| **xAI** | `grok-4-fast-reasoning` (Analyser) · `grok-4-1-fast-reasoning` (Apply) · `grok-3-mini-latest` (Dev Agent P2) | Paid | **$5/month** |
| **Perplexity** | `sonar` (Scraper) · `sonar` fallback (Dev Agent P2) | Paid | **$5/month** |
| **Groq** | `llama-3.3-70b-versatile` (Master + Tracker) | Free tier | — |
| **Cerebras** | `llama-3.3-70b` (All agent fallbacks) | Free tier | — |
| **SambaNova** | `Llama-3.1-70B` (Analyser + Apply fallback 1) | Free tier | — |
| **NVIDIA NIM** | `nv-embedqa-e5-v5` (Primary embeddings) | Free tier | — |
| **Google Gemini** | `text-embedding-004` (Fallback embeddings) | Free tier | — |

> **Hard cap: $10/month total LLM spend · $0.38/run xAI cap — both enforced in code. Run aborts gracefully if cap is hit.**

---

## 📊 Scoring & Routing

Every eligible job is scored on a **0.0 – 1.0 fit scale** by the Analyser Agent using skills match, experience alignment, role relevance, and seniority fit. Routing is a separate decision based solely on form complexity and platform compliance.

| Fit Score | Action |
|-----------|--------|
| `< 0.40` | ❌ Skip entirely — no Notion entry · no Postgres record |
| `0.40 – 0.49` | 📋 Manual queue only — low confidence, never auto-applied |
| `0.50 – 0.74` | ⚖️ Route by form complexity — auto-apply or manual |
| `≥ 0.75` | ✅ Route by form complexity — auto-apply or manual |
| `> 0.90` | 📋 **Force manual queue** — high match = high stakes, always manual |

> **Routing rule: Form complexity + platform compliance ONLY. Fit score is used for ranking and prioritisation — never for auto-apply routing decisions.**

---

## 🧠 RAG Resume System

The RAG system matches every scored job to the optimal resume variant using a local ChromaDB vector search against all 15 resume embeddings.

| Config | Detail |
|--------|--------|
| Vector Store | ChromaDB (fully local, persistent on disk) |
| Primary Embeddings | NVIDIA NIM `nv-embedqa-e5-v5` — 1024 dimensions |
| Fallback Embeddings | Gemini `text-embedding-004` — 768 dimensions |
| Total Resume Variants | **15** |
| RAG Failure Fallback | `Aarjun_Gen.pdf` assigned as default if ChromaDB query fails |

### Resume Variants

| # | File | Type |
|---|------|------|
| 1 | `Aarjun_Gen.pdf` | General |
| 2 | `Aarjun_OG.pdf` | Original / Base |
| 3 | `Aarjun_AIAutomation.pdf` | Domain-specific |
| 4 | `Aarjun_AIML.pdf` | Domain-specific |
| 5 | `Aarjun_AISolutionsArchitect.pdf` | Domain-specific |
| 6 | `Aarjun_AppliedML.pdf` | Domain-specific |
| 7 | `Aarjun_DataEngineering.pdf` | Domain-specific |
| 8 | `Aarjun_DataScience.pdf` | Domain-specific |
| 9 | `Aarjun_FeatureEngineering.pdf` | Domain-specific |
| 10 | `Aarjun_LLMFineTuning.pdf` | Domain-specific |
| 11 | `Aarjun_LLMGenAI.pdf` | Domain-specific |
| 12 | `Aarjun_MLDataEngineer.pdf` | Domain-specific |
| 13 | `Aarjun_MLOps.pdf` | Domain-specific |
| 14 | `Aarjun_PromptEngineer.pdf` | Domain-specific |
| 15 | `Aarjun_RAGEngineer.pdf` | Domain-specific |

---

## ⚙️ Infrastructure

### Data Layer

| Store | Technology | Role |
|-------|-----------|------|
| System of Record | PostgreSQL — Supabase (prod) / local Docker (dev) | `jobs` · `applications` · `run_sessions` · `audit_logs` |
| Vector Store | ChromaDB (local) | Resume + job embeddings — RAG only |
| Cache / Queue | Redis (local Docker) | Phase 1: response caching · Phase 2: full queue + inter-service cache |
| User UI | Notion API v1 | Applications DB · Job Tracker DB — read/write UI surface only |

Switch between prod and dev database via `ACTIVE_DB` environment variable (`supabase` or `local`).

### Notion Databases

**Applications DB** — manual queue jobs awaiting user review:
`Job Title` · `Company` · `Job URL` · `Application Deadline` · `Platform` · `Status` · `Fit Score` · `Resume Suggested` · `Match Reasoning` · `CTC` · `Priority` · `Run ID`

**Job Tracker DB** — applied jobs pipeline:
`Job Title` · `Company` · `Job URL` · `Stage` · `Date Applied` · `Platform` · `Applied Via` · `CTC` · `Notes` · `Run ID`

### Cron Schedule

```
GitHub Actions Cron:  0 6 * * 1,4,6
Timezone:             UTC → 12:00 PM IST
Days:                 Monday (1) · Thursday (4) · Saturday (6)
Approx runs/month:    ~13
```

### Secrets Management

All secrets live in `~/java.env` — a single, **git-ignored** file that is the source of truth for all API keys, DB URLs, and config values. Auto-loaded by the shell and passed to all Docker containers via `--env-file ~/java.env`.

---

## 🔧 Setup & Installation

### Prerequisites

- Python 3.11+
- Docker + Docker Compose
- Git
- Chrome / Chromium (for extension)

### Step 1 — Clone Repository

```bash
git clone https://github.com/aarjunm04/AI_Job_Automation_Agent.git
cd AI_Job_Automation_Agent
```

### Step 2 — Configure Environment

```bash
# Create the global env file (single source of truth — git-ignored)
touch ~/java.env
nano ~/java.env   # Add all required variables — see Environment Variables section below
```

### Step 3 — Boot Full Stack

```bash
# Start all services: Postgres, Redis, FastAPI
docker-compose --env-file ~/java.env up -d

# Verify all containers are healthy
docker-compose ps
```

### Step 4 — Install Python Dependencies

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
playwright install chromium
```

### Step 5 — Initialise Database & RAG

```bash
# Run Postgres schema migrations
python scripts/migrate_db.py

# Embed all 15 resume variants into local ChromaDB
python scripts/embed_resumes.py
```

### Step 6 — Verify Setup

```bash
python main.py --health-check
# Expected: ✅ All systems healthy
```

---

## 🔐 Environment Variables

All variables live in `~/java.env`. This file is never committed to git and is auto-loaded by your shell and Docker via `--env-file`.

```env
# ── LLM APIs ───────────────────────────────────────────────────────────────
XAI_API_KEY=                        # xAI — Analyser + Apply agents + Dev Agent (Phase 2)
PERPLEXITY_API_KEY=                  # Perplexity — Scraper sonar + Dev Agent fallback (P2)
GROQ_API_KEY=                        # Groq — Master + Tracker agents
CEREBRAS_API_KEY=                    # Cerebras — all agent fallbacks
SAMBANOVA_API_KEY=                   # SambaNova — Analyser + Apply fallback 1
NVIDIA_NIM_API_KEY=                  # NVIDIA NIM — primary embeddings (nv-embedqa-e5-v5)
GEMINI_API_KEY=                      # Google Gemini — fallback embeddings (text-embedding-004)

# ── Database ───────────────────────────────────────────────────────────────
ACTIVE_DB=supabase                   # supabase | local  (switches DB target)
SUPABASE_URL=                        # Supabase project URL
SUPABASE_KEY=                        # Supabase service/anon key
LOCAL_POSTGRES_URL=                  # postgresql://user:pass@localhost:5432/dbname

# ── Notion ─────────────────────────────────────────────────────────────────
NOTION_API_KEY=                      # Notion integration token
NOTION_APPLICATIONS_DB_ID=           # Applications DB — manual queue
NOTION_JOB_TRACKER_DB_ID=            # Job Tracker DB — applied pipeline

# ── Discovery ──────────────────────────────────────────────────────────────
SERPAPI_KEY_1=                       # SerpAPI account 1 (250 credits/month)
SERPAPI_KEY_2=                       # SerpAPI account 2 (250 credits/month)
SERPAPI_KEY_3=                       # SerpAPI account 3 (250 credits/month)
SERPAPI_KEY_4=                       # SerpAPI account 4 (250 credits/month)
JOOBLE_API_KEY=                      # Jooble REST API key (Phase 2 only)

# ── Proxies ────────────────────────────────────────────────────────────────
WEBSHARE_PROXY_LIST=                 # Comma-separated list of 20 static proxy addresses

# ── Platform Credentials ───────────────────────────────────────────────────
LINKEDIN_EMAIL=
LINKEDIN_PASSWORD=

# ── Observability ──────────────────────────────────────────────────────────
AGENTOPS_API_KEY=                    # AgentOps — traces + cost tracking

# ── Budget Enforcement ─────────────────────────────────────────────────────
XAI_BUDGET_CAP_PER_RUN=0.38          # Hard xAI spend cap per run (USD)
XAI_MONTHLY_BUDGET=5.00              # xAI monthly budget (USD)
PERPLEXITY_MONTHLY_BUDGET=5.00       # Perplexity monthly budget (USD)

# ── System ─────────────────────────────────────────────────────────────────
TIMEZONE=Asia/Kolkata
```

---

## 🗺️ Project Roadmap

### Phase 1 — Core System Build (4 Weeks)

| Milestone | Target | Deliverable |
|-----------|--------|-------------|
| **M1** | Week 1 | Repo structure · Docker Compose · `java.env` schema · Postgres schema + Supabase setup · ChromaDB init |
| **M2** | Week 1–2 | RAG system live — all 15 resumes embedded · NVIDIA NIM primary + Gemini fallback |
| **M3** | Week 2 | CrewAI framework setup · Master Agent · AgentOps integration |
| **M4** | Week 2–3 | Scraper Agent — all 10 platforms + safety-net · SerpAPI · Perplexity `sonar` normalisation |
| **M5** | Week 3 | Analyser Agent — eligibility filter · fit scoring · RAG match · routing decision |
| **M6** | Week 3–4 | Apply Agent — all platform Playwright scripts · retry logic · submission proof capture |
| **M7** | Week 4 | Tracker Agent · Notion sync · FastAPI slim server · Chrome Extension v1 |
| **🚀 M8** | **End of Week 4** | **Phase 1 Launch — first live fully autonomous run** |

### Phase 2 — Self-Improving System (8 Weeks)

| Milestone | Target | Deliverable |
|-----------|--------|-------------|
| **M9** | Week 5–6 | Redis full queue + inter-service cache system |
| **M10** | Week 6–7 | Developer Agent live — `grok-3-mini-latest` · AgentOps cross-session trace analysis |
| **M11** | Week 7–8 | Remotive + Jooble platform integration · all-round resume variant |
| **🏁 M12** | **End of Week 8** | **Phase 2 Complete — self-improving autonomous system** |

---

## 📁 Repository Structure

```
AI_Job_Automation_Agent/
│
├── main.py                         # Entry point — run session trigger
├── requirements.txt                # Python dependencies
├── docker-compose.yml              # Full local dev stack (Postgres · Redis · FastAPI)
│
├── .github/
│   └── workflows/
│       └── run_agent.yml           # GitHub Actions cron: 0 6 * * 1,4,6 (UTC = 12PM IST)
│
├── agents/                         # CrewAI Agent + Task definitions
│   ├── master_agent.py             # Crew Manager — session boot · budget enforcement · orchestration
│   ├── scraper_agent.py            # Discovery specialist — all platforms + normalisation
│   ├── analyser_agent.py           # Scoring + routing specialist — RAG match · fit scoring
│   ├── apply_agent.py              # Submission specialist — Playwright form filling
│   ├── tracker_agent.py            # Record keeper — Postgres · Notion · AgentOps
│   └── developer_agent.py          # Phase 2 — improvement analyst (runs outside pipeline)
│
├── tools/                          # @tool decorated functions registered to CrewAI agents
│   ├── postgres_tools.py           # DB read/write tools
│   ├── playwright_tools.py         # Browser automation tools
│   ├── rag_tools.py                # ChromaDB vector search + resume selection tools
│   ├── notion_tools.py             # Notion API read/write tools
│   ├── scraper_tools.py            # JobSpy · REST API · SerpAPI tools
│   └── agentops_tools.py           # Observability + cost tracking tools
│
├── scrapers/                       # Platform scraper implementations
│   ├── jobspy_scraper.py           # LinkedIn + Indeed via JobSpy library
│   ├── remoteok_api.py             # RemoteOK public JSON API
│   ├── himalayas_api.py            # Himalayas public REST API
│   └── playwright_scrapers/        # Per-platform Playwright scraper scripts
│       ├── wellfound.py
│       ├── weworkremotely.py
│       ├── yc_startup.py
│       ├── turing.py
│       ├── crossover.py
│       ├── arc_dev.py
│       ├── nodesk.py               # Safety-net platform
│       └── toptal.py               # Safety-net platform
│
├── rag/                            # RAG system
│   ├── embedder.py                 # NVIDIA NIM + Gemini fallback embedding logic
│   ├── chromadb_store.py           # ChromaDB vector store management
│   └── resume_matcher.py           # Job-to-resume cosine matching + selection
│
├── db/                             # Database layer
│   ├── schema.sql                  # Postgres table definitions (jobs · applications · run_sessions · audit_logs)
│   ├── supabase_client.py          # Supabase connection + query interface
│   └── local_postgres_client.py   # Local Postgres connection (dev)
│
├── api/                            # Slim FastAPI server
│   └── server.py                   # 3 endpoints: /extension · /rag-proxy · /manual-log
│
├── extension/                      # Chrome Extension v1 (canonical)
│   ├── manifest.json
│   ├── popup.html
│   ├── popup.js
│   └── content.js
│
├── config/                         # Static configuration files
│   ├── platforms.json              # Per-platform scraping config (rate limits · selectors · compliance)
│   └── search_params.json          # Job search parameters (titles · keywords · filters)
│
├── scripts/                        # Utility scripts
│   ├── migrate_db.py               # Run Postgres schema migrations
│   ├── embed_resumes.py            # Embed all 15 resume variants into ChromaDB
│   └── health_check.py             # System-wide health verification
│
└── resumes/                        # Resume variants — git-ignored
    ├── Aarjun_Gen.pdf              # General
    ├── Aarjun_OG.pdf               # Original / Base
    ├── Aarjun_AIAutomation.pdf
    ├── Aarjun_AIML.pdf
    ├── Aarjun_AISolutionsArchitect.pdf
    ├── Aarjun_AppliedML.pdf
    ├── Aarjun_DataEngineering.pdf
    ├── Aarjun_DataScience.pdf
    ├── Aarjun_FeatureEngineering.pdf
    ├── Aarjun_LLMFineTuning.pdf
    ├── Aarjun_LLMGenAI.pdf
    ├── Aarjun_MLDataEngineer.pdf
    ├── Aarjun_MLOps.pdf
    ├── Aarjun_PromptEngineer.pdf
    └── Aarjun_RAGEngineer.pdf
```

---

## 💰 Budget

| Service | Monthly Cost | Coverage |
|---------|-------------|----------|
| xAI | **$5.00** | `grok-4-fast-reasoning` (Analyser) · `grok-4-1-fast-reasoning` (Apply) · `grok-3-mini-latest` (Dev Agent Phase 2) |
| Perplexity | **$5.00** | `sonar` (Scraper) · `sonar` Dev Agent fallback (Phase 2) |
| Groq | Free tier | `llama-3.3-70b-versatile` — Master + Tracker agents |
| Cerebras | Free tier | `llama-3.3-70b` — all agent fallbacks |
| SambaNova | Free tier | `Llama-3.1-70B` — Analyser + Apply fallback 1 |
| NVIDIA NIM | Free tier | `nv-embedqa-e5-v5` — primary embeddings |
| Google Gemini | Free tier | `text-embedding-004` — fallback embeddings |
| Supabase | Free tier | 500 MB DB / 2 GB bandwidth |
| GitHub Actions | Free tier | ~13 runs/month — well within 2,000 min/month free limit |
| AgentOps | Free tier | Observability + cost tracking |
| Webshare Proxies | Separate budget | 20 static proxies — NOT included in $10 LLM cap |

> **Total LLM hard cap: $10/month · Per-run xAI cap: $0.38 — both enforced in application code.**

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

<div align="center">

Built by **Aarjun Mahule**

[⬆ Back to Top](#-ai-job-automation-agent)

</div>
