# Project Status Report
_Current snapshot: 2026-02-28 (Phase 1 · Sprint 1 Week 1)._ 

## Current project status report begin
- **Live layers:** `scrapers/scraper_engine.py` already orchestrates JobSpy / REST / Playwright sources, normalisation, logging, SerpAPI tracking, and CLI export; `rag_systems/*` exposes ingestion, embedding, and the FastAPI RAG server; `chrome_extension/` is Manifest V3 with content/background/popup UI; `integrations/llm_interface.py` handles Gemini → OpenRouter fallback.
- **Pending build:** CrewAI agents, apply automation, tools, platforms, and database wiring remain stubs, so `master_run.py` cannot orchestrate a run yet; config JSONs, migrations, and `scripts/`+`test_scripts/` do not contain runnable logic.
- **Blockers:** `docker-compose.yml` references absent `mcp/` build context and lacks `rag_systems/Dockerfile`; `narad.env.template` is missing, so secrets still rely on ad-hoc notes; resume PDFs in `rag_systems/resume_config.json` live outside the repo and must be supplied locally.

## File & Directory Status
| Path | Status | Notes |
| `.gitignore` | Active | Keeps secrets (`narad.env`) and caches ignored. |
| `.DS_Store` | macOS artifact | Present from Finder; safe to ignore. |
| `IDE_README.md` | Live spec | Sprint plan, agent choreography, and change log owned by PERPLEXITY_PRO. |
| `README.md` | Current-state doc | Explains implemented stack, notes missing components, and now points here (`PROJECT_STATUS.md`). |
| `PROJECT_STATUS.md` | This file | Captures today’s snapshot for quick reference; update each sprint. |
| `Utils/` | Empty stubs | `normalise_dedupe.py` and `proxy_ratelimit.py` have no logic yet. |
| `agents/` | Placeholders | Six CrewAI agents exist but all modules are empty skeletons. |
| `api/` | Placeholder | `api_server.py` contains no implementation. |
| `assets/` | Documentation | Architecture/system requirement docs plus diagrams (PNG/MDF). |
| `auto_apply/` | Planning | `documentation.md` exists, but `ats_detector.py` and `form_filler.py` are empty. |
| `chrome_extension/` | Implemented | Manifest V3, content/background scripts, sidebar/popup, Notion hooks, ATS detection selectors all wired. |
| `config/` | Placeholders | `platform_config.json`, `scoring_weights.json`, `search_queries.json`, `user_preferences.json` are empty. |
| `database/` | Empty | `schema.sql`, `init.sql`, and client helpers are blank; Postgres schema/migrations not defined. |
| `docker-compose.yml` | Infrastructure spec | Declares Postgres/Redis/MCP/RAG but `mcp/` directory and `rag_systems/Dockerfile` are missing, so the stack does not build. |
| `integrations/` | Mixed | `llm_interface.py` is functional (Gemini + OpenRouter), `notion.py` is empty. |
| `master_run.py` | Broken entry | Imports `config.settings`, `core`, and `mcp_client` that do not exist yet, so startup fails. |
| `platforms/` | Empty stubs | Platform-specific modules (LinkedIn, RemoteOK, etc.) exist but contain no code. |
| `postgres_data/` | Data volume | Persisted by Docker Compose; holds runtime Postgres files but not part of source. |
| `pyproject.toml` | Empty | Placeholder with no metadata or dependencies. |
| `rag_systems/` | Working core | Contains ChromaDB wrapper, ingestion pipeline, resume config, and production FastAPI server; requires on-disk resumes (paths point to `/Users/apple/TechStack/Resume/*.pdf`) and still lacks a Dockerfile. |
| `requirements.txt` | Ready | Pinned dependencies for scraping, CrewAI, Playwright, FastAPI, Chroma, embeddings, etc. |
| `scrapers/` | Active engine | `scraper_engine.py` is runnable, `jobspy_adapter.py`, `scraper_service.py`, and `job_filters.yaml` support normalization, dedupe, and proxy-aware Playwright. |
| `scripts/` | Empty | Directory currently has no scripts. |
| `test_scripts/` | Placeholders | Test files exist but contain no assertions or logic. |
| `tools/` | Empty | `agentops_tools.py`, `apply_tools.py`, `notion_tools.py`, `postgres_tools.py`, `rag_tools.py`, `scraper_tools.py`, and `serpapi_tool.py` are blank. |
| `venv/` | Local env | Contains the developer’s virtual environment; not committed. |
