# AI Job Automation Agent

Foundational work on a fully autonomous job application system with a strong scrape → analyze → RAG → extension feedback loop. The repository currently wires the data-fetching and resume recommendation layers, but the CrewAI-style agents, tooling, and database schemas referenced in the original spec remain scaffolds.

## Snapshot

- `PROJECT_STATUS.md` is the live build snapshot for the current sprint/phase. It lists completion percentage (~25% as of Feb 28, 2026), priority gaps, and infrastructure notes so you always know what is actually wired up right now.
- `IDE_README.md` and `assets/Architecture_v2_Specification.md` capture the aspirational plan. Treat them as documentation of the original spec rather than the current runnable stack.

## Implemented Systems

### 1. Scraper Engine (`scrapers/`)

- `scraper_engine.py` orchestrates JobSpy (LinkedIn + Indeed), RemoteOK/Himalayas APIs, SerpAPI, and Playwright-based scrapers, normalises jobs into a common schema, deduplicates them, and exposes a CLI (`python -m scrapers.scraper_engine`).
- `jobspy_adapter.py` patches JobSpy for production use (bad country strings, site allowlists, per-site limits) and hands raw job dictionaries to the engine.
- `scraper_service.py` contains the Playwright infrastructure with Webshare proxy rotation, shared browser manager, and site-specific scrapers (Wellfound, WWR, YC/WorkAtAStartup, Turing, Crossover, Arc.dev, Nodesk, Toptal).
- `job_filters.yaml` drives filtering and static scoring without any downstream routing logic.

### 2. RAG & Resume Stack (`rag_systems/`)

- `chromadb_store.py` wraps ChromaDB v1.x (PersistentClient) with retries, metadata sanitization, and fallback storage when the package is unavailable.
- `resume_engine.py` ingests the 20+ resume variants defined in `resume_config.json`, extracts PDF text via PyPDF2, chunks it, embeds it with NVIDIA NIM embeddings (with Gemini as fallback), and persists chunks+anchors.
- `rag_pipeline.py` and `rag_api.py` expose a singleton resume engine plus helpers for building grounding context and reranking.
- `production_server.py` spins up a FastAPI/uvicorn service (port 8090 by default) with endpoints for `/rag/query`, `/rag/select`, `/resumes`, `/resumes/list`, `/resumes/reindex/{resume_id}`, session management, cache invalidation, and metrics. The RAG server also enforces API keys, rate limiting, circuit breaking, cache TTLs, and Prometheus metrics.
- `ingest_all_resumes.py` has orchestration helpers for (re)embedding every resume variant when your config or embedding provider changes.
- **Heads up:** `resume_config.json` still points to `/Users/apple/TechStack/Resume/*.pdf`. Replace these paths with your own PDFs or symlink that directory before running the ingest scripts.

### 3. Chrome Extension (`chrome_extension/`)

- Manifest v3 extension with content script, background service worker, popup UI, and sidebar. The extension includes platform detection selectors for LinkedIn, Indeed, Naukri, Wellfound/AngelList, and a generic ATS detector.
- `content.js` extracts job metadata, watches for form state changes, and proxies events to the background worker. `background.js` brokers messages, talks to MCP/RAG/MCP proxies (via `mcp_client.js`), loads `extension_config.js`, logs to Notion, and exposes keyboard shortcuts (`Ctrl+Shift+A/F/S`).
- `sidebar.js/html/css` and `popup.js/html` provide the UI surface for match reasoning, resume suggestion, and manual apply hooks. Notion integration is wired via webhook fallbacks where the Notion SDK isn’t available.

### 4. LLM Interface (`integrations/llm_interface.py`)

- Centralised Gemini client with OpenRouter fallback. `LLMInterface.query()` tries Gemini first and only hits OpenRouter when Gemini fails, so downstream agents can rely on a two-tier LLM supply chain.

### 5. Orchestration & Infrastructure

- `master_run.py` is currently the single entry point described in the spec, but it imports `config.settings`, `mcp_client`, and `core` (including engines for notion, scraper, resume, automation) that do not exist yet. Running `python master_run.py` currently fails before it does anything meaningful.
- `docker-compose.yml` wires PostgreSQL 15, Redis 7, `mcp-server`, and `rag-server`. **Caution:** the `mcp/` directory and its Dockerfile are absent, and `rag_systems/Dockerfile` is missing too, so the compose stack cannot build the MCP/RAG services without those artifacts.
- Requirements list includes `crewai`, `playwright`, `fastapi`, `sentence-transformers`, `chroma`, `mcp`, `rasa`, and other dependencies needed once the missing modules are implemented.
- Secrets live in `~/narad.env` (per the original spec). That file is git-ignored and must be created manually. The template (`narad.env.template`) is currently missing, so copy the `Environment Variables` section from `IDE_README.md` or this README and fill in the keys yourself.

## Supporting Reference Documents

- `PROJECT_STATUS.md`: live sprint snapshot with priorities, missing files, and infrastructure notes.
- `IDE_README.md`: coordination doc / spec (not the live codebase). Use it to track what still needs building.
- `assets/Architecture_v2_Specification.md` and `assets/System_Requirements_and_Constraints.md`: architectural notes that inspired the repo layout.

## Known Gaps (Phase 1 is still early)

- `agents/`, `tools/`, `platforms/`, `auto_apply/`, `database/`, `api/`, `scripts/`, and `test_scripts/` directories exist but contain empty stubs; nothing is implemented inside yet.
- `config/platform_config.json`, `scoring_weights.json`, `search_queries.json`, and `user_preferences.json` are empty placeholders.
- `config/settings.py`, `core/`, `mcp_client.py`, and the `mcp/` directory are completely missing — master_run.py depends on them.
- `narad.env.template` is absent, so secrets must be documented elsewhere before instantiating `~/narad.env`.
- `database/schema.sql`, `database/init.sql`, and migration scripts are empty, leaving Postgres unprepared.
- `rag_systems/Dockerfile` is now present, allowing the `docker-compose` rag service to build.
- Resume PDFs referenced in `resume_config.json` live outside the repo; you must supply equivalent files.

## Getting Started (explore what exists today)

1. `python -m venv venv && source venv/bin/activate` (Windows: `venv\Scripts\activate`).
2. `pip install -r requirements.txt` plus `playwright install chromium` if you want to exercise the Playwright scrapers.
3. Create `~/narad.env` with at least the keys consumed by `scrapers/` and `rag_systems/`: `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB`, `LOCAL_POSTGRES_URL`, `REDIS_URL`, `SERPAPI_API_KEY_*`, `WEBSHARE_PROXY_*`, `NVIDIA_NIM_API_KEY`, `GEMINI_API_KEY`, `RAG_SERVER_API_KEY`, `NOTION_API_KEY`, and any Chrome extension secrets referenced in `chrome_extension/extension_config.js`.
4. Run the RAG server manually: `python rag_systems/production_server.py`. It will log API keys, rate limits, and metrics at startup and expose `/rag/query`, `/rag/select`, `/resumes/list`, `/resumes/reindex/{resume_id}`, session APIs, cache invalidation, and `/metrics`.
5. Run the scraper CLI: `python -m scrapers.scraper_engine`. Results, metrics, and SerpAPI usage are persisted under `logs/`.
6. Load the Chrome extension by pointing Chrome/Edge to `chrome_extension/` (developer mode). The content script, background script, sidebar/popup, and `extension_config.js` work against the RAG server + Notion integrations described above.

## Next Steps (Phase 1 priorities)

1. Implement the missing `config/settings.py`, `core/`, `mcp_client.py`, and CrewAI-style agent modules so `master_run.py` can orchestrate the scraper, analyser, apply, and tracker stages.
2. Flesh out `agents/`, `tools/`, `platforms/`, and `auto_apply/` with the CrewAI `@tool` functions, platform-specific Playwright scripts, and apply/ATS automation logic.
3. Populate `database/schema.sql`/`init.sql`, create migrations, and supply a `narad.env.template` so infrastructure can be bootstrapped reproducibly.
4. Provide a Dockerfile for `rag_systems/` (and deliver the missing `mcp/` build) so `docker-compose.yml` can stand up the whole stack.

Refer to `PROJECT_STATUS.md` for the current sprint-day targets if you want to align work with the IDE schedule.
