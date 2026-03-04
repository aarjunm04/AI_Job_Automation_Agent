---

# PROJECT_FILE_STATUS
<!-- AI Job Agent | Generated: 2026-03-04 | Author: GEMINI | DO NOT EDIT MANUALLY -->

## SUMMARY
| Metric | Count |
|--------|-------|
| Total files scanned | 88 |
| Clean (no stubs, no import risk) | 56 |
| Has TODO / stub | 32 |
| Has import risk | 0 |
| System readiness | 63% |

## SPRINT STATUS
| Field | Value |
|-------|-------|
| Sprint | Phase 1 — 2026-03-01 to 2026-03-14 |
| Today | Day 4 — 2026-03-04 |
| Last changelog | L068 — CLAUDE — agents/master_agent.py — MODIFY |
| Next milestone | M1 — Infra + 15 resumes in ChromaDB — TODAY |
| Phase 1 launch | 2026-03-14 |

## FILE STATUS TABLE
| File | Lines | Has Stub/TODO | Import Risk | Status |
|------|-------|--------------|-------------|--------|
| ./agents/__init__.py | 1 | NO | NO | ✅ CLEAN |
| ./agents/analyser_agent.py | 1137 | YES | NO | ⚠️ STUB |
| ./agents/apply_agent.py | 1308 | YES | NO | ⚠️ STUB |
| ./agents/developer_agent.py | 1399 | NO | NO | ✅ CLEAN |
| ./agents/master_agent.py | 1122 | YES | NO | ⚠️ STUB |
| ./agents/scraper_agent.py | 385 | YES | NO | ⚠️ STUB |
| ./agents/tracker_agent.py | 523 | YES | NO | ⚠️ STUB |
| ./api/__init__.py | 1 | NO | NO | ✅ CLEAN |
| ./api/api_server.py | 1261 | YES | NO | ⚠️ STUB |
| ./auto_apply/__init__.py | 0 | NO | NO | ✅ CLEAN |
| ./auto_apply/ats_detector.py | 822 | YES | NO | ⚠️ STUB |
| ./auto_apply/form_filler.py | 1212 | YES | NO | ⚠️ STUB |
| ./auto_apply/platforms/__init__.py | 30 | NO | NO | ✅ CLEAN |
| ./auto_apply/platforms/arc_dev.py | 493 | YES | NO | ⚠️ STUB |
| ./auto_apply/platforms/base_platform.py | 542 | YES | NO | ⚠️ STUB |
| ./auto_apply/platforms/greenhouse.py | 445 | NO | NO | ✅ CLEAN |
| ./auto_apply/platforms/indeed.py | 573 | YES | NO | ⚠️ STUB |
| ./auto_apply/platforms/lever.py | 669 | NO | NO | ✅ CLEAN |
| ./auto_apply/platforms/linkedin_easy_apply.py | 342 | YES | NO | ⚠️ STUB |
| ./auto_apply/platforms/native_form.py | 805 | YES | NO | ⚠️ STUB |
| ./auto_apply/platforms/wellfound.py | 482 | YES | NO | ⚠️ STUB |
| ./auto_apply/platforms/workday.py | 1130 | YES | NO | ⚠️ STUB |
| ./config/__init__.py | 1 | NO | NO | ✅ CLEAN |
| ./config/platform_config.json | 0 | NO | NO | ✅ CLEAN |
| ./config/platforms.json | 281 | NO | NO | ✅ CLEAN |
| ./config/scoring_weights.json | 21 | NO | NO | ✅ CLEAN |
| ./config/search_queries.json | 45 | NO | NO | ✅ CLEAN |
| ./config/settings.py | 238 | NO | NO | ✅ CLEAN |
| ./config/user_preferences.json | 42 | NO | NO | ✅ CLEAN |
| ./database/init.sql | 2 | NO | NO | ✅ CLEAN |
| ./database/local_postgres_client.py | 0 | NO | NO | ✅ CLEAN |
| ./database/migrations/v001_baseline.sql | 5 | NO | NO | ✅ CLEAN |
| ./database/migrations/v002_rename_tables.sql | 13 | NO | NO | ✅ CLEAN |
| ./database/schema.sql | 156 | YES | NO | ⚠️ STUB |
| ./database/supabase_client.py | 0 | NO | NO | ✅ CLEAN |
| ./docker-compose.yml | 186 | NO | NO | ✅ CLEAN |
| ./extension/manifest.json | 38 | NO | NO | ✅ CLEAN |
| ./integrations/__init__.py | 1 | NO | NO | ✅ CLEAN |
| ./integrations/llm_interface.py | 424 | YES | NO | ⚠️ STUB |
| ./integrations/notion.py | 342 | YES | NO | ⚠️ STUB |
| ./main.py | 273 | NO | NO | ✅ CLEAN |
| ./platforms/arcdev.py | 0 | NO | NO | ✅ CLEAN |
| ./platforms/crossover.py | 0 | NO | NO | ✅ CLEAN |
| ./platforms/himalayas.py | 0 | NO | NO | ✅ CLEAN |
| ./platforms/indeed.py | 0 | NO | NO | ✅ CLEAN |
| ./platforms/jooble.py | 0 | NO | NO | ✅ CLEAN |
| ./platforms/linkedin.py | 0 | NO | NO | ✅ CLEAN |
| ./platforms/nodesk.py | 0 | NO | NO | ✅ CLEAN |
| ./platforms/remoteok.py | 0 | NO | NO | ✅ CLEAN |
| ./platforms/remotive.py | 0 | NO | NO | ✅ CLEAN |
| ./platforms/toptal.py | 0 | NO | NO | ✅ CLEAN |
| ./platforms/turing.py | 0 | NO | NO | ✅ CLEAN |
| ./platforms/wellfound.py | 0 | NO | NO | ✅ CLEAN |
| ./platforms/wwr.py | 0 | NO | NO | ✅ CLEAN |
| ./platforms/yc.py | 0 | NO | NO | ✅ CLEAN |
| ./rag_systems/__init__.py | 1 | NO | NO | ✅ CLEAN |
| ./rag_systems/chromadb_store.py | 391 | YES | NO | ⚠️ STUB |
| ./rag_systems/ingestion.py | 280 | NO | NO | ✅ CLEAN |
| ./rag_systems/production_server.py | 1353 | YES | NO | ⚠️ STUB |
| ./rag_systems/rag_api.py | 139 | YES | NO | ⚠️ STUB |
| ./rag_systems/rag_pipeline.py | 494 | YES | NO | ⚠️ STUB |
| ./rag_systems/resume_config.json | 164 | NO | NO | ✅ CLEAN |
| ./rag_systems/resume_engine.py | 340 | NO | NO | ✅ CLEAN |
| ./scrapers/__init__.py | 75 | YES | NO | ⚠️ STUB |
| ./scrapers/jobspy_adapter.py | 408 | YES | NO | ⚠️ STUB |
| ./scrapers/scraper_engine.py | 1580 | YES | NO | ⚠️ STUB |
| ./scrapers/scraper_service.py | 855 | YES | NO | ⚠️ STUB |
| ./test_scripts/test_analyser.py | 0 | NO | NO | ✅ CLEAN |
| ./test_scripts/test_apply.py | 0 | NO | NO | ✅ CLEAN |
| ./test_scripts/test_e2e.py | 0 | NO | NO | ✅ CLEAN |
| ./test_scripts/test_fastapi.py | 0 | NO | NO | ✅ CLEAN |
| ./test_scripts/test_scraper.py | 0 | NO | NO | ✅ CLEAN |
| ./tools/__init__.py | 1 | NO | NO | ✅ CLEAN |
| ./tools/agentops_tools.py | 251 | YES | NO | ⚠️ STUB |
| ./tools/apply_tools.py | 975 | YES | NO | ⚠️ STUB |
| ./tools/budget_tools.py | 340 | NO | NO | ✅ CLEAN |
| ./tools/notion_tools.py | 447 | YES | NO | ⚠️ STUB |
| ./tools/postgres_tools.py | 848 | YES | NO | ⚠️ STUB |
| ./tools/rag_tools.py | 264 | NO | NO | ✅ CLEAN |
| ./tools/scraper_tools.py | 818 | YES | NO | ⚠️ STUB |
| ./tools/serpapi_tool.py | 140 | YES | NO | ⚠️ STUB |
| ./utils/__init__.py | 24 | NO | NO | ✅ CLEAN |
| ./utils/db_utils.py | 52 | NO | NO | ✅ CLEAN |
| ./utils/normalise_dedupe.py | 126 | YES | NO | ⚠️ STUB |
| ./utils/proxy_ratelimit.py | 156 | NO | NO | ✅ CLEAN |

## RISKS
| File | Issue | Recommended Fix |
|------|-------|----------------|
| ./agents/analyser_agent.py | Has TODO/stub | Implement logic |
| ./agents/apply_agent.py | Has TODO/stub | Implement logic |
| ./agents/master_agent.py | Has TODO/stub | Implement logic |
| ./agents/scraper_agent.py | Has TODO/stub | Implement logic |
| ./agents/tracker_agent.py | Has TODO/stub | Implement logic |
| ./api/api_server.py | Has TODO/stub | Implement logic |
| ./auto_apply/ats_detector.py | Has TODO/stub | Implement logic |
| ./auto_apply/form_filler.py | Has TODO/stub | Implement logic |
| ./auto_apply/platforms/arc_dev.py | Has TODO/stub | Implement logic |
| ./auto_apply/platforms/base_platform.py | Has TODO/stub | Implement logic |
| ./auto_apply/platforms/indeed.py | Has TODO/stub | Implement logic |
| ./auto_apply/platforms/linkedin_easy_apply.py | Has TODO/stub | Implement logic |
| ./auto_apply/platforms/native_form.py | Has TODO/stub | Implement logic |
| ./auto_apply/platforms/wellfound.py | Has TODO/stub | Implement logic |
| ./auto_apply/platforms/workday.py | Has TODO/stub | Implement logic |
| ./database/schema.sql | Has TODO/stub | Implement logic |
| ./integrations/llm_interface.py | Has TODO/stub | Implement logic |
| ./integrations/notion.py | Has TODO/stub | Implement logic |
| ./rag_systems/chromadb_store.py | Has TODO/stub | Implement logic |
| ./rag_systems/production_server.py | Has TODO/stub | Implement logic |
| ./rag_systems/rag_api.py | Has TODO/stub | Implement logic |
| ./rag_systems/rag_pipeline.py | Has TODO/stub | Implement logic |
| ./scrapers/__init__.py | Has TODO/stub | Implement logic |
| ./scrapers/jobspy_adapter.py | Has TODO/stub | Implement logic |
| ./scrapers/scraper_engine.py | Has TODO/stub | Implement logic |
| ./scrapers/scraper_service.py | Has TODO/stub | Implement logic |
| ./tools/agentops_tools.py | Has TODO/stub | Implement logic |
| ./tools/apply_tools.py | Has TODO/stub | Implement logic |
| ./tools/notion_tools.py | Has TODO/stub | Implement logic |
| ./tools/postgres_tools.py | Has TODO/stub | Implement logic |
| ./tools/scraper_tools.py | Has TODO/stub | Implement logic |
| ./tools/serpapi_tool.py | Has TODO/stub | Implement logic |
| ./utils/normalise_dedupe.py | Has TODO/stub | Implement logic |

## MISSING FILES
| Missing File | Referenced In | Impact |
|-------------|--------------|--------|

## READY TO RUN CHECKLIST
- [ ] All agent files present (master, scraper, analyser, apply, tracker)
- [ ] All tool files present (rag, scraper, apply, tracker, budget, postgres, notion)
- [ ] RAG system files present (pipeline, chromadb_store, resume_engine, ingestion, production_server)
- [ ] docker-compose.yml present
- [ ] Both Dockerfiles present (root + rag_systems/)
- [ ] .dockerignore present
- [ ] requirements.txt present
- [ ] requirements-dev.txt present
- [ ] database/schema.sql present
- [ ] config/platforms.json present
- [ ] main.py present with --dry-run flag
- [ ] No critical import risks remaining
- [ ] No TODO/stub in any agent or tool file

---
