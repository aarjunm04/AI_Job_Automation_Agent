# AI Job Application Automation Agent
## System Requirements & Constraints

**Version:** 1.0  
**Date:** February 12, 2026  
**Status:** Planning Phase  
**Target Deployment:** Week 6 (March 2026)

---

## Executive Summary

This document captures the complete functional and non-functional requirements for the AI Job Automation Agent. The system is designed around a **3-runs-per-week model** (Monday, Thursday, Saturday at 9 AM IST) targeting **150 jobs per week** (~600/month) with a **70% auto-apply, 30% manual review** strategy. The architecture emphasizes a **lean data stack** (PostgreSQL + Notion), **platform-safe automation**, **cost efficiency** ($6.45/month budget), and **observability** across the entire pipeline.

### Key Metrics Target (Month 3)
- **Applications:** 600/month (150/week)
- **Success Rate:** 90%+ for auto-applications
- **Interview Conversion:** 8-10% (48-60 interviews/month)
- **User Oversight:** <5 hours/week
- **Cost:** ≤$6.45/month
- **Response Time:** <90 minutes per batch run

---

## 1. Functional Requirements (FR)

### FR-1: Multi-Batch Job Discovery
**Priority:** MUST  
**Status:** Planned

**Description:**  
System executes 3 independent batch runs per week (Mon/Thu/Sat @ 9 AM IST). Each batch performs end-to-end workflow: job discovery → scoring → decision → application → logging.

**Acceptance Criteria:**
- Each batch discovers 100-200 unique jobs
- Jobs sourced from 10+ platforms: LinkedIn, Indeed, Glassdoor, Google Jobs, ZipRecruiter, Wellfound, Remotive, WeWorkRemotely, RemoteOK, Jooble, SimplyHired, Stack Overflow Jobs, YC Startup Jobs, Hiring Cafe
- Duplicate rate <5% across batches
- Batch completion time: 60-90 minutes
- Failed batches auto-alert user within 5 minutes

**Dependencies:** Scraper Agent, JobSpy library, platform APIs

**Validation Method:** Weekly metrics review showing 3 successful batch runs with job counts within target range

---

### FR-2: Intelligent Auto-Apply Threshold
**Priority:** MUST  
**Status:** Planned

**Description:**  
Analyzer Agent automatically flags 70% of jobs for autonomous application and 30% for manual review based on match score, complexity, and strategic value.

**Acceptance Criteria:**
- Auto-apply threshold: Match score ≥60
- Manual review: Score 50-100 with additional flags (salary >$150K, dream company, complex requirements)
- Distribution: 35 auto, 15 manual per 50-job batch (±5 tolerance)
- Flagging logic is tunable via configuration
- Manual jobs include: suggested resume, talking points, flag reason

**Dependencies:** Analyzer Agent, RAG service, user preferences config

**Validation Method:** Review of 10 batches showing 30% ±5% manual flagging rate with appropriate reasoning

---

### FR-3: Comprehensive Application Tracking
**Priority:** MUST  
**Status:** Planned

**Description:**  
All applications (auto and manual) are logged to PostgreSQL (via MCP) and mirrored to Notion with complete metadata, decision rationale, and timestamps.

**Acceptance Criteria:**
- 100% of applications logged within 5 minutes of submission
- Data includes: job title, company, URL, score, resume used, status, timestamp, proof data
- Notion databases sync automatically (no manual export)
- Historical data retained for 90 days in MCP, permanently in Notion
- Audit trail immutable and queryable

**Dependencies:** Tracker Agent, MCP Server, Notion API

**Validation Method:** Spot-check 20 random applications showing presence in both MCP and Notion with matching data

---

### FR-4: Hierarchical Agent Orchestration
**Priority:** MUST  
**Status:** Planned

**Description:**  
Master Agent coordinates 4 specialist worker agents (Scraper, Analyzer, Apply, Tracker) using CrewAI's hierarchical process pattern. Master delegates tasks, monitors progress, handles errors, and makes strategic decisions.

**Acceptance Criteria:**
- Master successfully delegates to all 4 workers per batch
- Sequential execution: Scraper → Analyzer → Apply → Tracker
- Master receives status updates from each worker
- Error escalation path: Worker → Master → Developer Agent (if pattern)
- Workflow completes even if individual jobs fail

**Dependencies:** CrewAI framework, all agent implementations

**Validation Method:** Review of MCP audit logs showing clear delegation chain and successful workflow completion

---

### FR-5: Multi-Source Job Scraping
**Priority:** MUST  
**Status:** Planned

**Description:**  
Scraper Agent collects jobs from 10+ platforms in parallel, normalizes different schemas, deduplicates using URL/title hashing, and caches results in Redis.

**Acceptance Criteria:**
- Minimum 8/10 platforms successfully scraped per batch
- Parallel scraping (not sequential) for speed
- Schema normalization: title, company, location, description, URL, salary, posted_date
- Deduplication rate: <5% duplicates
- Data completeness: 95%+ of jobs have all required fields
- Cache TTL: 1 hour (jobs don't need longer caching)

**Dependencies:** JobSpy library, platform APIs, Redis cache, Scraper Agent

**Validation Method:** Analysis of 5 batches showing 8+ platforms succeeded, <5% duplicates, >95% completeness

---

### FR-6: RAG-Powered Resume Selection
**Priority:** MUST  
**Status:** Planned

**Description:**  
Analyzer Agent calls RAG service to match job descriptions with optimal resume variant from user's portfolio (5-10 variants) using semantic embeddings and cosine similarity.

**Acceptance Criteria:**
- RAG returns top 3 resume matches with confidence scores
- Response time: <15 seconds per job
- Confidence score >0.6 for 90%+ of jobs
- Resume variants: ML-focused, Cloud-focused, Research-focused, Full-stack, Generalist
- Fallback to default resume if RAG confidence <0.4

**Dependencies:** RAG service, ChromaDB, sentence-transformers, resume portfolio

**Validation Method:** Spot-check 30 jobs showing appropriate resume selection and confidence scores >0.6

---

### FR-7: Multi-Criteria Job Scoring
**Priority:** MUST  
**Status:** Planned

**Description:**  
Analyzer Agent scores each job 0-100 using weighted criteria: required skills match (40%), RAG confidence (25%), salary alignment (20%), location fit (15%).

**Acceptance Criteria:**
- All jobs receive numeric score 0-100
- Scoring algorithm is deterministic (same inputs = same score)
- Weights are configurable in `job_preferences.json`
- Score breakdown logged to MCP for transparency
- Threshold rules: ≥60 auto-apply, 50-59 manual review, <50 reject

**Dependencies:** Analyzer Agent, RAG service, user preferences

**Validation Method:** Review of scoring logic showing correct weight application and score distribution matching expectations

---

### FR-8: Playwright-Based Form Automation
**Priority:** MUST  
**Status:** Planned

**Description:**  
Apply Agent triggers n8n workflows that launch Playwright browser automation to auto-fill and submit applications across major ATS platforms (Greenhouse, Lever, Workday, LinkedIn Easy Apply, Indeed Quick Apply).

**Acceptance Criteria:**
- Success rate: 90%+ for auto-applications
- Handles 5+ different ATS types with specific selectors
- Average submission time: <2 minutes per application
- Error handling: retry once, then flag for manual if still fails
- Captures submission confirmation for proof

**Dependencies:** Apply Agent, n8n, Playwright, browser instance, form selectors

**Validation Method:** Test suite covering 5 ATS types with 90%+ success rate over 50 test applications

---

### FR-9: Structured Submission Proof
**Priority:** MUST  
**Status:** Planned

**Description:**  
System captures lightweight JSON proof of application submission using multi-layer validation: confirmation number extraction, URL pattern matching, success message detection, form disappearance check.

**Acceptance Criteria:**
- Proof captured for 100% of applications
- Average proof size: <2 KB (vs 50 KB screenshots)
- Confidence score ≥50 for 95%+ of applications
- Validation layers: confirmation ID (40 pts), URL pattern (30 pts), success text (20 pts), form gone (10 pts)
- Stored in MCP evidence table with timestamp and metadata

**Dependencies:** Apply Agent, Chrome Extension, proof capture logic

**Validation Method:** Analysis of 100 applications showing proof data with appropriate confidence scores and validation flags

---

### FR-10: Chrome Extension Manual Flow
**Priority:** MUST  
**Status:** Planned

**Description:**  
Chrome extension (Manifest V3) integrates with Notion "Dream Jobs" database, auto-fills flagged jobs with data from MCP, allows user to complete custom questions, and captures submission proof.

**Acceptance Criteria:**
- Extension detects 90%+ of job application forms (5+ ATS types)
- Auto-fills standard fields: name, email, phone, LinkedIn, years of experience
- Fetches user data from MCP via webhook API
- Highlights custom questions requiring manual input
- Captures proof and syncs status back to Notion
- Supports Notion sync: shows badge with "ready to apply" count

**Dependencies:** Chrome Extension, MCP webhook endpoints, Notion API

**Validation Method:** User testing on 10 different job sites showing successful auto-fill and proof capture

---

### FR-11: Weekly System Optimization
**Priority:** MUST  
**Status:** Planned

**Description:**  
Developer Agent runs every Sunday at 8 PM, analyzes week's data from MCP (150 applications), identifies error patterns, optimizes workflows, and generates insights report with actionable recommendations.

**Acceptance Criteria:**
- Runs automatically every Sunday
- Analyzes ≥150 applications from past week
- Identifies top 5 error patterns with frequency and severity
- Proposes ≥2 concrete optimizations (prompt tweaks, workflow changes, scoring adjustments)
- Generates human-readable report posted to Notion "Analytics" database
- Execution time: <60 minutes

**Dependencies:** Developer Agent, MCP audit logs, Perplexity API (for research), statistical analysis

**Validation Method:** Review of 4 weekly reports showing meaningful pattern identification and improvement recommendations

---

### FR-12: Notion Dashboard Integration
**Priority:** MUST  
**Status:** Planned

**Description:**  
Tracker Agent maintains two Notion databases: "Applications" (all 600 applications) and "Dream Jobs" (30% flagged for manual review) with real-time status updates, filtering, and sorting.

**Acceptance Criteria:**
- "Applications" database properties: Job Title, Company, URL, Status, Applied Date, Match Score, Resume Used, Application Method, Confirmation ID, Proof Confidence
- "Dream Jobs" database additional properties: Flag Reason, Suggested Resume, Talking Points, Review Status (⏳/🎯/✅/❌)
- Updates within 5 minutes of application events
- Support filtering by: Status, Date Range, Match Score, Company
- Support sorting by: Applied Date (desc), Match Score (desc), Company (asc)

**Dependencies:** Tracker Agent, Notion API, MCP Server

**Validation Method:** User testing showing databases are up-to-date, filterable, and accurate

---

### FR-13: Rate Limit Protection
**Priority:** MUST  
**Status:** Planned

**Description:**  
System distributes applications across 10 job boards with platform-specific caps (LinkedIn: 5/run, Indeed: 10/run, others: 5/run), randomized delays (30-90s), and Redis-based rate limit tracking.

**Acceptance Criteria:**
- Hard caps enforced: LinkedIn 5, Indeed 10, others 5 per run
- Randomized delays: 30-90 seconds between applications on same platform
- User-agent rotation: 5+ realistic user-agent strings
- Respects robots.txt where applicable
- Redis tracks: `rate_limit:{platform}:{date}` counters
- Auto-alerts if approaching 80% of known platform limits

**Dependencies:** Scraper Agent, Apply Agent, Redis, rate limit configuration

**Validation Method:** 10 batch runs showing no rate limit violations, proper distribution, and appropriate delays

---

### FR-14: Session-Based Context Management
**Priority:** MUST  
**Status:** Planned

**Description:**  
MCP Server creates unique session for each batch run (Mon/Thu/Sat), stores all agent communications as context items, logs all actions to audit trail, and maintains 90-day retention.

**Acceptance Criteria:**
- Each batch gets unique `session_id` (UUID v4)
- All agent messages stored as context items with: role, content, timestamp, sequence
- Audit logs capture: actor, action, outcome, timestamp (immutable)
- Session metadata includes: batch date, goal, platform distribution, summary stats
- Context queryable via: `/v1/sessions/{id}/items`, `/v1/sessions/{id}/snapshot`
- Retention: 90 days for context, permanent for audit logs

**Dependencies:** MCP Server, PostgreSQL, session management logic

**Validation Method:** Database inspection showing proper session structure, complete context, and queryable audit trail

---

### FR-15: Error Recovery & Retry Logic
**Priority:** SHOULD  
**Status:** Planned

**Description:**  
All agents implement exponential backoff retry for transient failures (network errors, API rate limits, temporary unavailability). Master Agent receives error reports and decides whether to continue or abort batch.

**Acceptance Criteria:**
- Max 3 retry attempts per operation
- Exponential backoff: 2s, 4s, 8s delays
- Transient errors (5xx, timeouts, rate limits) trigger retry
- Permanent errors (4xx except 429, validation failures) escalate immediately
- Circuit breaker: 5 consecutive failures → pause agent, alert user
- Master Agent decides: continue with remaining jobs or abort entire batch

**Dependencies:** All agents, error handling framework, Master Agent decision logic

**Validation Method:** Fault injection testing showing appropriate retry behavior and error escalation

---

## 2. Non-Functional Requirements (NFR)

### NFR-1: Hands-Off Automation
**Priority:** MUST  
**Status:** Planned

**Description:**  
System operates with minimal human intervention. User only needs to: (1) Review 15 flagged jobs per batch (30 min), (2) Manually apply to approved dream jobs (2-3 hours), (3) Review weekly Developer Agent report (15 min).

**Acceptance Criteria:**
- 70% of applications require zero user interaction
- User oversight: <5 hours/week total
- Manual review queue clearly surfaced in Notion with priorities
- Automated notifications for: batch completion, errors, weekly reports
- No manual data entry required (auto-logging)

**Validation Method:** User time tracking over 4 weeks showing <5 hrs/week average oversight

---

### NFR-2: Lean Data Stack
**Priority:** SHOULD  
**Status:** Planned

**Description:**  
Minimize database complexity: PostgreSQL (primary), Notion (tracking/UI), Redis (optional caching). No MongoDB, Elasticsearch, or other databases unless absolutely necessary.

**Acceptance Criteria:**
- PostgreSQL handles: sessions, context items, audit logs, snapshots, evidence
- Notion handles: application tracking, dream job queue, analytics dashboard
- Redis handles: temporary caching (job data, RAG results), rate limiting counters
- Total monthly storage: <1 GB
- ChromaDB embedded in RAG service (not separate infrastructure)

**Validation Method:** Infrastructure audit showing only 3 data stores with <1 GB total usage

---

### NFR-3: Platform Safety Constraints
**Priority:** MUST  
**Status:** Planned

**Description:**  
Strict adherence to platform rate limits, ToS compliance, anti-detection measures (user-agent rotation, randomized delays, human-like behavior patterns).

**Acceptance Criteria:**
- Max 5 applications per platform per run (except Indeed: 10)
- Delays: 30-90 seconds between applications on same platform
- User-agent rotation: pool of 5+ realistic strings
- Respects robots.txt for scraping endpoints
- No CAPTCHA bypassing (flag for manual instead)
- Zero account bans or IP blocks over 6 months

**Validation Method:** 6-month operational history showing zero platform violations or blocks

---

### NFR-4: Cost Efficiency
**Priority:** MUST  
**Status:** Planned

**Description:**  
Total LLM API costs stay under $10/month, targeting $6.45/month. Groq ($5 credits) for most agents, Perplexity ($3 credits) reserved for Developer Agent research.

**Acceptance Criteria:**
- Monthly cost: ≤$6.45 (target), ≤$10 (hard cap)
- Cost per application: ≤$0.013
- Budget allocation: Master $1.50, Scraper $0.50, Analyzer $1.20, Apply $1.00, Tracker $0.25, Developer $2.00
- Automatic budget alerts at 80% ($5.16) and 95% ($6.13)
- Cost tracking dashboard in Notion

**Validation Method:** Monthly cost reports showing actual spend ≤$6.45 for 600 applications

---

### NFR-5: Execution Performance
**Priority:** MUST  
**Status:** Planned

**Description:**  
Each batch completes within 90 minutes. Average time per job: <2 minutes. No batch exceeds 120 minutes.

**Acceptance Criteria:**
- 90% of batches complete in 60-90 minutes
- Average time per job: <2 minutes
- No batch exceeds 120 minutes (timeout threshold)
- Performance breakdown logged: scraping 15 min, analysis 20 min, application 35 min, tracking 10 min
- Bottleneck identification in weekly Developer Agent report

**Validation Method:** Performance metrics over 12 batches showing 90%+ within 90-minute target

---

### NFR-6: System Observability
**Priority:** SHOULD  
**Status:** Planned

**Description:**  
Comprehensive logging, metrics export (Prometheus format), and real-time monitoring via Grafana dashboards. All services expose health and metrics endpoints.

**Acceptance Criteria:**
- All services expose: `/health` (health check), `/metrics` (Prometheus)
- Key metrics tracked: execution time, success rate, error count, LLM cost, rate limit usage
- Grafana dashboards: System Health, Cost Tracking, Application Pipeline
- Alerting rules: service down >5 min, batch failed, cost >80%, rate limit >80%
- Weekly automated reports to Notion

**Validation Method:** Grafana dashboards operational with all metrics flowing; test alerts firing correctly

---

### NFR-7: Data Privacy & Security
**Priority:** MUST  
**Status:** Planned

**Description:**  
All credentials encrypted and stored locally in `narad.env`. Personal data (resumes, contact info) never sent to cloud LLMs. Audit logs immutable.

**Acceptance Criteria:**
- `narad.env` file: 600 permissions (owner read/write only)
- API keys: Groq, Perplexity, Notion, LinkedIn session
- Resume files stored locally in encrypted Docker volume
- No PII in LLM prompts (use placeholders like "USER_NAME")
- Audit logs: append-only, no deletion allowed
- Backup strategy: weekly local backups to external drive

**Validation Method:** Security audit showing proper credential storage, no PII leakage, and backup verification

---

### NFR-8: Self-Improvement Capability
**Priority:** SHOULD  
**Status:** Planned

**Description:**  
Developer Agent continuously learns from outcomes, optimizes prompts, refines scoring weights, and improves system performance over time without manual intervention.

**Acceptance Criteria:**
- Measurable improvement: interview rate increases 5-10% per month
- Prompt evolution: ≥2 agent prompt updates per month based on data
- Scoring optimization: adjust weights based on observed correlation with interview success
- A/B testing: 10% of applications use experimental strategies
- Knowledge base: accumulate 50+ patterns after 3 months

**Validation Method:** 3-month trend analysis showing improvement in interview conversion rate and system reliability

---

### NFR-9: Containerized Deployment
**Priority:** MUST  
**Status:** Planned

**Description:**  
Full system runs in Docker containers orchestrated by Docker Compose. Reproducible deployments, easy rollback, version control for all configurations.

**Acceptance Criteria:**
- All 7 services containerized: mcp-server, rag-service, crewai-orchestrator, n8n, postgres, redis, prometheus
- Single command deployment: `docker-compose up -d`
- Health checks configured for all services
- Persistent volumes for: PostgreSQL data, resumes, n8n workflows
- Environment variables via `narad.env` (bind mount)
- Zero-downtime updates via rolling restart strategy

**Validation Method:** Fresh deployment on new machine completes successfully in <10 minutes

---

### NFR-10: Scalability Headroom
**Priority:** SHOULD  
**Status:** Planned

**Description:**  
Architecture supports scaling from 150 jobs/week to 500+ jobs/week without major refactoring. Identified bottlenecks have documented scaling paths.

**Acceptance Criteria:**
- System handles 3x load (150 jobs/batch) with <20% performance degradation
- Bottlenecks identified: Playwright concurrency (1 browser instance), LLM rate limits
- Scaling plan documented: horizontal scaling for Apply Agent, multiple browser instances
- Database can handle 10,000+ applications without performance issues
- Cost scales linearly: 3x jobs = ~3x LLM cost

**Validation Method:** Load testing with 150 jobs/batch showing acceptable performance and cost scaling

---

## 3. Constraints (CON)

### CON-1: Budget Constraint
**Priority:** MUST  
**Category:** Financial

**Hard cap at $10/month for LLM API costs.** Target: $6.45/month. No cloud infrastructure costs—system runs self-hosted on local machine or modest VPS (<$5/month if needed).

**Enforcement:**
- Budget tracker runs daily, checks Groq and Perplexity balance
- Automatic pause if spending exceeds $8/month (buffer before hard cap)
- Weekly cost reports to user via Notion
- Cost optimization is primary goal of Developer Agent

---

### CON-2: Solo Developer Constraint
**Priority:** MUST  
**Category:** Resources

**System must be maintainable by single person with minimal time investment (<5 hours/week).** No DevOps team, no on-call support, no complex operational procedures.

**Design Implications:**
- Extensive documentation (runbooks, troubleshooting guides)
- Automated monitoring reduces manual checks
- Self-healing where possible (retries, error recovery)
- Simple deployment (Docker Compose, not Kubernetes)
- Developer Agent handles routine optimizations

---

### CON-3: Platform Rate Limits
**Priority:** MUST  
**Category:** Technical

**Strict rate limits per platform per run:**
- LinkedIn: 5 applications
- Indeed: 10 applications
- All others: 5 applications each
- Total: 50 applications per batch across 10 platforms

**Enforcement:**
- Hardcoded limits in configuration
- Redis counters track real-time usage
- Automatic distribution algorithm ensures compliance
- Alerts if approaching 80% of limit
- No override mechanism (prevents accidental violations)

---

### CON-4: Three Weekly Execution Windows
**Priority:** MUST  
**Category:** Schedule

**System runs exactly 3 times per week: Monday, Thursday, Saturday at 9:00 AM IST.** No daily runs to respect rate limits, reduce oversight burden, and maintain platform safety.

**Enforcement:**
- Cron jobs: `0 9 * * 1,4,6` (Mon/Thu/Sat at 9 AM)
- Missed runs logged and alerted (not auto-retried same day)
- Manual trigger available via `crewai run --input '{...}'`
- No automatic daily rescheduling

---

### CON-5: Technology Stack Lock-In
**Priority:** SHOULD  
**Category:** Technical

**Core stack frozen for 6 months minimum:** CrewAI (agents), PostgreSQL (data), Notion (UI), n8n (workflows), Groq+Perplexity (LLMs). Changes require formal review and migration plan.

**Rationale:**
- Avoid churn and rewrite cycles
- Focus on optimization, not technology exploration
- Stability enables learning and improvement
- Migration possible after 6 months if justified

---

### CON-6: Local/Self-Hosted Infrastructure
**Priority:** SHOULD  
**Category:** Infrastructure

**All services run locally or on single VPS.** No managed cloud services (except Notion SaaS and LLM APIs). Minimize external dependencies and monthly costs.

**Enforcement:**
- Docker Compose sufficient for orchestration
- Data stored locally or on controlled VPS
- Backups to local external drive (no S3/cloud storage)
- No AWS/GCP/Azure managed services

---

### CON-7: 70/30 Auto-Manual Split
**Priority:** MUST  
**Category:** Business Logic

**70% of jobs auto-applied, 30% flagged for manual review.** This ratio balances automation efficiency with quality control for high-value opportunities.

**Criteria for Manual Flagging:**
- Salary >$150K (high compensation justifies personalization)
- Dream companies (pre-configured list: Google, Meta, OpenAI, etc.)
- Complex roles (multi-stage applications, custom essays)
- Score 50-100 but requires judgment call

**Tuning:** Ratio adjustable in configuration if data shows different optimal split.

---

### CON-8: No Follow-Up System (Phase 1)
**Priority:** MUST  
**Category:** Scope

**Follow-up emails deferred to Phase 2.** Current system focuses solely on application submission and tracking. User manually follows up using Notion reminders.

**Future Re-Integration:**
- Phase 2 (Month 4+): Add Gmail API integration
- Follow-up Agent (dormant code exists, commented out)
- Estimated cost: +$0.50/month LLM for email generation

---

### CON-9: No Company Enrichment (Phase 1)
**Priority:** MUST  
**Category:** Scope

**No Perplexity company research in Analyzer Agent to save $1.05/month.** Rely solely on job description content from job boards (which is typically sufficient).

**Rationale:**
- Job descriptions usually include company context
- Cost savings: $1.05/month (16% budget reduction)
- Time savings: 20 seconds per job
- Can re-enable if data shows enrichment improves interview rates

---

### CON-10: Storage Optimization
**Priority:** MUST  
**Category:** Technical

**No screenshot storage (saves 97% space).** Use structured JSON proof instead. Total storage budget <1 GB/month for 600 applications.

**Enforcement:**
- Average proof size: <2 KB (vs 50 KB screenshots)
- Monthly storage target: <1 GB total
- PostgreSQL database: <500 MB
- Optional mini-screenshots only for ultra-high-value jobs (salary >$180K)

---

## 4. Acceptance Testing Plan

### Phase 1: Component Testing (Week 1-2)
- [ ] Each agent runs independently with mock data
- [ ] MCP Server CRUD operations validated
- [ ] RAG service returns appropriate resumes
- [ ] n8n workflows successfully submit test applications
- [ ] Chrome extension auto-fills test forms

### Phase 2: Integration Testing (Week 3-4)
- [ ] Full batch run with 5 real jobs (manual trigger)
- [ ] Hierarchical orchestration validated (Master → Workers)
- [ ] Data flows correctly: Scraper → Analyzer → Apply → Tracker
- [ ] Notion databases update automatically
- [ ] Structured proof captured with 80%+ confidence

### Phase 3: Scale Testing (Week 5)
- [ ] Batch run with 50 jobs completes in <90 minutes
- [ ] Rate limits respected across all 10 platforms
- [ ] 90%+ success rate for auto-applications
- [ ] Cost tracking shows spend ≤$0.20 for 50 jobs
- [ ] No platform blocks or errors

### Phase 4: Production Validation (Week 6)
- [ ] 3 batches run successfully in production (Mon/Thu/Sat)
- [ ] 150 total applications logged correctly
- [ ] Developer Agent completes first weekly analysis
- [ ] User oversight <5 hours for the week
- [ ] All monitoring dashboards operational

---

## 5. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Platform bans (rate limit violations) | Medium | High | Strict rate limits, Redis tracking, randomized delays |
| Budget overrun (LLM costs) | Medium | Medium | Hard caps, daily monitoring, auto-pause at $8 |
| Low application success rate (<80%) | High | High | Extensive testing, multiple retry attempts, manual fallback |
| Developer Agent makes bad optimizations | Low | Medium | A/B testing, statistical significance, rollback capability |
| Chrome extension breaks (ATS changes) | High | Medium | Multiple ATS support, graceful degradation, user notifications |
| Data loss (database corruption) | Low | High | Weekly backups, Docker volume persistence, audit logs |
| User burnout (manual review fatigue) | Medium | Low | Clear prioritization, 30% ratio adjustable, skip option |

---

## 6. Success Metrics & KPIs

### Operational Metrics (Daily/Weekly)
- **Batch Execution Rate:** 100% (3/3 batches per week succeed)
- **Application Success Rate:** ≥90% for auto-applications
- **Average Execution Time:** 60-90 minutes per batch
- **Error Rate:** <5% of jobs encounter errors
- **Rate Limit Compliance:** Zero violations across all platforms

### Quality Metrics (Monthly)
- **Interview Conversion Rate:** 8-10% (48-60 interviews/month from 600 apps)
- **Response Rate:** 15-20% (companies reply to application)
- **Match Score Accuracy:** 75%+ (predicted fit matches actual interview invites)
- **False Positive Rate:** <10% (auto-applied to bad-fit jobs)

### Financial Metrics (Monthly)
- **Total Cost:** ≤$6.45/month
- **Cost per Application:** ≤$0.013
- **Cost per Interview:** <$0.15
- **ROI:** 1 job offer = infinite (system pays for itself)

### System Health Metrics (Continuous)
- **Service Uptime:** 99%+ (excluding planned maintenance)
- **Data Accuracy:** 100% (applications match across MCP and Notion)
- **Storage Usage:** <1 GB/month
- **Time to Recovery:** <15 minutes for service failures

### User Experience Metrics (Weekly)
- **Oversight Time:** <5 hours/week
- **Manual Review Queue:** 45 jobs/week (30% of 150)
- **User Satisfaction:** Qualitative (weekly notes on pain points)

---

## 7. Document Change History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-02-12 | Solo Developer | Initial requirements capture |

---

## 8. Appendices

### A. Glossary
- **ATS:** Applicant Tracking System (Greenhouse, Lever, Workday, etc.)
- **Batch Run:** Single execution of full workflow (Mon/Thu/Sat)
- **Dream Job:** High-value position flagged for manual review (30% of batch)
- **MCP:** Model Context Protocol (context/memory management system)
- **RAG:** Retrieval-Augmented Generation (resume selection via embeddings)

### B. Related Documents
- `README.md` - Project overview and setup guide
- `agents.yaml` - Agent role definitions
- `tasks.yaml` - Task descriptions and expected outputs
- `docker-compose.yml` - Infrastructure configuration
- `narad.env` - Environment variables and secrets (git-ignored)

### C. Contact & Support
- **Developer:** Solo (vibe coding with AI assistant)
- **Project Repository:** `github.com/aarjunm04/AI_Job_Automation_Agent`
- **Issue Tracking:** GitHub Issues
- **Documentation:** Notion workspace

---

**End of Document**
