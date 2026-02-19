# AI Job Automation Agent - Multi-Agent System Architecture v2.0

**Version:** 2.0  
**Date:** February 13, 2026  
**Status:** Design Complete - Ready for Implementation

---

## 🎯 Architecture Overview

This document provides a complete specification for the hierarchical multi-agent system that powers the AI Job Automation Agent. Use this as a blueprint to redraw the architecture diagram with all refinements incorporated.

---

## 📐 System Layers

```
┌─────────────────────────────────────────────────────────────────┐
│                     USER INTERACTION LAYER                       │
│  User → Notion Dashboard → Chrome Extension → Manual Review      │
└─────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────┐
│                    EXECUTION SCHEDULING LAYER                    │
│     Cron: Mon/Thu/Sat @ 9 AM IST + Developer: Sun @ 8 PM        │
└─────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────┐
│                      ORCHESTRATION LAYER                         │
│              Master Agent (Hierarchical Coordinator)             │
└─────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────┐
│                       EXECUTION LAYER                            │
│     Scraper → Analyzer → Apply → Tracker (Sequential Flow)      │
└─────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────┐
│                     OPTIMIZATION LAYER                           │
│         Developer Agent (Weekly Analysis & Improvement)          │
└─────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────┐
│                    INFRASTRUCTURE LAYER                          │
│    MCP | RAG | Redis | PostgreSQL | Notion | n8n | Prometheus   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🤖 Agent Specifications

### 🔵 MASTER AGENT (Orchestration Hub)

**Type:** Hierarchical Coordinator  
**LLM:** Grok-4-fast-reasoning ($0.20/M input)  
**Budget:** $1.50/month  
**Execution:** 3x/week (Mon/Thu/Sat @ 9 AM IST)

#### Core Responsibilities:
```
1. Context Management
   - Creates session for each batch run
   - Maintains shared memory across agents
   - Stores decisions and rationale in MCP

2. Task Delegation
   - Breaks down goal into sub-tasks
   - Assigns tasks to specialist agents
   - Monitors worker progress

3. Routing and Logic
   - Decides workflow branching (auto vs manual)
   - Handles conditional logic (if/then decisions)
   - Optimizes execution order

4. Security
   - Manages all API credentials (Groq, Perplexity, Notion, LinkedIn)
   - Enforces access control
   - Validates data integrity

5. Agents Control
   - Starts/stops worker agents
   - Handles agent failures
   - Coordinates parallel vs sequential execution

6. Data Handling
   - Aggregates results from workers
   - Transforms data between agent formats
   - Validates outputs before next step

7. Central Auth
   - Single source for all authentication
   - Token refresh and rotation
   - Secure credential injection to workers
```

#### External Connections:
```
Master Agent ←→ MCP Server (bidirectional, read/write context)
Master Agent  →  RAG System (query resume match)
Master Agent ←→ Redis (read/write cache)
Master Agent  →  Chrome Extension (webhook for manual jobs)
Master Agent  →  Databases (PostgreSQL + Notion writes)
Master Agent ←→ All Worker Agents (delegate tasks, receive status)
Master Agent  →  Budget Monitor (track LLM spend)
```

#### Error Handling:
```
Worker Failure → Master receives error report
                ↓
Master analyzes error (transient vs permanent)
                ↓
If transient: Retry with exponential backoff (2s, 4s, 8s)
If permanent: Skip job, log to MCP, continue workflow
                ↓
If pattern (3+ same errors): Escalate to Developer Agent
```

#### Decision Tree:
```
Master receives batch goal: "Apply to 50 jobs"
   ↓
Delegate to Scraper: "Find 50 ML Engineer jobs"
   ↓ (Scraper returns 48 jobs)
Master validates: 48 jobs received, 0 duplicates ✓
   ↓
Delegate to Analyzer: "Score and flag 48 jobs"
   ↓ (Analyzer returns 32 auto, 16 manual)
Master reviews split: 67% auto, 33% manual ✓ (within 70/30 ±5%)
   ↓
Delegate to Apply Agent: "Submit 32 auto-apply jobs"
   ↓ (Apply returns 29 success, 3 failed)
Master assesses: 91% success rate ✓ (exceeds 90% threshold)
   ↓
Delegate to Tracker: "Log all 48 jobs (29 applied, 16 queued, 3 failed)"
   ↓
Master generates summary: "29 applied, 16 await manual review, 3 need retry"
   ↓
Notify user via Notion: Update "Dream Jobs" database
```

---

### 🟢 SCRAPER AGENT (Data Acquisition Specialist)

**Type:** Worker Agent (Domain: Job Discovery)  
**LLM:** Grok-3-mini ($0.30/M input)  
**Budget:** $0.50/month  
**Tools:** JobSpy, platform APIs, Perplexity (optional)

#### Core Responsibilities:
```
1. Job Discovery (Multi-Platform)
   Platform Distribution (Per Run):
   - LinkedIn: 5 applications max
   - Indeed: 10 applications max
   - Glassdoor: 5 applications max
   - Google Jobs: 5 applications max
   - ZipRecruiter: 5 applications max
   - Wellfound: 5 applications max
   - Remotive: 5 applications max
   - WeWorkRemotely: 5 applications max
   - SimplyHired: 5 applications max
   - Stack Overflow Jobs: 5 applications max
   Total: 50 applications distributed across 10 platforms

2. Normalisation
   - Convert platform-specific schemas to unified format
   - Map fields: title, company, location, description, URL, salary, posted_date
   - Handle missing data gracefully

3. Retries
   - Exponential backoff: 2s, 4s, 8s
   - Max 3 attempts per platform
   - Switch to fallback platform if all retries fail

4. Fallback Strategy
   - If LinkedIn fails → try Indeed
   - If primary platforms exhausted → use backup (Remote.co, Hiring Cafe)
   - Always attempt to meet 50-job target

5. Security & Compliance
   - Rotate user-agents (pool of 5)
   - Respect robots.txt
   - Add randomized delays: 30-90 seconds between requests on same platform
   - Track rate limits in Redis

6. Rate Limit Protection (NEW)
   - Redis counters: rate_limit:{platform}:{date}
   - Check before each scrape: if count >= limit, skip platform
   - Alert if approaching 80% of daily limit
   - Hard stop at platform-specific caps

7. Proxy Management
   - Rotate IPs if available (optional)
   - Detect soft bans (CAPTCHA, 429 errors)
   - Switch to manual scraping if blocked
```

#### Data Flow:
```
Input (from Master):
{
  "goal": "Find 50 ML Engineer jobs",
  "location": "Remote",
  "hours_old": 24,
  "platforms": ["linkedin", "indeed", "glassdoor", ...]
}

Processing:
1. Parallel scraping across 10 platforms
2. Deduplication by URL hash
3. Schema normalization
4. Quality checks (all required fields present?)

Output (to Master):
{
  "jobs_found": 48,
  "platforms_succeeded": 9,
  "platforms_failed": ["wellfound"],
  "duplicates_removed": 3,
  "jobs": [
    {
      "id": "job_001",
      "title": "Senior ML Engineer",
      "company": "Google",
      "location": "Remote",
      "description": "...",
      "url": "https://linkedin.com/jobs/123",
      "salary_min": 180000,
      "salary_max": 220000,
      "posted_date": "2026-02-12",
      "source": "linkedin"
    },
    ...
  ]
}
```

---

### 🟢 ANALYZER AGENT (Intelligence & Decision Specialist)

**Type:** Worker Agent (Domain: Job Evaluation)  
**LLM:** Grok-4-fast-reasoning ($0.20/M input)  
**Budget:** $1.20/month  
**Tools:** RAG service, scoring algorithm, user preferences

#### Core Responsibilities:
```
1. Decision (Accept/Reject Threshold)
   - Score ≥60: Auto-apply (70% target)
   - Score 50-59 + special flags: Manual review (30% target)
   - Score <50: Reject (skip application)

2. Routing: 70% Auto / 30% Manual (NEW)
   Auto-apply criteria:
   - Score ≥60
   - No dream company flags
   - Salary <$150K
   - Standard application (single-page form)

   Manual review criteria:
   - Score 50-100 AND (salary ≥$150K OR dream company OR complex role)
   - Dream companies: [Google, Meta, OpenAI, Anthropic, DeepMind, ...]
   - Complex roles: Multi-stage apps, essays required, niche skills

3. Filtration
   - Remove jobs missing critical data (no description, no URL)
   - Filter by user blacklist (excluded companies)
   - Check against already-applied database (no re-applications)

4. Flagging (Dream Jobs) (ENHANCED)
   Flag reasons tracked:
   - "High Salary ($180K) - Personalize application"
   - "Dream Company (Google) - Showcase best work"
   - "Complex Role (requires ML paper) - Prepare custom materials"
   - "Niche Skills (Rust + ML) - Highlight rare combination"

5. Analysis (RAG + Scoring)
   RAG Phase:
   - Send job description to RAG service
   - Receive top 3 resume matches with confidence scores
   - Select resume with highest confidence (if >0.6)

   Scoring Phase:
   - Calculate 4-factor weighted score (see below)
   - Generate reasoning for score
   - Identify key selling points to highlight

6. Score Match (0-100 Weighted Algorithm)
   Formula:
   - Required Skills Match: 40 points
     * Extract required skills from job description
     * Match against user's skill profile
     * Score = (matched_skills / required_skills) × 40

   - RAG Confidence: 25 points
     * Score = RAG_confidence × 25
     * Example: 0.85 confidence = 21.25 points

   - Salary Alignment: 20 points
     * If salary ≥ user_min_salary: 20 points
     * If salary unknown: 10 points (default)
     * If salary < user_min_salary: 0 points

   - Location Fit: 15 points
     * Remote job + user prefers remote: 15 points
     * On-site + user accepts on-site: 10 points
     * Location mismatch: 0 points

   Total: Sum all factors (max 100)
```

#### Data Flow:
```
Input (from Master):
{
  "jobs": [48 jobs from Scraper],
  "user_preferences": {
    "must_have_skills": ["Python", "TensorFlow", "PyTorch", "ML"],
    "nice_to_have": ["AWS", "Docker", "Kubernetes"],
    "min_salary": 120000,
    "remote_only": true,
    "dream_companies": ["Google", "Meta", "OpenAI"]
  }
}

Processing:
For each job:
1. Call RAG service → get resume match
2. Calculate match score (4 factors)
3. Apply decision logic (auto vs manual)
4. If manual: generate flag reason and talking points
5. Store analysis in MCP context

Output (to Master):
{
  "jobs_analyzed": 48,
  "auto_apply_jobs": 32,  // 67% - within 70% ±5% tolerance
  "manual_review_jobs": 16,  // 33%
  "rejected_jobs": 0,

  "auto_apply_list": [
    {
      "job_id": "job_001",
      "score": 85,
      "resume_selected": "Resume_ML_Focus.pdf",
      "rag_confidence": 0.87,
      "reasoning": "Strong ML skills match (95%), remote, salary $150K",
      "key_points": ["PyTorch expert", "Published ML papers"]
    },
    ...
  ],

  "manual_review_list": [
    {
      "job_id": "job_010",
      "score": 92,
      "flag_reason": "Dream Company (Google) + High Salary ($180K)",
      "resume_suggested": "Resume_Research_Focus.pdf",
      "talking_points": [
        "Highlight Google Scholar citations",
        "Mention open-source contributions to TensorFlow",
        "Emphasize research background in RL"
      ],
      "application_tips": "Personalize cover letter with Google's AI research focus"
    },
    ...
  ]
}
```

---

### 🟢 APPLY AGENT (Execution Specialist)

**Type:** Worker Agent (Domain: Application Submission)  
**LLM:** Grok-code-fast-1 ($0.20/M input)  
**Budget:** $1.00/month  
**Tools:** n8n workflows, Playwright, Chrome Extension (via webhook)

#### Core Responsibilities (SPLIT PATHS):

**Path 1: Automated Applications (70%)**
```
1. Automation (n8n + Playwright)
   - Receive 32 auto-apply jobs from Master
   - For each job:
     a. Trigger n8n workflow via webhook
     b. n8n launches Playwright browser automation
     c. Navigate to job URL
     d. Detect ATS type (Greenhouse, Lever, Workday, LinkedIn, Indeed)
     e. Apply platform-specific form selectors
     f. Auto-fill standard fields
     g. Upload resume (suggested by Analyzer)
     h. Click Submit button
     i. Wait for confirmation page (5 sec timeout)

2. Form Fill (Automated)
   Standard fields:
   - First Name, Last Name
   - Email, Phone
   - LinkedIn URL, Portfolio URL
   - Years of Experience
   - Current Location
   - Work Authorization

3. Resume Attachment
   - Use resume path from Analyzer
   - Verify file exists before uploading
   - Handle different upload methods (browse, drag-drop, URL)

4. App Apply (Submit)
   - Click submit button
   - Handle loading states
   - Detect confirmation page

5. Success Tracking (Proof Capture - NEW)
   Multi-layer validation:
   Layer 1: Confirmation number (40 pts confidence)
   Layer 2: URL pattern match (30 pts)
   Layer 3: Success message text (20 pts)
   Layer 4: Form disappearance (10 pts)

   Capture structured JSON:
   {
     "timestamp": "2026-02-13T09:15:23Z",
     "job_url": "...",
     "confirmation": {
       "type": "confirmation_number",
       "value": "ABC123XYZ",
       "confidence_score": 95
     },
     "form_data": {...},
     "submission_indicators": {
       "success_url_matched": true,
       "success_text_found": "Application submitted",
       "http_status": 200
     }
   }

6. Data Tracking
   - Log attempt (timestamp, job_id, status)
   - Store proof in MCP evidence table
   - Update Redis cache with result
```

**Path 2: Manual Queue (30%) (NEW)**
```
1. Manual Queue (Chrome Extension)
   - Receive 16 manual-review jobs from Master
   - For each job:
     a. Create entry in Notion "Dream Jobs" database
     b. Status: "⏳ Awaiting Manual Review"
     c. Include: flag reason, suggested resume, talking points

   - User reviews in Notion (changes status to "🎯 Ready to Apply")

   - User opens Chrome Extension:
     a. Extension shows badge with count (16 ready)
     b. User clicks job URL from Notion
     c. Extension detects application page
     d. User clicks "🚀 Auto-fill from MCP"
     e. Extension calls MCP webhook:
        POST /webhook/chrome/fetch
        { "job_url": "...", "notion_id": "..." }
     f. MCP returns user data + resume path
     g. Extension auto-fills standard fields
     h. User manually completes custom questions
     i. User uploads resume
     j. User clicks Submit
     k. Extension captures proof (same as automated)
     l. Extension calls MCP webhook:
        POST /webhook/chrome/submission
        { "job_url": "...", "proof_data": {...} }

   - Tracker Agent receives submission event and logs to Notion
```

#### Error Handling:
```
Automated Path Errors:
- CAPTCHA detected → Skip, flag for manual (add to Dream Jobs queue)
- Form not found → Retry once with alternative selectors
- Timeout (>2 min) → Skip, log error pattern
- Network error → Retry with exponential backoff (3 attempts)

Manual Path Errors:
- Chrome Extension can't detect form → Show manual instructions
- MCP webhook fails → Retry once, fallback to cached data
- Proof capture fails → Accept low confidence (flag for user verification)
```

---

### 🟢 TRACKER AGENT (Record Keeper & Reporter)

**Type:** Worker Agent (Domain: Data Persistence)  
**LLM:** Grok-3-mini ($0.30/M input)  
**Budget:** $0.25/month  
**Tools:** Notion API, MCP Server, PostgreSQL

#### Core Responsibilities:
```
1. Tracking (Status Updates)
   - Receive results from Apply Agent
   - Update job status in real-time:
     * "Applied" - Auto-submitted successfully
     * "Queued" - In Dream Jobs, awaiting user
     * "Failed" - Submission error, needs retry
     * "Manual Applied" - User submitted via Chrome ext
     * "Skipped" - Rejected by Analyzer

2. Logging (MCP + Notion)
   MCP Database (PostgreSQL):
   - Create context_item for each application
   - Store full job object + analysis + result
   - Log to audit trail (immutable)

   Notion "Applications" Database:
   - Create page with properties:
     * Job Title (title)
     * Company (text)
     * URL (url)
     * Status (select: Applied/Queued/Failed)
     * Applied Date (date)
     * Match Score (number)
     * Resume Used (text)
     * Application Method (select: Auto/Manual Chrome)
     * Confirmation ID (text)
     * Proof Confidence (number: 0-100)

   Notion "Dream Jobs" Database (for 30% manual):
   - Same fields as Applications +
     * Flag Reason (text)
     * Suggested Resume (file)
     * Talking Points (text)
     * Review Status (select: ⏳/🎯/✅/❌)

3. Proof Capture (Evidence Storage) (NEW)
   - Receive structured JSON proof from Apply Agent
   - Store in MCP evidence table:
     INSERT INTO evidence (
       evidence_id,
       session_id,
       attached_to,  -- job context_item_id
       data,  -- JSON proof
       created_at,
       meta_json  -- { "confidence": 85, "method": "confirmation_number" }
     )

   - Link evidence to application context item
   - Query examples:
     * Find low-confidence proofs (need verification)
     * Find missing confirmation numbers
     * Generate proof quality report

4. Evidence Storage (Metadata Management) (NEW)
   - Track proof quality metrics:
     * Average confidence score
     * % with confirmation numbers
     * % with high confidence (>80)

   - Flag low-confidence applications for user verification:
     * Add "⚠️ Verify" tag in Notion
     * User manually checks email for confirmation
```

#### Data Flow:
```
Input (from Apply Agent):
{
  "applications": [
    {
      "job_id": "job_001",
      "status": "applied",
      "method": "auto",
      "timestamp": "2026-02-13T09:15:23Z",
      "proof": {
        "confidence_score": 95,
        "confirmation_number": "ABC123",
        ...
      }
    },
    {
      "job_id": "job_010",
      "status": "queued",
      "method": "manual_pending",
      "flag_reason": "Dream Company (Google)",
      ...
    }
  ]
}

Processing:
1. For each application:
   - Write to MCP context
   - Write to Notion Applications DB
   - Store proof in MCP evidence table
   - If manual: also write to Notion Dream Jobs DB

2. Generate summary stats:
   - Total applications: 48
   - Auto-applied: 29 (60%)
   - Queued for manual: 16 (33%)
   - Failed: 3 (6%)
   - Average proof confidence: 87%

Output (to Master):
{
  "tracking_complete": true,
  "mcp_items_created": 48,
  "notion_pages_created": 48,
  "evidence_stored": 29,  // Only for applied jobs
  "summary": {
    "applied": 29,
    "queued": 16,
    "failed": 3
  }
}
```

---

### 🟨 DEVELOPER AGENT (Meta-Optimization Specialist)

**Type:** Meta Agent (Advisory, Not in Execution Flow)  
**LLM:** Grok-4-fast-reasoning ($0.50) + Perplexity sonar-reasoning ($1.50)  
**Budget:** $2.00/month  
**Execution:** Weekly (Sunday @ 8 PM IST)

#### Core Responsibilities:
```
1. System Analysis (Weekly Data Review)
   - Query MCP audit logs for past 7 days
   - Analyze 150 applications (3 batches × 50 jobs)
   - Break down by:
     * Platform (which job boards performed best?)
     * Success rate (auto-apply %, manual %, errors)
     * Cost (LLM spend per application)
     * Time (execution duration per agent)
     * Quality (match score accuracy vs interview invites)

2. Performance Tracking (Metrics & KPIs)
   Track over time:
   - Application success rate (target: 90%+)
   - Interview conversion rate (target: 8-10%)
   - Response rate from companies (target: 15-20%)
   - Cost per application (target: <$0.013)
   - Execution time per batch (target: 60-90 min)
   - Error rate by type and frequency

3. Development Analytics (Pattern Recognition)
   Identify patterns:
   - Which platforms have highest success rate?
   - Which resume variants get most interviews?
   - Which companies respond fastest?
   - What time of day yields best response rates?
   - Which skills are most in-demand?
   - What salary ranges get most replies?

4. CI/CD (Continuous Improvement)
   Propose optimizations:
   - Agent prompt refinements (if errors detected)
   - Scoring algorithm adjustments (if predictions off)
   - Platform prioritization (focus on high-success boards)
   - Timing optimization (shift execution time if better results)
   - Budget reallocation (spend more on high-ROI agents)

5. Improvement Reports (Actionable Recommendations)
   Generate weekly report:
   ---
   Week of Feb 10-16, 2026

   📊 Performance Summary:
   - Applications: 150 (29+31+30 across 3 batches)
   - Success rate: 92% (138/150)
   - Interview invites: 12 (8% conversion) ✓
   - Cost: $1.85 ($0.012/application) ✓

   🔍 Key Findings:
   1. Indeed has 98% success rate (LinkedIn only 84%)
      → Recommendation: Increase Indeed quota to 15, reduce LinkedIn to 3

   2. Resume_ML_Focus.pdf gets 2x more responses than Resume_Generalist.pdf
      → Recommendation: Make ML_Focus the default resume

   3. Applications submitted 10-11 AM get 18% response rate (vs 12% overall)
      → Recommendation: Shift batch execution from 9 AM to 10 AM

   4. 5 CAPTCHA errors on LinkedIn (pattern detected)
      → Recommendation: Increase LinkedIn delay from 60s to 90s

   5. Match scores >75 have 15% interview rate (scores 60-75 only 5%)
      → Recommendation: Raise auto-apply threshold from 60 to 70

   ✅ Actions Taken:
   - Updated Scraper platform distribution
   - Modified Analyzer default resume
   - Adjusted cron schedule to 10 AM

   📈 Expected Impact:
   - +5% interview conversion rate
   - -$0.002 cost per application
   - -3 errors per week
   ---
```

#### A/B Testing Framework:
```
Experimental Changes (10% of applications):
- Test new resume variant on 5 applications
- Try alternative match scoring on 5 applications
- Experiment with different application timing on 5 applications

After 4 weeks (160 applications):
- Measure impact vs control group (90% baseline)
- If statistically significant improvement (p<0.05):
  → Roll out to 100% of applications
- If no improvement or worse:
  → Revert to baseline
```

#### Connection to Master Agent:
```
Developer Agent does NOT directly control execution
   ↓
Generates recommendations report
   ↓
Posts to Notion "Analytics" database
   ↓
User reviews recommendations (Sunday evening)
   ↓
User approves changes
   ↓
Developer Agent updates configuration files:
   - agents.yaml (agent prompts, backstories)
   - tasks.yaml (task descriptions)
   - job_preferences.json (scoring weights, platform distribution)
   - cron_schedule (execution timing)
   ↓
Changes take effect in next batch run (Monday 10 AM)
```

---

## 🔄 Data Flow Diagrams

### Batch Execution Flow (Mon/Thu/Sat @ 9 AM)

```
Cron Trigger (9:00 AM IST)
         ↓
[Master Agent Wakes Up]
         ↓
Master: Check budget (remaining: $4.60/month) ✓
Master: Create MCP session (session_id: abc-123)
         ↓
┌────────────────────────────────────────────────────────┐
│ STEP 1: Job Discovery                                  │
└────────────────────────────────────────────────────────┘
Master → Scraper: "Find 50 ML Engineer jobs, remote, 24h old"
         ↓
Scraper: Scrape 10 platforms in parallel
Scraper: Redis check rate limits before each scrape
Scraper: JobSpy returns 48 jobs
Scraper: Deduplicate (remove 3 duplicates) → 48 unique jobs
         ↓
Scraper → Master: "Found 48 jobs from 9 platforms (Wellfound failed)"
         ↓
Master: Log to MCP context
Master: Cache in Redis (TTL: 1 hour)

┌────────────────────────────────────────────────────────┐
│ STEP 2: Job Analysis                                   │
└────────────────────────────────────────────────────────┘
Master → Analyzer: "Score these 48 jobs, flag 30% for manual"
         ↓
Analyzer: For each job:
  1. Query RAG service (job description → resume match)
  2. Calculate match score (4 factors)
  3. Apply auto/manual decision logic
         ↓
Analyzer → RAG Service: "Match job_001 description"
RAG Service → Analyzer: "Resume_ML_Focus.pdf (confidence: 0.87)"
         ↓
Analyzer: Job_001 score = 85 (auto-apply)
Analyzer: Job_010 score = 92 (manual - Google, $180K salary)
Analyzer: ... (repeat for 48 jobs)
         ↓
Analyzer → Master: 
  "32 auto-apply (67%), 16 manual (33%), 0 rejected"
         ↓
Master: Validate 70/30 split (67%/33% is within ±5%) ✓
Master: Log analysis to MCP context

┌────────────────────────────────────────────────────────┐
│ STEP 3a: Auto-Application (70%)                        │
└────────────────────────────────────────────────────────┘
Master → Apply Agent: "Submit 32 auto-apply jobs"
         ↓
Apply Agent: For each of 32 jobs:
  1. POST to n8n webhook: /workflow/job-application
  2. n8n → Playwright: Launch browser, navigate to job URL
  3. Playwright: Detect ATS (Greenhouse), apply selectors
  4. Playwright: Auto-fill form fields
  5. Playwright: Upload Resume_ML_Focus.pdf
  6. Playwright: Click Submit button
  7. Playwright: Wait for confirmation page
  8. Playwright: Extract confirmation number "ABC123"
         ↓
Apply Agent: Capture structured proof (confidence: 95)
Apply Agent: Store in temporary result array
         ↓
Apply Agent: Job 1 done (2min), Job 2 done (2min), ...
         ↓ (after 32 jobs, ~64 min elapsed)
Apply Agent → Master: 
  "29 successful (91%), 3 failed (CAPTCHA)"
         ↓
Master: Log results to MCP
Master: Flag 3 CAPTCHA failures for Developer Agent review

┌────────────────────────────────────────────────────────┐
│ STEP 3b: Manual Queue (30%)                            │
└────────────────────────────────────────────────────────┘
Master → Tracker: "Queue 16 manual-review jobs in Notion"
         ↓
Tracker: For each of 16 jobs:
  1. Create Notion page in "Dream Jobs" database
  2. Set Status: "⏳ Awaiting Manual Review"
  3. Add flag reason, suggested resume, talking points
         ↓
Tracker → Master: "16 jobs queued in Notion Dream Jobs"
         ↓
Master: Notify user (Notion notification)

┌────────────────────────────────────────────────────────┐
│ STEP 4: Tracking & Logging                             │
└────────────────────────────────────────────────────────┘
Master → Tracker: "Log all 48 applications"
         ↓
Tracker: For each job:
  1. Write to MCP context (context_item)
  2. Write to MCP evidence table (if applied)
  3. Create Notion "Applications" page
  4. Update Redis metrics
         ↓
Tracker: Calculate summary stats:
  - 29 applied (auto)
  - 16 queued (manual pending)
  - 3 failed (need retry)
         ↓
Tracker → Master: "Tracking complete, 48 items logged"
         ↓
Master: Generate batch summary
Master: POST to Notion "Analytics" database
Master: Close MCP session

┌────────────────────────────────────────────────────────┐
│ STEP 5: User Notification                              │
└────────────────────────────────────────────────────────┘
Master → User (via Notion):
  "✅ Batch Complete (9:00 AM - 10:25 AM, 85 min)

   📊 Results:
   - Jobs Found: 48
   - Auto-Applied: 29 (91% success)
   - Queued for You: 16 (in Dream Jobs)
   - Failed: 3 (will retry next batch)

   💰 Cost: $0.61

   🎯 Action Needed:
   - Review 16 dream jobs in Notion
   - Apply manually using Chrome Extension"

[Master Agent Goes to Sleep]
```

### Manual Application Flow (User-Triggered)

```
User opens Notion "Dream Jobs" database (after batch run)
         ↓
User reviews 16 flagged jobs (30 min)
         ↓
User selects 10 jobs to apply to (skips 6)
         ↓
User changes status: ⏳ → 🎯 Ready to Apply (for 10 jobs)
         ↓
User opens Chrome Extension
         ↓
Extension: Badge shows "10" (ready to apply count)
         ↓
User clicks first job URL in Notion
         ↓
Chrome navigates to job application page (LinkedIn)
         ↓
Extension: Content script detects application form
Extension: Shows floating button "🚀 Auto-fill from MCP"
         ↓
User clicks "🚀 Auto-fill"
         ↓
Extension → MCP Server:
  POST /webhook/chrome/fetch
  {
    "job_url": "https://linkedin.com/jobs/123",
    "notion_page_id": "abc-def-123"
  }
         ↓
MCP Server: Fetch user profile + job metadata
MCP Server → Extension:
  {
    "first_name": "Your",
    "last_name": "Name",
    "email": "you@email.com",
    "phone": "+91-12345",
    "linkedin": "linkedin.com/in/you",
    "resume_path": "Resume_ML_Focus.pdf",
    "talking_points": [
      "Highlight PyTorch experience",
      "Mention published papers"
    ]
  }
         ↓
Extension: Auto-fills form fields (2 seconds)
Extension: Highlights custom questions in yellow
         ↓
User: Manually answers custom questions (2 min)
User: Downloads Resume_ML_Focus.pdf from Notion
User: Uploads resume to form (30 sec)
User: Reviews form (30 sec)
User: Clicks "Submit" button
         ↓
Extension: Wait for confirmation page (5 sec)
Extension: Extract proof data:
  - URL changed to /application-submitted
  - Found text: "Thank you for applying"
  - Extracted: "Application ID: XYZ789"
         ↓
Extension: Calculate confidence score:
  - Confirmation ID found: +40 pts
  - URL pattern matched: +30 pts
  - Success text found: +20 pts
  - Form disappeared: +10 pts
  Total: 100 pts (high confidence)
         ↓
Extension → MCP Server:
  POST /webhook/chrome/submission
  {
    "job_url": "...",
    "notion_page_id": "...",
    "proof_data": {
      "timestamp": "2026-02-13T11:30:15Z",
      "confirmation_number": "XYZ789",
      "confidence_score": 100,
      ...
    }
  }
         ↓
MCP Server: Store in evidence table
MCP Server: Trigger Tracker Agent webhook
         ↓
Tracker Agent: Update Notion
  - Status: 🎯 Ready to Apply → ✅ Applied
  - Applied Date: 2026-02-13
  - Method: Manual (Chrome Ext)
  - Confirmation: XYZ789
         ↓
Extension: Show success notification "✅ Application tracked!"
Extension: Badge count updates (10 → 9)
         ↓
User: Repeats for remaining 9 jobs (3 hours total)
```

### Weekly Optimization Flow (Sunday @ 8 PM)

```
Cron Trigger (Sunday, 8:00 PM IST)
         ↓
[Developer Agent Wakes Up]
         ↓
Developer Agent: Query MCP audit logs (past 7 days)
         ↓
Developer Agent: Fetch data:
  - 3 batch sessions (Mon/Thu/Sat)
  - 150 total applications
  - All agent execution logs
  - All errors and retries
  - All proof confidence scores
         ↓
┌────────────────────────────────────────────────────────┐
│ ANALYSIS PHASE (30 min)                                │
└────────────────────────────────────────────────────────┘
Developer Agent: Calculate metrics:
  1. Success Rate:
     - Auto-apply: 92% (132/144)
     - Manual: 100% (45/45) - user did 15/batch × 3

  2. Interview Conversion:
     - 12 interviews received
     - 12/150 = 8% ✓ (within target 8-10%)

  3. Platform Performance:
     - Indeed: 98% success (47/48)
     - LinkedIn: 84% success (38/45) ← Issue!
     - Other platforms: 90-95%

  4. Error Patterns:
     - 5 CAPTCHA errors on LinkedIn (pattern!)
     - 2 network timeouts on ZipRecruiter
     - 1 form selector outdated on Greenhouse

  5. Cost Analysis:
     - Total: $1.85 for 150 applications
     - $0.012 per application ✓ (under $0.013 target)
     - Budget remaining: $4.60/month

  6. Time Performance:
     - Average batch: 78 minutes ✓ (within 60-90 min)
     - Bottleneck: Apply Agent (40 min per batch)

  7. Resume Performance:
     - Resume_ML_Focus: 12 interviews from 90 uses (13.3%)
     - Resume_Generalist: 0 interviews from 60 uses (0%)
     - Clear winner: ML_Focus!

┌────────────────────────────────────────────────────────┐
│ RESEARCH PHASE (15 min)                                │
└────────────────────────────────────────────────────────┘
Developer Agent → Perplexity API:
  "How to avoid LinkedIn CAPTCHA when auto-applying to jobs?"
         ↓
Perplexity → Developer Agent:
  "Increase delays to 90+ seconds, rotate user-agents,
   randomize mouse movements, add human-like pauses"
         ↓
Developer Agent → Perplexity API:
  "Best job boards for ML engineer positions in 2026?"
         ↓
Perplexity → Developer Agent:
  "Indeed, LinkedIn, AI-Jobs.net, Kaggle Jobs emerging..."

┌────────────────────────────────────────────────────────┐
│ RECOMMENDATION PHASE (10 min)                          │
└────────────────────────────────────────────────────────┘
Developer Agent: Generate recommendations:

  Priority 1 (High Impact):
  1. Reduce LinkedIn quota from 5 to 3 applications/batch
     Increase Indeed quota from 10 to 12
     Rationale: Indeed has 98% success vs LinkedIn 84%
     Expected impact: +2% overall success rate

  2. Make Resume_ML_Focus.pdf the default resume
     Rationale: 13.3% interview rate vs 0% for Generalist
     Expected impact: +3% interview conversion rate

  3. Increase LinkedIn delay from 60s to 90s
     Rationale: Reduce CAPTCHA rate from 5/week to <2/week
     Expected impact: +2% success rate on LinkedIn

  Priority 2 (Medium Impact):
  4. Add AI-Jobs.net as 11th platform (quota: 3)
     Rationale: Emerging platform for ML roles
     Expected impact: +3 high-quality applications/batch

  5. Update Greenhouse form selectors
     Rationale: 1 outdated selector detected
     Expected impact: Fix 1 error/week

┌────────────────────────────────────────────────────────┐
│ IMPLEMENTATION PHASE (5 min)                           │
└────────────────────────────────────────────────────────┘
Developer Agent: Update configuration files:

  1. job_preferences.json:
     OLD: "linkedin_quota": 5, "indeed_quota": 10
     NEW: "linkedin_quota": 3, "indeed_quota": 12

  2. agents.yaml (Analyzer Agent):
     OLD: default_resume: "Resume_Generalist.pdf"
     NEW: default_resume: "Resume_ML_Focus.pdf"

  3. scraper_config.json:
     OLD: "linkedin_delay": 60
     NEW: "linkedin_delay": 90

  4. platforms.json:
     ADD: {"name": "AI-Jobs.net", "quota": 3, "api": "..."}

  5. form_selectors.json:
     UPDATE: Greenhouse selectors (latest version)
         ↓
Developer Agent: Git commit:
  "feat: optimize platform distribution and resume selection

   - Reduce LinkedIn quota (CAPTCHA issues)
   - Increase Indeed quota (98% success)
   - Default to ML_Focus resume (13.3% interview rate)
   - Increase LinkedIn delay to 90s
   - Add AI-Jobs.net platform"

┌────────────────────────────────────────────────────────┐
│ REPORTING PHASE (5 min)                                │
└────────────────────────────────────────────────────────┘
Developer Agent → Notion "Analytics" Database:
  Create new page with weekly report (formatted markdown)
         ↓
Developer Agent → User (Notion notification):
  "📊 Weekly Optimization Report Available

   5 improvements implemented for next batch run (Monday).
   Expected impact: +5% interview rate, -2 errors/week.

   Review report in Analytics database."

[Developer Agent Goes to Sleep until next Sunday]
```

---

## 🔌 External System Integrations

### MCP Server (Context & Memory Hub)

```
Purpose: Central memory and audit system for all agents

Endpoints Used:
- POST /v1/sessions (create session per batch)
- POST /v1/sessions/{id}/items (store agent messages)
- GET /v1/sessions/{id}/items (retrieve context)
- POST /v1/sessions/{id}/snapshot (weekly summaries)
- POST /evidence (store submission proofs)
- GET /metrics (Prometheus metrics)

Data Stored:
- Sessions: 3 per week × 4 weeks = 12 active sessions
- Context items: ~200 per session × 12 = 2,400 items/month
- Audit logs: ~1,000 entries/week (permanent)
- Evidence: 105 proofs/week (auto-applied jobs)
- Size: ~50 MB/month

Retention:
- Sessions: 90 days
- Context items: 90 days
- Audit logs: Permanent
- Evidence: 90 days
```

### RAG System (Resume Matching Service)

```
Purpose: Semantic resume-job matching via embeddings

Technology: ChromaDB + sentence-transformers

Endpoints Used:
- POST /select_resume (job description → resume match)
- POST /upload_resume (add new resume variant)
- GET /health

Data Stored:
- Resume embeddings: 5-10 variants × 384 dimensions
- Job cache: 30 days of job descriptions
- Match history: 600 matches/month
- Size: ~100 MB total

Performance:
- Query time: <15 seconds per job
- Batch processing: 50 jobs in <12 minutes
- Confidence threshold: >0.6 for 90%+ of jobs
```

### Redis (Caching & Rate Limiting)

```
Purpose: Fast caching and rate limit tracking

Keys Used:
- session:{id}:jobs - Scraped jobs cache (TTL: 1 hour)
- company:{name}:research - Company data (TTL: 7 days)
- rate_limit:{platform}:{date} - API call counters (TTL: 24 hours)
- rag:job:{hash}:resume - RAG results cache (TTL: 30 days)
- agent:context:{session} - Fast context access (TTL: 1 hour)

Memory Usage: ~50 MB (mostly ephemeral)

Benefits:
- 3x faster context retrieval vs PostgreSQL
- Real-time rate limit checking
- Avoid redundant RAG queries (30% cache hit rate)
```

### Chrome Extension (Manual Application UI)

```
Purpose: User interface for manual dream job applications

Architecture:
- Manifest V3 extension
- Content script: Runs on job pages
- Background worker: MCP API calls
- Popup UI: Shows ready-to-apply count

Webhook Endpoints (MCP Server):
- POST /webhook/chrome/fetch (get user data)
- POST /webhook/chrome/submission (log application)
- GET /webhook/chrome/sync (fetch Notion queue)

User Flow:
1. User opens Notion Dream Jobs
2. User marks jobs "🎯 Ready to Apply"
3. Extension shows badge count
4. User clicks job URL
5. Extension auto-fills form
6. User completes custom fields
7. Extension captures proof
8. Extension syncs to Notion

Supported ATS:
- Greenhouse
- Lever
- Workday
- LinkedIn Easy Apply
- Indeed Quick Apply
- Generic forms (fallback)
```

### Notion (Tracking & Reporting Dashboard)

```
Purpose: User-facing application tracking and analytics

Databases:
1. Applications (all 600/month)
   Properties:
   - Job Title (title)
   - Company (text)
   - URL (url)
   - Status (select: Applied/Queued/Failed)
   - Applied Date (date)
   - Match Score (number: 0-100)
   - Resume Used (text)
   - Method (select: Auto/Manual Chrome)
   - Confirmation ID (text)
   - Proof Confidence (number: 0-100)

2. Dream Jobs (30% = 45/week)
   Additional Properties:
   - Flag Reason (text)
   - Suggested Resume (file)
   - Talking Points (text)
   - Review Status (select: ⏳/🎯/✅/❌)
   - Time Spent (number: minutes)

3. Analytics (weekly reports)
   Properties:
   - Week (date)
   - Applications (number)
   - Interviews (number)
   - Response Rate (number: %)
   - Top Performers (text)
   - Recommendations (text)

API Usage:
- 150 page creates per batch (3 batches/week)
- 45 manual updates per week (user changes status)
- 1 analytics page per week
- Total: ~495 API calls/week (well under free tier limit)
```

### PostgreSQL (Primary Database)

```
Purpose: System of record for all application data

Schema:
- sessions (batch runs)
- context_items (agent communications)
- audit_logs (immutable event log)
- evidence (submission proofs)
- snapshots (weekly summaries)

Size Estimates:
- 12 sessions/month × 1 KB = 12 KB
- 2,400 context items/month × 2 KB = 4.8 MB
- 1,000 audit logs/week × 1 KB = 4 MB/month
- 420 evidence/month × 1.5 KB = 630 KB
- 4 snapshots/month × 5 KB = 20 KB
Total: ~10 MB/month, 120 MB/year

Retention:
- 90 days for sessions/context (auto-cleanup)
- Permanent for audit logs
- 90 days for evidence
```

### n8n (Workflow Automation Engine)

```
Purpose: Playwright automation orchestration

Workflows:
1. job-application (main automation)
   Trigger: Webhook from Apply Agent
   Steps:
   - Receive job data (URL, resume, user profile)
   - Launch Playwright browser
   - Navigate to URL
   - Detect ATS type
   - Execute form filling
   - Upload resume
   - Submit application
   - Capture proof
   - Return result to Apply Agent

   Duration: 90-120 seconds per job
   Success rate: 90%+

2. error-notification (optional)
   Trigger: Critical error detected
   Steps:
   - Format error message
   - Send Slack/email notification
   - Log to MCP

Storage:
- Workflow definitions: <1 MB
- Execution logs: 30 days retention
- Screenshots (if enabled): Disabled to save space
```

### Prometheus + Grafana (Monitoring)

```
Purpose: Real-time metrics and observability

Metrics Collected:
- agent_execution_time_seconds{agent="scraper"}
- agent_success_rate{agent="apply"}
- job_applications_total{platform="linkedin"}
- llm_cost_dollars{agent="analyzer"}
- rate_limit_usage{platform="indeed"}
- proof_confidence_score{method="confirmation_number"}
- batch_duration_minutes

Dashboards:
1. System Health
   - Service uptime
   - Error rate
   - Response times

2. Cost Tracking
   - Daily spend
   - Budget remaining
   - Cost per application

3. Application Pipeline
   - Jobs scraped
   - Jobs applied
   - Success rate
   - Interview conversion

Alerting Rules:
- Service down >5 min → Slack alert
- Batch failed → Email alert
- Cost >$8/month → Stop execution
- Rate limit >90% → Skip platform
```

---

## 🚨 Error Handling & Recovery Patterns

### Transient Errors (Auto-Retry)

```
Scenario: Network timeout, API rate limit (429), server error (5xx)

Retry Strategy:
Attempt 1: Immediate retry
Attempt 2: Wait 2 seconds, retry
Attempt 3: Wait 4 seconds, retry
Max attempts: 3

If all attempts fail:
- Log error to MCP audit trail
- Notify Master Agent
- Master decides: skip job or escalate

Example:
Scraper → LinkedIn API → 429 Too Many Requests
   ↓
Scraper: Wait 2 seconds
Scraper → LinkedIn API → 200 OK ✓
```

### Permanent Errors (Escalate Immediately)

```
Scenario: Invalid credentials (401), not found (404), validation error (400)

Action:
- No retry (permanent failure)
- Log error with full context
- Escalate to Master Agent immediately
- Master skips job, continues workflow

Example:
Apply Agent → Job URL → 404 Not Found
   ↓
Apply Agent → Master: "Job no longer available (404)"
   ↓
Master: Mark job as "Closed", move to next job
```

### Pattern-Based Errors (Escalate to Developer Agent)

```
Trigger: Same error occurs 3+ times in a week

Example:
Monday batch: 2 CAPTCHA errors on LinkedIn
Thursday batch: 3 CAPTCHA errors on LinkedIn
   ↓
Master Agent: Detected pattern (5 CAPTCHA errors)
   ↓
Master → Developer Agent: "CAPTCHA pattern on LinkedIn (5 occurrences)"
   ↓
Developer Agent (next Sunday):
- Analyzes pattern
- Researches solutions (Perplexity)
- Proposes fix: Increase LinkedIn delay to 90s
- Updates configuration
- Reports to user
```

### Catastrophic Errors (Abort Batch)

```
Trigger: Critical system failure, budget exceeded, security breach

Action:
- Immediately stop all agents
- Close MCP session
- Send urgent notification to user
- Log full state for debugging

Example:
Budget Monitor: Detected $8.50 spend (exceeds $8 cap)
   ↓
Budget Monitor → Master: "ABORT: Budget exceeded"
   ↓
Master: Stop all workers, close session
Master → User: "🚨 Batch aborted - budget cap reached"
```

### Circuit Breaker Pattern

```
Purpose: Prevent cascading failures

Threshold: 5 consecutive failures on same operation

Example:
Apply Agent → LinkedIn application → CAPTCHA (fail)
Apply Agent → LinkedIn application → CAPTCHA (fail)
Apply Agent → LinkedIn application → CAPTCHA (fail)
Apply Agent → LinkedIn application → CAPTCHA (fail)
Apply Agent → LinkedIn application → CAPTCHA (fail)
   ↓
Circuit breaker trips: "LinkedIn applications failing"
   ↓
Apply Agent → Master: "LinkedIn circuit breaker open"
   ↓
Master: Skip all remaining LinkedIn jobs in this batch
Master: Log pattern for Developer Agent
```

---

## 🔒 Security & Access Control

### Credential Management

```
Storage: ~/narad.env file (600 permissions, git-ignored)

Format:
# LLM APIs
GROQ_API_KEY=gsk-...
PERPLEXITY_API_KEY=pplx-...

# Job Platforms
LINKEDIN_EMAIL=you@email.com
LINKEDIN_PASSWORD=***
LINKEDIN_SESSION_COOKIE=***

# Integrations
NOTION_API_KEY=secret_...
MCP_API_KEY=mcp-...
RAG_API_KEY=rag-...

# Infrastructure
POSTGRES_PASSWORD=***
REDIS_PASSWORD=***

Loading:
Docker Compose: env_file: ~/narad.env
Master Agent: Loads on startup, injects to workers
Workers: Receive credentials via Master (not direct access)
```

### Access Control Matrix

```
| Resource        | Master | Scraper | Analyzer | Apply | Tracker | Developer |
|-----------------|--------|---------|----------|-------|---------|-----------|
| MCP (read)      | ✅     | ✅      | ✅       | ✅    | ✅      | ✅        |
| MCP (write)     | ✅     | ❌      | ❌       | ❌    | ✅      | ✅        |
| RAG (query)     | ✅     | ❌      | ✅       | ❌    | ❌      | ❌        |
| Redis (read)    | ✅     | ✅      | ✅       | ✅    | ✅      | ✅        |
| Redis (write)   | ✅     | ✅      | ❌       | ✅    | ✅      | ✅        |
| Notion (write)  | ✅     | ❌      | ❌       | ❌    | ✅      | ✅        |
| n8n (trigger)   | ✅     | ❌      | ❌       | ✅    | ❌      | ❌        |
| LLM APIs        | ✅     | ✅      | ✅       | ✅    | ✅      | ✅        |
| Job Platforms   | ✅     | ✅      | ❌       | ✅    | ❌      | ❌        |
| Config (write)  | ❌     | ❌      | ❌       | ❌    | ❌      | ✅        |

Principle: Least privilege - agents only access what they need
```

### API Key Rotation

```
Schedule:
- Groq/Perplexity: Never (unless compromised)
- Notion: Never (unless compromised)
- MCP internal: Every 90 days
- LinkedIn session: Every 30 days (auto-refresh)

Rotation Process:
1. Developer Agent detects expiration (weekly check)
2. Generate new key/token
3. Update narad.env file
4. Restart affected services (zero-downtime)
5. Revoke old key after 24-hour grace period
```

---

## 📊 Budget Monitoring & Cost Controls

### Real-Time Budget Tracking

```
Budget Monitor (runs every API call):

1. Track spend:
   - Groq: $0.20/M input tokens, $0.50/M output tokens
   - Perplexity: $1/M input tokens, $5/M output tokens

2. Calculate running total:
   - Start of month: $0
   - After batch 1: $0.58
   - After batch 2: $1.21
   - After batch 3: $1.85
   - ...

3. Check thresholds:
   - 80% ($6.40): Warning alert to user
   - 95% ($9.50): Urgent alert, disable Developer Agent
   - 100% ($10.00): Hard stop, abort all execution

4. Update dashboard:
   - Notion "Budget" widget shows: $1.85 / $10.00 (18.5%)
   - Projected end-of-month: $6.32 ✓
```

### Cost Optimization Strategies

```
1. Prompt Compression
   - Remove verbose instructions from prompts
   - Use abbreviations where clear
   - Expected savings: -10% tokens

2. Caching
   - Cache RAG queries (30-day TTL)
   - Cache company research (7-day TTL)
   - Expected savings: -20% Perplexity calls

3. Model Selection
   - Use Grok-3-mini for simple tasks (Scraper, Tracker)
   - Use Grok-4-fast only for complex reasoning (Master, Analyzer)
   - Expected savings: -15% overall cost

4. Batch Processing
   - Process 50 jobs at once (vs 10 jobs 5x)
   - Reuse context across jobs in same batch
   - Expected savings: -5% due to context reuse
```

---

## 📈 Success Metrics & Monitoring

### Real-Time Dashboards (Grafana)

```
Dashboard 1: System Health
- Service uptime (99%+ target)
- Error rate (5% target)
- Average batch duration (60-90 min target)
- Memory/CPU usage

Dashboard 2: Application Pipeline
- Jobs scraped per batch (50 target)
- Auto-apply success rate (90%+ target)
- Manual queue size (15 per batch)
- Interview conversion rate (8-10% target)

Dashboard 3: Cost Tracking
- Daily LLM spend
- Month-to-date total
- Budget remaining
- Cost per application ($0.013 target)
- Projected end-of-month cost

Dashboard 4: Platform Performance
- Success rate by platform (Indeed, LinkedIn, etc.)
- Average submission time by ATS type
- CAPTCHA/error frequency
```

### Weekly Reports (Developer Agent)

```
Generated every Sunday, posted to Notion Analytics:

---
📊 Weekly Performance Report
Week of Feb 10-16, 2026

🎯 Key Metrics:
- Applications: 150 (50 per batch × 3)
- Success rate: 92% (138/150) ✓
- Interview invites: 12 (8% conversion) ✓
- Cost: $1.85 ($0.012/application) ✓

📈 Trends:
- Success rate: +2% vs last week
- Interview rate: -1% vs last week (still in target range)
- Cost: -$0.15 vs last week (optimization working)

🏆 Top Performers:
- Best platform: Indeed (98% success, 8 interviews)
- Best resume: ML_Focus (13.3% interview rate)
- Best time: 10 AM applications (15% response rate)

⚠️ Issues Detected:
- 5 CAPTCHA errors on LinkedIn
- 2 network timeouts on ZipRecruiter

✅ Actions Taken:
- Reduced LinkedIn quota (5→3)
- Increased Indeed quota (10→12)
- Increased LinkedIn delay (60s→90s)

🎯 Next Week Goals:
- Maintain 90%+ success rate
- Achieve 10% interview rate
- Stay under $2.00 weekly cost
---
```

---

## ✅ Architecture Validation Checklist

```
Hierarchical Structure:
✅ Master Agent coordinates 4 workers
✅ Sequential workflow (Scraper→Analyzer→Apply→Tracker)
✅ Developer Agent advises (not in execution path)

70/30 Auto-Manual Split:
✅ Analyzer flags 30% for manual review
✅ Criteria explicit (salary, dream company, complex role)
✅ Chrome Extension handles manual flow

Rate Limit Protection:
✅ Platform-specific caps (LinkedIn: 5, Indeed: 10)
✅ Redis counters track usage
✅ Randomized delays (30-90s)

Proof Capture:
✅ Structured JSON (97% smaller than screenshots)
✅ Multi-layer validation (4 confidence factors)
✅ Stored in MCP evidence table

Error Handling:
✅ Retry logic (exponential backoff)
✅ Error escalation (Worker→Master→Developer)
✅ Circuit breakers (5 consecutive failures)

External Integrations:
✅ MCP (context), RAG (resumes), Redis (cache)
✅ Notion (tracking), n8n (automation), Chrome ext (manual)
✅ PostgreSQL (data), Prometheus (metrics)

Budget Controls:
✅ Real-time tracking
✅ Hard cap at $10/month
✅ Alerts at 80% and 95%

User Interaction:
✅ Notion dashboard (review dream jobs)
✅ Chrome extension (manual applications)
✅ Weekly reports (Developer Agent insights)

Execution Schedule:
✅ 3x/week (Mon/Thu/Sat @ 9 AM)
✅ Weekly optimization (Sun @ 8 PM)
✅ Cron-triggered (not manual)
```

---

## 🎨 Diagram Drawing Instructions

### Visual Elements to Include:

**1. Agents (6 boxes):**
```
- Master Agent (Blue, Large, Center-top)
- Scraper Agent (Green, Medium, Left)
- Analyzer Agent (Green, Medium, Center-left)
- Apply Agent (Green, Medium, Center-right)
- Tracker Agent (Green, Medium, Right)
- Developer Agent (Yellow, Medium, Top-left, separate)
```

**2. External Systems (7 boxes):**
```
- MCP Server (Purple, Top-right)
- RAG System (Purple, Right of MCP)
- Redis (Purple, Below MCP)
- Chrome Extension (Purple, Below Redis)
- Databases (Purple, Bottom-right)
- n8n + Playwright (Purple, Below Apply Agent)
- Notion (Purple, Below Tracker Agent)
```

**3. Data Flows (Arrows):**
```
Solid arrows: Primary data flow
Dashed arrows: Error feedback / Advisory
Thick arrows: High-volume data
Thin arrows: Low-volume data

Master → Workers: Task delegation (solid, thick)
Workers → Master: Status updates (dashed, thin)
Master ↔ MCP: Bidirectional context (solid, thick)
Master → RAG: Resume query (solid, thin)
Apply → n8n: Webhook trigger (solid, medium)
Tracker → Notion: Write data (solid, medium)
Developer ⇢ Master: Advisory (dashed, thin)
```

**4. Labels & Annotations:**
```
Each agent box:
- Agent name (bold)
- Core responsibilities (bullet list, 3-5 items)
- Key metrics (execution time, budget, etc.)

Each connection:
- Data direction (→ or ↔)
- Data type (jobs, scores, results, etc.)
- Volume if relevant (50 jobs, 150 applications)
```

**5. User Interaction Layer:**
```
Add User icon (top-center)
User → Notion (review dream jobs)
User → Chrome Extension (manual applications)
Notion → User (notifications, reports)
```

**6. Scheduling Layer:**
```
Add Cron icon (top-left)
Cron → Master: "Mon/Thu/Sat 9 AM"
Cron → Developer: "Sunday 8 PM"
```

**7. Budget Monitor:**
```
Add Budget icon (top-right, near MCP)
All agents → Budget Monitor (track spend)
Budget Monitor → Master (alert if exceeded)
```

---

## 🚀 Next Steps

1. **Redraw diagram** using this specification
   - Use tool like Excalidraw, draw.io, or Figma
   - Follow visual guidelines above
   - Add all 8 missing elements identified

2. **Validate against requirements**
   - Check all 35 requirements from CSV
   - Ensure all FR, NFR, CON are represented
   - Verify no critical gaps

3. **Review with stakeholders** (you!)
   - Does it match your mental model?
   - Is anything unclear or missing?
   - Ready to start implementation?

4. **Use as implementation blueprint**
   - Reference during coding
   - Update as system evolves
   - Maintain as living documentation

---

**This is your definitive v2.0 architecture specification. Everything you need to build the system is documented above.** 🎯

Let me know when you've redrawn the diagram and I can provide a final review! 🚀
