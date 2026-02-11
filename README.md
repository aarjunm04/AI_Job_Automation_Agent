# 🤖 AI Job Automation Agent - Agentic AI Job Application Ecosystem

<div align="center">

[![Python Version](https://img.shields.io/badge/Python-3.9+-blue?style=flat-square&logo=python)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
┌─────────────────────────────────────────────────────────────────────────────┐ │ AI JOB AUTOMATION AGENT ECOSYSTEM │ └─────────────────────────────────────────────────────────────────────────────┘

Code
                      ┌──────────────────────┐
                      │  Master Orchestrator │
                      │   (master_run.py)    │
                      └──────────┬───────────┘
                                 ��
              ┌──────────────────┼──────────────────┐
              │                  │                  │
    ┌─────────▼────────┐  ┌──────▼──────┐  ┌──────▼──────────┐
    │  Job Discovery   │  │ Job Analysis│  │ Automation      │
    │  Workflow        │  │ Workflow    │  │ Workflow        │
    └─────────┬────────┘  └──────┬──────┘  └──────┬──────────┘
              │                  │                  │
    ┌─────────▼────────────────────────────────────▼────┐
    │         CORE SYSTEM ENGINES (Modular)             │
    ├────────────────────────────────────────────────────┤
    │                                                    │
    │  ┌──────────────┐  ┌──────────────┐               │
    │  │ Scraper Eng. │  │  AI Engine   │               │
    │  └──────────────┘  └──────────────┘               │
    │                                                    │
    │  ┌──────────────┐  ┌──────────────┐               │
    │  │ Resume Eng.  │  │Notion Eng.   │               │
    │  └──────────────┘  └──────────────┘               │
    │                                                    │
    │  ┌──────────────┐  ┌──────────────┐               │
    │  │Automation Eng│  │MCP Client    │               │
    │  └──────────────┘  └──────────────┘               │
    │                                                    │
    └────────────────────────────────────────────────────┘
                        │
    ┌───────────────────┼───────────────────┐
    │                   │                   │
┌───▼────────┐  ┌──────▼──────┐  ┌────────▼───┐
│ Notion DBs │  │ External    │  │ N8N WF     │
│ (Job Data) │  │ Integrations│  │ Automation │
└────────────┘  └─────────────┘  └────────────┘
Code

### **Module Organization**

aarjunm04/AI_Job_Automation_Agent/ ├── master_run.py ← Main Entry Point (Orchestrator) ├── requirements.txt ← Python Dependencies ├── docker-compose.yml ← Docker Configuration │ ├── core/ ← Core Engine Implementations │ ├── ai_engine.py ← AI & Analysis Engine │ ├── scraper_engine.py ← Web Scraping Engine │ ├── resume_engine.py ← Resume Generation Engine │ ├── notion_engine.py ← Notion Database Engine │ ├── automation_engine.py ← Form Automation Engine │ └── init.py │ ├── agents/ ← AI Agent Implementations │ ├── discovery_agent.py ← Job Discovery Agent │ ├── analysis_agent.py ← Job Analysis Agent │ ├── application_agent.py ← Application Submission Agent │ └── coordinator_agent.py ← Multi-Agent Coordinator │ ├── scrapers/ ← Platform-Specific Scrapers │ ├── linkedin_scraper.py │ ├── indeed_scraper.py │ ├── glassdoor_scraper.py │ └── multi_platform_scraper.py │ ├── integrations/ ← External Service Integrations │ ├── notion_integration.py │ ├── openai_integration.py │ ├── anthropic_integration.py │ └── gmail_integration.py │ ├── mcp/ ← Model Context Protocol │ ├── mcp_client.py │ └── mcp_server.py │ ├── automation/ ← Automation Utilities │ ├── playwright_actions.py │ ├── form_filler.py │ └── browser_automation.py │ ├── rag_systems/ ← Retrieval Augmented Generation │ ├── job_rag_system.py │ ├── company_rag_system.py │ └── embeddings.py │ ├── config/ ← Configuration Management │ ├── settings.py │ ├── database_config.py │ └── prompts.yaml │ ├── chrome_extension/ ← Browser Extension (Frontend) │ ├── manifest.json │ ├── popup.html / popup.js │ └── content.js │ ├── n8n/ ← N8N Workflow Automation │ ├── discovery_workflow.json │ ├── analysis_workflow.json │ └── application_workflow.json │ ├── scripts/ ← Utility Scripts │ ├── setup_env.sh │ ├── migrate_data.py │ └── health_check.py │ └── assets/ ← Documentation Assets ├── diagrams/ ├── screenshots/ └── templates/

Code

---

## 🛠️ Tech Stack

### **Language Composition**
- **Python** 69.2% - Core automation engine, AI, and logic
- **JavaScript** 17.8% - Chrome extension, frontend interactions
- **CSS** 5.4% - Styling for UI components
- **HTML** 4.3% - Extension and web templates
- **Shell** 3.3% - Deployment and setup scripts

### **Key Technologies**

| Category | Technologies |
|----------|--------------|
| **AI & LLMs** | Anthropic Claude, OpenAI, Perplexity AI |
| **Web Scraping** | Scrapy, Selenium, Playwright, BeautifulSoup4, Splash |
| **Automation** | Playwright, Undetected-ChromeDriver |
| **Database** | Notion API, PostgreSQL, MongoDB, Redis |
| **Web Framework** | FastAPI, Flask, Starlette |
| **Data Processing** | Pandas, NumPy, Scikit-learn |
| **Document Processing** | PyPDF2, pdfplumber, PyLaTeX, ReportLab |
| **Vision/OCR** | EasyOCR, OpenCV, Pytesseract |
| **Protocol** | Model Context Protocol (MCP) |
| **Async** | asyncio, aiohttp, asyncpg |
| **Task Queue** | Celery |
| **DevOps** | Docker, Docker Compose |

### **Python Dependencies Highlights**
Core: anthropic, openai, perplexityai Scraping: scrapy, selenium, playwright, beautifulsoup4, lxml, selectolax Automation: undetected-chromedriver, playwright Data: pandas, numpy, scikit-learn, sentence-transformers AI/ML: tiktoken, nltk, spacy, textblob APIs: notion-client, google-api-python-client, requests Web: fastapi, flask, flask-cors, starlette, uvicorn Database: sqlalchemy, asyncpg, redis Document: PyPDF2, pdfplumber, pylatex, reportlab, pillow Testing: pytest, pytest-asyncio, pytest-cov, pytest-mock

Code

---

## 📦 Installation

### **Prerequisites**
- Python 3.9 or higher
- Docker & Docker Compose (optional but recommended)
- Git
- PostgreSQL (optional, for advanced features)
- Modern web browser (Chrome/Chromium for extensions)

### **Step 1: Clone Repository**
bash
git clone https://github.com/aarjunm04/AI_Job_Automation_Agent.git
cd AI_Job_Automation_Agent
Step 2: Set Up Python Environment
bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
Step 3: Install Dependencies
bash
pip install --upgrade pip
pip install -r requirements.txt

# Optional: Install development dependencies
pip install pytest pytest-cov pytest-asyncio
Step 4: Configure Environment Variables
bash
# Copy example environment file
cp .env.example .env

# Edit configuration with your API keys and credentials
nano .env  # or use your preferred editor
Required Environment Variables:

bash
# AI APIs
ANTHROPIC_API_KEY=your_anthropic_key
OPENAI_API_KEY=your_openai_key
PERPLEXITY_API_KEY=your_perplexity_key

# Job Platforms
LINKEDIN_EMAIL=your_email@example.com
LINKEDIN_PASSWORD=your_password
INDEED_API_KEY=your_indeed_key

# Database
NOTION_API_KEY=your_notion_token
DATABASE_URL=postgresql://user:password@localhost/db_name

# Integrations
GMAIL_CLIENT_ID=your_gmail_client_id
GMAIL_CLIENT_SECRET=your_gmail_client_secret
OVERLEAF_API_KEY=your_overleaf_key

# System
TIMEZONE=Asia/Kolkata
DAILY_RUN_TIME=09:00
Step 5: (Optional) Using Docker
bash
# Build and start services
docker-compose up -d

# Initialize database
docker exec ai_job_automation python scripts/migrate_data.py

# Check container status
docker-compose ps
Step 6: Verify Installation
bash
# Run health check
python master_run.py --mode health-check

# You should see output like:
# ✅ System Health Check: All components healthy
🚀 Quick Start
Basic Usage - Discover Jobs
bash
# Discover new jobs from all platforms
python master_run.py --mode discover

# Discover with custom limit
python master_run.py --mode discover --limit 100
Analyze Discovered Jobs
bash
# Analyze jobs for matching
python master_run.py --mode analyze

# Analyze with limit
python master_run.py --mode analyze --limit 20
Submit Applications
bash
# Apply to high-priority jobs
python master_run.py --mode apply

# Apply with limit
python master_run.py --mode apply --limit 5
Complete End-to-End Automation
bash
# Full automation: Discover → Analyze → Apply
python master_run.py --mode full-automation
Health Check
bash
# Verify all systems are operational
python master_run.py --mode health-check
Debug Mode
bash
# Run with detailed logging
python master_run.py --mode discover --debug
📊 Workflow Diagrams
1. Job Discovery Workflow
Code
┌─────────────────────────────────────────────────────────────────────┐
│  JOB DISCOVERY WORKFLOW - SCRAPE & COLLECT                          │
└─────────────────────────────────────────────────────────────────────┘

┌──────────────────┐
│  Start Discovery │
└────────┬─────────┘
         │
         ▼
    ┌─────────────────────────┐
    │ Load Search Criteria     │
    │ - Job Titles            │
    │ - Locations             │
    │ - Salary Range          │
    └────────────┬────────────┘
                 │
    ┌────────────▼────────────────────────────────────────┐
    │ Multi-Platform Scraping (Parallel)                 │
    ├────────────────────────────────────────────────────┤
    │ ┌─────────────┐  ┌──────────────┐  ┌────────────┐ │
    │ │  LinkedIn   │  │     Indeed   │  │ Glassdoor  │ │
    │ └──────┬──────┘  └───────┬──────┘  └─────┬──────┘ │
    │        │                 │                │        │
    │ ┌──────▼──────┐  ┌──────▼──────┐  ┌─────▼──────┐ │
    │ │ AngelList   │  │   Dice      │  │  ZipRecruit│ │
    │ └──────┬──────┘  └───────┬──────┘  └─────┬──────┘ │
    │        └──────────────┬──────────────────┘        │
    └──────────────────┬───────────────────────────────┘
                       │
                       ▼
            ┌──────────────────────────┐
            │ Deduplication & Cleaning │
            │ - Remove Duplicates      │
            │ - Normalize Data         │
            │ - Extract Metadata       │
            └────────────┬─────────────┘
                         │
                         ▼
            ┌──────────────────────────┐
            │ Convert to Standard      │
            │ JobApplication Format    │
            └────────────┬─────────────┘
                         │
                         ▼
            ┌──────────────────────────┐
            │ Save to Notion DB        │
            │ (Applications Database)  │
            └────────────┬─────────────┘
                         │
                         ▼
            ┌──────────────────────────┐
            │ Return Discovery Summary │
            │ - Jobs Found: X          │
            │ - Duplicates: Y          │
            │ - Stored: Z              │
            └──────────────────────────┘
2. Job Analysis & Matching Workflow
Code
┌─────────────────────────────────────────────────────────────────────┐
│  JOB ANALYSIS WORKFLOW - AI-POWERED MATCHING                        │
└─────────────────────────────────────────────────────────────────────┘

┌───────��──────────┐
│ Start Analysis   │
└────────┬─────────┘
         │
         ▼
  ┌─────────────────────────┐
  │ Load Jobs from Notion   │
  │ Filter: DISCOVERED      │
  │ Status                  │
  └────────────┬────────────┘
               │
               ▼
       ┌──────────────────┐
       │ Load User Profile│
       │ - Skills         │
       │ - Experience     │
       │ - Preferences    │
       └────────┬─────────┘
                │
                ▼
    ┌───────────────────────────┐
    │ AI Analysis Loop (Per Job)│
    ├───────────────────────────┤
    │                           │
    │  ┌─────────────────────┐  │
    │  │ Extract Job Details │  │
    │  │ - Title, Company    │  │
    │  │ - Description       │  │
    │  │ - Requirements      │  │
    │  └──────────┬──────────┘  │
    │             │             │
    │  ┌──────────▼──────────┐  │
    │  │ Semantic Matching   │  │
    │  │ - Compare Skills    │  │
    │  │ - Experience Level  │  │
    │  │ - Requirements      │  │
    │  └──────────┬──────────┘  │
    │             │             │
    │  ┌──────────▼──────────┐  │
    │  │ Calculate Match     │  │
    │  │ Score (0-100%)      │  │
    │  └──────────┬──────────┘  │
    │             │             │
    │  ┌──────────▼──────────┐  │
    │  │ Determine Status    │  │
    │  │ ≥85% → AUTO_APPLY   │  │
    │  │ 70-85% → REVIEW     │  │
    │  │ <70% → REJECT       │  │
    │  └──────────┬──────────┘  │
    │             │             │
    │  ┌──────────▼──────────┐  │
    │  │ Generate Strategy   │  │
    │  │ - Key Skills        │  │
    │  │ - Cover Letter Tips │  │
    │  └──────────┬──────────┘  │
    │             │             │
    │  ┌──────────▼──────────┐  │
    │  │ Update in Notion    │  │
    │  │ - Match Score       │  │
    │  │ - Status            │  │
    │  │ - Analysis          │  │
    │  └────────────────────┘  │
    │           │              │
    └───────────┼──────────────┘
                │
                ▼ (Next Job)
            ┌────────────┐
            │ More Jobs? │
            └──┬──────┬──┘
           Yes│      │No
              │      └────────────────────┐
              │                           │
              ▼                           ▼
        (Repeat Loop)         ┌──────────────────────┐
                              │ Return Analysis      │
                              │ Summary              │
                              │ - Analyzed: X        │
                              │ - High Priority: Y   │
                              │ - Ready to Apply: Z  │
                              └──────────────────────┘
3. Automated Application Submission Workflow
Code
┌──────────────────────────────��───────────────────────────────────────┐
│  AUTOMATED APPLICATION WORKFLOW - SUBMIT APPLICATIONS                │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────┐
│ Start Apply Mode │
└────────┬─────────┘
         │
         ▼
    ┌──────────────────────────┐
    │ Get High-Priority Jobs   │
    │ Filter: STAGED_FOR_APPLY │
    │ Sort: Match Score DESC   │
    └────────────┬─────────────┘
                 │
                 ▼
    ┌───────────────────────────────────┐
    │ Application Processing Loop       │
    ├───────────────────────────────────┤
    │                                   │
    │  ┌────────────────────────────┐   │
    │  │ Load Job & User Profile    │   │
    │  └────────────┬───────────────┘   │
    │               │                   │
    │  ┌────────────▼───────────────┐   │
    │  │ Generate Optimized Resume  │   │
    │  │ - Parse Job Description    │   │
    │  │ - Extract Keywords         │   │
    │  │ - Prioritize Skills        │   │
    │  │ - Generate ATS-Friendly    │   │
    │  │ - Calculate ATS Score      │   │
    │  │ - Export PDF               │   │
    │  └────────────┬───────────────┘   │
    │               │                   │
    │  ┌────────────▼───────────────┐   │
    │  │ Company Intelligence       │   │
    │  │ Research                   │   │
    │  │ - Company Overview         │   │
    │  │ - Culture Insights         │   │
    │  │ - Recent News              │   │
    │  └────────────┬───────────────┘   │
    │               │                   │
    │  ┌────────────▼───────────────┐   │
    │  │ Generate Cover Letter      │   │
    │  │ - AI-Personalized Content  │   │
    │  │ - Company-Specific         │   │
    │  │ - Role-Aligned             │   │
    │  └────────────┬───────────────┘   │
    │               │                   │
    │  ┌────────────▼───────────────┐   │
    │  │ Prepare Application Job    │   │
    │  │ Package Data               │   │
    │  └────────────┬───────────────┘   │
    │               │                   │
    │  ┌────────────▼───────────────┐   │
    │  │ Automated Form Submission  │   │
    │  │ - Open Application URL     │   │
    │  │ - Detect Form Fields       │   │
    │  │ - Fill with Parsed Data    │   │
    │  │ - Handle CAPTCHA           │   │
    │  │ - Submit Application       │   │
    │  └────────────┬───────────────┘   │
    │               │                   │
    │  ┌────────────▼───────────────┐   │
    │  │ Handle Result              │   │
    │  │ - Success: Move to Tracker │   │
    │  │ - Failure: Log & Notify    │   │
    │  └────────────┬───────────────┘   │
    │               │                   │
    │  ┌────────────▼───────────────┐   │
    │  │ Rate Limiting              │   │
    │  │ Wait 60 seconds             │   │
    │  └────────────┬────────���──────┘   │
    │               │                   │
    └───────────────┼───────────────────┘
                    │
                    ▼ (Next Job)
                ┌────────────┐
                │ More Jobs? │
                └──┬──────┬──┘
               Yes│      │No
                  │      └─────────────────────────┐
                  │                                │
                  ▼                                ▼
            (Repeat Loop)            ┌──────────────────────┐
                                     │ Return Summary       │
                                     │ - Attempted: X       │
                                     │ - Successful: Y      │
                                     │ - Success Rate: Z%   │
                                     └──────────────────────┘
4. End-to-End Automation Workflow
Code
┌─────────────────────────────────────────────────────────────────────┐
│  FULL AUTOMATION WORKFLOW - COMPLETE PIPELINE                       │
└─────────────────────────────────────────────────────────────────────┘

                    ┌──────────────────────┐
                    │ START FULL AUTOMATION │
                    └──────────┬───────────┘
                               │
                    ┌──────────▼──────────┐
                    │  PHASE 1: DISCOVER  │
                    ├──────────────────────┤
                    │ • Scrape Jobs       │
                    │ • Deduplicate       │
                    │ • Store in DB       │
                    └──────────┬──────────┘
                               │
                        ⏳ Wait 30s
                               │
                    ┌──────────▼──────────┐
                    │  PHASE 2: ANALYZE   │
                    ├──────────────────────┤
                    │ • Load Jobs         │
                    │ • AI Matching       │
                    │ • Score & Rank      │
                    │ • Update Status     │
                    └──────────┬──────────┘
                               │
                        ⏳ Wait 30s
                               │
                    ┌──────────▼──────────┐
                    │  PHASE 3: APPLY     │
                    ├──────────────────────┤
                    │ • Get High Priority │
                    │ • Generate Resumes  │
                    │ • Create CL         │
                    │ • Submit Apps       │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │ CONSOLIDATE RESULTS │
                    │ - Discovered: X     │
                    │ - Analyzed: Y       │
                    │ - Applied: Z        │
                    │ - Success Rate: W%  │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │ END FULL AUTOMATION  │
                    └──────────────────────┘
🔧 Components Documentation
1. Master Orchestrator (master_run.py)
The central command center that coordinates all system components.

Key Class: MasterOrchestrator

Main Methods:

initialize_system() - Boot all engines and verify health
run_job_discovery(max_jobs_per_platform=50) - Discover jobs
run_job_analysis(limit=20) - Analyze discovered jobs
run_automated_applications(max_applications=10) - Submit apps
run_full_automation() - Complete end-to-end workflow
run_system_health_check() - Verify system integrity
Usage:

bash
python master_run.py --mode discover --limit 50
python master_run.py --mode analyze --limit 20
python master_run.py --mode apply --limit 10
python master_run.py --mode full-automation
python master_run.py --mode health-check --debug
Output: JSON summary with execution statistics

2. Core Engine: Scraper Engine
Responsible for discovering and extracting job listings from multiple platforms.

Module: core/scraper_engine.py

Key Features:

Multi-platform scraping (20+ platforms)
Duplicate detection and removal
Data normalization
Real-time progress tracking
Configurable retry logic
Rate limiting and throttling
Supported Platforms:

LinkedIn Jobs
Indeed
Glassdoor
AngelList
Dice
ZipRecruiter
Monster
CareerBuilder
And 12+ more
Key Methods:

Python
async scrape_all_platforms(
    search_terms: List[str],
    max_jobs_per_platform: int = 50
) -> List[ScrapingResult]

convert_to_job_applications(jobs: List) -> List[JobApplication]
3. Core Engine: AI Engine
Provides AI-powered job analysis, matching, and content generation.

Module: core/ai_engine.py

Key Features:

Semantic job-to-profile matching
Match scoring algorithm (0-100%)
Company intelligence research
Personalized cover letter generation
Application strategy recommendations
Multi-LLM support (Anthropic, OpenAI, Perplexity)
Key Methods:

Python
async analyze_job_opportunity(
    job_data: Dict,
    user_profile: Dict,
    session_id: str
) -> JobAnalysis

async research_company_intelligence(
    company_name: str,
    job_role: str
) -> CompanyIntelligence

async generate_personalized_cover_letter(
    job_data: Dict,
    company_intel: CompanyIntelligence,
    user_profile: Dict,
    resume_highlights: List[str]
) -> CoverLetter
4. Core Engine: Resume Engine
Generates and optimizes resumes for each job application.

Module: core/resume_engine.py

Key Features:

Job-specific resume generation
ATS (Applicant Tracking System) optimization
Skill prioritization based on job description
Multiple export formats (PDF, DOCX, HTML)
Overleaf integration for advanced resumes
Performance scoring and recommendations
Key Methods:

Python
async generate_optimized_resume(
    request: ResumeGenerationRequest
) -> ResumeResult

async optimize_for_ats(resume_data: Dict) -> OptimizationResult
5. Core Engine: Notion Engine
Manages all database operations with Notion.

Module: core/notion_engine.py

Key Features:

Job application tracking database
Status management (Discovered, Staged, Applied, Rejected)
Batch operations for efficiency
Real-time updates and synchronization
Filtering and sorting capabilities
Database Schema:

Applications DB: Stores discovered jobs with metadata
Job Tracker DB: Tracks application status and results
Key Methods:

Python
async batch_create_job_applications(
    applications: List[JobApplication]
) -> List[str]

async get_jobs_by_status(status: ApplicationStatus) -> List[Dict]

async update_job_application(
    page_id: str,
    updates: Dict
) -> bool
6. Core Engine: Automation Engine
Handles browser automation and form filling for job applications.

Module: core/automation_engine.py

Key Features:

Playwright-based browser automation
Intelligent form field detection
Captcha handling and detection
Smart field mapping
Error handling and recovery
Screenshot capture on failure
Key Methods:

Python
async apply_to_job(
    application_job: ApplicationJob
) -> ApplicationAttempt

async fill_application_form(
    url: str,
    form_data: Dict
) -> bool
7. Supporting Systems
A. Configuration System (config/settings.py)
Centralized configuration management
Environment variable integration
Type-safe settings with Pydantic
Multiple configuration profiles
B. RAG Systems (rag_systems/)
Retrieval Augmented Generation for job analysis
Company knowledge base
Embeddings-based semantic search
Context-aware information retrieval
C. MCP Integration (mcp/)
Model Context Protocol support
Extended AI capabilities
Standardized tool interfaces
Integration with Claude and other AI models
D. N8N Workflows (n8n/)
Visual workflow automation
Complex multi-step automation
Conditional logic and branching
Integration with external services
E. Chrome Extension (chrome_extension/)
Browser-based user interface
Real-time job discovery notifications
One-click job application
Quick access to automation features
📘 Usage Guide
Scenario 1: Automated Daily Job Applications
bash
# Run every morning at 9 AM
0 9 * * * cd /path/to/AI_Job_Automation_Agent && python master_run.py --mode full-automation
Scenario 2: Manual Review Before Applying
bash
# Step 1: Discover new jobs
python master_run.py --mode discover --limit 100

# Step 2: Analyze and review in Notion (manual review)
python master_run.py --mode analyze

# Step 3: Submit high-confidence applications only
python master_run.py --mode apply --limit 5
Scenario 3: Targeted Job Search
bash
# Configure in .env or settings.yaml
SEARCH_CRITERIA={
  "job_titles": ["ML Engineer", "AI Engineer", "Data Scientist"],
  "locations": ["Remote", "San Francisco", "NYC"],
  "salary_min": 120000,
  "experience_level": ["Senior", "Lead"]
}

# Run discovery
python master_run.py --mode discover --limit 50
Scenario 4: Monitoring and Health Checks
bash
# Check system health
python master_run.py --mode health-check

# Get detailed logs
python master_run.py --mode discover --debug

# Monitor running process
tail -f automation.log
⚙️ Configuration
Environment Variables
Create a .env file in the project root:

bash
# ===== AI/LLM Configuration =====
ANTHROPIC_API_KEY=sk-ant-xxx
OPENAI_API_KEY=sk-xxx
PERPLEXITY_API_KEY=ppl-xxx

# ===== Job Platform Credentials =====
LINKEDIN_EMAIL=your.email@example.com
LINKEDIN_PASSWORD=your_secure_password
INDEED_API_KEY=indeed_key_xxx
GLASSDOOR_API_KEY=glassdoor_key_xxx

# ===== Database Configuration =====
NOTION_API_KEY=notion_token_xxx
NOTION_DATABASE_ID_APPLICATIONS=xxx
NOTION_DATABASE_ID_TRACKER=xxx

# ===== External Services =====
GMAIL_CLIENT_ID=client_id
GMAIL_CLIENT_SECRET=client_secret
OVERLEAF_API_KEY=overleaf_key

# ===== System Configuration =====
TIMEZONE=Asia/Kolkata
DAILY_RUN_TIME=09:00
MAX_APPLICATIONS_PER_DAY=15
SCRAPING_DELAY_MIN=2
SCRAPING_DELAY_MAX=5
PLAYWRIGHT_HEADLESS=true
Advanced Configuration
Edit config/settings.py for advanced settings:

Python
# Search criteria
SEARCH_CRITERIA = {
    'job_titles': ['ML Engineer', 'Data Scientist', 'AI Engineer'],
    'locations': ['Remote', 'San Francisco'],
    'salary_range': {'min': 100000, 'max': 200000},
    'experience_level': ['Mid', 'Senior', 'Lead']
}

# Automation settings
AUTOMATION = {
    'max_retries': 3,
    'timeout': 30,
    'headless': True,
    'rate_limit': 60  # seconds between applications
}

# AI/Analysis settings
AI_ANALYSIS = {
    'min_match_score_for_apply': 85,
    'min_match_score_for_review': 70,
    'use_multi_llm': True,
    'llm_models': ['claude-3-opus', 'gpt-4']
}
👥 Contributing
We welcome contributions! Here's how to get involved:

Development Setup
bash
# Clone and setup
git clone https://github.com/aarjunm04/AI_Job_Automation_Agent.git
cd AI_Job_Automation_Agent
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Install dev dependencies
pip install pytest pytest-cov black flake8 mypy

# Run tests
pytest tests/ -v --cov

# Format code
black . --line-length 100

# Type checking
mypy .
Contributing Guidelines
Fork the Repository
Create a Feature Branch: git checkout -b feature/amazing-feature
Make Changes: Follow PEP 8 style guide
Write Tests: Ensure new functionality is tested
Run Quality Checks:
bash
pytest tests/
black .
flake8 .
mypy .
Commit Changes: git commit -m 'Add amazing feature'
Push to Branch: git push origin feature/amazing-feature
Create Pull Request: Describe changes clearly
Areas for Contribution
 Additional job platform scrapers
 Improved AI matching algorithms
 Enhanced form filling accuracy
 Better CAPTCHA handling
 Additional language support
 Performance optimizations
 Documentation improvements
 Bug fixes and issue resolution
📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

In simple terms:

✅ You can use this project for personal and commercial purposes
✅ You can modify and distribute the code
✅ You must include the license and copyright notice
❌ The authors provide no warranty
📞 Support & Contact
Get Help
📧 Email: aarjun.mahule@example.com
🐛 Issues: GitHub Issues
💬 Discussions: GitHub Discussions
📖 Documentation: Wiki
Troubleshooting
Q: Application keeps failing A: Check automation.log for detailed error messages. Run health check:

bash
python master_run.py --mode health-check
Q: Notion API errors A: Verify your Notion API key and database permissions. Test with:

bash
python -c "from core.notion_engine import NotionEngine; print('✅ Connected')"
Q: Resume not generating A: Check OpenAI/Anthropic API keys. Verify Overleaf integration if using LaTeX.

Q: Jobs not being scraped A: Platform credentials may be invalid. Update credentials in .env and test individual scraper.

🌟 Acknowledgments
Built with ❤️ by the AI Job Automation Team
Powered by cutting-edge AI/ML technologies
Community contributions and feedback
Open-source libraries and frameworks
📈 Project Statistics
Code
┌─────────────────────────────────────────┐
│ Language Composition                    │
├─────────────────────────────────────────┤
│ Python        ████████████████░░░  69.2%│
│ JavaScript    ████░░░░░░░░░░░░░░░  17.8%│
│ CSS           ██░░░░░░░░���░░░░░░░░   5.4%│
│ HTML          ██░░░░░░░░░░░░░░░░░   4.3%│
│ Shell         █░░░░░░░░░░░░░░░░░░   3.3%│
└─────────────────────────────────────────┘

Key Metrics:
- 20+ Supported Job Platforms
- 100+ Python Dependencies
- 5 Core Engine Modules
- 3 Main Workflow Modes
- Async/Await Architecture
- Enterprise-Grade Reliability
<div align="center">
Made with ❤️ by Aarjun Mahule

⬆ Back to Top

</div> ```
