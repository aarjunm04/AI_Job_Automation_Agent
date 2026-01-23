# AI Job Automation Agent - Comprehensive Project Architecture

**Generated:** January 20, 2026  
**Project Location:** `/Users/apple/TechStack/Projects/AI_Job_Automation_Agent`  
**Version:** 2.0 Enterprise

---

## Table of Contents

1. [Directory Tree Structure](#directory-tree-structure)
2. [Runtime Dependencies](#runtime-dependencies)
3. [Configuration Files](#configuration-files)
4. [Port Mappings & Inter-Service Communication](#port-mappings--inter-service-communication)
5. [Persistent Data Locations](#persistent-data-locations)
6. [Environment Variables](#environment-variables)
7. [External API Dependencies & Credentials](#external-api-dependencies--credentials)

---

## 1. Directory Tree Structure

### Project Root Organization

```
/Users/apple/TechStack/Projects/AI_Job_Automation_Agent/
├── docker-compose.yml              # Container orchestration (PostgreSQL, N8N)
├── dump.rdb                        # Redis persistence dump
├── requirements.txt                # Python dependencies
├── master_run.py                   # Main entry point for entire system
├── rag_system_init.py             # RAG system initialization
├── README.md                       # Project documentation
├── SCRAPER_ENGINE_README.md       # Scraper documentation
│
├── agents/                         # AI Agent Implementation
│   ├── agent_service.py           # Agent service orchestration
│   ├── agents.py                  # Agent definitions and logic
│   └── __init__.py
│
├── automation/                     # Automation Engine
│   ├── __init__.py
│   ├── automation.py              # Main automation logic
│   └── cleanup_manager.py         # Cleanup and resource management
│
├── chrome_extension/              # Chrome Extension (Frontend)
│   ├── manifest.json              # Extension manifest v3
│   ├── popup.html                 # UI popup interface
│   ├── popup.js                   # Popup logic
│   ├── background.js              # Background service worker
│   ├── content.js                 # Content script for job sites
│   ├── sidebar.html               # Sidebar interface
│   ├── sidebar.js                 # Sidebar logic
│   ├── sidebar.css                # Sidebar styling
│   ├── extension_config.js        # Extension configuration (MCP, RAG)
│   ├── mcp_client.js              # MCP client implementation
│   ├── notion_api.js              # Notion API wrapper
│   └── icons/                     # Icon assets
│
├── config/                        # Configuration Management
│   ├── settings.py                # Centralized config (dataclasses)
│   ├── job_filters.yaml           # Job filtering rules
│   ├── proxy_pool.json            # Proxy configuration (empty)
│   └── resume_template.json       # Resume template (referenced)
│
├── core/                          # Core Business Logic
│   ├── __init__.py
│   ├── job_filters.yaml           # Job filtering rules
│   ├── jobspy_adapter.py          # JobSpy API adapter
│   ├── scraper_engine.py          # Web scraping orchestration
│   └── __pycache__/
│
├── docker/                        # Docker Configuration
│   ├── Dockerfile                 # Multi-stage build (Python + Node.js)
│   └── (nginx, postgres configs)
│
├── integrations/                  # External Service Integrations
│   ├── gmail_stub.py              # Gmail integration
│   ├── llm_interface.py           # LLM provider routing
│   ├── notion_integration.py      # Notion API integration
│   └── (additional integrations)
│
├── logs/                          # Application Logs
│   ├── latest_metrics.json        # Performance metrics
│   ├── latest_run.json            # Last execution summary
│   └── serpapi_usage.json         # API usage tracking
│
├── mcp/                           # Model Context Protocol (Enterprise)
│   ├── server.py                  # MCP FastAPI server (1978 lines)
│   ├── integrations.py            # MCP service integrations (931 lines)
│   ├── cache.py                   # Redis caching & metrics (667 lines)
│   ├── security.py                # Security & encryption
│   ├── mcp_context.db             # SQLite context database
│   ├── Dockerfile                 # MCP container build
│   └── __pycache__/
│
├── n8n/                           # N8N Workflow Automation
│   └── workflows/
│       ├── agent_importv2.json    # Agent import workflow
│       ├── ai_nodev3.json         # AI node workflow
│       ├── Job Automation.json    # Main job automation workflow
│       └── perp_aiagent.json      # Perplexity AI agent workflow
│
├── n8n_data/                      # N8N Persistent Storage
│   ├── config/                    # N8N configuration
│   ├── binaryData/                # Binary data storage
│   ├── git/                       # Git integration data
│   ├── ssh/                       # SSH keys
│   └── nodes/
│       └── package.json           # N8N custom nodes
│
├── postgres_data/                 # PostgreSQL Database Storage
│   ├── base/                      # Database cluster base files
│   │   ├── 1/, 4/, 5/, 16384/    # Tablespace directories
│   │   └── ...
│   ├── global/                    # Global database files
│   │   ├── 1213, 1260, etc       # System table files
│   │   └── ...
│   ├── pg_wal/                    # Write-ahead logs
│   ├── pg_xact/                   # Transaction status files
│   ├── pg_multixact/              # Multi-transaction files
│   ├── pg_logical/                # Logical replication
│   ├── pg_replslot/               # Replication slots
│   ├── pg_tblspc/                 # Tablespace links
│   ├── pg_stat_tmp/               # Statistics temporary
│   └── postgresql.conf            # PostgreSQL configuration
│
├── rag_systems/                   # RAG (Retrieval-Augmented Generation)
│   ├── production_server.py       # RAG HTTP server (1271 lines)
│   ├── rag_pipeline.py            # RAG pipeline orchestration (367 lines)
│   ├── rag_api.py                 # RAG API endpoints
│   ├── chromadb_store.py          # ChromaDB vector store
│   ├── resume_engine.py           # Resume processing & matching
│   ├── ingest_all_resumes.py      # Resume ingestion utility
│   ├── resume_config.json         # Resume configuration
│   └── __pycache__/
│
├── scrapers/                      # Web Scraping Services
│   ├── __init__.py
│   ├── scraper_service.py         # Playwright-based scraping
│   ├── apply_service.py           # Application submission service
│   ├── captcha_solver.py          # CAPTCHA solving utilities
│   ├── playwright_support.py      # Playwright extensions
│   └── __pycache__/
│
├── scripts/                       # Utility Scripts
│   ├── build_extension.sh         # Chrome extension build script
│   ├── daily_cron.sh              # Daily job run scheduler
│   ├── run_agent.sh               # Agent execution script
│   ├── run_scraper.sh             # Scraper execution script
│   ├── setup.sh                   # Initial setup script
│   └── stress_test.py             # Load testing utility
│
├── assets/                        # Static Assets & Resources
│   └── (images, icons, templates)
│
└── venv/                          # Python Virtual Environment
    └── bin/activate
```

### Component Breakdown

#### **MCP (Model Context Protocol) - Lines: 2,500+ total**
- Enterprise-grade context management service
- Session lifecycle management with TTL and versioning
- Context item storage with sequence tracking
- Snapshot/summarization with LLM integration
- Evidence attachment system
- Audit trail for all operations
- Redis caching and compression
- Rate limiting and security
- Prometheus metrics

#### **RAG System - Lines: 1,500+ total**
- Production-ready HTTP server (FastAPI)
- Session management with conversation history
- Resume-to-job matching using vector search
- ChromaDB for embeddings storage
- Multiple embedding providers (Gemini, OpenAI)
- Circuit breaker patterns
- Rate limiting & throttling

#### **Scraper Engine - Lines: 1,000+ total**
- Playwright-based browser automation
- API-based scraping (JobSpy, Jooble, SerpAPI)
- Proxy management (10 static Webshare proxies)
- Captcha solving integration
- Concurrent scraping coordination

#### **Chrome Extension - 10 files**
- Content scripts for job site interaction
- MCP client for real-time AI assistance
- Resume management and optimization
- Notion integration
- Job tracking and analysis

#### **Automation Engine**
- Master orchestration script
- Multi-mode operation (discover, apply, full-automation)
- Job analysis and matching
- Dynamic resume generation
- Application submission

#### **N8N Workflows - 4 workflows**
- Agent import and configuration
- AI node orchestration
- Job automation pipeline
- Perplexity AI integration

---

## 2. Runtime Dependencies

### Python Dependencies (requirements.txt)

#### **Core Framework & Web**
```
fastapi==0.118.0
flask==3.1.2
uvicorn[standard]==0.31.1
gunicorn==23.0.0
starlette==0.48.0
```

#### **Async & HTTP**
```
aiohttp==3.12.15
aiofiles==24.1.0
httpx[http2]==0.27.2
requests==2.32.3
websockets==15.0.1
```

#### **Database & ORM**
```
sqlalchemy==2.0.43
asyncpg==0.30.0
alembic==1.16.5
```

#### **Redis & Caching**
```
redis==5.1.0
diskcache==5.6.3
```

#### **AI & LLM Integration**
```
anthropic==0.34.2
openai==1.102.0
perplexityai==0.13.0
sentence-transformers==3.1.1
```

#### **Vector Database & RAG**
```
chromadb==2.0.0  # (implicit from chromadb_store.py)
```

#### **Data Processing**
```
pandas==2.3.3
numpy==2.2.2
beautifulsoup4==4.14.2
lxml==5.3.0
pyyaml==6.0.3
```

#### **Web Scraping & Automation**
```
playwright==1.55.0
selenium==4.25.0
scrapy==2.13.3
scrapy-splash==0.8.0
undetected-chromedriver==3.5.5
requests-toolbelt==1.0.0
```

#### **Document Processing**
```
pdfplumber==0.11.4
PyPDF2==3.0.1
openpyxl==3.1.5
pytesseract==0.3.13
easyocr==1.7.1
```

#### **Computer Vision**
```
opencv-python-headless==4.12.0.88
pillow==11.3.0
```

#### **Job Scraping APIs**
```
google-api-python-client==2.147.0
google-auth==2.34.0
google-auth-oauthlib==1.2.1
```

#### **Configuration & Environment**
```
python-dotenv==1.1.1
python-decouple==3.8
pydantic==2.11.0
pydantic-settings==2.11.0
dynaconf==3.2.11
```

#### **Authentication & Security**
```
cryptography==46.0.2
bcrypt==4.3.0
pyjwt==2.10.1
passlib[bcrypt]==1.7.4
pyotp==2.9.0
```

#### **Email & Communication**
```
imapclient==3.0.1
email-validator==2.2.0
```

#### **Notion Integration**
```
notion-client==2.3.0
```

#### **NLP & Text Processing**
```
nltk==3.9.1
textblob==0.19.0
spacy==3.8.0
```

#### **ML & Embeddings**
```
scikit-learn==1.7.2
tiktoken==0.11.0
```

#### **Monitoring & Logging**
```
loguru==0.7.3
structlog==25.4.0
colorlog==6.8.2
prometheus-client==0.23.1
rich==13.8.1
```

#### **Task Scheduling**
```
celery==5.5.3
```

#### **Utility Libraries**
```
click==8.2.2
typer==0.12.5
jsonschema==4.25.1
marshmallow==4.0.1
mcp==1.16.0
```

#### **Development & Testing**
```
pytest==8.4.2
pytest-asyncio==1.2.0
pytest-cov==7.0.0
pytest-mock==3.14.0
black==25.9.0
flake8==7.3.0
isort==5.13.2
mypy==1.18.2
```

#### **Other Utilities**
```
fake-useragent==1.5.1
pylatex==1.4.2
reportlab==4.2.5
supervisor==4.2.5
docker==7.1.0
psutil==6.0.0
pympler==0.9
memory-profiler==0.61.0
```

### System Requirements

```
Python: 3.11+
Node.js: 18+
PostgreSQL: 15
Redis: latest
Docker: 20.10+
Docker Compose: 1.29+
```

### Node.js Components (package.json referenced in scripts)

```json
{
  "dependencies": {
    "playwright": "^1.55.0",
    "express": "^4.x",
    "cors": "^2.x",
    "axios": "^1.x"
  },
  "devDependencies": {
    "@types/node": "^20.x"
  }
}
```

---

## 3. Configuration Files

### Critical Configuration Files

#### **A. Environment Configuration (.env)**

**Location:** `.env` (root)

**Primary Variables:**
```ini
# N8N Configuration
N8N_PORT=5678
N8N_HOST=localhost
WEBHOOK_URL=http://localhost:5678
GENERIC_TIMEZONE=Asia/Kolkata
N8N_BASIC_AUTH_ACTIVE=true
N8N_BASIC_AUTH_USER=admin
N8N_BASIC_AUTH_PASSWORD=ai_job_2025
N8N_ENCRYPTION_KEY=<generated-key>
N8N_USER_MANAGEMENT_ENABLED=false

# PostgreSQL Configuration
POSTGRES_DB=n8n_ai_job
POSTGRES_USER=n8n_user
POSTGRES_PASSWORD=n8n_password_2025
DB_TYPE=postgresdb

# AI API Keys (REQUIRED)
OPENAI_API_KEY=<your-key>
OPENAI_MODEL=gpt-4-turbo-preview
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_TEMPERATURE=0.7
OPENAI_MAX_TOKENS=4000

PERPLEXITY_API_KEY=<your-key>
PERPLEXITY_MODEL=llama-3.1-sonar-huge-128k-online
PERPLEXITY_TEMPERATURE=0.5

GEMINI_API_KEY_RAG=<your-key>

# Notion Integration (REQUIRED)
NOTION_API_KEY=<your-key>
NOTION_APPLICATIONS_DB_ID=<your-id>
NOTION_JOB_TRACKER_DB_ID=<your-id>
NOTION_VERSION=2022-06-28
NOTION_TIMEOUT=30
NOTION_MAX_RETRIES=3

# Resume Generation (OPTIONAL)
OVERLEAF_API_KEY=<your-key>
OVERLEAF_PROJECT_ID=<your-id>
OVERLEAF_COMPILE_URL=https://api.overleaf.com/docs/compile

# Gmail Integration (OPTIONAL)
GMAIL_CLIENT_ID=<your-id>
GMAIL_CLIENT_SECRET=<your-secret>
GMAIL_REDIRECT_URI=http://localhost:8080/auth/callback
GMAIL_SCOPES=https://www.googleapis.com/auth/gmail.readonly
GMAIL_CHECK_INTERVAL=300

# Scraping Configuration
SCRAPING_DELAY_MIN=2
SCRAPING_DELAY_MAX=5
MAX_CONCURRENT_SCRAPERS=5
SCRAPING_TIMEOUT=30
SCRAPING_MAX_RETRIES=3
USER_AGENT=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36

LINKEDIN_EMAIL=<your-email>
LINKEDIN_PASSWORD=<your-password>
INDEED_API_KEY=<your-key>
JOOBLE_API_KEY=<your-key>
SERPAPI_API_KEY_1=<your-key>

# Playwright Configuration
PLAYWRIGHT_SERVICE_URL=http://localhost:3000
PLAYWRIGHT_HEADLESS=true
PLAYWRIGHT_TIMEOUT=30000
PLAYWRIGHT_VIEWPORT_WIDTH=1920
PLAYWRIGHT_VIEWPORT_HEIGHT=1080
SAVE_SCREENSHOTS=false

# Webshare Proxies (10 static proxies)
WEBSHARE_PROXY_2_1=http://username:password@ip:port
WEBSHARE_PROXY_2_2=http://username:password@ip:port
... (up to WEBSHARE_PROXY_2_10)

# System Configuration
TIMEZONE=Asia/Kolkata
DAILY_RUN_TIME=09:00
MAX_APPLICATIONS_PER_DAY=50
MIN_JOB_MATCH_SCORE=70
HIGH_PRIORITY_DAYS=3
MEDIUM_PRIORITY_DAYS=7
LOW_PRIORITY_DAYS=14

# Security
ENCRYPTION_KEY=<generated-key>
JWT_SECRET=<generated-secret>
SESSION_TIMEOUT=3600
EXTENSION_API_SECRET=<generated-secret>

# MCP Server Configuration
MCP_SERVER_URL=http://localhost:3001
MCP_CLIENT_TIMEOUT=30
MCP_MAX_RETRIES=3
MCP_CONTEXT_WINDOW_SIZE=32000
MCP_API_KEY=<your-key>
JWT_EXPIRY_HOURS=24
DATABASE_URL=sqlite+aiosqlite:///mcp/mcp_context.db
REDIS_URL=redis://localhost:6379/0
REDIS_ENABLED=true
RATE_LIMIT_ENABLED=true
RATE_LIMIT_REQUESTS=1000
RATE_LIMIT_WINDOW=60
SESSION_TTL_HOURS=168
MCP_DEV_MODE=false

# RAG Server Configuration
RAG_SERVER_HOST=localhost
RAG_SERVER_PORT=8090
RAG_SERVER_RELOAD=false
RAG_SERVER_WORKERS=1
RAG_BASE_URL=http://localhost:8090
RAG_API_KEY=master-bcb8c7fe4d2e1a0526130f5f2f954bfd
SESSION_TIMEOUT_MINUTES=30
SESSION_MAX_HISTORY=50
SESSION_CLEANUP_INTERVAL=300

# Monitoring (OPTIONAL)
GRAFANA_USER=admin
GRAFANA_PASSWORD=ai_job_grafana_2025
DEBUG_MODE=false
TEST_MODE=false
LOG_LEVEL=INFO
VERBOSE_LOGGING=false

# File Storage (OPTIONAL)
MINIO_ROOT_USER=ai_job_admin
MINIO_ROOT_PASSWORD=ai_job_minio_2025

# Notifications (OPTIONAL)
SLACK_WEBHOOK_URL=<your-url>
DISCORD_WEBHOOK_URL=<your-url>

# Mock & Testing
MOCK_APPLICATIONS=false
```

#### **B. Job Filters Configuration**

**Location:** `config/job_filters.yaml`

**Purpose:** Define job matching rules, location filters, salary ranges, required/excluded keywords

**Example Structure:**
```yaml
platforms:
  - linkedin
  - indeed
  - glassdoor

filters:
  locations:
    - USA
    - Remote
  salary_min: 100000
  experience_level: mid-senior
  
keywords:
  required:
    - AI
    - machine-learning
  excluded:
    - blockchain
```

#### **C. Resume Configuration**

**Location:** `rag_systems/resume_config.json`

**Purpose:** Resume template and generation parameters

**Contains:**
- Resume sections (contact, experience, skills, education)
- Formatting preferences
- Keyword optimization settings

#### **D. Proxy Pool Configuration**

**Location:** `config/proxy_pool.json`

**Status:** Empty (proxies managed via environment variables)

**Note:** Uses WEBSHARE_PROXY_2_1 through WEBSHARE_PROXY_2_10 environment variables

#### **E. Docker Compose Configuration**

**Location:** `docker-compose.yml`

**Services Defined:**
- `postgres`: Database service (port 5432)
- `n8n`: Workflow automation (port 5678)
- `redis`: Cache store (port 6379)

#### **F. Docker Dockerfile**

**Location:** `docker/Dockerfile`

**Build Stages:**
1. Python builder (3.11-slim base)
2. Node.js builder (18-slim base)
3. Final production image
   - Python 3.11
   - Chromium browser
   - All dependencies
   - Exposed ports: 8080, 3001, 3002

#### **G. MCP Dockerfile**

**Location:** `mcp/Dockerfile`

**Purpose:** Containerize MCP service separately

**Environment Variables Set:**
```
PYTHONUNBUFFERED=1
PYTHONDONTWRITEBYTECODE=1
PIP_NO_CACHE_DIR=1
MCP_DEV_MODE=false
```

#### **H. Settings Configuration**

**Location:** `config/settings.py` (356 lines)

**Dataclasses Defined:**
```python
AIConfig
NotionConfig
OverleafConfig
GmailConfig
ScrapingConfig
PlaywrightConfig
SystemConfig
SecurityConfig
LoggingConfig
```

#### **I. Extension Configuration**

**Location:** `chrome_extension/extension_config.js`

**Exports:**
```javascript
MCP_CONFIG           // MCP server endpoints
API_CONFIG          // RAG endpoints
STORAGE_KEYS        // Chrome storage keys
FEATURES            // Feature flags
DEFAULT_USER_SETTINGS
```

---

## 4. Port Mappings & Inter-Service Communication

### Port Allocations

| Service | Port | Protocol | Purpose | Status |
|---------|------|----------|---------|--------|
| **N8N Workflows** | 5678 | HTTP | Workflow UI & API | Docker |
| **PostgreSQL** | 5432 | TCP | Database | Docker (internal) |
| **Redis** | 6379 | TCP | Cache/Sessions | Default/Local |
| **MCP Server** | 3001 | HTTP | Context Management | FastAPI |
| **RAG Server** | 8090 | HTTP | Resume Matching | FastAPI |
| **Backend API** | 8080 | HTTP | Main API | FastAPI/Flask |
| **Playwright Service** | 3000 | HTTP | Browser Automation | Optional |
| **Grafana** | 3000 | HTTP | Monitoring | Optional |
| **Gmail Redirect** | 8080 | HTTP | OAuth Callback | Embedded |

### Inter-Service Communication Patterns

#### **1. Chrome Extension → Backend Stack**

```
┌─────────────────────────────────────────────────────────────┐
│                    Chrome Extension                         │
│          (popup.js, content.js, background.js)             │
└────────┬────────────────────────────────────────────────────┘
         │
    ┌────▼──────────────────────────────────────────────────┐
    │  1. MCP Client (mcp_client.js)                        │
    │     → POST http://localhost:8080/llm/complete         │
    │     → Sends: API key, job description, user resume   │
    └────┬──────────────────────────────────────────────────┘
         │
    ┌────▼──────────────────────────────────────────────────┐
    │  2. RAG API (notion_api.js)                           │
    │     → POST http://localhost:8090/match                │
    │     → Sends: Resume, job posting                      │
    └────┬──────────────────────────────────────────────────┘
         │
    ┌────▼──────────────────────────────────────────────────┐
    │  3. Notion Integration (notion_api.js)                │
    │     → POST https://api.notion.com/v1/pages            │
    │     → Sends: Job tracking updates                     │
    └─────────────────────────────────────────────────────────┘
```

#### **2. Master Run → Core Services**

```
┌─────────────────────────────────┐
│      master_run.py              │
│  (Main Orchestration Script)    │
└────────┬────────────────────────┘
         │
    ┌────▼────────────────────────────────────────┐
    │  Scraper Engine → Job Sites                 │
    │  (Playwright/Selenium/API)                  │
    │  Output: Job List                           │
    └────┬──────────────────────────────────────┬─┘
         │                                      │
    ┌────▼─────────────────┐          ┌────────▼──────────┐
    │  AI Engine           │          │  MCP Server       │
    │  (RAG + LLM)         │          │  (Context Mgmt)   │
    │  ↓                   │          │  ↓                │
    │  Match scoring       │          │  Session tracking │
    │  Analysis            │          │  Audit trail      │
    └────┬─────────────────┘          └────┬──────────────┘
         │                                  │
    ┌────▼──────────────────────────────────▼────┐
    │  Notion Engine                             │
    │  (Job Tracking Database)                   │
    │  ↓                                         │
    │  Store applications                        │
    │  Update status                             │
    └────────────────────────────────────────────┘
```

#### **3. N8N Workflows → External Services**

```
N8N Workflows (5678)
    │
    ├─→ PostgreSQL (5432) - State storage
    ├─→ MCP Server (3001) - Context queries
    ├─→ RAG Server (8090) - Resume matching
    ├─→ LLM Providers:
    │   ├─→ OpenAI API
    │   ├─→ Perplexity API
    │   └─→ Gemini API
    ├─→ Notion API - Job tracking
    └─→ Gmail API - Email notifications
```

#### **4. MCP Server Architecture**

```
MCP Server (3001) - FastAPI
├─ Database Layer
│  └─ SQLAlchemy ORM
│     └─ PostgreSQL (5432) | SQLite (dev)
│
├─ Cache Layer
│  └─ Redis (6379)
│     ├─ Session cache
│     ├─ Context item cache
│     └─ Rate limiting
│
├─ Integration Layer
│  ├─ RAG Client → RAG Server (8090)
│  ├─ LLM Router
│  │  ├─ OpenAI
│  │  ├─ Perplexity
│  │  └─ Grok
│  ├─ Notion Client
│  └─ Scraper Client
│
└─ API Endpoints
   ├─ Session management
   ├─ Context operations
   ├─ Snapshot/summarization
   └─ Audit trails
```

#### **5. RAG Server Architecture**

```
RAG Server (8090) - FastAPI
├─ Session Management
│  └─ In-memory + optional Redis
│
├─ Resume-to-Job Matching
│  ├─ Embedding Pipeline
│  │  ├─ Gemini (text-embedding-004)
│  │  └─ OpenAI (text-embedding-3-small)
│  │
│  └─ Vector Search
│     └─ ChromaDB Store
│
├─ Rate Limiting
│  └─ Token bucket per API key
│
└─ Circuit Breaker Patterns
   └─ For LLM provider failover
```

### Data Flow Diagram

```
[Job Sites] ──────────────┐
                          │
[Playwright Scraper]◄─────┤
                          │
                    ┌─────▼──────────────┐
                    │   Job Analysis     │
                    │   (AI Engine)      │
                    │   ↓                │
                    │   Match Scoring    │
                    └─────┬──────────────┘
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
   ┌────▼─────┐      ┌────▼────┐      ┌───▼──────┐
   │ MCP Cache │      │ RAG Store│      │ Notion DB │
   │ (Redis)   │      │(ChromaDB)│      │(PostgreSQL)
   └────┬─────┘      └────┬────┘      └───┬──────┘
        │                 │                │
        └─────────────────┼────────────────┘
                          │
                    ┌─────▼──────────────┐
                    │  Apply Service     │
                    │  (Automated Form   │
                    │   Filling)         │
                    └────────────────────┘
```

---

## 5. Persistent Data Locations

### Database Storage

#### **A. PostgreSQL Database** (`postgres_data/`)

**Size:** Multi-gigabyte (contains N8N state + custom schemas)

**Directory Structure:**
```
postgres_data/
├── base/                    # Database cluster base
│   ├── 1/, 4/, 5/          # System databases
│   ├── 16384/              # Main application tablespace
│   └── ...
├── global/                  # Global objects
│   ├── 1213, 1260, etc    # System table files
│   ├── *_fsm              # Free space maps
│   └── *_vm               # Visibility maps
├── pg_wal/                  # Write-ahead logs (critical)
├── pg_xact/                 # Transaction status
├── pg_multixact/            # Multi-transaction data
├── pg_logical/              # Logical replication
├── pg_replslot/             # Replication slots
├── pg_tblspc/               # Tablespace links
├── pg_stat_tmp/             # Statistics temporary
├── pg_dynshmem/             # Dynamic shared memory
├── pg_notify/               # Notification data
├── pg_serial/               # Serializable transactions
├── pg_snapshots/            # Snapshot files
├── pg_subtrans/             # Subtransaction data
└── postgresql.conf          # Main configuration
```

**Content:**
- N8N workflows and execution history
- N8N user data and credentials
- Custom application schemas (if any)
- PostgreSQL system tables

**Backup Strategy:** Volume mount persists across container restarts

#### **B. Redis Persistence** (`dump.rdb`)

**Location:** Root directory

**Format:** RDB (Redis Database) binary format

**Contains:**
- Session cache
- Context item cache
- Rate limiting counters
- Real-time data

**Configuration:** Default Redis persistence

#### **C. MCP Context Database** (`mcp/mcp_context.db`)

**Location:** `mcp/mcp_context.db`

**Type:** SQLite (development) | PostgreSQL (production)

**Tables:**
- `session` - Session lifecycle
- `context_item` - Context storage
- `snapshot` - Summarizations
- `evidence` - Attachments
- `audit_log` - Operation trails

#### **D. Vector Store** (ChromaDB)

**Location:** `rag_systems/.chroma/`

**Contains:**
- Resume embeddings
- Job description embeddings
- Similarity index
- Metadata

**Embedding Model:** Gemini text-embedding-004 (768 dimensions)

#### **E. N8N Data Storage** (`n8n_data/`)

**Size:** 100MB - 1GB+

**Directory Structure:**
```
n8n_data/
├── config/                  # N8N configuration
├── binaryData/              # File storage (uploads, exports)
├── git/                     # Git integration data
├── ssh/                     # SSH keys for integrations
└── nodes/
    ├── package.json        # Custom node definitions
    └── (node modules)
```

**Persistence:** Named volume in Docker

### Application Logs

#### **A. Application Logs** (`logs/`)

**Files:**
```
logs/
├── latest_metrics.json      # Performance metrics
│   └── Contains: API calls, response times, errors
├── latest_run.json          # Last execution summary
│   └── Contains: Jobs found, applied, failed
├── serpapi_usage.json       # API usage statistics
│   └── Contains: API calls, quota remaining
└── (generated *.log files)
```

**Rotation:** Log files managed by Python logging handler (10MB per file, 5 backups)

**Path:** `logs/automation.log`

#### **B. Chrome Extension Logs** (Browser Console)

**Location:** DevTools → Console

**Contains:** Extension execution traces, API calls, errors

#### **C. Playwright Logs** (`playwright_scrapers.log`)

**Location:** Root directory

**Contains:** Browser automation logs, scraping traces, timing info

---

## 6. Environment Variables

### Environment Variable Categories

#### **A. AI & LLM Configuration**

| Variable | Default | Type | Required | Purpose |
|----------|---------|------|----------|---------|
| `OPENAI_API_KEY` | - | string | Yes* | OpenAI API authentication |
| `OPENAI_MODEL` | gpt-4-turbo-preview | string | No | Model selection |
| `OPENAI_EMBEDDING_MODEL` | text-embedding-3-small | string | No | Embedding model |
| `OPENAI_TEMPERATURE` | 0.7 | float | No | Output randomness (0-1) |
| `OPENAI_MAX_TOKENS` | 4000 | int | No | Max completion tokens |
| `PERPLEXITY_API_KEY` | - | string | Yes* | Perplexity API key |
| `PERPLEXITY_MODEL` | llama-3.1-sonar-huge-128k-online | string | No | Perplexity model |
| `PERPLEXITY_TEMPERATURE` | 0.5 | float | No | Output randomness |
| `GEMINI_API_KEY_RAG` | - | string | Yes* | Gemini embeddings API |
| `GEMINI_API_KEY` | - | string | No | Gemini chat API (MCP) |
| `OPENROUTER_API_KEY` | - | string | No | OpenRouter fallback |
| `GROK_API_KEY` | - | string | No | Grok AI fallback |

#### **B. Database Configuration**

| Variable | Default | Type | Purpose |
|----------|---------|------|---------|
| `DATABASE_URL` | sqlite+aiosqlite:///mcp/mcp_context.db | string | MCP database connection |
| `DB_POOL_SIZE` | 20 | int | Connection pool size |
| `DB_MAX_OVERFLOW` | 10 | int | Extra connections allowed |
| `DB_POOL_TIMEOUT` | 30 | int | Pool timeout (seconds) |
| `POSTGRES_DB` | n8n_ai_job | string | PostgreSQL database name |
| `POSTGRES_USER` | n8n_user | string | PostgreSQL user |
| `POSTGRES_PASSWORD` | - | string | PostgreSQL password |
| `DB_TYPE` | postgresdb | string | N8N database type |

#### **C. Redis Configuration**

| Variable | Default | Type | Purpose |
|----------|---------|------|---------|
| `REDIS_URL` | redis://localhost:6379/0 | string | Redis connection URL |
| `REDIS_ENABLED` | true | bool | Enable Redis caching |

#### **D. Notion Integration**

| Variable | Default | Type | Required | Purpose |
|----------|---------|------|----------|---------|
| `NOTION_API_KEY` | - | string | Yes | Notion API token |
| `NOTION_APPLICATIONS_DB_ID` | - | string | Yes | Applications database ID |
| `NOTION_JOB_TRACKER_DB_ID` | - | string | Yes | Job tracker database ID |
| `NOTION_VERSION` | 2022-06-28 | string | No | Notion API version |
| `NOTION_TIMEOUT` | 30 | int | No | Request timeout (seconds) |
| `NOTION_MAX_RETRIES` | 3 | int | No | Retry count |

#### **E. Scraping Configuration**

| Variable | Default | Type | Purpose |
|----------|---------|------|---------|
| `SCRAPING_DELAY_MIN` | 2 | int | Min delay between requests (seconds) |
| `SCRAPING_DELAY_MAX` | 5 | int | Max delay between requests (seconds) |
| `MAX_CONCURRENT_SCRAPERS` | 5 | int | Concurrent scraper instances |
| `SCRAPING_TIMEOUT` | 30 | int | Scraper timeout (seconds) |
| `SCRAPING_MAX_RETRIES` | 3 | int | Max retries per request |
| `USER_AGENT` | Mozilla/5.0... | string | Browser user agent |
| `LINKEDIN_EMAIL` | - | string | LinkedIn login email |
| `LINKEDIN_PASSWORD` | - | string | LinkedIn password |
| `INDEED_API_KEY` | - | string | Indeed API key |
| `JOOBLE_API_KEY` | - | string | Jooble API key |
| `SERPAPI_API_KEY_1` | - | string | SerpAPI key for Google Jobs |

#### **F. Web Proxies**

| Variable | Default | Type | Purpose |
|----------|---------|------|---------|
| `WEBSHARE_PROXY_2_1` to `WEBSHARE_PROXY_2_10` | - | string | Static Webshare proxies |
| `PROXY_1` to `PROXY_10` | - | string | Alternative proxy format |

#### **G. Playwright Configuration**

| Variable | Default | Type | Purpose |
|----------|---------|------|---------|
| `PLAYWRIGHT_SERVICE_URL` | http://localhost:3000 | string | Browser service URL |
| `PLAYWRIGHT_HEADLESS` | true | bool | Run headless browser |
| `PLAYWRIGHT_TIMEOUT` | 30000 | int | Navigation timeout (ms) |
| `PLAYWRIGHT_VIEWPORT_WIDTH` | 1920 | int | Browser viewport width |
| `PLAYWRIGHT_VIEWPORT_HEIGHT` | 1080 | int | Browser viewport height |
| `SAVE_SCREENSHOTS` | false | bool | Save screenshots on error |

#### **H. Gmail Integration**

| Variable | Default | Type | Purpose |
|----------|---------|------|---------|
| `GMAIL_CLIENT_ID` | - | string | Gmail OAuth client ID |
| `GMAIL_CLIENT_SECRET` | - | string | Gmail OAuth secret |
| `GMAIL_REDIRECT_URI` | http://localhost:8080/auth/callback | string | OAuth callback URL |
| `GMAIL_SCOPES` | gmail.readonly | string | Gmail API scopes |
| `GMAIL_CHECK_INTERVAL` | 300 | int | Check interval (seconds) |

#### **I. System Configuration**

| Variable | Default | Type | Purpose |
|----------|---------|------|---------|
| `TIMEZONE` | Asia/Kolkata | string | Application timezone |
| `DAILY_RUN_TIME` | 09:00 | string | Daily execution time |
| `MAX_APPLICATIONS_PER_DAY` | 50 | int | Rate limit for applications |
| `MIN_JOB_MATCH_SCORE` | 70 | int | Minimum match score (0-100) |
| `HIGH_PRIORITY_DAYS` | 3 | int | High priority threshold |
| `MEDIUM_PRIORITY_DAYS` | 7 | int | Medium priority threshold |
| `LOW_PRIORITY_DAYS` | 14 | int | Low priority threshold |

#### **J. Security & Encryption**

| Variable | Default | Type | Required | Purpose |
|----------|---------|------|----------|---------|
| `ENCRYPTION_KEY` | - | string | Yes | Encryption key for sensitive data |
| `JWT_SECRET` | - | string | Yes | JWT signing secret |
| `JWT_EXPIRY_HOURS` | 24 | int | No | Token expiration hours |
| `SESSION_TIMEOUT` | 3600 | int | No | Session timeout (seconds) |
| `EXTENSION_API_SECRET` | - | string | No | Extension API secret |
| `MCP_API_KEY` | - | string | No | MCP server API key |

#### **K. MCP Server Configuration**

| Variable | Default | Type | Purpose |
|----------|---------|------|---------|
| `MCP_SERVER_URL` | http://localhost:3001 | string | MCP server address |
| `MCP_CLIENT_TIMEOUT` | 30 | int | Client timeout (seconds) |
| `MCP_MAX_RETRIES` | 3 | int | Max retry attempts |
| `MCP_CONTEXT_WINDOW_SIZE` | 32000 | int | Context window tokens |
| `MCP_DEV_MODE` | false | bool | Development mode flag |

#### **L. N8N Configuration**

| Variable | Default | Type | Purpose |
|----------|---------|------|---------|
| `N8N_PORT` | 5678 | int | N8N service port |
| `N8N_HOST` | localhost | string | N8N host |
| `WEBHOOK_URL` | http://localhost:5678 | string | Webhook URL |
| `GENERIC_TIMEZONE` | Asia/Kolkata | string | Timezone |
| `N8N_BASIC_AUTH_ACTIVE` | true | bool | Enable basic auth |
| `N8N_BASIC_AUTH_USER` | admin | string | Admin username |
| `N8N_BASIC_AUTH_PASSWORD` | - | string | Admin password |
| `N8N_ENCRYPTION_KEY` | - | string | Data encryption key |
| `N8N_USER_MANAGEMENT_ENABLED` | false | bool | User management |

#### **M. RAG Server Configuration**

| Variable | Default | Type | Purpose |
|----------|---------|------|---------|
| `RAG_SERVER_HOST` | localhost | string | RAG server host |
| `RAG_SERVER_PORT` | 8090 | int | RAG server port |
| `RAG_SERVER_WORKERS` | 1 | int | Worker processes |
| `RAG_SERVER_RELOAD` | false | bool | Hot reload |
| `RAG_BASE_URL` | http://localhost:8090 | string | RAG base URL |
| `RAG_API_KEY` | master-bcb8c7fe4d2e1a0526130f5f2f954bfd | string | Master API key |
| `SESSION_TIMEOUT_MINUTES` | 30 | int | Session timeout |
| `SESSION_MAX_HISTORY` | 50 | int | Max history items |

#### **N. Rate Limiting**

| Variable | Default | Type | Purpose |
|----------|---------|------|---------|
| `RATE_LIMIT_ENABLED` | true | bool | Enable rate limiting |
| `RATE_LIMIT_REQUESTS` | 1000 | int | Requests per window |
| `RATE_LIMIT_WINDOW` | 60 | int | Window size (seconds) |

#### **O. Logging & Debugging**

| Variable | Default | Type | Purpose |
|----------|---------|------|---------|
| `LOG_LEVEL` | INFO | string | Logging level |
| `LOG_FILE_PATH` | ./logs/automation.log | string | Log file path |
| `LOG_MAX_FILE_SIZE` | 10485760 | int | Max log file size (bytes) |
| `LOG_BACKUP_COUNT` | 5 | int | Backup log count |
| `VERBOSE_LOGGING` | false | bool | Verbose output |
| `DEBUG_MODE` | false | bool | Debug mode |
| `TEST_MODE` | false | bool | Test mode |

#### **P. Optional Services**

| Variable | Default | Type | Purpose |
|----------|---------|------|---------|
| `SLACK_WEBHOOK_URL` | - | string | Slack notifications |
| `DISCORD_WEBHOOK_URL` | - | string | Discord notifications |
| `GRAFANA_USER` | admin | string | Grafana admin user |
| `GRAFANA_PASSWORD` | - | string | Grafana admin password |
| `MINIO_ROOT_USER` | - | string | MinIO admin user |
| `MINIO_ROOT_PASSWORD` | - | string | MinIO admin password |
| `OVERLEAF_API_KEY` | - | string | Overleaf resume API |
| `OVERLEAF_PROJECT_ID` | - | string | Overleaf project ID |
| `MOCK_APPLICATIONS` | false | bool | Use mock data |

---

## 7. External API Dependencies & Credentials

### AI & LLM Providers

#### **1. OpenAI**
- **Authentication:** API Key (`OPENAI_API_KEY`)
- **Models Used:**
  - `gpt-4-turbo-preview` - Main LLM for job analysis
  - `text-embedding-3-small` - Embeddings for job/resume matching
- **Endpoints:**
  - Chat completions: `POST /v1/chat/completions`
  - Embeddings: `POST /v1/embeddings`
- **Rate Limits:** Token-based (depends on tier)
- **Usage:** Job analysis, resume optimization, matching scoring
- **Cost:** Pay-as-you-go (input/output tokens)

#### **2. Perplexity AI**
- **Authentication:** API Key (`PERPLEXITY_API_KEY`)
- **Models Used:**
  - `llama-3.1-sonar-huge-128k-online` - Company research
- **Endpoints:**
  - Chat completions: `POST /v1/chat/completions`
- **Rate Limits:** Token-based
- **Usage:** Real-time company information, job market research
- **Cost:** Token-based pricing

#### **3. Google Gemini**
- **Authentication:** API Key (`GEMINI_API_KEY_RAG`)
- **Models Used:**
  - `text-embedding-004` - Resume embeddings (768 dims)
- **Endpoints:**
  - Embed content: `POST /v1beta/models/text-embedding-004:embedContent`
- **Rate Limits:** Token-based with MRL support
- **Usage:** Resume-to-job matching via embeddings
- **Cost:** Token-based (embeddings cheaper than LLM)

#### **4. OpenRouter (Fallback)**
- **Authentication:** API Key (`OPENROUTER_API_KEY`)
- **Purpose:** Fallback provider for LLM calls
- **Base URL:** `https://api.openrouter.ai`

#### **5. Grok AI (Optional)**
- **Authentication:** API Key (`GROK_API_KEY`)
- **Purpose:** Alternative LLM provider

### Job Scraping APIs

#### **1. JobSpy (Free, No Auth)**
- **Purpose:** Multi-platform job aggregation
- **Platforms Supported:** 50+ job sites
- **Rate Limits:** Reasonable (no auth required)

#### **2. Jooble API**
- **Authentication:** API Key (`JOOBLE_API_KEY`)
- **Endpoint:** RESTful API
- **Rate Limits:** Per tier
- **Returns:** Job listings with details

#### **3. SerpAPI (Google Jobs)**
- **Authentication:** API Key (`SERPAPI_API_KEY_1`)
- **Endpoint:** `https://serpapi.com/search`
- **Rate Limits:** Per tier
- **Returns:** Google Jobs results

#### **4. Indeed API**
- **Authentication:** API Key (`INDEED_API_KEY`)
- **Endpoint:** RESTful API
- **Rate Limits:** Per tier

#### **5. Direct Scraping (LinkedIn, Glassdoor, etc.)**
- **Authentication:** Email/Password credentials
  - `LINKEDIN_EMAIL`, `LINKEDIN_PASSWORD`
- **Method:** Browser automation (Playwright)
- **Rate Limits:** Anti-bot detection (delay + proxies)

### Data & Integration Services

#### **1. Notion API**
- **Authentication:** API Key (`NOTION_API_KEY`)
- **Endpoint:** `https://api.notion.com/v1/`
- **Version:** `2022-06-28` (configurable)
- **Resources Used:**
  - Databases: Read/Write
  - Pages: Create/Update/Query
  - Properties: Edit
- **Rate Limits:** 3 req/sec per integration
- **Databases:**
  - Applications DB (`NOTION_APPLICATIONS_DB_ID`)
  - Job Tracker DB (`NOTION_JOB_TRACKER_DB_ID`)

#### **2. Gmail API**
- **Authentication:** OAuth 2.0
  - Client ID: `GMAIL_CLIENT_ID`
  - Client Secret: `GMAIL_CLIENT_SECRET`
  - Redirect URI: `GMAIL_REDIRECT_URI`
- **Scopes:** `gmail.readonly` (configurable)
- **Endpoint:** `https://www.googleapis.com/gmail/v1/`
- **Rate Limits:** 250 requests/day per user
- **Usage:** Email-based job notifications, application confirmations

#### **3. Google OAuth**
- **Scopes Required:**
  - `gmail.readonly` - Read emails
  - Additional scopes in `GMAIL_SCOPES`
- **Provider:** Google Cloud Console

### Proxy Services

#### **1. Webshare Static Proxies**
- **Configuration:** 10 static proxies
- **Environment Variables:** `WEBSHARE_PROXY_2_1` through `WEBSHARE_PROXY_2_10`
- **Format:** `http://username:password@ip:port`
- **Purpose:** Rotate through proxies to avoid IP bans during scraping
- **Fallback:** Direct connection if proxies fail

### External Services (Optional)

#### **1. Slack Notifications**
- **Webhook URL:** `SLACK_WEBHOOK_URL`
- **Purpose:** Job alerts, application status updates
- **Endpoint:** Custom webhook URL

#### **2. Discord Notifications**
- **Webhook URL:** `DISCORD_WEBHOOK_URL`
- **Purpose:** Job alerts, application status
- **Endpoint:** Custom webhook URL

#### **3. Overleaf Resume Generation**
- **API Key:** `OVERLEAF_API_KEY`
- **Project ID:** `OVERLEAF_PROJECT_ID`
- **Endpoint:** `https://api.overleaf.com/docs/compile`
- **Purpose:** Dynamic resume generation and formatting

### Monitoring & Analytics (Optional)

#### **1. Prometheus Metrics**
- **Endpoint:** `http://localhost:9090` (if enabled)
- **Metrics Collected:** Request latency, error rates, job processing stats

#### **2. Grafana Dashboards**
- **Credentials:** `GRAFANA_USER`, `GRAFANA_PASSWORD`
- **Port:** 3000 (if enabled)
- **Dashboards:** Custom job automation metrics

#### **3. MinIO Object Storage (Optional)**
- **Credentials:** `MINIO_ROOT_USER`, `MINIO_ROOT_PASSWORD`
- **Purpose:** Resume and screenshot storage

### API Summary Table

| API | Auth Type | Key Variable | Endpoint | Rate Limit | Cost |
|-----|-----------|--------------|----------|-----------|------|
| OpenAI | API Key | OPENAI_API_KEY | api.openai.com | Tier-dependent | Tokens |
| Perplexity | API Key | PERPLEXITY_API_KEY | api.perplexity.ai | Tier-dependent | Tokens |
| Gemini | API Key | GEMINI_API_KEY_RAG | generativelanguage.googleapis.com | Tier-dependent | Tokens |
| Notion | API Key | NOTION_API_KEY | api.notion.com | 3 req/sec | Free |
| Gmail | OAuth 2.0 | GMAIL_CLIENT_ID/SECRET | googleapis.com | 250/day | Free |
| JobSpy | None | - | -  | Reasonable | Free |
| Jooble | API Key | JOOBLE_API_KEY | jooble.com | Tier-dependent | Free/Paid |
| SerpAPI | API Key | SERPAPI_API_KEY_1 | serpapi.com | Tier-dependent | Paid |
| Indeed | API Key | INDEED_API_KEY | indeed.com | Tier-dependent | Paid |
| Webshare | Credentials | WEBSHARE_PROXY_* | proxy service | Per proxy | Paid |
| Slack | Webhook | SLACK_WEBHOOK_URL | hooks.slack.com | Custom | Free |
| Discord | Webhook | DISCORD_WEBHOOK_URL | discordapp.com | Custom | Free |
| Overleaf | API Key | OVERLEAF_API_KEY | api.overleaf.com | Custom | Paid |

### Credential Management Best Practices

1. **Never commit .env file** - Add to .gitignore
2. **Rotate sensitive keys** regularly
3. **Use secure vaults** for production credentials
4. **Limit API key scopes** to minimum required
5. **Monitor API usage** via provider dashboards
6. **Set up alerts** for unusual activity
7. **Use environment-specific keys** (dev/prod/staging)
8. **Backup credentials securely** (encrypted storage)

---

## Service Health Checks

### Health Check Endpoints

```bash
# MCP Server
curl http://localhost:3001/health

# RAG Server
curl http://localhost:8090/health

# PostgreSQL
pg_isready -U n8n_user -d n8n_ai_job -h localhost

# Redis
redis-cli ping

# N8N
curl http://localhost:5678/api/health
```

### Startup Order (Recommended)

1. PostgreSQL (waits for health check)
2. Redis
3. N8N (depends on PostgreSQL)
4. MCP Server
5. RAG Server
6. Master automation engine

---

## Testing & Development

### Environment for Development

```bash
# Set test mode
export TEST_MODE=true
export DEBUG_MODE=true
export LOG_LEVEL=DEBUG
export MCP_DEV_MODE=true

# Use mock data
export MOCK_APPLICATIONS=true

# Development AI keys (if available)
export OPENAI_API_KEY="test-key"
export PERPLEXITY_API_KEY="test-key"
```

### Common Configuration Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| "OPENAI_API_KEY is required" | Missing API key | Add to .env and reload |
| PostgreSQL connection failed | DB not running | `docker-compose up -d postgres` |
| Redis connection refused | Redis not running | Start Redis service |
| MCP port 3001 already in use | Another service running | Change MCP_SERVER_URL or kill process |
| Notion 401 Unauthorized | Invalid API key | Verify NOTION_API_KEY in .env |
| Gmail OAuth redirect fails | Wrong redirect URI | Update GMAIL_REDIRECT_URI |

---

## Production Deployment Checklist

- [ ] Set `DEBUG_MODE=false`
- [ ] Set `MCP_DEV_MODE=false`
- [ ] Generate strong `ENCRYPTION_KEY`
- [ ] Generate strong `JWT_SECRET`
- [ ] Configure PostgreSQL with persistent volume
- [ ] Enable Redis persistence (dump.rdb)
- [ ] Set up automated backups
- [ ] Configure monitoring (Prometheus/Grafana)
- [ ] Set up log aggregation
- [ ] Configure rate limiting
- [ ] Test circuit breaker failovers
- [ ] Set appropriate timeouts
- [ ] Configure all external API keys
- [ ] Test Notion integration
- [ ] Verify proxy configuration
- [ ] Set up CI/CD pipeline
- [ ] Document runbook for operations
- [ ] Plan disaster recovery

---

**Generated:** January 20, 2026  
**Last Updated:** January 20, 2026  
**Maintained By:** AI Job Automation Team
