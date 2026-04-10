-- =============================================================================
-- AI Job Application Agent — Postgres Schema
-- Single source of truth. Last synced: 2026-04-10
-- Run:  psql $LOCAL_POSTGRES_URL -f database/schema.sql
-- Reset: DROP SCHEMA public CASCADE; CREATE SCHEMA public; then re-run.
-- =============================================================================

-- =============================================================================
-- EXTENSIONS
-- =============================================================================
CREATE EXTENSION IF NOT EXISTS pgcrypto;   -- gen_random_uuid()
CREATE EXTENSION IF NOT EXISTS vector;     -- pgvector: VECTOR(1024) job embeddings

-- =============================================================================
-- TABLE 1: users
-- One row per candidate profile. Holds all per-user config as JSONB blobs
-- so we never need schema migrations for preference changes.
-- =============================================================================
CREATE TABLE IF NOT EXISTS users (
    id                UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    email             TEXT        UNIQUE NOT NULL,
    name              TEXT        NOT NULL,
    preferences_json  JSONB       NOT NULL DEFAULT '{}',
    user_settings     JSONB       NOT NULL DEFAULT '{}',   -- apply limits, blacklists, etc.
    platform_settings JSONB       NOT NULL DEFAULT '{}'    -- per-platform credentials + caps
);

-- =============================================================================
-- TABLE 2: resumes
-- Each row is one resume variant (slug ID = TEXT, not UUID).
-- id is a human-readable slug: resume_generic, resume_ml_data_engineer, etc.
-- Matches ChromaDB metadata.resume_id for RAG lookups.
-- =============================================================================
CREATE TABLE IF NOT EXISTS resumes (
    id           TEXT        PRIMARY KEY,                  -- slug, e.g. "resume_data_science"
    user_id      UUID        NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    label        TEXT        NOT NULL,                     -- display name
    storage_path TEXT        NOT NULL,                     -- relative path to PDF
    is_active    BOOLEAN     NOT NULL DEFAULT TRUE,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- =============================================================================
-- TABLE 3: run_sessions
-- One row per scheduled pipeline run (3x/week).
-- Counters are incremented by triggers on jobs and applications inserts.
-- =============================================================================
CREATE TABLE IF NOT EXISTS run_sessions (
    id                  UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    run_date            DATE        NOT NULL DEFAULT CURRENT_DATE,
    run_index_in_week   INTEGER     NOT NULL,              -- 1, 2, or 3
    jobs_discovered     INTEGER     NOT NULL DEFAULT 0,
    jobs_auto_applied   INTEGER     NOT NULL DEFAULT 0,
    jobs_queued         INTEGER     NOT NULL DEFAULT 0,
    started_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    closed_at           TIMESTAMPTZ                        -- set by trg_close_run_session
);

-- =============================================================================
-- TABLE 4: run_batches
-- Tracks each discrete pipeline execution batch (MasterAgent level).
-- run_batch_id is the UUID string passed through all agents for correlation.
-- =============================================================================
CREATE TABLE IF NOT EXISTS run_batches (
    id              UUID           PRIMARY KEY DEFAULT gen_random_uuid(),
    run_batch_id    TEXT           UNIQUE NOT NULL,        -- correlation ID across all agents
    mode            TEXT           NOT NULL DEFAULT 'full',
    status          TEXT           NOT NULL DEFAULT 'running'
                                   CHECK (status IN ('running','completed','failed','aborted')),
    dry_run         BOOLEAN        NOT NULL DEFAULT FALSE,
    jobs_found      INTEGER        NOT NULL DEFAULT 0,
    jobs_applied    INTEGER        NOT NULL DEFAULT 0,
    jobs_queued     INTEGER        NOT NULL DEFAULT 0,
    total_cost_usd  NUMERIC(10,6)  NOT NULL DEFAULT 0.0,
    error_message   TEXT,
    started_at      TIMESTAMPTZ    NOT NULL DEFAULT NOW(),
    completed_at    TIMESTAMPTZ,
    updated_at      TIMESTAMPTZ    NOT NULL DEFAULT NOW()
);

-- =============================================================================
-- TABLE 5: jobs
-- Raw scraped job postings. One row per unique URL.
-- embedding_vector holds the 1024-dim NVIDIA NIM embedding for similarity.
-- run_batch_id FK → run_sessions (the session that discovered this job).
-- =============================================================================
CREATE TABLE IF NOT EXISTS jobs (
    id               UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    run_batch_id     UUID        NOT NULL REFERENCES run_sessions(id) ON DELETE CASCADE,
    source_platform  TEXT        NOT NULL,                 -- e.g. "linkedin", "remotive"
    title            TEXT        NOT NULL,
    company          TEXT        NOT NULL,
    location         TEXT,
    url              TEXT        UNIQUE NOT NULL,
    posted_at        TIMESTAMPTZ,
    embedding_vector VECTOR(1024),                         -- pgvector job embedding
    created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- =============================================================================
-- TABLE 6: job_scores
-- One row per job. Written by AnalyserAgent after LLM + RAG scoring.
-- fit_score    = LLM-derived 0.0–1.0 match score
-- rag_score    = raw ChromaDB cosine similarity from best matching chunk
-- similarity_score = weighted aggregation across top-k chunks (60/40)
-- matched_resume_id = slug of the best-matching resume variant
-- =============================================================================
CREATE TABLE IF NOT EXISTS job_scores (
    id                UUID             PRIMARY KEY DEFAULT gen_random_uuid(),
    job_post_id       UUID             NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    resume_id         TEXT             REFERENCES resumes(id) ON DELETE SET NULL,
    fit_score         DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    eligibility_pass  BOOLEAN          NOT NULL DEFAULT FALSE,
    similarity_score  DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    matched_resume_id TEXT,                                -- slug, may differ from resume_id
    matched_label     TEXT,                                -- human label from ChromaDB metadata
    rag_score         DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    reasons_json      JSONB            NOT NULL DEFAULT '{}', -- LLM reasoning payload
    scored_at         TIMESTAMPTZ      NOT NULL DEFAULT NOW(),
    CONSTRAINT job_scores_job_post_id_unique UNIQUE (job_post_id)
);

-- =============================================================================
-- TABLE 7: applications
-- One row per application attempt (auto or manual queue entry).
-- status flow: manual_queued → applied | failed
-- =============================================================================
CREATE TABLE IF NOT EXISTS applications (
    id           UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    job_post_id  UUID        NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    resume_id    TEXT        REFERENCES resumes(id) ON DELETE SET NULL,
    user_id      UUID        NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    mode         TEXT        NOT NULL CHECK (mode IN ('auto', 'manual')),
    status       TEXT        NOT NULL CHECK (status IN ('applied', 'failed', 'manual_queued')),
    platform     TEXT        NOT NULL,
    submitted_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    error_code   TEXT
);

-- =============================================================================
-- TABLE 8: queued_jobs
-- Manual review queue. Created when analyser routes a job to manual.
-- Priority 1 (highest) → 10 (lowest). Default 5.
-- =============================================================================
CREATE TABLE IF NOT EXISTS queued_jobs (
    id             UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    application_id UUID        NOT NULL REFERENCES applications(id) ON DELETE CASCADE,
    job_post_id    UUID        NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    priority       INTEGER     NOT NULL DEFAULT 5,
    notes          TEXT,
    queued_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- =============================================================================
-- TABLE 9: audit_logs
-- Append-only structured event log for all agent actions and errors.
-- Used by MasterAgent for health checks and budget abort decisions.
-- =============================================================================
CREATE TABLE IF NOT EXISTS audit_logs (
    id             UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    run_batch_id   UUID        NOT NULL REFERENCES run_sessions(id) ON DELETE CASCADE,
    application_id UUID        REFERENCES applications(id) ON DELETE SET NULL,
    job_post_id    UUID        REFERENCES jobs(id) ON DELETE SET NULL,
    level          TEXT        NOT NULL CHECK (level IN ('INFO','WARNING','ERROR','CRITICAL')),
    event_type     TEXT        NOT NULL,
    message        TEXT        NOT NULL,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- =============================================================================
-- TABLE 10: system_config
-- Key-value store for runtime config that changes between runs.
-- e.g. active query sets, budget overrides, feature flags.
-- =============================================================================
CREATE TABLE IF NOT EXISTS system_config (
    id           UUID                PRIMARY KEY DEFAULT gen_random_uuid(),
    config_key   VARCHAR(255)        UNIQUE NOT NULL,
    config_value JSONB               NOT NULL,
    updated_at   TIMESTAMPTZ         NOT NULL DEFAULT NOW()
);

-- =============================================================================
-- TABLE 11: query_cache
-- Stores LLM-generated search query sets between runs.
-- Tracks yield baseline vs improved to decide when to regenerate.
-- =============================================================================
CREATE TABLE IF NOT EXISTS query_cache (
    id                        UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    queries                   JSONB       NOT NULL,
    generated_by_run_id       UUID,
    yield_baseline            INTEGER,
    yield_improved            INTEGER,
    consecutive_low_yield_count INTEGER   NOT NULL DEFAULT 0,
    created_at                TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at                TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- =============================================================================
-- TABLE 12: schema_versions
-- Migration version tracking. Insert a row on every schema change.
-- =============================================================================
CREATE TABLE IF NOT EXISTS schema_versions (
    version    VARCHAR(16) PRIMARY KEY,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- =============================================================================
-- INDEXES
-- =============================================================================

-- jobs
CREATE INDEX IF NOT EXISTS idx_jobs_run_batch_id    ON jobs(run_batch_id);
CREATE INDEX IF NOT EXISTS idx_jobs_source_platform ON jobs(source_platform);
CREATE INDEX IF NOT EXISTS idx_jobs_url_hash        ON jobs USING hash(url);
CREATE INDEX IF NOT EXISTS idx_jobs_run_platform    ON jobs(run_batch_id, source_platform);
CREATE INDEX IF NOT EXISTS idx_jobs_created_at      ON jobs(created_at DESC);

-- job_scores
CREATE INDEX IF NOT EXISTS idx_job_scores_job_post_id ON job_scores(job_post_id);
CREATE INDEX IF NOT EXISTS idx_job_scores_fit_score   ON job_scores(fit_score DESC);
CREATE INDEX IF NOT EXISTS idx_job_scores_route       ON job_scores(eligibility_pass, fit_score DESC);

-- applications
CREATE INDEX IF NOT EXISTS idx_applications_job_post_id  ON applications(job_post_id);
CREATE INDEX IF NOT EXISTS idx_applications_user_id      ON applications(user_id);
CREATE INDEX IF NOT EXISTS idx_applications_status       ON applications(status);
CREATE INDEX IF NOT EXISTS idx_applications_run_platform ON applications(platform, status);

-- queued_jobs
CREATE INDEX IF NOT EXISTS idx_queued_jobs_priority    ON queued_jobs(priority DESC);
CREATE INDEX IF NOT EXISTS idx_queued_jobs_job_post_id ON queued_jobs(job_post_id);

-- audit_logs
CREATE INDEX IF NOT EXISTS idx_audit_logs_run_batch_id ON audit_logs(run_batch_id);
CREATE INDEX IF NOT EXISTS idx_audit_logs_level        ON audit_logs(level);
CREATE INDEX IF NOT EXISTS idx_audit_logs_event_type   ON audit_logs(event_type);

-- run_sessions
CREATE INDEX IF NOT EXISTS idx_run_sessions_date ON run_sessions(run_date DESC);

-- run_batches
CREATE INDEX IF NOT EXISTS idx_run_batches_run_batch_id ON run_batches(run_batch_id);
CREATE INDEX IF NOT EXISTS idx_run_batches_status       ON run_batches(status);
CREATE INDEX IF NOT EXISTS idx_run_batches_started_at   ON run_batches(started_at DESC);

-- =============================================================================
-- FUNCTION: update_run_session_counts()
-- Fired after every INSERT/UPDATE on applications.
-- Increments the parent run_session auto/queued counters atomically.
-- =============================================================================
CREATE OR REPLACE FUNCTION update_run_session_counts()
RETURNS TRIGGER AS $$
BEGIN
    UPDATE run_sessions
    SET
        jobs_auto_applied = jobs_auto_applied +
            CASE WHEN NEW.mode = 'auto'   AND NEW.status = 'applied'       THEN 1 ELSE 0 END,
        jobs_queued       = jobs_queued +
            CASE WHEN NEW.status = 'manual_queued'                          THEN 1 ELSE 0 END
    WHERE id = (
        SELECT run_batch_id FROM jobs WHERE id = NEW.job_post_id LIMIT 1
    );
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_update_run_session_counts ON applications;
CREATE TRIGGER trg_update_run_session_counts
    AFTER INSERT OR UPDATE ON applications
    FOR EACH ROW
    EXECUTE FUNCTION update_run_session_counts();

-- =============================================================================
-- FUNCTION: update_run_session_discovered()
-- Fired after every INSERT on jobs.
-- Increments jobs_discovered on the parent run_session.
-- =============================================================================
CREATE OR REPLACE FUNCTION update_run_session_discovered()
RETURNS TRIGGER AS $$
BEGIN
    UPDATE run_sessions
    SET jobs_discovered = jobs_discovered + 1
    WHERE id = NEW.run_batch_id;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_update_run_session_discovered ON jobs;
CREATE TRIGGER trg_update_run_session_discovered
    AFTER INSERT ON jobs
    FOR EACH ROW
    EXECUTE FUNCTION update_run_session_discovered();

-- =============================================================================
-- FUNCTION: close_run_session()
-- Fired before UPDATE on run_sessions.
-- Auto-sets closed_at when the session records its first real output.
-- =============================================================================
CREATE OR REPLACE FUNCTION close_run_session()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.jobs_auto_applied > 0 OR NEW.jobs_queued > 0 THEN
        NEW.closed_at = NOW();
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_close_run_session ON run_sessions;
CREATE TRIGGER trg_close_run_session
    BEFORE UPDATE ON run_sessions
    FOR EACH ROW
    WHEN (OLD.closed_at IS NULL AND (NEW.jobs_auto_applied > 0 OR NEW.jobs_queued > 0))
    EXECUTE FUNCTION close_run_session();

-- =============================================================================
-- SEED: schema version record
-- =============================================================================
INSERT INTO schema_versions (version) VALUES ('001')
    ON CONFLICT (version) DO NOTHING;

-- =============================================================================
-- END OF SCHEMA
-- =============================================================================
