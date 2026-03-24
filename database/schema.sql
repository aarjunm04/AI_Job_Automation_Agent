CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE EXTENSION IF NOT EXISTS vector;

-- AI Job Application Agent — Postgres Schema
-- Run: psql $LOCAL_POSTGRES_URL -f database/schema.sql

-- =============================================================================
-- TABLE 1: users
-- =============================================================================
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email TEXT UNIQUE NOT NULL,
    name TEXT NOT NULL,
    preferences_json JSONB DEFAULT '{}',
    user_settings JSONB NOT NULL DEFAULT '{}'::jsonb,
    platform_settings JSONB NOT NULL DEFAULT '{}'::jsonb
);

-- =============================================================================
-- TABLE 2: resumes
-- =============================================================================
CREATE TABLE IF NOT EXISTS resumes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    label TEXT NOT NULL,
    storage_path TEXT NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- =============================================================================
-- TABLE 3: run_sessions
-- =============================================================================
CREATE TABLE IF NOT EXISTS run_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_date DATE NOT NULL DEFAULT CURRENT_DATE,
    run_index_in_week INTEGER NOT NULL,
    jobs_discovered INTEGER DEFAULT 0,
    jobs_auto_applied INTEGER DEFAULT 0,
    jobs_queued INTEGER DEFAULT 0,
    started_at TIMESTAMPTZ DEFAULT NOW(),
    closed_at TIMESTAMPTZ
);

-- =============================================================================
-- TABLE 4: config_limits
-- =============================================================================
CREATE TABLE IF NOT EXISTS config_limits (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    platform TEXT UNIQUE NOT NULL,
    max_per_run INTEGER NOT NULL DEFAULT 50,
    max_per_day INTEGER NOT NULL DEFAULT 100,
    max_concurrent_sessions INTEGER NOT NULL DEFAULT 1
);

-- =============================================================================
-- TABLE 5: jobs
-- =============================================================================
CREATE TABLE IF NOT EXISTS jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_batch_id UUID NOT NULL REFERENCES run_sessions(id) ON DELETE CASCADE,
    source_platform TEXT NOT NULL,
    title TEXT NOT NULL,
    company TEXT NOT NULL,
    location TEXT,
    url TEXT UNIQUE NOT NULL,
    posted_at TIMESTAMPTZ,
    embedding_vector VECTOR(1024),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- =============================================================================
-- TABLE 6: job_scores
-- =============================================================================
CREATE TABLE IF NOT EXISTS job_scores (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_post_id UUID NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    resume_id UUID REFERENCES resumes(id) ON DELETE SET NULL,
    fit_score FLOAT NOT NULL DEFAULT 0.0,
    eligibility_pass BOOLEAN DEFAULT FALSE,
    reasons_json JSONB DEFAULT '{}',
    scored_at TIMESTAMPTZ DEFAULT NOW()
);

-- =============================================================================
-- TABLE 7: applications
-- =============================================================================
CREATE TABLE IF NOT EXISTS applications (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_post_id UUID NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    resume_id UUID REFERENCES resumes(id) ON DELETE SET NULL,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    mode TEXT NOT NULL CHECK (mode IN ('auto', 'manual')),
    status TEXT NOT NULL CHECK (status IN ('applied', 'failed', 'manual_queued')),
    platform TEXT NOT NULL,
    submitted_at TIMESTAMPTZ DEFAULT NOW(),
    error_code TEXT
);

-- =============================================================================
-- TABLE 8: queued_jobs
-- =============================================================================
CREATE TABLE IF NOT EXISTS queued_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    application_id UUID NOT NULL REFERENCES applications(id) ON DELETE CASCADE,
    job_post_id UUID NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    priority INTEGER NOT NULL DEFAULT 5,
    notes TEXT,
    queued_at TIMESTAMPTZ DEFAULT NOW()
);

-- =============================================================================
-- TABLE 9: audit_logs
-- =============================================================================
CREATE TABLE IF NOT EXISTS audit_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_batch_id UUID NOT NULL REFERENCES run_sessions(id) ON DELETE CASCADE,
    application_id UUID REFERENCES applications(id) ON DELETE SET NULL,
    job_post_id UUID REFERENCES jobs(id) ON DELETE SET NULL,
    level TEXT NOT NULL CHECK (level IN ('INFO', 'WARNING', 'ERROR', 'CRITICAL')),
    event_type TEXT NOT NULL,
    message TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- =============================================================================
-- TABLE 10: system_config
-- =============================================================================
CREATE TABLE IF NOT EXISTS system_config (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    config_key VARCHAR(255) UNIQUE NOT NULL,
    config_value JSONB NOT NULL,
    updated_at TIMESTAMPTZ DEFAULT now()
);

-- =============================================================================
-- TABLE 11: query_cache
-- =============================================================================
CREATE TABLE IF NOT EXISTS query_cache (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    queries JSONB NOT NULL,
    generated_by_run_id UUID,
    yield_baseline INT,
    yield_improved INT,
    consecutive_low_yield_count INT DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

-- =============================================================================
-- INDEXES
-- =============================================================================
CREATE INDEX IF NOT EXISTS idx_jobs_run_batch_id ON jobs(run_batch_id);
CREATE INDEX IF NOT EXISTS idx_jobs_source_platform ON jobs(source_platform);
CREATE INDEX IF NOT EXISTS idx_jobs_url_hash ON jobs USING hash(url);
CREATE INDEX IF NOT EXISTS idx_jobs_run_platform ON jobs(run_batch_id, source_platform);
CREATE INDEX IF NOT EXISTS idx_jobs_created_at ON jobs(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_job_scores_job_post_id ON job_scores(job_post_id);
CREATE INDEX IF NOT EXISTS idx_job_scores_fit_score ON job_scores(fit_score DESC);
CREATE INDEX IF NOT EXISTS idx_job_scores_route ON job_scores(eligibility_pass, fit_score DESC);
CREATE INDEX IF NOT EXISTS idx_applications_job_post_id ON applications(job_post_id);
CREATE INDEX IF NOT EXISTS idx_applications_user_id ON applications(user_id);
CREATE INDEX IF NOT EXISTS idx_applications_status ON applications(status);
CREATE INDEX IF NOT EXISTS idx_applications_run_platform ON applications(platform, status);
CREATE INDEX IF NOT EXISTS idx_queued_jobs_priority ON queued_jobs(priority DESC);
CREATE INDEX IF NOT EXISTS idx_queued_jobs_job_post_id ON queued_jobs(job_post_id);
CREATE INDEX IF NOT EXISTS idx_audit_logs_run_batch_id ON audit_logs(run_batch_id);
CREATE INDEX IF NOT EXISTS idx_audit_logs_level ON audit_logs(level);
CREATE INDEX IF NOT EXISTS idx_audit_logs_event_type ON audit_logs(event_type);
CREATE INDEX IF NOT EXISTS idx_run_sessions_date ON run_sessions(run_date DESC);

-- =============================================================================
-- TRIGGER: auto-update run_sessions counts from applications
-- =============================================================================
CREATE OR REPLACE FUNCTION update_run_session_counts()
RETURNS TRIGGER AS $$
BEGIN
    UPDATE run_sessions
    SET jobs_auto_applied = (
            SELECT COUNT(*) FROM applications a
            JOIN jobs j ON j.id = a.job_post_id
            WHERE j.run_batch_id = (
                SELECT run_batch_id FROM jobs WHERE id = NEW.job_post_id LIMIT 1
            )
            AND a.status = 'applied'
        ),
        jobs_queued = (
            SELECT COUNT(*) FROM applications a
            JOIN jobs j ON j.id = a.job_post_id
            WHERE j.run_batch_id = (
                SELECT run_batch_id FROM jobs WHERE id = NEW.job_post_id LIMIT 1
            )
            AND a.status = 'manual_queued'
        )
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
-- TRIGGER: auto-update run_sessions.jobs_discovered from jobs inserts
-- =============================================================================
CREATE OR REPLACE FUNCTION update_run_session_discovered()
RETURNS TRIGGER AS $$
BEGIN
    UPDATE run_sessions
    SET jobs_discovered = (
        SELECT COUNT(*) FROM jobs WHERE run_batch_id = NEW.run_batch_id
    )
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
-- TABLE 12: schema_versions
-- =============================================================================
CREATE TABLE IF NOT EXISTS schema_versions (
    version     VARCHAR(16) PRIMARY KEY,
    applied_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Migrations for single-source-of-truth JSONB settings
ALTER TABLE users ADD COLUMN IF NOT EXISTS user_settings JSONB NOT NULL DEFAULT '{}'::jsonb;
ALTER TABLE users ADD COLUMN IF NOT EXISTS platform_settings JSONB NOT NULL DEFAULT '{}'::jsonb;
-- config_limits dropped in favour of users.platform_settings JSONB
DROP TABLE IF EXISTS config_limits;

-- Add closed_at auto-set for run_sessions
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

-- Record initial schema version
INSERT INTO schema_versions (version) VALUES ('001')
ON CONFLICT (version) DO NOTHING;

-- run_batches: tracks each pipeline execution batch
CREATE TABLE IF NOT EXISTS run_batches (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_batch_id    TEXT UNIQUE NOT NULL,
    mode            TEXT NOT NULL DEFAULT 'full',
    status          TEXT NOT NULL DEFAULT 'running',
    dry_run         BOOLEAN NOT NULL DEFAULT FALSE,
    jobs_found      INTEGER DEFAULT 0,
    jobs_applied    INTEGER DEFAULT 0,
    jobs_queued     INTEGER DEFAULT 0,
    total_cost_usd  NUMERIC(10,6) DEFAULT 0.0,
    error_message   TEXT,
    started_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at    TIMESTAMPTZ,
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_run_batches_run_batch_id ON run_batches(run_batch_id);
CREATE INDEX IF NOT EXISTS idx_run_batches_status ON run_batches(status);
CREATE INDEX IF NOT EXISTS idx_run_batches_started_at ON run_batches(started_at DESC);
