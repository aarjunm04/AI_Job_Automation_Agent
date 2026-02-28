-- AI Job Application Agent — Postgres Schema
-- Run: psql $LOCAL_POSTGRES_URL -f database/schema.sql

-- Enable pgvector extension for embedding vectors
CREATE EXTENSION IF NOT EXISTS vector;

-- =============================================================================
-- TABLE 1: users
-- =============================================================================
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email TEXT UNIQUE NOT NULL,
    name TEXT NOT NULL,
    preferences_json JSONB DEFAULT '{}'
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
-- TABLE 3: run_batches
-- =============================================================================
CREATE TABLE IF NOT EXISTS run_batches (
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
-- TABLE 5: job_posts
-- =============================================================================
CREATE TABLE IF NOT EXISTS job_posts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_batch_id UUID NOT NULL REFERENCES run_batches(id) ON DELETE CASCADE,
    source_platform TEXT NOT NULL,
    title TEXT NOT NULL,
    company TEXT NOT NULL,
    location TEXT,
    url TEXT UNIQUE NOT NULL,
    posted_at TIMESTAMPTZ,
    embedding_vector vector(1024),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- =============================================================================
-- TABLE 6: job_scores
-- =============================================================================
CREATE TABLE IF NOT EXISTS job_scores (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_post_id UUID NOT NULL REFERENCES job_posts(id) ON DELETE CASCADE,
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
    job_post_id UUID NOT NULL REFERENCES job_posts(id) ON DELETE CASCADE,
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
    job_post_id UUID NOT NULL REFERENCES job_posts(id) ON DELETE CASCADE,
    priority INTEGER NOT NULL DEFAULT 5,
    notes TEXT,
    queued_at TIMESTAMPTZ DEFAULT NOW()
);

-- =============================================================================
-- TABLE 9: logs_events
-- =============================================================================
CREATE TABLE IF NOT EXISTS logs_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_batch_id UUID NOT NULL REFERENCES run_batches(id) ON DELETE CASCADE,
    application_id UUID REFERENCES applications(id) ON DELETE SET NULL,
    job_post_id UUID REFERENCES job_posts(id) ON DELETE SET NULL,
    level TEXT NOT NULL CHECK (level IN ('INFO', 'WARNING', 'ERROR', 'CRITICAL')),
    event_type TEXT NOT NULL,
    message TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- =============================================================================
-- INDEXES
-- =============================================================================
CREATE INDEX IF NOT EXISTS idx_job_posts_run_batch_id ON job_posts(run_batch_id);
CREATE INDEX IF NOT EXISTS idx_job_posts_source_platform ON job_posts(source_platform);
CREATE INDEX IF NOT EXISTS idx_job_scores_job_post_id ON job_scores(job_post_id);
CREATE INDEX IF NOT EXISTS idx_job_scores_fit_score ON job_scores(fit_score DESC);
CREATE INDEX IF NOT EXISTS idx_applications_job_post_id ON applications(job_post_id);
CREATE INDEX IF NOT EXISTS idx_applications_user_id ON applications(user_id);
CREATE INDEX IF NOT EXISTS idx_applications_status ON applications(status);
CREATE INDEX IF NOT EXISTS idx_queued_jobs_priority ON queued_jobs(priority DESC);
CREATE INDEX IF NOT EXISTS idx_logs_events_run_batch_id ON logs_events(run_batch_id);
CREATE INDEX IF NOT EXISTS idx_logs_events_level ON logs_events(level);
CREATE INDEX IF NOT EXISTS idx_config_limits_platform ON config_limits(platform);
