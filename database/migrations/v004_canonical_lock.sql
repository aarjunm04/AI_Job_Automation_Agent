-- ============================================================
-- MIGRATION v004 — Canonical Schema Lock (FIXED)
-- AI Job Automation Agent
-- Date: 2026-04-13
-- Run: docker exec -i ai_postgres psql -U aarjunm04 -d ai_job_db -f /tmp/v004_canonical_lock.sql
-- ============================================================

BEGIN;

-- Safety check: abort if already applied
DO $$
BEGIN
  IF EXISTS (SELECT 1 FROM schema_versions WHERE version = '004') THEN
    RAISE EXCEPTION 'Migration v004 already applied. Aborting.';
  END IF;
END $$;

-- ============================================================
-- STEP 1: DROP OLD TRIGGERS FIRST — before any column is touched
-- These triggers reference run_batch_id / run_sessions which
-- we are about to rename/drop. Must die before anything else.
-- ============================================================

DROP TRIGGER IF EXISTS trg_update_run_session_counts ON applications;
DROP TRIGGER IF EXISTS trg_update_run_session_discovered ON jobs;
DROP TRIGGER IF EXISTS trg_close_run_session ON run_sessions;
DROP FUNCTION IF EXISTS update_run_session_counts();
DROP FUNCTION IF EXISTS update_run_session_discovered();
DROP FUNCTION IF EXISTS close_run_session();

-- ============================================================
-- STEP 2: CREATE pipeline_runs (merges run_batches + run_sessions)
-- ============================================================

CREATE TABLE IF NOT EXISTS pipeline_runs (
  id                  UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
  run_id              TEXT         NOT NULL UNIQUE,
  run_date            DATE         NOT NULL DEFAULT CURRENT_DATE,
  run_index_in_week   INTEGER,
  mode                TEXT         NOT NULL DEFAULT 'full',
  status              TEXT         NOT NULL DEFAULT 'running'
                      CHECK (status IN ('running', 'completed', 'failed')),
  dry_run             BOOLEAN      NOT NULL DEFAULT FALSE,
  jobs_found          INTEGER      DEFAULT 0,
  jobs_applied        INTEGER      DEFAULT 0,
  jobs_queued         INTEGER      DEFAULT 0,
  jobs_skipped        INTEGER      DEFAULT 0,
  jobs_failed         INTEGER      DEFAULT 0,
  total_cost_usd      NUMERIC(10,6) DEFAULT 0.0,
  error_message       TEXT,
  started_at          TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
  completed_at        TIMESTAMPTZ,
  updated_at          TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

-- ============================================================
-- STEP 3: MIGRATE run_sessions → pipeline_runs
-- ============================================================

INSERT INTO pipeline_runs (
  id, run_id, run_date, run_index_in_week,
  mode, status, dry_run,
  jobs_found, jobs_applied, jobs_queued,
  jobs_skipped, jobs_failed,
  total_cost_usd, started_at, completed_at, updated_at
)
SELECT
  rs.id,
  'run_' || TO_CHAR(rs.run_date, 'YYYYMMDD') || '_' || LPAD(rs.run_index_in_week::TEXT, 3, '0') AS run_id,
  rs.run_date,
  rs.run_index_in_week,
  'full'                                                    AS mode,
  CASE WHEN rs.closed_at IS NOT NULL THEN 'completed' ELSE 'running' END AS status,
  FALSE                                                     AS dry_run,
  rs.jobs_discovered                                        AS jobs_found,
  rs.jobs_auto_applied                                      AS jobs_applied,
  rs.jobs_queued                                            AS jobs_queued,
  0                                                         AS jobs_skipped,
  0                                                         AS jobs_failed,
  0.0                                                       AS total_cost_usd,
  rs.started_at,
  rs.closed_at                                              AS completed_at,
  COALESCE(rs.closed_at, rs.started_at)                    AS updated_at
FROM run_sessions rs
ON CONFLICT (id) DO NOTHING;

-- ============================================================
-- STEP 4: MIGRATE run_batches → pipeline_runs (any that aren't
--         already in pipeline_runs from run_sessions above)
-- ============================================================

INSERT INTO pipeline_runs (
  id, run_id, run_date, mode, status, dry_run,
  jobs_found, jobs_applied, jobs_queued, total_cost_usd,
  error_message, started_at, completed_at, updated_at
)
SELECT
  rb.id,
  rb.run_batch_id                                           AS run_id,
  CURRENT_DATE                                              AS run_date,
  rb.mode,
  rb.status,
  rb.dry_run,
  rb.jobs_found,
  rb.jobs_applied,
  rb.jobs_queued,
  rb.total_cost_usd,
  rb.error_message,
  rb.started_at,
  rb.completed_at,
  rb.updated_at
FROM run_batches rb
WHERE NOT EXISTS (
  SELECT 1 FROM pipeline_runs pr WHERE pr.id = rb.id
)
ON CONFLICT (id) DO NOTHING;

-- ============================================================
-- STEP 5: JOBS TABLE — add 3 missing columns + rename FK
-- ============================================================

ALTER TABLE jobs
  ADD COLUMN IF NOT EXISTS description   TEXT,
  ADD COLUMN IF NOT EXISTS salary        TEXT,
  ADD COLUMN IF NOT EXISTS job_type      TEXT;

ALTER TABLE jobs
  ADD COLUMN IF NOT EXISTS pipeline_run_id UUID;

UPDATE jobs SET pipeline_run_id = run_batch_id;

ALTER TABLE jobs
  ADD CONSTRAINT jobs_pipeline_run_id_fkey
  FOREIGN KEY (pipeline_run_id) REFERENCES pipeline_runs(id) ON DELETE CASCADE;

ALTER TABLE jobs
  DROP CONSTRAINT IF EXISTS jobs_run_batch_id_fkey,
  DROP CONSTRAINT IF EXISTS jobs_runbatchid_fkey;

ALTER TABLE jobs DROP COLUMN IF EXISTS run_batch_id;

ALTER TABLE jobs ALTER COLUMN pipeline_run_id SET NOT NULL;

-- ============================================================
-- STEP 6: JOB_SCORES TABLE — add route, routed_at, resume_used
-- ============================================================

ALTER TABLE job_scores
  ADD COLUMN IF NOT EXISTS route       TEXT CHECK (route IN ('auto_apply', 'manual_queue', 'skip')),
  ADD COLUMN IF NOT EXISTS routed_at   TIMESTAMPTZ,
  ADD COLUMN IF NOT EXISTS resume_used TEXT;

UPDATE job_scores
SET
  route     = CASE
                WHEN fit_score >= 0.60 AND eligibility_pass = TRUE THEN 'auto_apply'
                WHEN fit_score >= 0.45 THEN 'manual_queue'
                ELSE 'skip'
              END,
  routed_at = scored_at
WHERE route IS NULL;

-- ============================================================
-- STEP 7: APPLICATIONS TABLE — add 5 missing columns
-- ============================================================

ALTER TABLE applications
  ADD COLUMN IF NOT EXISTS fit_score          FLOAT,
  ADD COLUMN IF NOT EXISTS proof_json         JSONB,
  ADD COLUMN IF NOT EXISTS proof_confidence   FLOAT,
  ADD COLUMN IF NOT EXISTS notion_synced      BOOLEAN NOT NULL DEFAULT FALSE,
  ADD COLUMN IF NOT EXISTS notion_synced_at   TIMESTAMPTZ;

UPDATE applications a
SET fit_score = js.fit_score
FROM job_scores js
WHERE js.job_post_id = a.job_post_id
  AND a.fit_score IS NULL;

-- ============================================================
-- STEP 8: RESUMES TABLE — add uuid column
-- ============================================================

ALTER TABLE resumes
  ADD COLUMN IF NOT EXISTS uuid UUID NOT NULL DEFAULT gen_random_uuid();

-- ============================================================
-- STEP 9: QUEUED_JOBS — change priority from INTEGER to TEXT
-- ============================================================

ALTER TABLE queued_jobs
  ADD COLUMN IF NOT EXISTS priority_text TEXT
  CHECK (priority_text IN ('low', 'mid', 'high'));

UPDATE queued_jobs
SET priority_text = CASE
  WHEN priority <= 3 THEN 'high'
  WHEN priority <= 6 THEN 'mid'
  ELSE 'low'
END;

ALTER TABLE queued_jobs DROP COLUMN priority;
ALTER TABLE queued_jobs RENAME COLUMN priority_text TO priority;
ALTER TABLE queued_jobs ALTER COLUMN priority SET NOT NULL;
ALTER TABLE queued_jobs ALTER COLUMN priority SET DEFAULT 'mid';

-- ============================================================
-- STEP 10: AUDIT_LOGS — rename FK column to pipeline_run_id
-- ============================================================

ALTER TABLE audit_logs
  ADD COLUMN IF NOT EXISTS pipeline_run_id UUID;

UPDATE audit_logs SET pipeline_run_id = run_batch_id;

ALTER TABLE audit_logs
  ADD CONSTRAINT audit_logs_pipeline_run_id_fkey
  FOREIGN KEY (pipeline_run_id) REFERENCES pipeline_runs(id) ON DELETE CASCADE;

ALTER TABLE audit_logs
  DROP CONSTRAINT IF EXISTS auditlogs_run_batch_id_fkey,
  DROP CONSTRAINT IF EXISTS auditlogs_runbatchid_fkey;

ALTER TABLE audit_logs DROP COLUMN IF EXISTS run_batch_id;

ALTER TABLE audit_logs ALTER COLUMN pipeline_run_id SET NOT NULL;

-- ============================================================
-- STEP 11: DROP old tables (triggers already gone from Step 1)
-- ============================================================

DROP TABLE IF EXISTS run_batches;
DROP TABLE IF EXISTS run_sessions;

-- ============================================================
-- STEP 12: DROP old indexes, CREATE new canonical indexes
-- ============================================================

DROP INDEX IF EXISTS idx_jobs_run_batch_id;
DROP INDEX IF EXISTS idx_jobs_run_platform;
DROP INDEX IF EXISTS idx_job_scores_route;
DROP INDEX IF EXISTS idx_audit_logs_run_batch_id;
DROP INDEX IF EXISTS idx_run_batches_run_batch_id;
DROP INDEX IF EXISTS idx_run_batches_status;
DROP INDEX IF EXISTS idx_run_batches_started_at;
DROP INDEX IF EXISTS idx_run_sessions_date;
DROP INDEX IF EXISTS idx_queued_jobs_priority;

CREATE INDEX IF NOT EXISTS idx_jobs_pipeline_run_id      ON jobs(pipeline_run_id);
CREATE INDEX IF NOT EXISTS idx_jobs_run_platform          ON jobs(pipeline_run_id, source_platform);
CREATE INDEX IF NOT EXISTS idx_jobs_url_hash              ON jobs USING hash(url);
CREATE INDEX IF NOT EXISTS idx_jobs_created_at            ON jobs(created_at DESC);

CREATE INDEX IF NOT EXISTS idx_job_scores_route           ON job_scores(route, fit_score DESC);
CREATE INDEX IF NOT EXISTS idx_job_scores_routed_at       ON job_scores(routed_at);
CREATE INDEX IF NOT EXISTS idx_job_scores_fit_score       ON job_scores(fit_score DESC);
CREATE INDEX IF NOT EXISTS idx_job_scores_job_post_id     ON job_scores(job_post_id);

CREATE INDEX IF NOT EXISTS idx_applications_notion_synced ON applications(notion_synced, status);
CREATE INDEX IF NOT EXISTS idx_applications_status        ON applications(status);
CREATE INDEX IF NOT EXISTS idx_applications_platform_status ON applications(platform, status);

CREATE INDEX IF NOT EXISTS idx_audit_logs_pipeline_run_id ON audit_logs(pipeline_run_id);
CREATE INDEX IF NOT EXISTS idx_audit_logs_level           ON audit_logs(level);
CREATE INDEX IF NOT EXISTS idx_audit_logs_event_type      ON audit_logs(event_type);

CREATE INDEX IF NOT EXISTS idx_pipeline_runs_run_id       ON pipeline_runs(run_id);
CREATE INDEX IF NOT EXISTS idx_pipeline_runs_status       ON pipeline_runs(status);
CREATE INDEX IF NOT EXISTS idx_pipeline_runs_started_at   ON pipeline_runs(started_at DESC);

CREATE INDEX IF NOT EXISTS idx_queued_jobs_priority       ON queued_jobs(priority);
CREATE INDEX IF NOT EXISTS idx_queued_jobs_job_post_id    ON queued_jobs(job_post_id);

-- ============================================================
-- STEP 13: RECORD MIGRATION
-- ============================================================

INSERT INTO schema_versions (version) VALUES ('004')
ON CONFLICT (version) DO NOTHING;

COMMIT;

-- ============================================================
-- POST-MIGRATION VERIFICATION — run these after COMMIT
-- ============================================================
-- SELECT version, applied_at FROM schema_versions ORDER BY version;
-- SELECT COUNT(*) FROM pipeline_runs;
-- SELECT COUNT(*) FROM jobs WHERE pipeline_run_id IS NULL;      -- must be 0
-- SELECT COUNT(*) FROM audit_logs WHERE pipeline_run_id IS NULL; -- must be 0
-- SELECT DISTINCT route FROM job_scores;
-- SELECT DISTINCT priority FROM queued_jobs;
-- SELECT column_name, data_type FROM information_schema.columns WHERE table_name='jobs' ORDER BY ordinal_position;
