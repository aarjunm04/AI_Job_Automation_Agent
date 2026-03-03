-- ============================================================
-- Migration: v002
-- Description: Rename tables to match agent query expectations
-- Apply: psql $LOCAL_POSTGRES_URL -f database/migrations/v002_rename_tables.sql
-- ============================================================

-- v002: rename tables to match agent query expectations
ALTER TABLE job_posts RENAME TO jobs;
ALTER TABLE logs_events RENAME TO audit_logs;
ALTER TABLE run_batches RENAME TO run_sessions;

INSERT INTO schema_versions (version, applied_at)
VALUES ('002', NOW()) ON CONFLICT (version) DO NOTHING;
