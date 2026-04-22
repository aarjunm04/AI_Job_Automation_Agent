-- v005: Add missing `agent` column to audit_logs
-- Root cause: v004_canonical_lock.sql never added this column
-- Every audit event since v004 has been silently discarded

BEGIN;

ALTER TABLE audit_logs
    ADD COLUMN IF NOT EXISTS agent TEXT;

INSERT INTO schema_versions (version, description, applied_at)
VALUES ('005', 'add_agent_column_to_audit_logs', NOW())
ON CONFLICT (version) DO NOTHING;

COMMIT;
