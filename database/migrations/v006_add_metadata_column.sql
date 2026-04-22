-- v006: Add missing metadata column to audit_logs
-- Root cause: v005 only added agent; metadata referenced but missing
ALTER TABLE audit_logs ADD COLUMN IF NOT EXISTS metadata TEXT;
INSERT INTO schema_versions (version, applied_at)
VALUES ('006', NOW())
ON CONFLICT (version) DO NOTHING;
