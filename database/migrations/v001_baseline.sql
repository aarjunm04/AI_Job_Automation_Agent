-- v001 baseline: marks initial schema deployment
-- No DDL — schema.sql IS v001
INSERT INTO schema_versions (version, applied_at)
VALUES ('001', NOW())
ON CONFLICT (version) DO NOTHING;
