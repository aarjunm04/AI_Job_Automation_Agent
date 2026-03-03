# Schema Migrations

## Convention
- Files named: v{NNN}_{description}.sql (e.g. v002_add_index.sql)
- Apply in numeric order, never skip versions
- Never modify schema.sql directly after first deployment

## Applying a Migration
psql $LOCAL_POSTGRES_URL -f database/migrations/vNNN_description.sql

## Creating a New Migration
1. Copy the last migration file as a template
2. Increment the version number
3. Write only the delta DDL (ALTER TABLE, CREATE INDEX, etc.)
4. Test on local Docker Postgres before committing

## Version History
| Version | File | Description |
|---------|------|-------------|
| v001 | v001_baseline.sql | Initial schema baseline marker |
