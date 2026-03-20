CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgcrypto;
\i /docker-entrypoint-initdb.d/schema.sql
