CREATE TABLE IF NOT EXISTS sessions (
  session_id TEXT PRIMARY KEY,
  owner TEXT,
  created_at TIMESTAMP,
  last_active_at TIMESTAMP,
  meta_json TEXT,
  ttl_hours INTEGER,
  is_active BOOLEAN DEFAULT 1,
  version INTEGER DEFAULT 1
);
CREATE INDEX IF NOT EXISTS ix_sessions_last_active ON sessions(last_active_at);

CREATE TABLE IF NOT EXISTS context_items (
  item_id TEXT PRIMARY KEY,
  session_id TEXT,
  role TEXT,
  content TEXT,
  vector_id TEXT,
  created_at TIMESTAMP,
  meta_json TEXT,
  trusted BOOLEAN DEFAULT 0,
  deprecated BOOLEAN DEFAULT 0,
  sequence INTEGER
);
CREATE INDEX IF NOT EXISTS ix_ctx_session_sequence ON context_items(session_id, sequence);
CREATE INDEX IF NOT EXISTS ix_ctx_session_created_at ON context_items(session_id, created_at);

CREATE TABLE IF NOT EXISTS snapshots (
  snapshot_id TEXT PRIMARY KEY,
  session_id TEXT,
  summary_text TEXT,
  method TEXT,
  created_at TIMESTAMP,
  meta_json TEXT
);

CREATE TABLE IF NOT EXISTS evidence (
  evidence_id TEXT PRIMARY KEY,
  session_id TEXT,
  attached_to TEXT,
  data TEXT,
  created_at TIMESTAMP,
  meta_json TEXT
);