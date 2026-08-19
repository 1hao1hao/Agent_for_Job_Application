CREATE TABLE IF NOT EXISTS conversation_sessions (
  session_id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL,
  title TEXT NOT NULL DEFAULT '',
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS conversation_sessions_user_idx
  ON conversation_sessions(user_id, updated_at DESC);

CREATE TABLE IF NOT EXISTS conversation_messages (
  message_id TEXT PRIMARY KEY,
  session_id TEXT NOT NULL REFERENCES conversation_sessions(session_id) ON DELETE CASCADE,
  user_id TEXT NOT NULL,
  role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
  content TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL
);
CREATE INDEX IF NOT EXISTS conversation_messages_scope_idx
  ON conversation_messages(user_id, session_id, created_at);

CREATE TABLE IF NOT EXISTS conversation_summaries (
  session_id TEXT PRIMARY KEY REFERENCES conversation_sessions(session_id) ON DELETE CASCADE,
  user_id TEXT NOT NULL,
  version INTEGER NOT NULL CHECK (version > 0),
  summary TEXT NOT NULL,
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS user_profiles (
  user_id TEXT PRIMARY KEY,
  version INTEGER NOT NULL CHECK (version > 0),
  facts_json JSONB NOT NULL,
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS memory_items (
  memory_id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL,
  session_id TEXT REFERENCES conversation_sessions(session_id) ON DELETE SET NULL,
  memory_type TEXT NOT NULL CHECK (memory_type IN ('fact','preference','experience','decision')),
  content TEXT NOT NULL,
  source TEXT NOT NULL,
  importance DOUBLE PRECISION NOT NULL CHECK (importance BETWEEN 0 AND 1),
  version INTEGER NOT NULL CHECK (version > 0),
  expires_at TIMESTAMPTZ,
  confirmed BOOLEAN NOT NULL DEFAULT false,
  active BOOLEAN NOT NULL DEFAULT true,
  embedding vector(512),
  created_at TIMESTAMPTZ NOT NULL,
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS memory_items_scope_idx
  ON memory_items(user_id, active, created_at DESC);
CREATE INDEX IF NOT EXISTS memory_items_hnsw_cosine
  ON memory_items USING hnsw (embedding vector_cosine_ops)
  WHERE embedding IS NOT NULL;
