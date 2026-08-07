CREATE TABLE IF NOT EXISTS rag_requests (
  request_id TEXT PRIMARY KEY,
  trace_id TEXT NOT NULL,
  query TEXT NOT NULL,
  status TEXT NOT NULL CHECK (status IN ('answered', 'insufficient_evidence', 'error')),
  response_json JSONB NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS rag_requests_trace_id_idx ON rag_requests(trace_id);

CREATE TABLE IF NOT EXISTS agent_traces (
  trace_id TEXT PRIMARY KEY,
  request_id TEXT NOT NULL,
  trace_json JSONB NOT NULL,
  created_at TIMESTAMPTZ NOT NULL
);

CREATE INDEX IF NOT EXISTS agent_traces_request_id_idx ON agent_traces(request_id);

CREATE TABLE IF NOT EXISTS evaluation_jobs (
  job_id TEXT PRIMARY KEY,
  dataset_version TEXT NOT NULL,
  split TEXT NOT NULL CHECK (split IN ('dev', 'test')),
  status TEXT NOT NULL CHECK (status IN ('queued', 'running', 'succeeded', 'failed')),
  run_config JSONB NOT NULL,
  idempotency_key TEXT UNIQUE,
  attempt_count INTEGER NOT NULL DEFAULT 0,
  max_retries INTEGER NOT NULL DEFAULT 1 CHECK (max_retries BETWEEN 0 AND 1),
  report_path TEXT,
  error_type TEXT,
  error_message TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  started_at TIMESTAMPTZ,
  completed_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS evaluation_jobs_status_idx ON evaluation_jobs(status);

CREATE TABLE IF NOT EXISTS evaluation_runs (
  run_id TEXT PRIMARY KEY,
  job_id TEXT NOT NULL REFERENCES evaluation_jobs(job_id),
  config_json JSONB NOT NULL,
  summary_json JSONB NOT NULL,
  report_path TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
