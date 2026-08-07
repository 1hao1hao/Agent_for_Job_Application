from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from uuid import uuid4

from intern_rag.agent.schemas import RagRequest, RagResponse
from intern_rag.persistence.models import EvaluationJob, EvaluationRunRecord
from intern_rag.tracing import AgentTrace


class PostgresRepository:
    """使用 PostgreSQL 保存请求、Trace、Job 和 Run 元数据。

    每个公开方法使用短连接和事务，避免把连接生命周期泄漏给 FastAPI/Worker。
    完整 report 不写入 JSONB，只保存摘要与持久化 volume 路径。
    """

    def __init__(self, database_url: str, migrations_dir: Path) -> None:
        if not database_url.strip():
            raise ValueError("database_url must not be empty")
        self.database_url = database_url
        self.migrations_dir = migrations_dir

    def _connect(self):
        import psycopg

        return psycopg.connect(self.database_url)

    def initialize(self) -> None:
        """按文件名顺序执行未应用 migration，并记录版本。"""

        with self._connect() as connection:
            # API 与 Worker 可能同时启动，事务级 advisory lock 防止重复执行 migration。
            connection.execute("SELECT pg_advisory_xact_lock(20260807)")
            connection.execute(
                "CREATE TABLE IF NOT EXISTS schema_migrations ("
                "version TEXT PRIMARY KEY, applied_at TIMESTAMPTZ NOT NULL DEFAULT now())"
            )
            applied = {
                row[0]
                for row in connection.execute(
                    "SELECT version FROM schema_migrations"
                ).fetchall()
            }
            for migration_path in sorted(self.migrations_dir.glob("*.sql")):
                if migration_path.name in applied:
                    continue
                connection.execute(migration_path.read_text(encoding="utf-8"))
                connection.execute(
                    "INSERT INTO schema_migrations(version) VALUES (%s)",
                    (migration_path.name,),
                )

    def ping(self) -> bool:
        """执行轻量查询；连接失败时返回 False，避免泄漏连接错误细节。"""

        try:
            with self._connect() as connection:
                return bool(connection.execute("SELECT 1").fetchone())
        except Exception:
            return False

    def save_request(self, request: RagRequest, response: RagResponse) -> None:
        """幂等保存请求及最终响应 JSON。"""

        from psycopg.types.json import Jsonb

        payload = {
            "request_id": response.request_id,
            "trace_id": response.trace_id,
            "answer": response.answer,
            "citations": [citation.to_dict() for citation in response.citations],
            "routed_sources": response.routed_sources,
            "status": response.status,
            "latency_ms": response.latency_ms,
            "error_type": response.error_type,
        }
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO rag_requests(request_id, trace_id, query, status, response_json)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (request_id) DO UPDATE SET
                  trace_id=EXCLUDED.trace_id, status=EXCLUDED.status,
                  response_json=EXCLUDED.response_json
                """,
                (
                    request.request_id,
                    response.trace_id,
                    request.query,
                    response.status,
                    Jsonb(payload),
                ),
            )

    def save_trace(self, trace: AgentTrace) -> None:
        """按 trace_id 幂等保存完整 AgentTrace JSON。"""

        from psycopg.types.json import Jsonb

        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO agent_traces(trace_id, request_id, trace_json, created_at)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (trace_id) DO UPDATE SET trace_json=EXCLUDED.trace_json
                """,
                (trace.trace_id, trace.request_id, Jsonb(trace.to_dict()), trace.created_at),
            )

    def get_trace(self, trace_id: str) -> AgentTrace | None:
        """读取 JSONB 并恢复已有 AgentTrace 契约。"""

        with self._connect() as connection:
            row = connection.execute(
                "SELECT trace_json FROM agent_traces WHERE trace_id=%s", (trace_id,)
            ).fetchone()
        if row is None:
            return None
        payload = row[0] if isinstance(row[0], dict) else json.loads(row[0])
        return AgentTrace.from_dict(payload)

    def create_job(
        self,
        *,
        dataset_version: str,
        split: str,
        run_config: dict[str, object],
        idempotency_key: str | None,
        max_retries: int,
    ) -> tuple[EvaluationJob, bool]:
        """事务内创建 job；重复 idempotency key 返回原 job 且不重复入队。"""

        from psycopg.types.json import Jsonb

        job_id = str(uuid4())
        with self._connect() as connection:
            if idempotency_key:
                existing = connection.execute(
                    "SELECT * FROM evaluation_jobs WHERE idempotency_key=%s",
                    (idempotency_key,),
                ).fetchone()
                if existing is not None:
                    return _job_from_row(existing), False
            row = connection.execute(
                """
                INSERT INTO evaluation_jobs(
                  job_id, dataset_version, split, run_config, status,
                  idempotency_key, max_retries)
                VALUES (%s, %s, %s, %s, 'queued', %s, %s)
                RETURNING *
                """,
                (
                    job_id,
                    dataset_version,
                    split,
                    Jsonb(run_config),
                    idempotency_key,
                    max_retries,
                ),
            ).fetchone()
        return _job_from_row(row), True

    def get_job(self, job_id: str) -> EvaluationJob | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM evaluation_jobs WHERE job_id=%s", (job_id,)
            ).fetchone()
        return _job_from_row(row) if row is not None else None

    def mark_job_running(self, job_id: str) -> EvaluationJob:
        """只允许 queued -> running，防止两个 Worker 同时执行同一 job。"""

        return self._transition(
            """
            UPDATE evaluation_jobs SET status='running', attempt_count=attempt_count+1,
              started_at=now(), completed_at=NULL, updated_at=now(),
              error_type=NULL, error_message=NULL
            WHERE job_id=%s AND status='queued' RETURNING *
            """,
            (job_id,),
            "job is not queued",
        )

    def mark_job_succeeded(self, job_id: str, report_path: str) -> EvaluationJob:
        return self._transition(
            """
            UPDATE evaluation_jobs SET status='succeeded', report_path=%s,
              completed_at=now(), updated_at=now()
            WHERE job_id=%s AND status='running' RETURNING *
            """,
            (report_path, job_id),
            "job is not running",
        )

    def mark_job_failed(
        self, job_id: str, error_type: str, error_message: str
    ) -> EvaluationJob:
        return self._transition(
            """
            UPDATE evaluation_jobs SET status='failed', error_type=%s,
              error_message=%s, completed_at=now(), updated_at=now()
            WHERE job_id=%s AND status IN ('queued', 'running') RETURNING *
            """,
            (error_type, error_message[:2000], job_id),
            "job cannot transition to failed",
        )

    def retry_failed_job(self, job_id: str) -> EvaluationJob:
        """只允许未超过 max_retries 的失败任务显式重入队。"""

        return self._transition(
            """
            UPDATE evaluation_jobs SET status='queued', error_type=NULL,
              error_message=NULL, completed_at=NULL, updated_at=now()
            WHERE job_id=%s AND status='failed' AND attempt_count <= max_retries
            RETURNING *
            """,
            (job_id,),
            "retry budget exhausted or job is not failed",
        )

    def recover_interrupted_jobs(self) -> list[str]:
        """恢复有预算的 running job；预算耗尽的中断任务直接标记 failed。"""

        with self._connect() as connection:
            connection.execute(
                """
                UPDATE evaluation_jobs SET status='failed',
                  error_type='retry_exhausted',
                  error_message='worker interrupted after retry budget was exhausted',
                  completed_at=now(), updated_at=now()
                WHERE status='running' AND attempt_count > max_retries
                """
            )
            rows = connection.execute(
                """
                UPDATE evaluation_jobs SET status='queued', error_type='worker_interrupted',
                  error_message='worker restarted before completion', updated_at=now()
                WHERE status='running' AND attempt_count <= max_retries
                RETURNING job_id
                """
            ).fetchall()
        return [str(row[0]) for row in rows]

    def save_run(self, run: EvaluationRunRecord) -> None:
        from psycopg.types.json import Jsonb

        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO evaluation_runs(
                  run_id, job_id, config_json, summary_json, report_path)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (run_id) DO UPDATE SET
                  summary_json=EXCLUDED.summary_json, report_path=EXCLUDED.report_path
                """,
                (
                    run.run_id,
                    run.job_id,
                    Jsonb(run.config),
                    Jsonb(run.summary),
                    run.report_path,
                ),
            )

    def _transition(
        self, sql: str, params: tuple[object, ...], error_message: str
    ) -> EvaluationJob:
        with self._connect() as connection:
            row = connection.execute(sql, params).fetchone()
        if row is None:
            raise ValueError(error_message)
        return _job_from_row(row)


def _job_from_row(row: Any) -> EvaluationJob:
    """按 migration 的固定列顺序把数据库行转换为 EvaluationJob。"""

    run_config = row[4] if isinstance(row[4], dict) else json.loads(row[4])
    return EvaluationJob(
        job_id=str(row[0]),
        dataset_version=str(row[1]),
        split=str(row[2]),
        status=str(row[3]),  # type: ignore[arg-type]
        run_config=run_config,
        idempotency_key=str(row[5]) if row[5] is not None else None,
        attempt_count=int(row[6]),
        max_retries=int(row[7]),
        report_path=str(row[8]) if row[8] is not None else None,
        error_type=str(row[9]) if row[9] is not None else None,
        error_message=str(row[10]) if row[10] is not None else None,
        created_at=row[11].isoformat(),
        updated_at=row[12].isoformat(),
        started_at=row[13].isoformat() if row[13] is not None else None,
        completed_at=row[14].isoformat() if row[14] is not None else None,
    )
