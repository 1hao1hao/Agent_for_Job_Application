from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from uuid import uuid4

from intern_rag.agent.context_engine import (
    ConversationMessage,
    MemoryItem,
    ProfileFact,
    UserProfile,
)
from intern_rag.agent.schemas import RagRequest, RagResponse
from intern_rag.persistence.models import EvaluationJob, EvaluationRunRecord, SessionRecord
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

    def create_session(self, user_id: str, title: str) -> SessionRecord:
        """创建用户会话并返回数据库时间戳。"""

        if not user_id.strip():
            raise ValueError("user_id must not be empty")
        session_id = str(uuid4())
        with self._connect() as connection:
            row = connection.execute(
                """
                INSERT INTO conversation_sessions(session_id, user_id, title)
                VALUES (%s, %s, %s) RETURNING *
                """,
                (session_id, user_id, title[:200]),
            ).fetchone()
        return _session_from_row(row)

    def get_session(self, user_id: str, session_id: str) -> SessionRecord | None:
        """同时使用 user_id/session_id 查询，跨用户访问表现为不存在。"""

        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM conversation_sessions WHERE user_id=%s AND session_id=%s",
                (user_id, session_id),
            ).fetchone()
        return _session_from_row(row) if row is not None else None

    def append_message(self, message: ConversationMessage) -> None:
        """只有会话归属匹配时才能写入消息。"""

        with self._connect() as connection:
            row = connection.execute(
                "SELECT 1 FROM conversation_sessions WHERE user_id=%s AND session_id=%s",
                (message.user_id, message.session_id),
            ).fetchone()
            if row is None:
                raise PermissionError("session does not belong to user")
            connection.execute(
                """
                INSERT INTO conversation_messages(
                  message_id, session_id, user_id, role, content, created_at)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (message_id) DO NOTHING
                """,
                (
                    message.message_id,
                    message.session_id,
                    message.user_id,
                    message.role,
                    message.content,
                    message.created_at,
                ),
            )
            connection.execute(
                "UPDATE conversation_sessions SET updated_at=now() WHERE session_id=%s",
                (message.session_id,),
            )

    def list_messages(self, user_id: str, session_id: str, limit: int = 50) -> list[ConversationMessage]:
        """读取最后 N 条消息后恢复为正序。"""

        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT message_id, session_id, user_id, role, content, created_at
                FROM conversation_messages
                WHERE user_id=%s AND session_id=%s
                ORDER BY created_at DESC, message_id DESC LIMIT %s
                """,
                (user_id, session_id, limit),
            ).fetchall()
        return [
            ConversationMessage(
                message_id=str(row[0]), session_id=str(row[1]), user_id=str(row[2]),
                role=str(row[3]), content=str(row[4]), created_at=row[5].isoformat(),  # type: ignore[arg-type]
            )
            for row in reversed(rows)
        ]

    def save_summary(self, user_id: str, session_id: str, summary: str, version: int) -> None:
        """幂等保存会话摘要；不触碰 user_profiles。"""

        with self._connect() as connection:
            owner = connection.execute(
                "SELECT 1 FROM conversation_sessions WHERE user_id=%s AND session_id=%s",
                (user_id, session_id),
            ).fetchone()
            if owner is None:
                raise PermissionError("session does not belong to user")
            connection.execute(
                """
                INSERT INTO conversation_summaries(session_id, user_id, version, summary)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (session_id) DO UPDATE SET
                  version=EXCLUDED.version, summary=EXCLUDED.summary, updated_at=now()
                WHERE conversation_summaries.user_id=EXCLUDED.user_id
                """,
                (session_id, user_id, version, summary),
            )

    def get_summary(self, user_id: str, session_id: str) -> str | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT summary FROM conversation_summaries WHERE user_id=%s AND session_id=%s",
                (user_id, session_id),
            ).fetchone()
        return str(row[0]) if row is not None else None

    def upsert_profile(self, profile: UserProfile, expected_version: int | None) -> UserProfile:
        """显式写入画像；expected_version 防止并发覆盖。"""

        from psycopg.types.json import Jsonb

        facts = [fact.__dict__ for fact in profile.facts]
        with self._connect() as connection:
            current = connection.execute(
                "SELECT version FROM user_profiles WHERE user_id=%s FOR UPDATE",
                (profile.user_id,),
            ).fetchone()
            current_version = int(current[0]) if current is not None else 0
            if expected_version is not None and current_version != expected_version:
                raise ValueError("profile version conflict")
            next_version = current_version + 1
            row = connection.execute(
                """
                INSERT INTO user_profiles(user_id, version, facts_json)
                VALUES (%s, %s, %s)
                ON CONFLICT (user_id) DO UPDATE SET
                  version=EXCLUDED.version, facts_json=EXCLUDED.facts_json, updated_at=now()
                RETURNING user_id, version, facts_json, updated_at
                """,
                (profile.user_id, next_version, Jsonb(facts)),
            ).fetchone()
        return _profile_from_row(row)

    def get_profile(self, user_id: str) -> UserProfile | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT user_id, version, facts_json, updated_at FROM user_profiles WHERE user_id=%s",
                (user_id,),
            ).fetchone()
        return _profile_from_row(row) if row is not None else None

    def save_memory(self, item: MemoryItem, embedding: list[float] | None = None) -> None:
        """保存已确认记忆；同 ID 更新时版本必须递增。"""

        if not item.confirmed:
            raise ValueError("unconfirmed memory must not be persisted")
        vector_value = (
            "[" + ",".join(f"{float(value):.10g}" for value in embedding) + "]"
            if embedding is not None
            else None
        )
        with self._connect() as connection:
            if item.session_id is not None:
                owner = connection.execute(
                    "SELECT 1 FROM conversation_sessions WHERE user_id=%s AND session_id=%s",
                    (item.user_id, item.session_id),
                ).fetchone()
                if owner is None:
                    raise PermissionError("memory session does not belong to user")
            connection.execute(
                """
                INSERT INTO memory_items(
                  memory_id, user_id, session_id, memory_type, content, source,
                  importance, version, expires_at, confirmed, active, embedding, created_at)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::vector,%s)
                ON CONFLICT (memory_id) DO UPDATE SET
                  content=EXCLUDED.content, source=EXCLUDED.source,
                  importance=EXCLUDED.importance, version=EXCLUDED.version,
                  expires_at=EXCLUDED.expires_at, active=EXCLUDED.active,
                  embedding=EXCLUDED.embedding
                WHERE memory_items.user_id=EXCLUDED.user_id
                  AND EXCLUDED.version > memory_items.version
                """,
                (
                    item.memory_id, item.user_id, item.session_id, item.memory_type,
                    item.content, item.source, item.importance, item.version,
                    item.expires_at, item.confirmed, item.active, vector_value,
                    item.created_at,
                ),
            )

    def list_memories(self, user_id: str, limit: int = 20) -> list[MemoryItem]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT memory_id,user_id,memory_type,content,source,importance,created_at,
                       version,session_id,expires_at,confirmed,active
                FROM memory_items WHERE user_id=%s AND active=true AND confirmed=true
                  AND (expires_at IS NULL OR expires_at > now())
                ORDER BY importance DESC, created_at DESC LIMIT %s
                """,
                (user_id, limit),
            ).fetchall()
        return [_memory_from_row(row) for row in rows]

    def search_memories(self, user_id: str, query_embedding: list[float], top_k: int = 5) -> list[MemoryItem]:
        """使用 cosine HNSW 检索指定用户记忆，不允许跨用户召回。"""

        vector = "[" + ",".join(f"{float(value):.10g}" for value in query_embedding) + "]"
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT memory_id,user_id,memory_type,content,source,importance,created_at,
                       version,session_id,expires_at,confirmed,active
                FROM memory_items
                WHERE user_id=%s AND active=true AND confirmed=true
                  AND embedding IS NOT NULL AND (expires_at IS NULL OR expires_at > now())
                ORDER BY embedding <=> %s::vector, importance DESC, memory_id LIMIT %s
                """,
                (user_id, vector, top_k),
            ).fetchall()
        return [_memory_from_row(row) for row in rows]

    def delete_memory(self, user_id: str, memory_id: str) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                "UPDATE memory_items SET active=false WHERE user_id=%s AND memory_id=%s RETURNING memory_id",
                (user_id, memory_id),
            ).fetchone()
        return row is not None

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


def _session_from_row(row: Any) -> SessionRecord:
    return SessionRecord(str(row[0]), str(row[1]), str(row[2]), row[3].isoformat(), row[4].isoformat())


def _profile_from_row(row: Any) -> UserProfile:
    values = row[2] if isinstance(row[2], list) else json.loads(row[2])
    return UserProfile(
        user_id=str(row[0]),
        version=int(row[1]),
        facts=tuple(ProfileFact(**value) for value in values),
        updated_at=row[3].isoformat(),
    )


def _memory_from_row(row: Any) -> MemoryItem:
    return MemoryItem(
        memory_id=str(row[0]), user_id=str(row[1]), memory_type=str(row[2]),  # type: ignore[arg-type]
        content=str(row[3]), source=str(row[4]), importance=float(row[5]),
        created_at=row[6].isoformat(), version=int(row[7]),
        session_id=str(row[8]) if row[8] is not None else None,
        expires_at=row[9].isoformat() if row[9] is not None else None,
        confirmed=bool(row[10]), active=bool(row[11]),
    )
