from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from intern_rag.agent.context_engine import ConversationMessage, MemoryItem, UserProfile


EvaluationJobStatus = Literal["queued", "running", "succeeded", "failed"]


@dataclass(frozen=True)
class EvaluationJob:
    """一个可持久化、可恢复的批量评测任务。"""

    job_id: str
    dataset_version: str
    split: str
    run_config: dict[str, object]
    status: EvaluationJobStatus
    idempotency_key: str | None
    attempt_count: int
    max_retries: int
    report_path: str | None
    error_type: str | None
    error_message: str | None
    created_at: str
    updated_at: str
    started_at: str | None = None
    completed_at: str | None = None


@dataclass(frozen=True)
class EvaluationRunRecord:
    """Worker 成功后写入 PostgreSQL 的 Run 摘要索引。"""

    run_id: str
    job_id: str
    config: dict[str, object]
    summary: dict[str, object]
    report_path: str


@dataclass(frozen=True)
class SessionRecord:
    """归属于单一用户的会话索引。"""

    session_id: str
    user_id: str
    title: str
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class SessionContext:
    """Context Engine 从持久化层读取的画像、历史、摘要和记忆集合。"""

    session: SessionRecord
    profile: UserProfile | None
    messages: tuple[ConversationMessage, ...]
    summary: str | None
    memories: tuple[MemoryItem, ...]
    history_source: Literal["redis", "postgres"] = "postgres"
