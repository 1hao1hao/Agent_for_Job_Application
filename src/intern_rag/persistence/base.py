from __future__ import annotations

from typing import Protocol

from intern_rag.agent.schemas import RagRequest, RagResponse
from intern_rag.agent.context_engine import ConversationMessage, MemoryItem, UserProfile
from intern_rag.persistence.models import EvaluationJob, EvaluationRunRecord, SessionRecord
from intern_rag.tracing import AgentTrace


class PersistenceRepository(Protocol):
    """FastAPI 与 Worker 共同依赖的最小持久化接口。"""

    def initialize(self) -> None:
        """执行尚未应用的数据库 migration。"""

    def ping(self) -> bool:
        """检查数据库是否可用。"""

    def save_request(self, request: RagRequest, response: RagResponse) -> None:
        """保存请求索引和最终响应。"""

    def save_trace(self, trace: AgentTrace) -> None:
        """保存请求级 AgentTrace。"""

    def get_trace(self, trace_id: str) -> AgentTrace | None:
        """按 trace id 查询完整 Trace。"""

    def create_job(
        self,
        *,
        dataset_version: str,
        split: str,
        run_config: dict[str, object],
        idempotency_key: str | None,
        max_retries: int,
    ) -> tuple[EvaluationJob, bool]:
        """创建 job；返回 job 与是否新建，重复 key 返回已有 job。"""

    def get_job(self, job_id: str) -> EvaluationJob | None:
        """查询一个评测任务。"""

    def mark_job_running(self, job_id: str) -> EvaluationJob:
        """原子地把 queued job 转成 running 并增加 attempt。"""

    def mark_job_succeeded(self, job_id: str, report_path: str) -> EvaluationJob:
        """记录成功和标准报告路径。"""

    def mark_job_failed(
        self, job_id: str, error_type: str, error_message: str
    ) -> EvaluationJob:
        """记录受控失败，不删除原 job。"""

    def retry_failed_job(self, job_id: str) -> EvaluationJob:
        """在重试预算内把 failed job 重新置为 queued。"""

    def recover_interrupted_jobs(self) -> list[str]:
        """把进程中断遗留的 running job 恢复为 queued。"""

    def save_run(self, run: EvaluationRunRecord) -> None:
        """保存 Run 配置、摘要和工件索引。"""

    def create_session(self, user_id: str, title: str) -> SessionRecord:
        """创建绑定用户的会话。"""

    def get_session(self, user_id: str, session_id: str) -> SessionRecord | None:
        """按用户作用域查询会话，禁止跨用户读取。"""

    def append_message(self, message: ConversationMessage) -> None:
        """保存一条会话消息。"""

    def list_messages(self, user_id: str, session_id: str, limit: int = 50) -> list[ConversationMessage]:
        """按时间顺序读取会话消息。"""

    def save_summary(self, user_id: str, session_id: str, summary: str, version: int) -> None:
        """保存历史摘要，不能修改 Profile。"""

    def get_summary(self, user_id: str, session_id: str) -> str | None:
        """读取最新摘要。"""

    def upsert_profile(self, profile: UserProfile, expected_version: int | None) -> UserProfile:
        """显式写入画像并执行乐观锁校验。"""

    def get_profile(self, user_id: str) -> UserProfile | None:
        """读取用户画像。"""

    def save_memory(self, item: MemoryItem, embedding: list[float] | None = None) -> None:
        """保存已确认长期记忆及可选向量。"""

    def list_memories(self, user_id: str, limit: int = 20) -> list[MemoryItem]:
        """只读取指定用户有效记忆。"""

    def search_memories(self, user_id: str, query_embedding: list[float], top_k: int = 5) -> list[MemoryItem]:
        """在用户作用域内执行 pgvector 语义记忆检索。"""

    def delete_memory(self, user_id: str, memory_id: str) -> bool:
        """软删除指定用户的记忆。"""
