from __future__ import annotations

from typing import Protocol

from intern_rag.agent.schemas import RagRequest, RagResponse
from intern_rag.persistence.models import EvaluationJob, EvaluationRunRecord
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
