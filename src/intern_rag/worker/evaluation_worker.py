from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import sys
import socket
from typing import Protocol

from intern_rag.persistence import (
    EvaluationJob,
    EvaluationRunRecord,
    PersistenceRepository,
)
from intern_rag.worker.queue import JobQueue, QueueMessage
from intern_rag.runtime import AgentRuntime, RunContext


@dataclass(frozen=True)
class EvaluationExecutionResult:
    """Evaluation Executor 成功后交给 Worker 持久化的结果。"""

    run_id: str
    config: dict[str, object]
    summary: dict[str, object]
    report_path: str


class EvaluationExecutor(Protocol):
    """Worker 可注入的评测执行器，自动化测试使用 Fake。"""

    def execute(self, job: EvaluationJob) -> EvaluationExecutionResult:
        """执行一个 job 并返回标准 Run 摘要。"""


class WorkerExecutionError(RuntimeError):
    """携带稳定 error type 的 Worker 受控失败。"""

    def __init__(self, error_type: str, message: str) -> None:
        super().__init__(message)
        self.error_type = error_type


class SubprocessEvaluationExecutor:
    """通过现有 CLI 执行评测，复用其标签校验、配置和标准工件输出。

    输入 job.run_config 必须包含仓库内的 retriever config；执行时使用参数列表而非
    shell 字符串，并设置超时。成功后读取 Runner 实际生成的 summary，不手写预测。
    """

    def __init__(self, project_root: Path, timeout_seconds: int = 1800) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than 0")
        self.project_root = project_root.resolve()
        self.timeout_seconds = timeout_seconds

    def execute(self, job: EvaluationJob) -> EvaluationExecutionResult:
        config_path = self._safe_config_path(
            str(job.run_config.get("retriever_config_path", "")),
            "configs/retrieval",
        )
        router_value = str(job.run_config.get("router_config_path", ""))
        router_path = (
            self._safe_config_path(router_value, "configs/routing")
            if router_value
            else None
        )
        run_id = str(job.run_config.get("run_id") or f"p1-d1-job-{job.job_id}")
        command = [
            sys.executable,
            "scripts/run_evaluation.py",
            "--dataset-version",
            job.dataset_version,
            "--split",
            job.split,
            "--config",
            str(config_path.relative_to(self.project_root)),
            "--run-id",
            run_id,
        ]
        if router_path is not None:
            command.extend(
                ["--router-config", str(router_path.relative_to(self.project_root))]
            )
        if job.split == "test":
            if not bool(job.run_config.get("allow_frozen_test", False)):
                raise WorkerExecutionError(
                    "frozen_test_not_allowed",
                    "test split requires explicit allow_frozen_test=true",
                )
            command.append("--allow-frozen-test")

        environment = os.environ.copy()
        environment["PYTHONPATH"] = "src"
        try:
            completed = subprocess.run(
                command,
                cwd=self.project_root,
                env=environment,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            raise WorkerExecutionError(
                "evaluation_timeout", "evaluation exceeded worker timeout"
            ) from error
        if completed.returncode != 0:
            message = (completed.stderr or completed.stdout or "evaluation failed")[-2000:]
            raise WorkerExecutionError("evaluation_failed", message)

        report_path = Path("reports/runs") / run_id
        summary_path = self.project_root / report_path / "summary.json"
        if not summary_path.exists():
            raise WorkerExecutionError(
                "artifact_missing", "evaluation completed without summary.json"
            )
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        config = json.loads(
            (self.project_root / report_path / "run_config.json").read_text(
                encoding="utf-8"
            )
        )
        return EvaluationExecutionResult(
            run_id=run_id,
            config=config,
            summary=summary,
            report_path=str(report_path),
        )

    def _safe_config_path(self, value: str, expected_dir: str) -> Path:
        """只允许读取仓库声明配置目录中的 JSON，避免任意路径注入。"""

        if not value:
            raise WorkerExecutionError("invalid_config", "config path is required")
        path = (self.project_root / value).resolve()
        allowed_root = (self.project_root / expected_dir).resolve()
        if path.suffix != ".json" or allowed_root not in path.parents or not path.exists():
            raise WorkerExecutionError("invalid_config", f"invalid config path: {value}")
        return path


class EvaluationWorker:
    """执行 Streams pending -> PostgreSQL 状态机 -> ACK 的可靠消费流程。

    ``run_once`` 先尝试 reclaim stale pending，再读取新消息。queued Job 只有在
    PostgreSQL 条件更新成功后才执行；成功或失败终态持久化后才 ACK。Worker 在
    任一 ACK 前 crash 时消息仍可重投，重复执行由数据库状态机和幂等 Run 写入约束。
    """

    def __init__(
        self,
        repository: PersistenceRepository,
        queue: JobQueue,
        executor: EvaluationExecutor,
        runtime: AgentRuntime | None = None,
        *,
        consumer_name: str | None = None,
        stale_min_idle_ms: int = 1_860_000,
    ) -> None:
        if stale_min_idle_ms < 0:
            raise ValueError("stale_min_idle_ms must not be negative")
        self.repository = repository
        self.queue = queue
        self.executor = executor
        self.runtime = runtime
        self.consumer_name = consumer_name or f"{socket.gethostname()}-{os.getpid()}"
        self.stale_min_idle_ms = stale_min_idle_ms

    def run_once(self, timeout_seconds: int = 5) -> bool:
        """恢复或读取一条消息，依据 PG 状态执行，终态落库后 ACK。

        返回值只表示本轮是否取到消息。Redis/PG 异常向上抛出，使进程入口能够记录并
        退避；异常发生在 ACK 前时 pending 消息保持可恢复。
        """

        reclaimed = self.queue.reclaim_stale(
            self.consumer_name, self.stale_min_idle_ms, count=1
        )
        message = reclaimed[0] if reclaimed else self.queue.dequeue(
            self.consumer_name, timeout_seconds
        )
        if message is None:
            return False
        self._process_message(message, reclaimed=bool(reclaimed))
        return True

    def _process_message(self, message: QueueMessage, *, reclaimed: bool) -> None:
        """按消息来源和数据库状态决定执行、恢复或安全 ACK。"""

        job = self.repository.get_job(message.job_id)
        if job is None:
            # 不存在的 job 无法执行，ACK 防止坏消息永久占据 pending。
            self.queue.ack(message.message_id)
            return
        if job.status in {"succeeded", "failed"}:
            self.queue.ack(message.message_id)
            return
        if job.status == "running":
            if not reclaimed:
                # 同一 job 的另一条消息正在由健康 Worker 驱动，当前重复消息可安全 ACK。
                self.queue.ack(message.message_id)
                return
            job = self.repository.recover_interrupted_job(job.job_id)
            if job.status == "failed":
                self.queue.ack(message.message_id)
                return
        if job.status != "queued":
            return
        try:
            job = self.repository.mark_job_running(job.job_id)
        except ValueError:
            current = self.repository.get_job(message.job_id)
            if current is not None and current.status in {
                "running", "succeeded", "failed"
            }:
                self.queue.ack(message.message_id)
            return
        try:
            if self.runtime is None:
                result = self.executor.execute(job)
            else:
                result = self.runtime.execute_operation(
                    RunContext(
                        run_id=f"evaluation-job-{job.job_id}",
                        request_id=job.job_id,
                        entrypoint="worker",
                        config=job.run_config,
                        dataset_version=job.dataset_version,
                    ),
                    "evaluation.worker",
                    lambda: self.executor.execute(job),
                )
            self.repository.save_run(
                EvaluationRunRecord(
                    run_id=result.run_id,
                    job_id=job.job_id,
                    config=result.config,
                    summary=result.summary,
                    report_path=result.report_path,
                )
            )
            self.repository.mark_job_succeeded(job.job_id, result.report_path)
        except WorkerExecutionError as error:
            self.repository.mark_job_failed(job.job_id, error.error_type, str(error))
        except Exception as error:
            self.repository.mark_job_failed(
                job.job_id, "worker_unexpected_error", str(error)
            )
        # ACK 失败时 PG 已是终态；消息保留 pending，后续 reclaim 会读取终态并补 ACK。
        self.queue.ack(message.message_id)
