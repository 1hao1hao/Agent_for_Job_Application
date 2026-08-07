from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Protocol

from intern_rag.persistence import (
    EvaluationJob,
    EvaluationRunRecord,
    PersistenceRepository,
)
from intern_rag.worker.queue import JobQueue


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
    """执行 `queue -> PostgreSQL claim -> Evaluation -> final status` 状态机。

    `run_once` 每次最多处理一个 job，便于测试和优雅退出。只有 PostgreSQL 成功把
    queued 原子转换为 running 后才执行；超时或异常统一落为 failed。进程重启时
    `recover_interrupted` 把未完成 job 恢复入队，重试次数仍受数据库预算约束。
    """

    def __init__(
        self,
        repository: PersistenceRepository,
        queue: JobQueue,
        executor: EvaluationExecutor,
    ) -> None:
        self.repository = repository
        self.queue = queue
        self.executor = executor

    def recover_interrupted(self) -> list[str]:
        job_ids = self.repository.recover_interrupted_jobs()
        for job_id in job_ids:
            self.queue.enqueue(job_id)
        return job_ids

    def run_once(self, timeout_seconds: int = 5) -> bool:
        job_id = self.queue.dequeue(timeout_seconds)
        if job_id is None:
            return False
        try:
            job = self.repository.mark_job_running(job_id)
        except ValueError:
            return False
        try:
            result = self.executor.execute(job)
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
        return True
