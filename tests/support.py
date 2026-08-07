from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from uuid import uuid4

from intern_rag.agent import RagRequest, RagResponse
from intern_rag.persistence import EvaluationJob, EvaluationRunRecord
from intern_rag.tracing import AgentTrace


class InMemoryPersistenceRepository:
    """服务和 Worker 测试共用的确定性内存 Fake，不用于生产。"""

    def __init__(self) -> None:
        self.requests: dict[str, tuple[RagRequest, RagResponse]] = {}
        self.traces: dict[str, AgentTrace] = {}
        self.jobs: dict[str, EvaluationJob] = {}
        self.idempotency: dict[str, str] = {}
        self.runs: dict[str, EvaluationRunRecord] = {}
        self.available = True

    def initialize(self) -> None:
        return None

    def ping(self) -> bool:
        return self.available

    def save_request(self, request: RagRequest, response: RagResponse) -> None:
        self.requests[request.request_id] = (request, response)

    def save_trace(self, trace: AgentTrace) -> None:
        self.traces[trace.trace_id] = trace

    def get_trace(self, trace_id: str) -> AgentTrace | None:
        return self.traces.get(trace_id)

    def create_job(
        self,
        *,
        dataset_version: str,
        split: str,
        run_config: dict[str, object],
        idempotency_key: str | None,
        max_retries: int,
    ) -> tuple[EvaluationJob, bool]:
        if idempotency_key and idempotency_key in self.idempotency:
            return self.jobs[self.idempotency[idempotency_key]], False
        now = datetime.now(timezone.utc).isoformat()
        job = EvaluationJob(
            job_id=str(uuid4()),
            dataset_version=dataset_version,
            split=split,
            run_config=run_config,
            status="queued",
            idempotency_key=idempotency_key,
            attempt_count=0,
            max_retries=max_retries,
            report_path=None,
            error_type=None,
            error_message=None,
            created_at=now,
            updated_at=now,
        )
        self.jobs[job.job_id] = job
        if idempotency_key:
            self.idempotency[idempotency_key] = job.job_id
        return job, True

    def get_job(self, job_id: str) -> EvaluationJob | None:
        return self.jobs.get(job_id)

    def mark_job_running(self, job_id: str) -> EvaluationJob:
        job = self._require(job_id)
        if job.status != "queued":
            raise ValueError("job is not queued")
        job = replace(
            job,
            status="running",
            attempt_count=job.attempt_count + 1,
            started_at=_now(),
            updated_at=_now(),
            error_type=None,
            error_message=None,
        )
        self.jobs[job_id] = job
        return job

    def mark_job_succeeded(self, job_id: str, report_path: str) -> EvaluationJob:
        job = self._require(job_id)
        job = replace(
            job,
            status="succeeded",
            report_path=report_path,
            completed_at=_now(),
            updated_at=_now(),
        )
        self.jobs[job_id] = job
        return job

    def mark_job_failed(
        self, job_id: str, error_type: str, error_message: str
    ) -> EvaluationJob:
        job = self._require(job_id)
        job = replace(
            job,
            status="failed",
            error_type=error_type,
            error_message=error_message,
            completed_at=_now(),
            updated_at=_now(),
        )
        self.jobs[job_id] = job
        return job

    def retry_failed_job(self, job_id: str) -> EvaluationJob:
        job = self._require(job_id)
        if job.status != "failed" or job.attempt_count > job.max_retries:
            raise ValueError("retry budget exhausted or job is not failed")
        job = replace(
            job,
            status="queued",
            error_type=None,
            error_message=None,
            completed_at=None,
            updated_at=_now(),
        )
        self.jobs[job_id] = job
        return job

    def recover_interrupted_jobs(self) -> list[str]:
        recovered: list[str] = []
        for job_id, job in list(self.jobs.items()):
            if job.status == "running" and job.attempt_count > job.max_retries:
                self.jobs[job_id] = replace(
                    job,
                    status="failed",
                    error_type="retry_exhausted",
                    error_message="worker interrupted after retry budget was exhausted",
                )
                continue
            if job.status == "running" and job.attempt_count <= job.max_retries:
                self.jobs[job_id] = replace(
                    job,
                    status="queued",
                    error_type="worker_interrupted",
                    error_message="worker restarted before completion",
                )
                recovered.append(job_id)
        return recovered

    def save_run(self, run: EvaluationRunRecord) -> None:
        self.runs[run.run_id] = run

    def _require(self, job_id: str) -> EvaluationJob:
        job = self.jobs.get(job_id)
        if job is None:
            raise ValueError("job does not exist")
        return job


class InMemoryJobQueue:
    def __init__(self) -> None:
        self.job_ids: list[str] = []
        self.available = True

    def enqueue(self, job_id: str) -> None:
        if not self.available:
            raise RuntimeError("queue unavailable")
        self.job_ids.append(job_id)

    def dequeue(self, timeout_seconds: int = 5) -> str | None:
        del timeout_seconds
        return self.job_ids.pop(0) if self.job_ids else None

    def ping(self) -> bool:
        return self.available


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
