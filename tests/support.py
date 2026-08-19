from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from uuid import uuid4

from intern_rag.agent import RagRequest, RagResponse
from intern_rag.agent.context_engine import ConversationMessage, MemoryItem, UserProfile
from intern_rag.persistence import EvaluationJob, EvaluationRunRecord, SessionRecord
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
        self.sessions: dict[str, SessionRecord] = {}
        self.messages: dict[str, list[ConversationMessage]] = {}
        self.summaries: dict[str, str] = {}
        self.profiles: dict[str, UserProfile] = {}
        self.memories: dict[str, MemoryItem] = {}

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

    def create_session(self, user_id: str, title: str) -> SessionRecord:
        now = _now()
        value = SessionRecord(str(uuid4()), user_id, title, now, now)
        self.sessions[value.session_id] = value
        return value

    def get_session(self, user_id: str, session_id: str) -> SessionRecord | None:
        value = self.sessions.get(session_id)
        return value if value is not None and value.user_id == user_id else None

    def append_message(self, message: ConversationMessage) -> None:
        if self.get_session(message.user_id, message.session_id) is None:
            raise PermissionError("session does not belong to user")
        self.messages.setdefault(message.session_id, []).append(message)

    def list_messages(self, user_id: str, session_id: str, limit: int = 50) -> list[ConversationMessage]:
        if self.get_session(user_id, session_id) is None:
            return []
        return self.messages.get(session_id, [])[-limit:]

    def save_summary(self, user_id: str, session_id: str, summary: str, version: int) -> None:
        del version
        if self.get_session(user_id, session_id) is None:
            raise PermissionError("session does not belong to user")
        self.summaries[session_id] = summary

    def get_summary(self, user_id: str, session_id: str) -> str | None:
        if self.get_session(user_id, session_id) is None:
            return None
        return self.summaries.get(session_id)

    def upsert_profile(self, profile: UserProfile, expected_version: int | None) -> UserProfile:
        current = self.profiles.get(profile.user_id)
        current_version = current.version if current else 0
        if expected_version is not None and expected_version != current_version:
            raise ValueError("profile version conflict")
        value = replace(profile, version=current_version + 1, updated_at=_now())
        self.profiles[profile.user_id] = value
        return value

    def get_profile(self, user_id: str) -> UserProfile | None:
        return self.profiles.get(user_id)

    def save_memory(self, item: MemoryItem, embedding: list[float] | None = None) -> None:
        del embedding
        if not item.confirmed:
            raise ValueError("unconfirmed memory")
        current = self.memories.get(item.memory_id)
        if current is None or item.version > current.version:
            self.memories[item.memory_id] = item

    def list_memories(self, user_id: str, limit: int = 20) -> list[MemoryItem]:
        values = [item for item in self.memories.values() if item.user_id == user_id and item.is_available]
        return sorted(values, key=lambda item: (-item.importance, item.memory_id))[:limit]

    def search_memories(self, user_id: str, query_embedding: list[float], top_k: int = 5) -> list[MemoryItem]:
        del query_embedding
        return self.list_memories(user_id, top_k)

    def delete_memory(self, user_id: str, memory_id: str) -> bool:
        item = self.memories.get(memory_id)
        if item is None or item.user_id != user_id:
            return False
        self.memories[memory_id] = replace(item, active=False)
        return True

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
