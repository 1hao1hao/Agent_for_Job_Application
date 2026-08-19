from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from intern_rag.agent.schemas import RagResponse
from intern_rag.agent.context_engine import ProfileFact, UserProfile
from intern_rag.persistence import EvaluationJob, SessionRecord


class RagRequestBody(BaseModel):
    """HTTP 对现有 RagRequest 的校验 adapter。"""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=4000)
    request_id: str | None = None
    top_k: int = Field(default=5, ge=1, le=50)
    retriever: Literal[
        "keyword", "bm25", "dense", "hybrid", "bm25_hybrid"
    ] = "bm25"
    user_id: str | None = Field(default=None, min_length=1, max_length=100)
    session_id: str | None = Field(default=None, min_length=1, max_length=100)

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        """去除首尾空白并拒绝只有空格的 Query。"""

        normalized = value.strip()
        if not normalized:
            raise ValueError("query must not be empty")
        return normalized

    @model_validator(mode="after")
    def validate_session_pair(self):
        if (self.user_id is None) != (self.session_id is None):
            raise ValueError("user_id and session_id must be provided together")
        return self


class CitationBody(BaseModel):
    chunk_id: str
    source_path: str
    source_type: str
    title: str
    rank: int
    score: float


class RagResponseBody(BaseModel):
    """HTTP 输出字段直接来自现有 RagResponse，不改变 Pipeline 语义。"""

    request_id: str
    trace_id: str
    answer: str
    citations: list[CitationBody]
    routed_sources: list[str]
    status: Literal["answered", "insufficient_evidence", "error"]
    latency_ms: float
    error_type: str | None = None

    @classmethod
    def from_domain(cls, response: RagResponse) -> "RagResponseBody":
        return cls(
            request_id=response.request_id,
            trace_id=response.trace_id,
            answer=response.answer,
            citations=[CitationBody(**item.to_dict()) for item in response.citations],
            routed_sources=response.routed_sources,
            status=response.status,
            latency_ms=response.latency_ms,
            error_type=response.error_type,
        )


class EvaluationJobRequestBody(BaseModel):
    """创建异步评测任务所需的版本、split 和运行配置。"""

    model_config = ConfigDict(extra="forbid")

    dataset_version: str = Field(default="evalrag_v0.2", min_length=1)
    split: Literal["dev", "test"] = "dev"
    run_config: dict[str, object]
    max_retries: int = Field(default=1, ge=0, le=1)


class EvaluationJobBody(BaseModel):
    job_id: str
    dataset_version: str
    split: str
    run_config: dict[str, object]
    status: str
    idempotency_key: str | None
    attempt_count: int
    max_retries: int
    report_path: str | None
    error_type: str | None
    error_message: str | None
    created_at: str
    updated_at: str
    started_at: str | None
    completed_at: str | None

    @classmethod
    def from_domain(cls, job: EvaluationJob) -> "EvaluationJobBody":
        return cls(**job.__dict__)


class ErrorBody(BaseModel):
    error_type: str
    message: str
    request_id: str | None = None
    trace_id: str | None = None


class SessionCreateBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: str = Field(default="", max_length=200)


class SessionBody(BaseModel):
    session_id: str
    user_id: str
    title: str
    created_at: str
    updated_at: str

    @classmethod
    def from_domain(cls, session: SessionRecord) -> "SessionBody":
        return cls(**session.__dict__)


class ProfileFactBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    key: str = Field(min_length=1, max_length=100)
    value: str = Field(min_length=1, max_length=2000)
    source: str = Field(min_length=1, max_length=200)
    confirmed: bool = True


class ProfileUpdateBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    facts: list[ProfileFactBody]
    expected_version: int | None = Field(default=None, ge=0)


class ProfileBody(BaseModel):
    user_id: str
    facts: list[ProfileFactBody]
    version: int
    updated_at: str

    @classmethod
    def from_domain(cls, profile: UserProfile) -> "ProfileBody":
        return cls(
            user_id=profile.user_id,
            facts=[ProfileFactBody(**fact.__dict__) for fact in profile.facts],
            version=profile.version,
            updated_at=profile.updated_at,
        )

    def to_domain(self) -> UserProfile:
        return UserProfile(
            user_id=self.user_id,
            facts=tuple(ProfileFact(**fact.model_dump()) for fact in self.facts),
            version=self.version,
            updated_at=self.updated_at,
        )
