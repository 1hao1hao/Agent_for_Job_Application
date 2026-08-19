from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Annotated

from fastapi import FastAPI, Header, Response
from fastapi.responses import JSONResponse

from intern_rag.agent import RagRequest
from intern_rag.persistence import PersistenceRepository
from intern_rag.serving.schemas import (
    ErrorBody,
    EvaluationJobBody,
    EvaluationJobRequestBody,
    RagRequestBody,
    RagResponseBody,
    ProfileBody,
    ProfileUpdateBody,
    SessionBody,
    SessionCreateBody,
)
from intern_rag.agent.context_engine import ProfileFact, UserProfile
from intern_rag.serving.service import QueryService
from intern_rag.worker import JobQueue


@dataclass(frozen=True)
class AppServices:
    """FastAPI 依赖集合；生产使用真实 adapter，测试注入 Fake。"""

    query_service: QueryService
    repository: PersistenceRepository
    queue: JobQueue
    query_timeout_seconds: float = 90.0


def create_app(services: AppServices) -> FastAPI:
    """构建 HTTP 服务并定义稳定错误映射。

    Query 使用线程执行现有同步 Pipeline，并设置请求级 timeout；Evaluation 创建只
    写 PostgreSQL 和 Redis 后立即返回 202；Trace/Job 查询始终读取持久化 repository。
    Pipeline 的回答、拒答和错误状态不会被 HTTP 层重新推断。
    """

    app = FastAPI(title="EvalRAG API", version="1.0.0")

    @app.get("/health")
    def health() -> Response:
        database_ok = services.repository.ping()
        queue_ok = services.queue.ping()
        payload = {
            "status": "ok" if database_ok and queue_ok else "degraded",
            "database": "ok" if database_ok else "unavailable",
            "queue": "ok" if queue_ok else "unavailable",
        }
        return JSONResponse(payload, status_code=200 if database_ok and queue_ok else 503)

    @app.post("/v1/query", response_model=RagResponseBody)
    async def query(body: RagRequestBody) -> Response:
        request_kwargs: dict[str, object] = {
            "query": body.query,
            "top_k": body.top_k,
            "retriever": body.retriever,
        }
        if body.request_id is not None:
            request_kwargs["request_id"] = body.request_id
        if body.user_id is not None:
            request_kwargs["user_id"] = body.user_id
            request_kwargs["session_id"] = body.session_id
        request = RagRequest(**request_kwargs)  # type: ignore[arg-type]
        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(services.query_service.execute, request),
                timeout=services.query_timeout_seconds,
            )
        except asyncio.TimeoutError:
            return _error_response(
                504, "request_timeout", "query exceeded service timeout", request.request_id
            )
        except PermissionError:
            return _error_response(
                404,
                "session_not_found",
                "session does not exist",
                request.request_id,
            )
        except Exception:
            return _error_response(
                500,
                "internal_error",
                "query service failed; inspect server logs and persisted job state",
                request.request_id,
            )
        if response.status == "error":
            status_code = _pipeline_error_status(response.error_type)
            return JSONResponse(
                RagResponseBody.from_domain(response).model_dump(),
                status_code=status_code,
            )
        return JSONResponse(RagResponseBody.from_domain(response).model_dump())

    @app.get("/v1/traces/{trace_id}")
    def get_trace(trace_id: str) -> Response:
        trace = services.repository.get_trace(trace_id)
        if trace is None:
            return _error_response(404, "trace_not_found", "trace does not exist")
        return JSONResponse(trace.to_dict())

    @app.post("/v1/evaluation-jobs", response_model=EvaluationJobBody)
    def create_evaluation_job(
        body: EvaluationJobRequestBody,
        response: Response,
        idempotency_key: Annotated[
            str | None, Header(alias="Idempotency-Key", max_length=200)
        ] = None,
    ) -> Response:
        job, created = services.repository.create_job(
            dataset_version=body.dataset_version,
            split=body.split,
            run_config=body.run_config,
            idempotency_key=idempotency_key,
            max_retries=body.max_retries,
        )
        if created:
            try:
                services.queue.enqueue(job.job_id)
            except Exception:
                services.repository.mark_job_failed(
                    job.job_id, "queue_unavailable", "failed to enqueue evaluation job"
                )
                return _error_response(
                    503, "queue_unavailable", "evaluation queue is unavailable"
                )
            response.status_code = 202
        return JSONResponse(
            EvaluationJobBody.from_domain(
                services.repository.get_job(job.job_id) or job
            ).model_dump(),
            status_code=202 if created else 200,
        )

    @app.get("/v1/evaluation-jobs/{job_id}", response_model=EvaluationJobBody)
    def get_evaluation_job(job_id: str) -> Response:
        job = services.repository.get_job(job_id)
        if job is None:
            return _error_response(404, "job_not_found", "evaluation job does not exist")
        return JSONResponse(EvaluationJobBody.from_domain(job).model_dump())

    @app.post("/v1/evaluation-jobs/{job_id}/retry", response_model=EvaluationJobBody)
    def retry_evaluation_job(job_id: str) -> Response:
        try:
            job = services.repository.retry_failed_job(job_id)
            services.queue.enqueue(job.job_id)
        except ValueError as error:
            return _error_response(409, "retry_not_allowed", str(error))
        except Exception:
            services.repository.mark_job_failed(
                job_id, "queue_unavailable", "failed to enqueue retry"
            )
            return _error_response(503, "queue_unavailable", "evaluation queue is unavailable")
        return JSONResponse(EvaluationJobBody.from_domain(job).model_dump(), status_code=202)

    @app.post("/v1/users/{user_id}/sessions", response_model=SessionBody, status_code=201)
    def create_session(user_id: str, body: SessionCreateBody) -> Response:
        session = services.repository.create_session(user_id, body.title)
        return JSONResponse(SessionBody.from_domain(session).model_dump(), status_code=201)

    @app.get("/v1/users/{user_id}/sessions/{session_id}", response_model=SessionBody)
    def get_session(user_id: str, session_id: str) -> Response:
        session = services.repository.get_session(user_id, session_id)
        if session is None:
            return _error_response(404, "session_not_found", "session does not exist")
        return JSONResponse(SessionBody.from_domain(session).model_dump())

    @app.get("/v1/users/{user_id}/profile", response_model=ProfileBody)
    def get_profile(user_id: str) -> Response:
        profile = services.repository.get_profile(user_id)
        if profile is None:
            return _error_response(404, "profile_not_found", "profile does not exist")
        return JSONResponse(ProfileBody.from_domain(profile).model_dump())

    @app.put("/v1/users/{user_id}/profile", response_model=ProfileBody)
    def update_profile(user_id: str, body: ProfileUpdateBody) -> Response:
        if any(not fact.confirmed for fact in body.facts):
            return _error_response(422, "unconfirmed_profile", "profile facts must be confirmed")
        try:
            profile = services.repository.upsert_profile(
                UserProfile(
                    user_id=user_id,
                    facts=tuple(ProfileFact(**fact.model_dump()) for fact in body.facts),
                    version=body.expected_version or 0,
                    updated_at="",
                ),
                body.expected_version,
            )
        except ValueError:
            return _error_response(409, "profile_version_conflict", "profile version conflict")
        return JSONResponse(ProfileBody.from_domain(profile).model_dump())

    return app


def _pipeline_error_status(error_type: str | None) -> int:
    if error_type == "llm_timeout":
        return 504
    if error_type in {"llm_error", "llm_format_error", "citation_invalid"}:
        return 502
    return 500


def _error_response(
    status_code: int,
    error_type: str,
    message: str,
    request_id: str | None = None,
    trace_id: str | None = None,
) -> JSONResponse:
    payload = ErrorBody(
        error_type=error_type,
        message=message,
        request_id=request_id,
        trace_id=trace_id,
    )
    return JSONResponse(payload.model_dump(), status_code=status_code)
