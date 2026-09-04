from __future__ import annotations

from dataclasses import replace
from typing import Protocol
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from intern_rag.agent import ConversationMessage, RagPipeline, RagRequest, RagResponse
from intern_rag.persistence import PersistenceRepository, SessionMemoryService
from intern_rag.runtime import AgentRuntime, RunContext, build_run_context_snapshot


class QueryService(Protocol):
    """FastAPI 可注入的同步查询服务，测试无需启动真实模型。"""

    def execute(self, request: RagRequest) -> RagResponse:
        """执行 Pipeline 并持久化请求索引。"""


class PipelineQueryService:
    """把 HTTP 请求交给现有 Pipeline，并保存最终请求/响应索引。"""

    def __init__(
        self,
        pipeline: RagPipeline,
        repository: PersistenceRepository,
        memory_service: SessionMemoryService | None = None,
        runtime: AgentRuntime | None = None,
        run_context_template: RunContext | None = None,
    ) -> None:
        self.pipeline = pipeline
        self.repository = repository
        self.memory_service = memory_service
        self.runtime = runtime
        self.run_context_template = run_context_template

    def execute(self, request: RagRequest) -> RagResponse:
        if self.runtime is None:
            response = self.pipeline.run(request)
        else:
            context = self._build_run_context(request)
            execution = self.runtime.execute(
                request,
                context,
            )
            response = execution.response
            trace = getattr(execution, "trace", None)
            if trace is not None:
                trace = replace(trace, run_context=build_run_context_snapshot(context))
                self.pipeline.last_trace = trace
                self.repository.save_trace(trace)
        self.repository.save_request(request, response)
        if request.user_id is not None and request.session_id is not None and self.memory_service is not None:
            now = datetime.now(timezone.utc)
            self.memory_service.append_message(
                ConversationMessage(
                    str(uuid4()), request.session_id, request.user_id, "user",
                    request.query, now.isoformat(),
                )
            )
            self.memory_service.append_message(
                ConversationMessage(
                    str(uuid4()), request.session_id, request.user_id, "assistant",
                    response.answer, (now + timedelta(microseconds=1)).isoformat(),
                )
            )
        return response

    def _build_run_context(self, request: RagRequest) -> RunContext:
        """合并服务模板与本次请求参数，形成可恢复的版本化运行快照。"""

        template = self.run_context_template or RunContext()
        config = {
            **template.config,
            "router": self.pipeline.config.router_name,
            "retriever": request.retriever,
            "top_k": request.top_k,
            "temperature": getattr(self.pipeline.config, "temperature", 0.0),
            "context_max_chars": getattr(self.pipeline.config, "context_max_chars", 4000),
            "context_strategy": getattr(self.pipeline.config, "context_strategy", "rank_prefix"),
            "context_token_budget": getattr(self.pipeline.config, "context_token_budget", 1800),
            "context_mode": getattr(self.pipeline.config, "context_mode", "recent_window"),
            "max_source_retries": getattr(self.pipeline.config, "max_source_retries", 1),
            "max_format_retries": getattr(self.pipeline.config, "max_format_retries", 1),
        }
        return RunContext(
            request_id=request.request_id,
            entrypoint="http",
            config=config,
            artifact_refs=dict(template.artifact_refs),
            dataset_version=template.dataset_version or "service-runtime",
            model_version=self.pipeline.config.model,
            prompt_version=self.pipeline.config.prompt_version,
            index_version=template.index_version,
        )
