from __future__ import annotations

from typing import Protocol
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from intern_rag.agent import ConversationMessage, RagPipeline, RagRequest, RagResponse
from intern_rag.persistence import PersistenceRepository, SessionMemoryService
from intern_rag.runtime import AgentRuntime, RunContext


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
    ) -> None:
        self.pipeline = pipeline
        self.repository = repository
        self.memory_service = memory_service
        self.runtime = runtime

    def execute(self, request: RagRequest) -> RagResponse:
        if self.runtime is None:
            response = self.pipeline.run(request)
        else:
            response = self.runtime.execute(
                request,
                RunContext(
                    request_id=request.request_id,
                    entrypoint="http",
                    config={
                        "router": self.pipeline.config.router_name,
                        "retriever": request.retriever,
                        "top_k": request.top_k,
                    },
                    dataset_version="service-runtime",
                    model_version=self.pipeline.config.model,
                    prompt_version=self.pipeline.config.prompt_version,
                ),
            ).response
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
