from __future__ import annotations

from typing import Protocol

from intern_rag.agent import RagPipeline, RagRequest, RagResponse
from intern_rag.persistence import PersistenceRepository


class QueryService(Protocol):
    """FastAPI 可注入的同步查询服务，测试无需启动真实模型。"""

    def execute(self, request: RagRequest) -> RagResponse:
        """执行 Pipeline 并持久化请求索引。"""


class PipelineQueryService:
    """把 HTTP 请求交给现有 Pipeline，并保存最终请求/响应索引。"""

    def __init__(
        self, pipeline: RagPipeline, repository: PersistenceRepository
    ) -> None:
        self.pipeline = pipeline
        self.repository = repository

    def execute(self, request: RagRequest) -> RagResponse:
        response = self.pipeline.run(request)
        self.repository.save_request(request, response)
        return response
