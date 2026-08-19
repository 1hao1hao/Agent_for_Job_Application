from __future__ import annotations

import json
import os
from pathlib import Path
import re

from intern_rag.agent import ContextEngine, ContextInputs, PipelineConfig, RagPipeline
from intern_rag.agent.generation import DeepSeekChatClient
from intern_rag.evaluation import load_chunks_jsonl
from intern_rag.persistence import PostgresRepository, RedisRecentHistoryCache, SessionMemoryService
from intern_rag.retrieval import (
    build_bm25_index,
    build_retriever_from_config,
    retrieve_top_k,
    save_bm25_index,
)
from intern_rag.routing import route_query
from intern_rag.serving.api import AppServices, create_app
from intern_rag.serving.service import PipelineQueryService
from intern_rag.worker import RedisJobQueue
from intern_rag.runtime import AgentRuntime, JsonlSpanSink, PipelineRuntimeExecutor


class DeterministicDemoLlmClient:
    """Docker/自动化 smoke 使用的无网络生成器，不用于正式效果评测。"""

    last_token_usage: dict[str, int] | None = None

    def generate(self, prompt: str, *, model: str, temperature: float) -> str:
        del model, temperature
        match = re.search(r"允许引用的 chunk id：([^\n]+)", prompt)
        first_id = match.group(1).split(",")[0].strip() if match else ""
        if not first_id or first_id == "无":
            payload = {
                "answer": "当前证据不足，无法可靠回答。",
                "cited_chunk_ids": [],
                "sufficient": False,
                "reason": "没有可引用证据",
            }
        else:
            payload = {
                "answer": "根据本轮检索证据，可以确认该问题存在相关资料。",
                "cited_chunk_ids": [first_id],
                "sufficient": True,
                "reason": "使用排名最高的完整证据",
            }
        return json.dumps(payload, ensure_ascii=False)


def create_runtime_app():
    """从环境变量组装生产 adapter；密钥不进入配置、Trace 或错误响应。"""

    project_root = Path(os.environ.get("EVALRAG_PROJECT_ROOT", ".")).resolve()
    database_url = os.environ["DATABASE_URL"]
    redis_url = os.environ["REDIS_URL"]
    repository = PostgresRepository(database_url, project_root / "migrations")
    repository.initialize()
    queue = RedisJobQueue(redis_url)
    memory_service = SessionMemoryService(
        repository,
        RedisRecentHistoryCache(redis_url),
    )

    def context_provider(request):
        """按 RagRequest 的 user/session scope 获取 Context；无会话时返回空输入。"""

        if request.user_id is None or request.session_id is None:
            return ContextInputs()
        value = memory_service.load_context(request.user_id, request.session_id)
        return ContextInputs(
            profile=value.profile,
            history=value.messages,
            memories=value.memories,
            history_summary=value.summary,
            history_source=value.history_source,
        )

    dataset_version = os.environ.get("EVALRAG_DATASET_VERSION", "evalrag_v0.2")
    chunks = load_chunks_jsonl(
        project_root / "data/processed/chunks" / f"{dataset_version}.jsonl"
    )
    bm25_index_path = (
        project_root / "data/processed/indexes" / dataset_version / "bm25-v1/index.json"
    )
    if not bm25_index_path.exists():
        save_bm25_index(
            build_bm25_index(chunks, dataset_version), bm25_index_path
        )
    bm25_config = json.loads(
        (project_root / "configs/retrieval/bm25_v0.2.json").read_text(
            encoding="utf-8"
        )
    )
    bm25_config["bm25_index_path"] = str(bm25_index_path)
    bm25 = build_retriever_from_config(bm25_config)

    llm_backend = os.environ.get("EVALRAG_LLM_BACKEND", "fake")
    llm_client = (
        DeepSeekChatClient(
            timeout_seconds=float(os.environ.get("LLM_TIMEOUT_SECONDS", "60"))
        )
        if llm_backend == "deepseek"
        else DeterministicDemoLlmClient()
    )
    trace_path = Path(os.environ.get("TRACE_PATH", "traces/service/agent_trace.jsonl"))
    pipeline = RagPipeline(
        chunks,
        llm_client,
        PipelineConfig(
            model=os.environ.get("EVALRAG_MODEL", "deterministic-demo"),
            router_name="rule",
            context_strategy=os.environ.get(
                "EVALRAG_CONTEXT_STRATEGY", "source_balanced"
            ),
            context_token_budget=int(os.environ.get("EVALRAG_CONTEXT_TOKEN_BUDGET", "1800")),
            context_mode=os.environ.get("EVALRAG_CONTEXT_MODE", "recent_window"),  # type: ignore[arg-type]
        ),
        trace_path=trace_path,
        router=route_query,
        retriever=retrieve_top_k,
        retrievers={"keyword": retrieve_top_k, "bm25": bm25},
        trace_sink=repository.save_trace,
        context_engine=ContextEngine(),
        context_provider=context_provider,
    )
    agent_runtime = AgentRuntime(
        PipelineRuntimeExecutor(pipeline),
        span_sinks=(JsonlSpanSink(Path("traces/service/runtime_spans.jsonl")),),
    )
    services = AppServices(
        query_service=PipelineQueryService(
            pipeline, repository, memory_service, runtime=agent_runtime
        ),
        repository=repository,
        queue=queue,
        query_timeout_seconds=float(os.environ.get("QUERY_TIMEOUT_SECONDS", "90")),
    )
    return create_app(services)
