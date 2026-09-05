from __future__ import annotations

import json
import os
from pathlib import Path
import re
from typing import Mapping

from intern_rag.agent import (
    ContextEngine,
    ContextInputs,
    ContextSignalExtractor,
    PipelineConfig,
    RagPipeline,
    load_evidence_config,
)
from intern_rag.agent.generation import DeepSeekChatClient
from intern_rag.evaluation import load_chunks_jsonl
from intern_rag.ingestion import Chunk
from intern_rag.persistence import (
    DenseMemoryEmbeddingProvider,
    PostgresRepository,
    RedisRecentHistoryCache,
    SessionMemoryService,
)
from intern_rag.retrieval import (
    RetrievalResult,
    Retriever,
    build_retriever_from_config,
    load_dense_index,
    retrieve_top_k,
)
from intern_rag.routing import route_query
from intern_rag.serving.api import AppServices, create_app
from intern_rag.serving.service import PipelineQueryService
from intern_rag.worker import RedisJobQueue
from intern_rag.runtime import (
    AgentRuntime,
    JsonlSpanSink,
    PipelineRuntimeExecutor,
    RunContext,
    SavedRun,
)
from intern_rag.agent.generation import LlmClient


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


class RuntimeRetrieverBinding:
    """为在线 Retriever 补充实际配置和受控降级信息。"""

    def __init__(
        self,
        retriever: Retriever,
        *,
        configured_name: str,
        effective_name: str,
        config_version: str,
        config_path: Path,
        fallback_reason: str | None = None,
    ) -> None:
        self.retriever = retriever
        self.configured_name = configured_name
        self.effective_name = effective_name
        self.config_version = config_version
        self.config_path = config_path
        self.fallback_reason = fallback_reason

    def __call__(
        self,
        query: str,
        chunks: list[Chunk],
        top_k: int = 5,
        source_types: set[str] | None = None,
    ) -> list[RetrievalResult]:
        return self.retriever(query, chunks, top_k=top_k, source_types=source_types)

    def get_last_trace(self) -> dict[str, object]:
        """合并底层策略决策与在线配置身份，避免静默降级。"""

        getter = getattr(self.retriever, "get_last_trace", None)
        inner = dict(getter()) if callable(getter) else {}
        return {
            **inner,
            "runtime_configured_retriever": self.configured_name,
            "runtime_effective_retriever": self.effective_name,
            "runtime_config_version": self.config_version,
            "runtime_config_path": str(self.config_path),
            "runtime_fallback_reason": self.fallback_reason,
        }


def load_runtime_retriever(
    project_root: Path,
    config_path: Path | None = None,
) -> tuple[RuntimeRetrieverBinding, dict[str, object]]:
    """加载在线锁定 Retriever；失败时只接受显式 fallback。

    相对工件路径统一基于项目根目录解析。默认使用 Adaptive v2；若模型、Dense
    或 Graph 工件缺失会 fail-fast。设置 `EVALRAG_RETRIEVER_FALLBACK_CONFIG`
    后才允许降级，并在返回 adapter 的 Trace 中保存原因。
    """

    selected_path = config_path or Path(
        os.environ.get(
            "EVALRAG_RETRIEVER_CONFIG",
            str(project_root / "configs/retrieval/adaptive_v2_v0.3.json"),
        )
    )
    selected_path = _absolute_path(project_root, selected_path)
    configured = _load_retriever_config(project_root, selected_path)
    configured_name = str(configured.get("retriever_name", "unknown"))
    try:
        retriever = build_retriever_from_config(configured)
        return RuntimeRetrieverBinding(
            retriever,
            configured_name=configured_name,
            effective_name=configured_name,
            config_version=str(configured.get("config_version", "unversioned")),
            config_path=selected_path,
        ), configured
    except Exception as error:
        fallback_value = os.environ.get("EVALRAG_RETRIEVER_FALLBACK_CONFIG", "").strip()
        if not fallback_value:
            raise RuntimeError(
                f"failed to build configured retriever '{configured_name}' from {selected_path}"
            ) from error
        fallback_path = _absolute_path(project_root, Path(fallback_value))
        fallback = _load_retriever_config(project_root, fallback_path)
        effective_name = str(fallback.get("retriever_name", "unknown"))
        fallback_retriever = build_retriever_from_config(fallback)
        reason = f"{type(error).__name__}: configured retriever unavailable"
        return RuntimeRetrieverBinding(
            fallback_retriever,
            configured_name=configured_name,
            effective_name=effective_name,
            config_version=str(fallback.get("config_version", "unversioned")),
            config_path=fallback_path,
            fallback_reason=reason,
        ), fallback


def _load_retriever_config(project_root: Path, path: Path) -> dict[str, object]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    for key in ("bm25_index_path", "index_dir", "graph_index_path"):
        value = raw.get(key)
        if isinstance(value, str) and value:
            raw[key] = str(_absolute_path(project_root, Path(value)))
    return raw


def _absolute_path(project_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else project_root / path


def _retriever_artifact_refs(config: Mapping[str, object]) -> dict[str, str]:
    return {
        key: str(config[key])
        for key in ("bm25_index_path", "index_dir", "graph_index_path")
        if config.get(key)
    }


def create_runtime_app():
    """从环境变量组装生产 adapter；密钥不进入配置、Trace 或错误响应。"""

    project_root = Path(os.environ.get("EVALRAG_PROJECT_ROOT", ".")).resolve()
    database_url = os.environ["DATABASE_URL"]
    redis_url = os.environ["REDIS_URL"]
    repository = PostgresRepository(database_url, project_root / "migrations")
    repository.initialize()
    queue = RedisJobQueue(redis_url)
    memory_embedding_model = None
    memory_index_dir = Path(
        os.environ.get(
            "EVALRAG_MEMORY_INDEX_DIR",
            str(
                project_root
                / "data/processed/indexes/evalrag_v0.3/bge-small-zh-v1.5"
            ),
        )
    )
    if memory_index_dir.exists():
        _, memory_embedding_model = load_dense_index(memory_index_dir)
    memory_service = SessionMemoryService(
        repository,
        RedisRecentHistoryCache(redis_url),
        DenseMemoryEmbeddingProvider(memory_embedding_model)
        if memory_embedding_model is not None
        else None,
    )
    context_signal_extractor = ContextSignalExtractor(memory_embedding_model)

    def context_provider(request):
        """按 RagRequest 的 user/session scope 获取 Context；无会话时返回空输入。"""

        if request.user_id is None or request.session_id is None:
            return ContextInputs()
        value = memory_service.load_context_for_query(
            request.user_id, request.session_id, request.query
        )
        return ContextInputs(
            profile=value.profile,
            history=value.messages,
            memories=value.memories,
            history_summary=value.summary,
            history_source=value.history_source,
        )

    runtime_retriever, retrieval_config = load_runtime_retriever(project_root)
    configured_dataset = str(
        retrieval_config.get("dataset_version", "evalrag_v0.3")
    )
    dataset_version = os.environ.get("EVALRAG_DATASET_VERSION", configured_dataset)
    if dataset_version != configured_dataset:
        raise ValueError(
            "EVALRAG_DATASET_VERSION must match the configured retriever dataset"
        )
    chunks = load_chunks_jsonl(
        project_root / "data/processed/chunks" / f"{dataset_version}.jsonl"
    )
    bm25_config_path = project_root / "configs/retrieval/bm25_v0.3.json"
    bm25_config = _load_retriever_config(project_root, bm25_config_path)
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
    evidence_path = Path(
        os.environ.get(
            "EVALRAG_EVIDENCE_CONFIG",
            str(project_root / "configs/evidence/gate_calibrated_v0.3.json"),
        )
    )
    evidence_config = load_evidence_config(evidence_path) if evidence_path.exists() else None
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
            context_mode=os.environ.get("EVALRAG_CONTEXT_MODE", "adaptive"),  # type: ignore[arg-type]
            config_versions={
                "retriever": str(
                    retrieval_config.get("config_version", "unversioned")
                )
            },
            **({"evidence": evidence_config} if evidence_config is not None else {}),
        ),
        trace_path=trace_path,
        router=route_query,
        retriever=retrieve_top_k,
        retrievers={
            "keyword": retrieve_top_k,
            "bm25": bm25,
            "adaptive": runtime_retriever,
        },
        trace_sink=repository.save_trace,
        context_engine=ContextEngine(
            semantic_similarity=context_signal_extractor.similarities
        ),
        context_provider=context_provider,
        context_signal_extractor=context_signal_extractor,
    )
    agent_runtime = AgentRuntime(
        PipelineRuntimeExecutor(pipeline),
        span_sinks=(JsonlSpanSink(Path("traces/service/runtime_spans.jsonl")),),
    )
    context_template = RunContext(
        config={
            "retriever_config_path": str(runtime_retriever.config_path),
            "retriever_config_version": runtime_retriever.config_version,
            "retriever_fallback_reason": runtime_retriever.fallback_reason,
        },
        artifact_refs={
            "chunks": str(project_root / "data/processed/chunks" / f"{dataset_version}.jsonl"),
            "retriever_config": str(runtime_retriever.config_path),
            **_retriever_artifact_refs(retrieval_config),
            **({"evidence_config": str(evidence_path)} if evidence_path.exists() else {}),
        },
        dataset_version=dataset_version,
        index_version=runtime_retriever.config_version,
    )
    services = AppServices(
        query_service=PipelineQueryService(
            pipeline,
            repository,
            memory_service,
            runtime=agent_runtime,
            run_context_template=context_template,
        ),
        repository=repository,
        queue=queue,
        query_timeout_seconds=float(os.environ.get("QUERY_TIMEOUT_SECONDS", "90")),
        default_query_retriever="adaptive",
    )
    return create_app(services)


def build_runtime_for_replay(
    saved: SavedRun, llm_client: LlmClient
) -> AgentRuntime:
    """仅从历史 RunContext 和本地工件重建离线 Pipeline。

    当前服务公开支持 rule router 与版本化 Retriever 配置。若历史请求依赖会话记忆或
    未保存的 adapter 配置，函数明确失败，不使用当前默认值猜测历史行为。
    """

    if saved.request.user_id is not None:
        raise ValueError("session memory snapshot is unavailable for deterministic replay")
    config = saved.context.config
    if str(config.get("router", "rule")) != "rule":
        raise ValueError("replay runtime currently supports persisted rule router only")
    chunks_path = Path(saved.context.artifact_refs.get("chunks", ""))
    chunks = load_chunks_jsonl(chunks_path)
    retrievers: dict[str, Retriever] = {"keyword": retrieve_top_k}
    if saved.request.retriever != "keyword":
        config_path = Path(saved.context.artifact_refs.get("retriever_config", ""))
        retriever_config = json.loads(config_path.read_text(encoding="utf-8"))
        legacy_bm25 = saved.context.artifact_refs.get("bm25_index")
        if legacy_bm25:
            retriever_config["bm25_index_path"] = legacy_bm25
        for key in ("bm25_index_path", "index_dir", "graph_index_path"):
            if saved.context.artifact_refs.get(key):
                retriever_config[key] = saved.context.artifact_refs[key]
        retrievers[saved.request.retriever] = build_retriever_from_config(
            retriever_config
        )
    evidence_path = saved.context.artifact_refs.get("evidence_config")
    evidence = load_evidence_config(Path(evidence_path)) if evidence_path else None
    pipeline = RagPipeline(
        chunks,
        llm_client,
        PipelineConfig(
            model=saved.context.model_version,
            temperature=float(config.get("temperature", 0.0)),
            prompt_version=saved.context.prompt_version,
            context_max_chars=int(config.get("context_max_chars", 4000)),
            context_strategy=str(config.get("context_strategy", "rank_prefix")),  # type: ignore[arg-type]
            router_name="rule",
            context_token_budget=int(config.get("context_token_budget", 1800)),
            context_mode=str(config.get("context_mode", "recent_window")),  # type: ignore[arg-type]
            max_source_retries=int(config.get("max_source_retries", 1)),
            max_format_retries=int(config.get("max_format_retries", 1)),
            max_action_steps=int(config.get("max_action_steps", 4)),
            agent_controller_version=str(
                config.get("agent_controller_version", "bounded-agent-controller-v1")
            ),
            **({"evidence": evidence} if evidence is not None else {}),
        ),
        trace_path=Path("/tmp/evalrag_replay_trace.jsonl"),
        router=route_query,
        retriever=retrieve_top_k,
        retrievers=retrievers,
    )
    return AgentRuntime(PipelineRuntimeExecutor(pipeline))
