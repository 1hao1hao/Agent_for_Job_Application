from __future__ import annotations

import json
import os
from pathlib import Path
from threading import Lock

from intern_rag.agent import PipelineConfig, RagPipeline, load_evidence_config
from intern_rag.agent.generation import DeepSeekChatClient
from intern_rag.evaluation import load_chunks_jsonl
from intern_rag.retrieval import build_retriever_from_config, retrieve_top_k
from intern_rag.routing import route_query
from intern_rag.serving.api import AppServices, create_app
from intern_rag.serving.runtime import DeterministicDemoLlmClient
from intern_rag.serving.service import PipelineQueryService


class LoadTestRepository:
    """压测专用线程安全 adapter，只隔离 PostgreSQL I/O，不替换 Query Pipeline。"""

    def __init__(self) -> None:
        self._lock = Lock()
        self.request_count = 0
        self.trace_count = 0

    def ping(self) -> bool:
        return True

    def save_request(self, request, response) -> None:
        del request, response
        with self._lock:
            self.request_count += 1

    def save_trace(self, trace) -> None:
        del trace
        with self._lock:
            self.trace_count += 1

    def get_trace(self, trace_id: str):
        del trace_id
        return None


class LoadTestQueue:
    """健康检查使用的可用队列占位；压测不创建 Evaluation Job。"""

    @staticmethod
    def ping() -> bool:
        return True


def create_loadtest_app():
    """构造保留真实 Router/BM25/Gate/Context 的轻量压测服务。

    deterministic 模式只替换外部 LLM 和 PostgreSQL/Redis；real 模式使用现有
    DeepSeek adapter，且仍要求密钥仅从环境变量读取。
    """

    root = Path(os.environ.get("EVALRAG_PROJECT_ROOT", ".")).resolve()
    dataset_version = os.environ.get("EVALRAG_DATASET_VERSION", "evalrag_v0.3")
    chunks = load_chunks_jsonl(root / "data/processed/chunks" / f"{dataset_version}.jsonl")
    config_path = root / "configs/retrieval/bm25_v0.3.json"
    retriever_config = json.loads(config_path.read_text(encoding="utf-8"))
    retriever = build_retriever_from_config(retriever_config)
    evidence_path = root / "configs/evidence/gate_calibrated_v0.3.json"
    mode = os.environ.get("EVALRAG_LOADTEST_MODE", "deterministic")
    if mode == "real" and not os.environ.get("DEEPSEEK_API_KEY", "").strip():
        raise RuntimeError("DEEPSEEK_API_KEY is required for real load-test mode")
    llm_client = DeepSeekChatClient() if mode == "real" else DeterministicDemoLlmClient()
    repository = LoadTestRepository()
    trace_path = Path(os.environ.get("EVALRAG_LOADTEST_TRACE", "/dev/null"))
    pipeline = RagPipeline(
        chunks,
        llm_client,
        PipelineConfig(
            model=os.environ.get("EVALRAG_MODEL", "deterministic-demo"),
            router_name="rule",
            context_strategy="source_balanced",
            evidence=load_evidence_config(evidence_path),
        ),
        trace_path=trace_path,
        router=route_query,
        retriever=retrieve_top_k,
        retrievers={"keyword": retrieve_top_k, "bm25": retriever},
        trace_sink=repository.save_trace,
    )
    return create_app(
        AppServices(
            query_service=PipelineQueryService(pipeline, repository),  # type: ignore[arg-type]
            repository=repository,  # type: ignore[arg-type]
            queue=LoadTestQueue(),  # type: ignore[arg-type]
            query_timeout_seconds=float(os.environ.get("QUERY_TIMEOUT_SECONDS", "90")),
        )
    )
