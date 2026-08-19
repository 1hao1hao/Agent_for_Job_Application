from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from intern_rag.agent import FakeLlmClient, PipelineConfig, RagPipeline, RagRequest  # noqa: E402
from intern_rag.evaluation import load_chunks_jsonl  # noqa: E402
from intern_rag.runtime import AgentRuntime, JsonlSpanSink, PipelineRuntimeExecutor, RunContext  # noqa: E402
from intern_rag.retrieval import retrieve_top_k  # noqa: E402
from intern_rag.routing import route_query  # noqa: E402


def main() -> int:
    """从 CLI 经同一 AgentRuntime 运行离线 Fake Case，输出响应与 span 路径。"""

    parser = argparse.ArgumentParser()
    parser.add_argument("query", nargs="?", default="分析岗位要求")
    args = parser.parse_args()
    chunks = load_chunks_jsonl(ROOT / "data/processed/chunks/evalrag_v0.2.jsonl")
    route = route_query(args.query)
    selected = retrieve_top_k(
        args.query, chunks, top_k=5,
        source_types=set(route.routed_sources) if route.routed_sources else None,
    )
    cited_chunk_id = selected[0].chunk_id if selected else chunks[0].id
    client = FakeLlmClient([
        json.dumps({
            "answer": "根据证据可确认存在岗位要求。",
            "cited_chunk_ids": [cited_chunk_id],
            "sufficient": True,
            "reason": "Fake replay fixture",
        }, ensure_ascii=False)
    ])
    pipeline = RagPipeline(chunks, client, PipelineConfig(model="fake-runtime"))
    runtime = AgentRuntime(
        PipelineRuntimeExecutor(pipeline),
        span_sinks=(JsonlSpanSink(ROOT / "traces/runtime/cli_spans.jsonl"),),
    )
    request = RagRequest(args.query)
    execution = runtime.execute(
        request,
        RunContext(request_id=request.request_id, entrypoint="cli", config={"llm": "fake"}),
    )
    print(json.dumps({
        "run_id": execution.run_context.run_id,
        "status": execution.response.status,
        "trace_id": execution.response.trace_id,
        "span_count": len(execution.spans),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
