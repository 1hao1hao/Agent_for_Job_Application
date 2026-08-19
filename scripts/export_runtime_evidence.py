from __future__ import annotations

import json
from pathlib import Path
from statistics import median
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from intern_rag.agent import Citation, RagRequest, RagResponse  # noqa: E402
from intern_rag.runtime import AgentRuntime, RunContext  # noqa: E402
from intern_rag.runtime.langgraph_adapter import build_langgraph_app  # noqa: E402
from intern_rag.tracing import AgentTrace  # noqa: E402


class FixtureExecutor:
    """生成 answered/retry/error 三种脱敏固定 Trace。"""

    def __init__(self, case_type: str) -> None:
        self.case_type = case_type

    def execute(self, request: RagRequest) -> tuple[RagResponse, AgentTrace]:
        citation = Citation("chunk-demo", "data/public/demo.md", "jd", "脱敏证据", 1, 0.9)
        if self.case_type == "error":
            status, answer, citations, error_type = "error", "模型输出格式错误。", [], "llm_format_error"
            attempts = [
                {"attempt": 1, "type": "generation", "status": "error", "reason": "llm_format_error"},
                {"attempt": 2, "type": "format_retry", "status": "error", "reason": "llm_format_error"},
            ]
        elif self.case_type == "retry":
            status, answer, citations, error_type = "answered", "扩源后依据证据回答。", [citation], None
            attempts = [
                {"attempt": 1, "type": "retrieval", "status": "retry", "reason": "missing_required_sources"},
                {"attempt": 2, "type": "source_retry", "status": "succeeded", "reason": "broaden_sources"},
                {"attempt": 3, "type": "generation", "status": "succeeded"},
            ]
        else:
            status, answer, citations, error_type = "answered", "依据证据回答。", [citation], None
            attempts = [{"attempt": 1, "type": "generation", "status": "succeeded"}]
        response = RagResponse(request.request_id, f"trace-{self.case_type}", answer, citations, ["jd"], status, 4.0, error_type)
        trace = AgentTrace(
            request_id=request.request_id, query="[redacted]", intent="analyze_jd",
            routed_sources=["jd"], retrieved_chunks=[{"chunk_id": "chunk-demo"}],
            latency_ms={"routing": 0.1, "retrieval": 1.0, "evidence": 0.1, "context": 0.1, "generation": 2.0, "validation": 0.1},
            trace_id=response.trace_id, citations=[item.to_dict() for item in citations],
            answer="[redacted]", response_status=status,
            error_type=error_type or "none", error_message="controlled format error" if error_type else "",
            attempts=attempts, token_usage={"input_tokens": 20, "output_tokens": 8},
        )
        return response, trace


def main() -> int:
    """导出三条脱敏 span Trace、故障矩阵和 LangGraph 固定 Case 对照。"""

    trace_dir = ROOT / "traces/sanitized_examples"
    trace_dir.mkdir(parents=True, exist_ok=True)
    for case_type in ("answered", "retry", "error"):
        request = RagRequest("[redacted]", request_id=f"request-{case_type}")
        execution = AgentRuntime(FixtureExecutor(case_type)).execute(
            request,
            RunContext(
                run_id=f"p1-d6-{case_type}", request_id=request.request_id,
                entrypoint="cli", config={"fixture": case_type},
                dataset_version="sanitized-fixture-v0.1", model_version="fake-v1",
                prompt_version="runtime-v1", index_version="fixture-v1",
            ),
        )
        payload = {
            "run_context": {
                "run_id": execution.run_context.run_id,
                "entrypoint": execution.run_context.entrypoint,
                "config_fingerprint": execution.run_context.fingerprint,
                "dataset_version": execution.run_context.dataset_version,
                "model_version": execution.run_context.model_version,
            },
            "response": {
                "status": execution.response.status,
                "citation_ids": [item.chunk_id for item in execution.response.citations],
                "error_type": execution.response.error_type,
            },
            "spans": [span.to_dict() for span in execution.spans],
        }
        (trace_dir / f"p1-d6-{case_type}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    direct, graph = _benchmark_langgraph()
    report_dir = ROOT / "reports/runtime/p1-d6-runtime-v01"
    report_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "runtime_version": "agent-runtime-v0.1",
        "fault_matrix": [
            {"fault": "llm_timeout", "expected": "controlled llm_timeout + failed generation span", "test": "tests.integration.test_pipeline.PipelineIntegrationTests.test_timeout_is_recorded", "covered": True},
            {"fault": "invalid_generation_json", "expected": "one format retry then controlled error", "test": "tests.integration.test_pipeline", "covered": True},
            {"fault": "invalid_citation", "expected": "validator rejects citation", "test": "tests.integration.test_pipeline", "covered": True},
            {"fault": "span_sink_failure", "expected": "business response preserved", "test": "test_trace_sink_failure_does_not_replace_business_response", "covered": True},
            {"fault": "checkpoint_interruption", "expected": "resume skips completed side effect", "test": "test_checkpoint_resume_skips_completed_side_effect", "covered": True},
            {"fault": "config_or_artifact_drift", "expected": "discard stale checkpoint and rerun", "test": "test_config_or_artifact_drift_restarts_workflow", "covered": True}
        ],
        "langgraph_comparison": {
            "case_count": 100,
            "state_equal": True,
            "native_runtime_latency_ms": direct,
            "langgraph_latency_ms": graph,
            "decision": "keep native runtime as default; LangGraph adapter adds dependency and overhead without changing this fixed one-node state",
        },
        "sanitized_traces": [str(path.relative_to(ROOT)) for path in sorted(trace_dir.glob("p1-d6-*.json"))],
    }
    (report_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["langgraph_comparison"], ensure_ascii=False, indent=2))
    return 0


def _benchmark_langgraph() -> tuple[dict[str, float], dict[str, float]]:
    handler = lambda state: {**state, "status": "answered", "trace_id": "trace-fixed"}
    graph_app = build_langgraph_app(handler)
    direct_values, graph_values = [], []
    for _ in range(100):
        started = perf_counter(); direct = handler({"query": "fixed"}); direct_values.append((perf_counter() - started) * 1000)
        started = perf_counter(); graph = dict(graph_app.invoke({"query": "fixed"})); graph_values.append((perf_counter() - started) * 1000)
        if direct != graph:
            raise RuntimeError("LangGraph adapter changed fixed-case state")
    return _latency(direct_values), _latency(graph_values)


def _latency(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {"p50": median(ordered), "p95": ordered[int((len(ordered) - 1) * 0.95)]}


if __name__ == "__main__":
    raise SystemExit(main())
