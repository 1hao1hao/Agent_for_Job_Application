from __future__ import annotations

import json
import os
from pathlib import Path
from statistics import mean
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from intern_rag.agent import (  # noqa: E402
    EvidenceConfig,
    PipelineConfig,
    RagPipeline,
    RagRequest,
    build_model_gateway_from_config,
)
from intern_rag.evaluation import load_chunks_jsonl  # noqa: E402
from intern_rag.evaluation.knowledge_dataset import load_knowledge_dataset  # noqa: E402
from intern_rag.evaluation.metrics import calculate_key_point_coverage, calculate_recall_at_k  # noqa: E402
from intern_rag.retrieval import build_retriever_from_config  # noqa: E402
from intern_rag.routing import build_active_router_from_registry  # noqa: E402


RUN_ID = "p1-d7-v03-frozen-20260816-live-e2e"


def main() -> int:
    """在冻结 v0.3/test 上运行最终 Pipeline，并保存逐 Case 响应与 Trace。

    该 Run 计算回答状态、引用合法性、检索 Recall 与词面 Key-Point Coverage；
    未运行 Claim-Level Grounding，因此不会输出 Unsupported Answer Rate。
    """

    if not os.environ.get("DEEPSEEK_API_KEY", "").strip():
        print("ERROR: DEEPSEEK_API_KEY is not configured")
        return 2
    run_dir = ROOT / "reports/runs" / RUN_ID
    if run_dir.exists():
        raise FileExistsError(f"frozen E2E run already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    trace_path = run_dir / "traces.jsonl"
    chunks = load_chunks_jsonl(ROOT / "data/processed/chunks/evalrag_v0.3.jsonl")
    cases = [case for case in load_knowledge_dataset(ROOT / "data/evaluation/evalrag_v0.3.jsonl") if case.split == "test"]
    retriever_raw = json.loads((ROOT / "configs/retrieval/graph_adaptive_final_v0.3.json").read_text(encoding="utf-8"))
    retriever = build_retriever_from_config(retriever_raw)
    router = build_active_router_from_registry(ROOT / "configs/routing/router_registry_v0.1.json")
    gateway = build_model_gateway_from_config(ROOT / "configs/model_gateway/gateway_v0.1.json")
    pipeline = RagPipeline(
        chunks,
        gateway,
        PipelineConfig(
            model="model-gateway-v0.1",
            temperature=0.0,
            prompt_version="p1-final-json-v1",
            context_max_chars=4000,
            router_name="hybrid",
            evidence=EvidenceConfig(min_results=1, require_source_coverage=True),
            max_source_retries=1,
            max_format_retries=1,
        ),
        trace_path=trace_path,
        router=router,
        routers={"hybrid": router},
        retriever=retriever,
        retrievers={"adaptive": retriever},
    )
    rows = []
    for index, case in enumerate(cases, 1):
        response = pipeline.run(RagRequest(query=case.query, request_id=case.case_id, top_k=5, retriever="adaptive"))
        trace = pipeline.last_trace
        assert trace is not None
        trace_data = trace.to_dict()
        cited = [item.chunk_id for item in response.citations]
        context_ids = list(trace_data["context"].get("used_chunk_ids", []))
        retrieved_ids = list(trace_data["retrieval"].get("chunk_ids", []))
        coverage, covered = calculate_key_point_coverage(response.answer, case.expected_points)
        citation_valid = bool(cited) and all(item in context_ids for item in cited) if response.status == "answered" else response.status == "insufficient_evidence"
        success = (
            response.status == "answered" and citation_valid and coverage >= 0.5
            if case.answerable
            else response.status == "insufficient_evidence"
        )
        row = {
            "case_id": case.case_id,
            "category": case.category,
            "answerable": case.answerable,
            "status": response.status,
            "answer": response.answer,
            "citation_ids": cited,
            "citation_valid": citation_valid,
            "key_point_coverage_lexical": coverage,
            "covered_points": covered,
            "retrieval_recall_at_5": calculate_recall_at_k(retrieved_ids, case.relevant_chunk_ids, 5) if case.answerable else None,
            "end_to_end_success_without_grounding": success,
            "latency_ms": response.latency_ms,
            "error_type": response.error_type,
            "trace_id": response.trace_id,
        }
        rows.append(row)
        with (run_dir / "case_results.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"PROGRESS {index}/{len(cases)} {case.case_id} {response.status}", flush=True)
    answerable = [row for row in rows if row["answerable"]]
    unanswerable = [row for row in rows if not row["answerable"]]
    traces = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    gateway_calls = [
        dict(attempt.get("model_gateway", {}))
        for trace in traces
        for attempt in trace.get("attempts", [])
        if attempt.get("model_gateway")
    ]
    summary = {
        "run_id": RUN_ID,
        "dataset_version": "evalrag_v0.3",
        "split": "test",
        "case_count": len(rows),
        "answered_count": sum(row["status"] == "answered" for row in rows),
        "insufficient_count": sum(row["status"] == "insufficient_evidence" for row in rows),
        "error_count": sum(row["status"] == "error" for row in rows),
        "citation_validity": mean(float(row["citation_valid"]) for row in rows),
        "lexical_key_point_coverage": mean(float(row["key_point_coverage_lexical"]) for row in answerable),
        "abstention_accuracy": mean(float(row["status"] == "insufficient_evidence") for row in unanswerable),
        "answerable_success_without_grounding": mean(float(row["end_to_end_success_without_grounding"]) for row in answerable),
        "overall_success_without_grounding": mean(float(row["end_to_end_success_without_grounding"]) for row in rows),
        "latency_ms": {"p50": _percentile([float(row["latency_ms"]) for row in rows], 0.50), "p95": _percentile([float(row["latency_ms"]) for row in rows], 0.95)},
        "model_gateway": {
            "call_count": len(gateway_calls),
            "fallback_count": sum(bool(call.get("fallback_used")) for call in gateway_calls),
            "tokens": _sum_gateway_tokens(gateway_calls),
        },
        "limitations": [
            "Key-Point Coverage is lexical in this frozen E2E run.",
            "Claim-Level Grounding was not run, so Unsupported Answer Rate and full grounded E2E success are unavailable.",
            "The same test predictions will not be regenerated for tuning.",
        ],
    }
    (run_dir / "run_config.json").write_text(json.dumps({"release": "configs/final/p1_v0.3.json", "retriever": "configs/retrieval/graph_adaptive_final_v0.3.json", "router": "configs/routing/router_registry_v0.1.json", "gateway": "configs/model_gateway/gateway_v0.1.json"}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    failures = [row for row in rows if not row["end_to_end_success_without_grounding"]]
    (run_dir / "failures.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in failures), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _sum_gateway_tokens(calls: list[dict[str, object]]) -> dict[str, int]:
    totals = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    for call in calls:
        for attempt in call.get("attempts", []):
            tokens = attempt.get("tokens") if isinstance(attempt, dict) else None
            if isinstance(tokens, dict):
                for key in totals:
                    totals[key] += int(tokens.get(key, 0))
    return totals


def _percentile(values: list[float], ratio: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, int(len(ordered) * ratio + 0.999999) - 1)] if ordered else 0.0


if __name__ == "__main__":
    raise SystemExit(main())
