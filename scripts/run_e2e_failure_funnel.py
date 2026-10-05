from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from intern_rag.agent import (  # noqa: E402
    ContextEngine,
    PipelineConfig,
    RagPipeline,
    RagRequest,
    build_model_gateway_from_config,
    load_evidence_config,
)
from intern_rag.evaluation import load_chunks_jsonl  # noqa: E402
from intern_rag.evaluation.failure_funnel import (  # noqa: E402
    FailureFunnelCase,
    classify_failure_stage,
    summarize_failure_funnel,
)
from intern_rag.evaluation.knowledge_dataset import (  # noqa: E402
    load_knowledge_dataset,
)
from intern_rag.retrieval import build_retriever_from_config  # noqa: E402
from intern_rag.routing import build_active_router_from_registry  # noqa: E402


def main() -> int:
    """运行完整真实 LLM 链路并生成逐 Case 失败漏斗工件。

    输入固定 benchmark split、Retriever/Gate/Router/Gateway 配置；逐条执行真实
    Pipeline，结合保存的 Trace 和 gold labels 判断终止阶段；输出 case results、
    failures、summary、report 与请求级 traces。该脚本只读 benchmark 标签进行离线
    评分，不会把 relevant ids 注入在线 Pipeline。
    """

    args = _parse_args()
    if not os.environ.get("DEEPSEEK_API_KEY", "").strip():
        print("ERROR: DEEPSEEK_API_KEY is not configured")
        return 2
    run_dir = ROOT / "reports" / "runs" / args.run_id
    if run_dir.exists():
        raise FileExistsError(f"run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)

    dataset_path = ROOT / "data/evaluation/evalrag_v0.3.jsonl"
    chunks_path = ROOT / "data/processed/chunks/evalrag_v0.3.jsonl"
    retriever_path = ROOT / args.retriever_config
    evidence_path = ROOT / args.evidence_config
    router_path = ROOT / "configs/routing/router_registry_v0.1.json"
    gateway_path = ROOT / "configs/model_gateway/gateway_v0.1.json"
    chunks = load_chunks_jsonl(chunks_path)
    cases = [
        case for case in load_knowledge_dataset(dataset_path)
        if case.split == args.split
    ]
    retriever_raw = _absolute_artifact_paths(
        json.loads(retriever_path.read_text(encoding="utf-8"))
    )
    retriever = build_retriever_from_config(retriever_raw)
    router = build_active_router_from_registry(router_path)
    gateway = build_model_gateway_from_config(gateway_path)
    pipeline = RagPipeline(
        chunks,
        gateway,
        PipelineConfig(
            model="model-gateway-v0.1",
            temperature=0.0,
            prompt_version=args.prompt_version,
            context_max_chars=4000,
            context_strategy="source_balanced",
            router_name="hybrid",
            evidence=load_evidence_config(evidence_path),
            max_source_retries=1,
            max_format_retries=1,
            context_token_budget=1800,
            context_mode="adaptive",
            config_versions={
                "retriever": str(retriever_raw.get("config_version", "unknown")),
                "evidence": load_evidence_config(evidence_path).config_version,
            },
        ),
        trace_path=run_dir / "traces.jsonl",
        router=router,
        routers={"hybrid": router},
        retriever=retriever,
        retrievers={"adaptive": retriever},
        context_engine=ContextEngine(),
    )

    rows: list[FailureFunnelCase] = []
    case_path = run_dir / "case_results.jsonl"
    for index, case in enumerate(cases, 1):
        response = pipeline.run(
            RagRequest(
                query=case.query,
                request_id=case.case_id,
                top_k=5,
                retriever="adaptive",
            )
        )
        trace = pipeline.last_trace
        if trace is None:
            raise RuntimeError(f"pipeline did not expose trace for {case.case_id}")
        row = classify_failure_stage(case, response, trace)
        rows.append(row)
        with case_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")
        print(
            f"PROGRESS {index}/{len(cases)} {case.case_id} "
            f"{row.terminal_stage}",
            flush=True,
        )

    summary = summarize_failure_funnel(rows)
    summary.update({
        "run_id": args.run_id,
        "dataset_version": "evalrag_v0.3",
        "split": args.split,
        "retriever_config": args.retriever_config,
        "retriever_config_version": retriever_raw.get("config_version", "unknown"),
        "evidence_config": args.evidence_config,
        "evidence_config_version": load_evidence_config(evidence_path).config_version,
        "prompt_version": args.prompt_version,
        "category_counts": dict(Counter(row.category for row in rows)),
        "model_gateway": _gateway_usage(run_dir / "traces.jsonl"),
    })
    failures = [row for row in rows if row.terminal_stage != "answered"]
    (run_dir / "failures.jsonl").write_text(
        "".join(json.dumps(row.to_dict(), ensure_ascii=False) + "\n" for row in failures),
        encoding="utf-8",
    )
    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    run_config = {
        "dataset": str(dataset_path.relative_to(ROOT)),
        "chunks": str(chunks_path.relative_to(ROOT)),
        "split": args.split,
        "top_k": 5,
        "router": str(router_path.relative_to(ROOT)),
        "retriever": args.retriever_config,
        "evidence": args.evidence_config,
        "gateway": str(gateway_path.relative_to(ROOT)),
        "prompt_version": args.prompt_version,
        "temperature": 0.0,
        "context_strategy": "source_balanced",
        "context_mode": "adaptive",
        "context_token_budget": 1800,
    }
    (run_dir / "run_config.json").write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (run_dir / "report.md").write_text(
        _render_report(summary), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run EvalRAG E2E failure funnel")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument(
        "--retriever-config",
        default="configs/retrieval/adaptive_v2_v0.3.json",
    )
    parser.add_argument(
        "--evidence-config",
        default="configs/evidence/gate_e2e_v0.3.json",
    )
    parser.add_argument("--prompt-version", default="p1-e2e-funnel-v2")
    return parser.parse_args()


def _absolute_artifact_paths(raw: dict[str, object]) -> dict[str, object]:
    config = dict(raw)
    for key in ("bm25_index_path", "index_dir", "graph_index_path"):
        value = config.get(key)
        if isinstance(value, str) and value:
            config[key] = str(ROOT / value)
    return config


def _gateway_usage(trace_path: Path) -> dict[str, object]:
    calls: list[dict[str, object]] = []
    for line in trace_path.read_text(encoding="utf-8").splitlines():
        trace = json.loads(line)
        for attempt in trace.get("attempts", []):
            gateway = attempt.get("model_gateway")
            if isinstance(gateway, dict) and gateway:
                calls.append(gateway)
    tokens = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    estimated_cost = 0.0
    provider_attempts = 0
    for call in calls:
        estimated_cost += float(call.get("estimated_cost_usd", 0.0))
        for attempt in call.get("attempts", []):
            if not isinstance(attempt, dict):
                continue
            provider_attempts += 1
            usage = attempt.get("tokens")
            if isinstance(usage, dict):
                for key in tokens:
                    tokens[key] += int(usage.get(key, 0))
    return {
        "generation_calls": len(calls),
        "provider_attempts": provider_attempts,
        "tokens": tokens,
        "estimated_cost_usd": estimated_cost,
    }


def _render_report(summary: dict[str, object]) -> str:
    stages = summary["terminal_stage_counts"]
    answerable_stages = summary["answerable_terminal_stage_counts"]
    lines = [
        f"# {summary['run_id']}",
        "",
        f"- Dataset: `evalrag_v0.3/{summary['split']}`",
        f"- Cases: {summary['case_count']} (answerable={summary['answerable_case_count']}, unanswerable={summary['unanswerable_case_count']})",
        f"- Retriever: `{summary['retriever_config_version']}`",
        f"- Evidence Gate: `{summary['evidence_config_version']}`",
        f"- Prompt: `{summary['prompt_version']}`",
        "",
        "## Failure Funnel",
        "",
        "| Terminal stage | All | Answerable |",
        "|---|---:|---:|",
    ]
    for stage in (
        "router_abstain", "retrieval_miss", "gate_reject",
        "context_evidence_dropped", "generator_abstain", "citation_invalid",
        "model_error", "answered",
    ):
        lines.append(
            f"| {stage} | {dict(stages).get(stage, 0)} | "
            f"{dict(answerable_stages).get(stage, 0)} |"
        )
    lines.extend([
        "",
        "## Metrics",
        "",
        f"- Answerable E2E Success: {float(summary['answerable_e2e_success']):.2%}",
        f"- Unexpected Abstention Rate: {float(summary['unexpected_abstention_rate']):.2%}",
        f"- Unanswerable Abstention Accuracy: {float(summary['unanswerable_abstention_accuracy']):.2%}",
        f"- Citation Validity (answered): {float(summary['citation_validity_answered']):.2%}",
        f"- Error count: {summary['error_count']}",
        "",
        "逐 Case 证据流见 `case_results.jsonl`，非 answered Case 见 `failures.jsonl`。",
    ])
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
