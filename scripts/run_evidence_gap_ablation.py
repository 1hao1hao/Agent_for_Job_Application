from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from statistics import mean
import sys
from time import perf_counter


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from intern_rag.agent import check_evidence, load_evidence_config  # noqa: E402
from intern_rag.evaluation import (  # noqa: E402
    calculate_ndcg_at_k,
    calculate_recall_at_k,
    load_chunks_jsonl,
)
from intern_rag.evaluation.knowledge_dataset import load_knowledge_dataset  # noqa: E402
from intern_rag.retrieval import (  # noqa: E402
    EvidenceGapGuidedRetriever,
    build_retriever_from_config,
)
from intern_rag.routing import build_active_router_from_registry  # noqa: E402


BASE_CONFIG = ROOT / "configs/retrieval/adaptive_evidence_rerank_v0.3.json"
RESCUE_CONFIG = ROOT / "configs/retrieval/adaptive_evidence_gap_v0.3.json"
GATE_CONFIG = ROOT / "configs/evidence/gate_e2e_v0.3.json"
HISTORICAL_RUN = ROOT / (
    "reports/runs/p1-task2-evidence-rerank-context-v03-dev-20261006/"
    "case_results.jsonl"
)


def main() -> int:
    """运行 current/fixed-multi-route/on-demand-rescue 的 dev-only 对照。"""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-id",
        default="p1-evidence-gap-v03-dev-" + datetime.now(timezone.utc).strftime("%Y%m%d"),
    )
    parser.add_argument(
        "--reuse-reference-run",
        help="复用代码未变化的 current/fixed 逐 Case prediction，只重跑候选策略。",
    )
    args = parser.parse_args()
    chunks = load_chunks_jsonl(ROOT / "data/processed/chunks/evalrag_v0.3.jsonl")
    cases = [
        item for item in load_knowledge_dataset(
            ROOT / "data/evaluation/evalrag_v0.3.jsonl"
        )
        if item.split == "dev"
    ]
    router = build_active_router_from_registry(
        ROOT / "configs/routing/router_registry_v0.1.json"
    )
    base_config = _read_json(BASE_CONFIG)
    rescue_config = _read_json(RESCUE_CONFIG)
    base = build_retriever_from_config(base_config)
    rescue = build_retriever_from_config(rescue_config)
    if not isinstance(rescue, EvidenceGapGuidedRetriever):
        raise TypeError("rescue config did not build EvidenceGapGuidedRetriever")
    gate = load_evidence_config(GATE_CONFIG)
    historical_misses = _historical_miss_ids()

    rows = {}
    for name, selected, mode in (
        ("current_adaptive", base, "current"),
        ("fixed_multi_route", rescue, "fixed"),
        ("on_demand_rescue", rescue, "rescue"),
    ):
        if args.reuse_reference_run and name != "on_demand_rescue":
            reused = (
                ROOT / "reports/ablations" / args.reuse_reference_run
                / f"{name}_case_results.jsonl"
            )
            rows[name] = _read_jsonl(reused)
            print(f"reused {name} from {reused.relative_to(ROOT)}", flush=True)
            continue
        print(f"running {name} ({len(cases)} dev cases)", flush=True)
        rows[name] = _run_strategy(cases, chunks, router, selected, gate, mode)
        print(f"completed {name}", flush=True)
    summaries = {
        name: _summarize(values, historical_misses)
        for name, values in rows.items()
    }
    payload = {
        "run_id": args.run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_version": "evalrag_v0.3",
        "split": "dev",
        "case_count": len(cases),
        "answerable_case_count": sum(item.answerable for item in cases),
        "base_config": str(BASE_CONFIG.relative_to(ROOT)),
        "rescue_config": str(RESCUE_CONFIG.relative_to(ROOT)),
        "gate_config": str(GATE_CONFIG.relative_to(ROOT)),
        "historical_retrieval_miss_count": len(historical_misses),
        "strategies": summaries,
        "differences": _differences(rows)[:20],
        "decision": _decision(summaries),
        "boundary": (
            "dev-only deterministic retrieval ablation; no LLM and no frozen test; "
            "gold ids are used only after prediction for metrics"
        ),
    }
    output = ROOT / "reports/ablations" / args.run_id
    output.mkdir(parents=True, exist_ok=True)
    for name, values in rows.items():
        _write_jsonl(output / f"{name}_case_results.jsonl", values)
    _write_json(output / "summary.json", payload)
    (output / "report.md").write_text(_report(payload), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def _run_strategy(cases, chunks, router, retriever, gate, mode):
    rows = []
    for case in cases:
        route = router(case.query)
        source_types = set(route.routed_sources) or None
        started = perf_counter()
        if mode == "fixed":
            results = retriever.retrieve_fixed_multi_route(
                case.query, chunks, top_k=5
            )
            trace = {
                "rescue_invoked": True,
                "rescue_call_count": 2 + int(retriever.graph_retriever is not None),
                "rescue_paths": ["bm25", "dense", "graph_path"],
            }
        else:
            results = retriever(
                case.query, chunks, top_k=5, source_types=source_types
            )
            trace = _trace(retriever)
            if mode == "rescue":
                requirement = trace.get("evidence_requirement", {})
                decision = check_evidence(
                    route,
                    results,
                    retriever_name="adaptive",
                    retry_count=0,
                    max_retries=1,
                    config=gate,
                    evidence_requirement=(
                        requirement if isinstance(requirement, dict) else {}
                    ),
                    retrieval_trace=trace,
                )
                if decision.status == "retryable":
                    results = retriever.rescue(
                        case.query,
                        chunks,
                        initial_results=results,
                        top_k=5,
                        evidence_requirement=(
                            requirement if isinstance(requirement, dict) else {}
                        ),
                        initial_trace=trace,
                    )
                    trace = _trace(retriever)
        latency = (perf_counter() - started) * 1000
        ids = [item.chunk_id for item in results]
        relevant = list(case.relevant_chunk_ids)
        observed_edges = _observed_edges(results)
        rows.append({
            "case_id": case.case_id,
            "category": case.category,
            "query": case.query,
            "answerable": case.answerable,
            "routed_sources": sorted(source_types or ()),
            "relevant_chunk_ids": relevant,
            "retrieved_chunk_ids": ids,
            "recall_at_5": (
                calculate_recall_at_k(ids, relevant, 5) if case.answerable else None
            ),
            "mrr": _reciprocal_rank(ids, relevant) if case.answerable else None,
            "ndcg_at_5": (
                calculate_ndcg_at_k(ids, relevant, 5) if case.answerable else None
            ),
            "path_complete": (
                set(case.graph_edge_ids).issubset(observed_edges)
                if case.graph_edge_ids else None
            ),
            "latency_ms": latency,
            "rescue_invoked": bool(trace.get("rescue_invoked", False)),
            "extra_retrieval_calls": int(trace.get("rescue_call_count", 0)),
            "rescue_paths": list(trace.get("rescue_paths", [])),
            "evidence_gaps": list(trace.get("evidence_gaps", [])),
            "selection_trace": trace,
        })
    return rows


def _summarize(rows, historical_misses):
    answerable = [item for item in rows if item["answerable"]]
    relation = [
        item for item in rows if item["category"] in {"two_hop", "three_hop"}
    ]
    recovered = sorted(
        item["case_id"] for item in answerable
        if item["case_id"] in historical_misses and float(item["recall_at_5"] or 0) > 0
    )
    latencies = [float(item["latency_ms"]) for item in rows]
    return {
        "metrics": {
            "recall_at_5": mean(float(item["recall_at_5"]) for item in answerable),
            "mrr": mean(float(item["mrr"]) for item in answerable),
            "ndcg_at_5": mean(float(item["ndcg_at_5"]) for item in answerable),
            "path_completeness": mean(
                float(bool(item["path_complete"])) for item in relation
            ),
        },
        "latency_ms": {"p50": _percentile(latencies, .50), "p95": _percentile(latencies, .95)},
        "extra_retrieval_call_rate": mean(
            float(int(item["extra_retrieval_calls"]) > 0) for item in rows
        ),
        "mean_extra_retrieval_calls": mean(
            int(item["extra_retrieval_calls"]) for item in rows
        ),
        "historical_miss_recovered_count": len(recovered),
        "historical_miss_recovered_ids": recovered,
        "rescue_path_counts": dict(Counter(
            path for item in rows for path in item["rescue_paths"]
        )),
    }


def _differences(rows):
    indexed = {
        name: {item["case_id"]: item for item in values}
        for name, values in rows.items()
    }
    output = []
    for case_id, base in indexed["current_adaptive"].items():
        if not base["answerable"]:
            continue
        rescue = indexed["on_demand_rescue"][case_id]
        delta = float(rescue["recall_at_5"] or 0) - float(base["recall_at_5"] or 0)
        mrr_delta = float(rescue["mrr"] or 0) - float(base["mrr"] or 0)
        if delta or mrr_delta or rescue["rescue_invoked"]:
            output.append({
                "case_id": case_id,
                "category": base["category"],
                "query": base["query"],
                "recall_at_5_delta": delta,
                "mrr_delta": mrr_delta,
                "rescue_invoked": rescue["rescue_invoked"],
                "rescue_paths": rescue["rescue_paths"],
                "evidence_gaps": rescue["evidence_gaps"],
                "base_ids": base["retrieved_chunk_ids"],
                "rescue_ids": rescue["retrieved_chunk_ids"],
            })
    return sorted(
        output,
        key=lambda item: (
            -abs(float(item["recall_at_5_delta"])) - abs(float(item["mrr_delta"])),
            item["case_id"],
        ),
    )


def _decision(summaries):
    base = summaries["current_adaptive"]
    rescue = summaries["on_demand_rescue"]
    quality_gain = (
        rescue["metrics"]["recall_at_5"] > base["metrics"]["recall_at_5"]
        or rescue["metrics"]["mrr"] > base["metrics"]["mrr"]
        or rescue["metrics"]["path_completeness"] > base["metrics"]["path_completeness"]
    )
    return {
        "worth_real_e2e": quality_gain,
        "selected": "on_demand_rescue" if quality_gain else "current_adaptive",
        "reason": (
            "deterministic dev retrieval/path metric improved"
            if quality_gain else "no deterministic retrieval/path gain; keep current adaptive"
        ),
    }


def _historical_miss_ids():
    if not HISTORICAL_RUN.exists():
        return set()
    return {
        str(item["case_id"])
        for item in _read_jsonl(HISTORICAL_RUN)
        if item.get("terminal_stage") == "retrieval_miss"
    }


def _observed_edges(results):
    edges = set()
    for result in results:
        raw = result.details.get("graph_edge_ids")
        if isinstance(raw, str):
            edges.update(item for item in raw.split("|") if item)
    return edges


def _trace(retriever):
    getter = getattr(retriever, "get_last_trace", None)
    value = getter() if callable(getter) else {}
    return value if isinstance(value, dict) else {}


def _reciprocal_rank(ids, relevant):
    expected = set(relevant)
    return next((1.0 / rank for rank, item in enumerate(ids, 1) if item in expected), 0.0)


def _percentile(values, quantile):
    ordered = sorted(values)
    return ordered[max(0, int(len(ordered) * quantile + .999999) - 1)] if ordered else 0.0


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_json(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _report(payload):
    lines = [
        "# Evidence-Gap Guided Retrieval Dev Ablation", "",
        "- Dataset: `evalrag_v0.3`; split: `dev`; 160 cases / 120 answerable.",
        "- Gold ids are used only for offline metrics; selector and rescue never receive labels.",
        "- No LLM and no frozen test in this stage.", "",
        "| Strategy | Recall@5 | MRR | NDCG@5 | Path completeness | P95 ms | Extra-call rate | Historical miss recovered |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, value in payload["strategies"].items():
        metrics = value["metrics"]
        lines.append(
            f"| {name} | {metrics['recall_at_5']:.4f} | {metrics['mrr']:.4f} | "
            f"{metrics['ndcg_at_5']:.4f} | {metrics['path_completeness']:.4f} | "
            f"{value['latency_ms']['p95']:.2f} | {value['extra_retrieval_call_rate']:.2%} | "
            f"{value['historical_miss_recovered_count']} |"
        )
    lines.extend(["", "## Decision", "", f"- {payload['decision']['reason']}.", "", "## Representative differences", ""])
    for item in payload["differences"][:10]:
        lines.append(
            f"- `{item['case_id']}` ({item['category']}): Recall@5 "
            f"{item['recall_at_5_delta']:+.3f}, MRR {item['mrr_delta']:+.3f}, "
            f"paths={item['rescue_paths']}."
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
