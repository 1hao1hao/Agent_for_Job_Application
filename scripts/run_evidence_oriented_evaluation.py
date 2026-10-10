"""集中 dev 检索对照及可断点恢复的真实 LLM 小样本对照。"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from intern_rag.agent import (ContextEngine, PipelineConfig, RagPipeline, RagRequest,
                              build_model_gateway_from_config, check_evidence, load_evidence_config)
from intern_rag.evaluation import load_chunks_jsonl
from intern_rag.evaluation.knowledge_dataset import load_knowledge_dataset
from intern_rag.evaluation.failure_funnel import FailureFunnelCase, classify_failure_stage, summarize_failure_funnel
from intern_rag.retrieval.adaptive import AdaptiveRetriever
from intern_rag.retrieval.evidence_gap import EvidenceGapGuidedRetriever, EvidenceGapConfig
from intern_rag.retrieval.factory import build_retriever_from_config, _build_query_analyzer
from intern_rag.routing import build_active_router_from_registry
from run_evidence_gap_ablation import _summarize, _reciprocal_rank, _observed_edges
from intern_rag.evaluation import calculate_recall_at_k, calculate_ndcg_at_k

CONFIG = ROOT / "configs/retrieval/evidence_oriented_v0.3.json"
GATE = ROOT / "configs/evidence/gate_e2e_v0.3.json"
REFERENCE = ROOT / "reports/ablations/p1-evidence-gap-v03-dev-20261009-r2"


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append(path, value):
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + "\n")


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def build_systems():
    """三个策略共享固定模型/索引，分别保留旧 Analyzer 和新规划器。"""
    cfg = json.loads(CONFIG.read_text())
    new = build_retriever_from_config(cfg)
    old_cfg = json.loads((ROOT / "configs/retrieval/adaptive_evidence_gap_v0.3.json").read_text())
    old = AdaptiveRetriever(new.base.retrievers, new.base.scorer,
                            analyzer=_build_query_analyzer(old_cfg),
                            config=new.base.config, graph_retriever=new.base.graph_retriever)
    # 旧配置的重排条件保持原样；共享模型不会改变配置语义。
    from dataclasses import replace
    old.config = replace(old.config, confidence_threshold=0.65, rerank_policy="evidence_need", rerank_need_types=(
        "exact_fact", "semantic_explanation", "multi_source_synthesis"))
    gap = EvidenceGapGuidedRetriever(old, {k: new.base.retrievers[k] for k in ("bm25", "dense")},
                                   graph_retriever=new.graph_retriever,
                                   config=EvidenceGapConfig(candidate_k=20, max_paths=2))
    return {"adaptive": old, "evidence_gap": gap, "evidence_oriented": new}


def evaluate(cases, chunks, router, systems, gate, out):
    """逐策略运行 dev；online 不拿标签，返回后才按 gold 评分并保存断点。"""
    all_rows = {}
    for name, retriever in systems.items():
        path = out / f"{name}_case_results.jsonl"
        rows = read_rows(path)
        completed = {r["case_id"] for r in rows}
        for i, case in enumerate(cases, 1):
            if case.case_id in completed:
                continue
            route = router(case.query)
            start = perf_counter()
            results = retriever(case.query, chunks, 5, set(route.routed_sources) or None)
            trace = retriever.get_last_trace()
            decision = check_evidence(route, results, retriever_name="adaptive", retry_count=0,
                                      max_retries=1, config=gate, retrieval_trace=trace)
            if name != "adaptive" and decision.status == "retryable":
                results = retriever.rescue(case.query, chunks, initial_results=results, top_k=5,
                                          evidence_requirement=trace.get("evidence_requirement", {}), initial_trace=trace)
                trace = retriever.get_last_trace()
                decision = check_evidence(route, results, retriever_name="adaptive", retry_count=1,
                                          max_retries=1, config=gate, retrieval_trace=trace)
            elapsed = (perf_counter() - start) * 1000
            ids = [r.chunk_id for r in results]
            row = {"case_id": case.case_id, "category": case.category, "query": case.query,
                   "answerable": case.answerable, "relevant_chunk_ids": list(case.relevant_chunk_ids),
                   "retrieved_chunk_ids": ids,
                   "recall_at_5": calculate_recall_at_k(ids, case.relevant_chunk_ids, 5) if case.answerable else None,
                   "mrr": _reciprocal_rank(ids, case.relevant_chunk_ids) if case.answerable else None,
                   "ndcg_at_5": calculate_ndcg_at_k(ids, case.relevant_chunk_ids, 5) if case.answerable else None,
                   "path_complete": set(case.graph_edge_ids) <= _observed_edges(results) if case.graph_edge_ids else None,
                   "latency_ms": elapsed, "rescue_invoked": trace.get("rescue_invoked", False),
                   "extra_retrieval_calls": trace.get("rescue_call_count", 0),
                   "rescue_paths": trace.get("rescue_paths", []),
                   "gate_status": decision.status, "selection_trace": trace}
            append(path, row)
            rows.append(row)
            if i % 20 == 0:
                print(f"{name} {i}/{len(cases)}", flush=True)
        all_rows[name] = rows
    historical = {r["case_id"] for r in read_rows(REFERENCE / "on_demand_rescue_case_results.jsonl")
                  if r["answerable"] and not r["recall_at_5"]}
    summaries = {name: _summarize(rows, historical) for name, rows in all_rows.items()}
    before = {r["case_id"]: r for r in all_rows["evidence_gap"]}
    differences = [{"case_id": r["case_id"], "category": r["category"],
                    "before_recall": before[r["case_id"]]["recall_at_5"], "after_recall": r["recall_at_5"],
                    "before_path": before[r["case_id"]]["path_complete"], "after_path": r["path_complete"],
                    "gate_status": r["gate_status"], "slots": r["selection_trace"].get("slot_verification", {})}
                   for r in all_rows["evidence_oriented"] if r["answerable"]]
    recovered = [r["case_id"] for r in differences if r["case_id"] in historical and r["after_recall"] > 0]
    remaining = [r["case_id"] for r in differences if r["case_id"] in historical and r["after_recall"] == 0]
    write_json(out / "summary.json", {"dataset_version": "evalrag_v0.3", "split": "dev",
        "case_count": len(cases), "answerable_count": sum(c.answerable for c in cases),
        "strategies": summaries, "historical_25_recovered": recovered,
        "historical_25_remaining": remaining, "differences": differences,
        "gate_distribution": {name: dict(Counter(r["gate_status"] for r in rows)) for name, rows in all_rows.items()},
        "limits": "structural readiness is not answer correctness; relation completeness uses labeled edges offline only"})


def real_e2e(cases, chunks, router, systems, gate, out):
    """先短超时连接检查；稳定后一次分层 paired dev，每个完成请求立刻保存。

    断点只跳过已完成 case/strategy；连续连接错误两次停止，不重新执行160条。
    Frozen 不会自动触发：新策略尚未通过可靠性/质量门槛时不得发布默认。
    """
    if not os.environ.get("DEEPSEEK_API_KEY"):
        write_json(out / "e2e_status.json", {"status": "NOT TESTED", "reason": "missing API key"})
        return
    from intern_rag.agent.generation import DeepSeekChatClient
    smoke_path = out / "smoke.json"
    if not smoke_path.exists():
        smoke = DeepSeekChatClient(timeout_seconds=10.0, max_retries=0)
        try:
            answer = smoke.generate('只输出 JSON: {"status":"OK"}', model="deepseek-v4-flash", temperature=0)
            write_json(smoke_path, {"status": "connected", "response": answer,
                                   "usage": smoke.last_token_usage})
        except Exception as error:
            write_json(smoke_path, {"status": "failed", "error_type": type(error).__name__})
            write_json(out / "e2e_status.json", {"status": "NOT TESTED", "reason": "bounded connection smoke failed",
                                                "generation_calls": 1, "frozen": "NOT TESTED"})
            return
    if json.loads(smoke_path.read_text())["status"] != "connected":
        return
    # 直接使用现有单 provider adapter，短超时；不用全局长时间重试配置。
    gateway_config = json.loads((ROOT / "configs/model_gateway/gateway_v0.1.json").read_text())
    gateway_config["gateway"].update({"timeout_seconds": 15.0, "max_attempts_per_provider": 1})
    gateway_config["providers"] = gateway_config["providers"][:1]
    client = build_model_gateway_from_config(gateway_config)
    write_json(out / "e2e_gateway_config.json", gateway_config)
    selected = []
    for category in dict.fromkeys(c.category for c in cases):
        selected.extend([c for c in cases if c.category == category][:2])
    saved = read_rows(out / "e2e_case_results.jsonl")
    completed = {(r["strategy"], r["case_id"]) for r in saved}
    failures = 0
    for name in ("evidence_gap", "evidence_oriented"):
        pipeline = RagPipeline(chunks, client, PipelineConfig(
            model="deepseek-v4-flash", router_name="hybrid", evidence=gate,
            context_mode="adaptive", context_strategy="source_balanced", context_token_budget=1800),
            trace_path=out / f"{name}_e2e_traces.jsonl", router=router, routers={"hybrid": router},
            retriever=systems[name], retrievers={"adaptive": systems[name]}, context_engine=ContextEngine())
        for case in selected:
            if (name, case.case_id) in completed:
                continue
            response = pipeline.run(RagRequest(query=case.query, request_id=case.case_id, retriever="adaptive"))
            trace = pipeline.last_trace
            row = {**classify_failure_stage(case, response, trace).to_dict(), "strategy": name,
                   "tokens": trace.token_usage, "generation_calls": sum("generation" in a.get("type", "") or a.get("type") == "format_repair" for a in trace.attempts)}
            append(out / "e2e_case_results.jsonl", row)
            failures = failures + 1 if response.error_type in {"llm_error", "llm_timeout"} else 0
            if failures >= 2:
                write_json(out / "e2e_status.json", {"status": "partial", "reason": "two consecutive provider failures",
                                                    "frozen": "NOT TESTED"})
                return
    saved = read_rows(out / "e2e_case_results.jsonl")
    from dataclasses import fields
    columns = {f.name for f in fields(FailureFunnelCase)}
    summary = {
        name: summarize_failure_funnel([FailureFunnelCase(**{k: v for k, v in r.items() if k in columns})
                                       for r in saved if r["strategy"] == name])
        for name in ("evidence_gap", "evidence_oriented")
    }
    summary["generation_calls"] = sum(r["generation_calls"] for r in saved)
    summary["input_tokens"] = sum(int(u.get("input_tokens") or 0) for r in saved
                                 for u in r["tokens"].get("attempts", []) if isinstance(u, dict))
    summary["output_tokens"] = sum(int(u.get("output_tokens") or 0) for r in saved
                                  for u in r["tokens"].get("attempts", []) if isinstance(u, dict))
    summary["cost"] = "NOT AVAILABLE: configured price is unspecified; not a zero-cost assertion"
    write_json(out / "e2e_summary.json", summary)
    write_json(out / "e2e_status.json", {"status": "completed", "cases_per_strategy": len(selected),
                                        "frozen": "NOT TESTED", "scope": "stratified paired dev; not full E2E benchmark"})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="p1-evidence-oriented-v03-dev-20261010")
    parser.add_argument("--real-e2e", action="store_true")
    parser.add_argument("--e2e-only", action="store_true")
    args = parser.parse_args()
    out = ROOT / "reports/ablations" / args.run_id
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"config": json.loads(CONFIG.read_text()), "commit_sha": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True).strip(), "dirty_implementation": True,
        "artifact_hashes": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in (CONFIG, ROOT / "data/evaluation/evalrag_v0.3.jsonl",
                                      ROOT / "data/processed/chunks/evalrag_v0.3.jsonl")}}
    if (out / "manifest.json").exists() and json.loads((out / "manifest.json").read_text())["artifact_hashes"] != manifest["artifact_hashes"]:
        raise ValueError("resume requires unchanged config and dataset")
    write_json(out / "manifest.json", manifest)
    chunks = load_chunks_jsonl(ROOT / "data/processed/chunks/evalrag_v0.3.jsonl")
    cases = [c for c in load_knowledge_dataset(ROOT / "data/evaluation/evalrag_v0.3.jsonl") if c.split == "dev"]
    router = build_active_router_from_registry(ROOT / "configs/routing/router_registry_v0.1.json")
    systems = build_systems()
    gate = load_evidence_config(GATE)
    if not args.e2e_only:
        evaluate(cases, chunks, router, systems, gate, out)
    if args.real_e2e:
        real_e2e(cases, chunks, router, systems, gate, out)


if __name__ == "__main__":
    main()
