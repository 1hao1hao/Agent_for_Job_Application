"""最后一次三阶段公平 dev 对照；保存负结果，冻结后不再调参。"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from statistics import mean
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from run_evidence_gap_ablation import _observed_edges, _percentile, _reciprocal_rank
from run_evidence_oriented_evaluation import (
    CONFIG,
    GATE,
    append,
    build_systems,
    read_rows,
    write_json,
)

from intern_rag.agent import (
    ContextEngine,
    ContextEngineConfig,
    check_evidence,
    load_evidence_config,
)
from intern_rag.agent.context import context_item_from_result, format_context_item
from intern_rag.agent.generation import build_generation_prompt
from intern_rag.agent.schemas import BuiltContext
from intern_rag.evaluation import calculate_recall_at_k, load_chunks_jsonl
from intern_rag.evaluation.knowledge_dataset import load_knowledge_dataset
from intern_rag.retrieval.evidence_plan import EvidenceAssembler, EvidencePlanner
from intern_rag.routing import build_active_router_from_registry

OUT = ROOT / "reports/ablations/p1-evidence-final-v03-dev-20261010"
SYSTEM = "仅依据提供的证据回答；证据不足时明确拒答。"


def reserved(query, ids, engine):
    """和实际 Pipeline 使用同一 Prompt 预算公式。"""
    empty = BuiltContext(query, "", [], ids, [], 0, 0)
    return engine.estimator.count(build_generation_prompt(query, empty, "p0-v1")) + 128


def metrics(case, rows, raw_ids):
    """只在预测之后读取标签，分别统计 top-5 和完整阶段集合丢失。"""
    ids = [r.chunk_id for r in rows]
    gold = set(case.relevant_chunk_ids)
    return {
        "chunk_ids": ids,
        "recall_at_5": calculate_recall_at_k(ids, gold, 5) if case.answerable else None,
        "mrr": _reciprocal_rank(ids, gold) if case.answerable else None,
        "path_complete": set(case.graph_edge_ids) <= _observed_edges(rows) if case.graph_edge_ids else None,
        "any_gold": bool(gold & set(ids)),
        "post_recall_lost": bool(gold & set(raw_ids)) and not bool(gold & set(ids)),
    }


def main():
    """统一预算/格式器，依次测候选、Assembler、Context，并锁定选择。

    参考策略使用相同 Assembler 审计适配器与完整候选池，不拿扁平 top-k 与
    裁剪后输出比较。每阶段单独评分；Gate 标准、ACL 和标签均不更改。
    """
    OUT.mkdir(parents=True, exist_ok=True)
    files = [CONFIG, Path(__file__), ROOT / "data/evaluation/evalrag_v0.3.jsonl",
             ROOT / "data/processed/chunks/evalrag_v0.3.jsonl"]
    files += [ROOT / name for name in subprocess.check_output(
        ["git", "ls-files", "src/intern_rag", "scripts/run_evidence_freeze.py"], text=True).splitlines()]
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    manifest = {"hashes": hashes, "parent_commit": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True).strip(), "implementation_dirty": True,
        "dataset_version": "evalrag_v0.3", "split": "dev", "evidence_tokens": 1200,
        "prompt_tokens": 1800, "evidence_chars": 4000, "threads": 2,
        "reference_adapter": "same budgeted assembler over ranked full candidates; not a change to production defaults"}
    if (OUT / "manifest.json").exists() and json.loads((OUT / "manifest.json").read_text())["hashes"] != hashes:
        raise ValueError("cannot resume changed implementation/config/labels")
    write_json(OUT / "manifest.json", manifest)
    chunks = load_chunks_jsonl(ROOT / "data/processed/chunks/evalrag_v0.3.jsonl")
    cases = [c for c in load_knowledge_dataset(ROOT / "data/evaluation/evalrag_v0.3.jsonl") if c.split == "dev"]
    router = build_active_router_from_registry(ROOT / "configs/routing/router_registry_v0.1.json")
    systems = build_systems()
    gate = load_evidence_config(GATE)
    engine = ContextEngine()
    formatter = lambda r: format_context_item(context_item_from_result(r))
    all_rows = {}
    for name, retriever in systems.items():
        path = OUT / f"{name}_case_results.jsonl"
        rows = read_rows(path)
        complete = {r["case_id"] for r in rows}
        for index, case in enumerate(cases, 1):
            if case.case_id in complete:
                continue
            def available(ids, query=case.query):
                return 1800 - reserved(query, ids, engine) - engine.estimator.count(SYSTEM) - engine.estimator.count(query)
            if hasattr(retriever, "configure_packing"):
                retriever.configure_packing(engine.estimator.count, formatter, max_chars=4000, available_tokens=available)
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
            candidates = retriever.get_candidates()
            plan = EvidencePlanner().plan(case.query, set(route.routed_sources))
            if name == "evidence_oriented":
                assembled = results
                dropped = trace["evidence_bundle"]["dropped"]
            else:
                assembler = EvidenceAssembler()
                bundle = assembler.assemble(plan, candidates, 1200, engine.estimator.count,
                                            max_chars=4000, format_result=formatter)
                effective = max(0, min(1200, available([r.chunk_id for r in bundle.results])))
                if effective < 1200:
                    bundle = assembler.assemble(plan, candidates, effective, engine.estimator.count,
                                                max_chars=4000, format_result=formatter)
                assembled = list(bundle.results)
                dropped = list(bundle.dropped)
            managed = engine.build(query=case.query, system_prompt=SYSTEM,
                                   retrieved_results=assembled,
                                   config=ContextEngineConfig(token_budget=1800, mode="no_memory",
                                       evidence_char_budget=4000, evidence_strategy="source_balanced",
                                       reserved_token_count=reserved(case.query, [r.chunk_id for r in assembled], engine)),
                                   required_source_types=plan.requirement.required_source_types)
            used = set(managed.evidence.used_chunk_ids)
            context = [r for r in assembled if r.chunk_id in used]
            raw_ids = [r.chunk_id for r in candidates]
            row = {"case_id": case.case_id, "category": case.category, "answerable": case.answerable,
                   "stages": {"candidates": metrics(case, candidates, raw_ids),
                              "assembled": metrics(case, assembled, raw_ids),
                              "context": metrics(case, context, raw_ids)},
                   "latency_ms": (perf_counter() - start) * 1000,
                   "logical_calls": trace.get("retrieval_call_count", 1 + trace.get("rescue_call_count", 0)),
                   "assembly_dropped": dropped, "context_dropped": list(managed.dropped),
                   "selection_trace": trace}
            append(path, row)
            rows.append(row)
            if index % 40 == 0:
                print(f"{name} {index}/{len(cases)}", flush=True)
        all_rows[name] = rows
    summary = {}
    for name, rows in all_rows.items():
        answerable = [r for r in rows if r["answerable"]]
        summary[name] = {
            "stages": {stage: {
                "recall_at_5": mean(r["stages"][stage]["recall_at_5"] for r in answerable),
                "mrr": mean(r["stages"][stage]["mrr"] for r in answerable),
                "path_completeness": mean(float(r["stages"][stage]["path_complete"])
                                          for r in rows if r["category"] in {"two_hop", "three_hop"}),
                "zero_at_5": sum(r["stages"][stage]["recall_at_5"] == 0 for r in answerable),
                "lost_after_candidate_hit": sum(r["stages"][stage]["post_recall_lost"] for r in answerable),
            } for stage in ("candidates", "assembled", "context")},
            "p95_ms": _percentile([r["latency_ms"] for r in rows], .95),
            "mean_logical_calls": mean(r["logical_calls"] for r in rows),
        }
    before = {r["case_id"]: r for r in all_rows["evidence_gap"]}
    regressions = [{"case_id": r["case_id"], "stages": {
        s: {"reference": before[r["case_id"]]["stages"][s], "candidate": r["stages"][s]}
        for s in ("candidates", "assembled", "context")}}
        for r in all_rows["evidence_oriented"] if r["answerable"] and any(
            r["stages"][s]["recall_at_5"] < before[r["case_id"]]["stages"][s]["recall_at_5"]
            for s in ("candidates", "assembled", "context"))]
    ref, new = summary["evidence_gap"], summary["evidence_oriented"]
    noninferior = all(new["stages"][s][m] >= ref["stages"][s][m]
                     for s in ("candidates", "context")
                     for m in ("recall_at_5", "mrr", "path_completeness"))
    selected = "evidence_oriented" if noninferior and new["p95_ms"] <= 1.25 * ref["p95_ms"] else "evidence_gap"
    old = read_rows(ROOT / "reports/ablations/p1-evidence-oriented-v03-dev-20261010/evidence_oriented_case_results.jsonl")
    dropped41 = {r["case_id"] for r in old if r["answerable"] and not r["recall_at_5"]
                 and set(r["relevant_chunk_ids"]) & {cid for h in r["selection_trace"]["retrieval_history"] for cid in h["chunk_ids"]}}
    audit41 = [{"case_id": r["case_id"], "recovered_at_5": r["stages"]["context"]["recall_at_5"] > 0,
                "stages": r["stages"]} for r in all_rows["evidence_oriented"] if r["case_id"] in dropped41]
    write_json(OUT / "summary.json", {"dataset_version": "evalrag_v0.3", "split": "dev", "cases": 160,
        "answerable": 120, "strategies": summary, "regressions": regressions, "audit_previous_41": audit41,
        "selected_candidate": selected, "production_default_changed": False, "status": "PROJECT_FROZEN",
        "frozen_test": "NOT RUN by task requirement", "e2e": "E2E NOT VERIFIED"})
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
