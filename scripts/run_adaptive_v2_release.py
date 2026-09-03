from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from intern_rag.agent import EvidenceConfig, check_evidence, load_evidence_config  # noqa: E402
from intern_rag.evaluation import (  # noqa: E402
    MetricGate, calibrate_score_gate, evaluate_ci_gate, is_structurally_positive,
    load_chunks_jsonl, load_regression_cases, run_regression_suite,
)
from intern_rag.evaluation.knowledge_dataset import load_knowledge_dataset  # noqa: E402
from intern_rag.evaluation.knowledge_runner import (  # noqa: E402
    KnowledgeRunConfig, run_knowledge_evaluation, save_knowledge_run,
)
from intern_rag.retrieval import QueryAnalyzer, RetrievalResult, build_retriever_from_config  # noqa: E402
from intern_rag.routing import RouteDecision, route_query  # noqa: E402


DEV_ID = "p1-adaptive-v2-v03-dev-20260903-r1"
RELEASE_ID = "p1-adaptive-v2-v03-release-test-20260903-r1"
CONFIGS = {
    "bm25": "configs/retrieval/bm25_v0.3.json",
    "dense": "configs/retrieval/dense_v0.3.json",
    "hybrid": "configs/retrieval/bm25_hybrid_v0.3.json",
    "graph_vector": "configs/retrieval/graph_vector_v0.3.json",
    "adaptive_v1": "configs/retrieval/adaptive_v1_replay_v0.3.json",
    "adaptive_v2": "configs/retrieval/adaptive_v2_v0.3.json",
}


def main() -> int:
    """按固定顺序执行 dev 锁定或锁定后的 release test。"""

    phase = sys.argv[1] if len(sys.argv) > 1 else "dev"
    if phase not in {"dev", "frozen"}:
        raise ValueError("phase must be dev or frozen")
    return run_dev() if phase == "dev" else run_frozen()


def run_dev() -> int:
    """运行同集检索、校准、Gate、Regression 与 CI，并锁定配置哈希。"""

    chunks = load_chunks_jsonl(ROOT / "data/processed/chunks/evalrag_v0.3.jsonl")
    cases = load_knowledge_dataset(ROOT / "data/evaluation/evalrag_v0.3.jsonl")
    rows_by_name, summaries = _run_matrix(cases, chunks, "dev", DEV_ID, CONFIGS)
    output = ROOT / "reports/ablations" / DEV_ID
    output.mkdir(parents=True, exist_ok=True)

    analyzer = QueryAnalyzer()
    grouped = {
        name: _group_by_need(cases, rows, analyzer)
        for name, rows in rows_by_name.items()
        if name in {"bm25", "dense", "hybrid", "graph_vector"}
    }
    calibrations = {
        name: calibrate_score_gate(name, [c for c in cases if c.split == "dev"], rows)
        for name, rows in rows_by_name.items()
        if name in {"bm25", "dense", "hybrid", "graph_vector"}
    }
    evidence_payload = {
        "config_version": "evidence-gate-v2.0-dev-locked",
        "dataset_version": "evalrag_v0.3",
        "calibration_split": "dev",
        "far_target": 0.05,
        "max_frr_for_score_gate": 0.25,
        "min_results": 1,
        "legacy_source_coverage": False,
        "need_min_results": {
            "exact_fact": 1, "semantic_explanation": 1,
            "balanced_retrieval": 1, "multi_source_synthesis": 2,
            "relation_reasoning": 1,
        },
        "retrievers": {
            name: {
                "enabled": result.enabled, "threshold": result.threshold,
                "status": result.status,
                "operating_point": result.operating_point,
            }
            for name, result in calibrations.items()
        },
    }
    evidence_path = ROOT / "configs/evidence/gate_calibrated_v0.3.json"
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(evidence_path, evidence_payload)
    _write_json(output / "threshold_calibration.json", {
        "dataset_version": "evalrag_v0.3", "split": "dev",
        "retrievers": {name: value.to_dict() for name, value in calibrations.items()},
    })

    old_gate = _evaluate_gate(cases, rows_by_name["adaptive_v2"], chunks, EvidenceConfig())
    new_gate = _evaluate_gate(cases, rows_by_name["adaptive_v2"], chunks, load_evidence_config(evidence_path))
    _write_jsonl(output / "gate_v2_case_results.jsonl", new_gate.pop("case_results"))
    _write_jsonl(output / "gate_legacy_case_results.jsonl", old_gate.pop("case_results"))
    regressions = _run_fixed_regressions()
    reference = _flat_metrics(summaries["adaptive_v1"])
    candidate = _flat_metrics(summaries["adaptive_v2"])
    ci = evaluate_ci_gate(
        reference, candidate,
        [
            MetricGate("recall_at_5", "higher_is_better", dynamic_one_case_tolerance=True),
            MetricGate("mrr", "higher_is_better", dynamic_one_case_tolerance=True),
            MetricGate("p95_ms", "lower_is_better", max_increase_ratio=1.25),
            MetricGate("ndcg_at_5", "higher_is_better", blocking=False),
        ],
        fixed_regression_pass_rate=float(regressions["fixed_pass_rate"]),
        failed_case_ids=[str(x["case_id"]) for x in regressions["case_results"] if x.get("passed") is False],
        case_count=160, answerable_case_count=120,
    )
    comparison = {
        "run_id": DEV_ID, "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_version": "evalrag_v0.3", "split": "dev",
        "case_count": 160, "answerable_case_count": 120,
        "semantic_mapping": _semantic_mapping(rows_by_name),
        "strategies": summaries,
        "evidence_need_groups": grouped,
        "selector": _selector_summary(rows_by_name["adaptive_v2"]),
        "adaptive_comparison": {
            "adaptive_v1": summaries["adaptive_v1"],
            "adaptive_v2": summaries["adaptive_v2"],
            "always_graph": summaries["graph_vector"],
        },
        "representative_differences": _differences(rows_by_name, 10),
        "gate_comparison": {"legacy": old_gate, "v2": new_gate},
        "fixed_regression": regressions,
        "ci_gate_v2": ci.to_dict(),
        "boundary": "dev-only configuration selection; no frozen test was read",
    }
    _write_json(output / "summary.json", comparison)
    (output / "report.md").write_text(_report(comparison), encoding="utf-8")
    _write_json(output / "ci_gate_v2_summary.json", ci.to_dict())
    if not ci.passed:
        raise RuntimeError(f"CI gate failed: {ci.reasons}")

    manifest = {
        "release_config_version": "adaptive-v2-release-v0.3",
        "dataset_version": "evalrag_v0.3",
        "dev_run": f"reports/ablations/{DEV_ID}/summary.json",
        "retrieval_config": "configs/retrieval/adaptive_v2_v0.3.json",
        "evidence_config": "configs/evidence/gate_calibrated_v0.3.json",
        "sha256": _hashes([
            "data/evaluation/evalrag_v0.3.jsonl",
            "data/processed/chunks/evalrag_v0.3.jsonl",
            "data/processed/graphs/evalrag_v0.3/job-skill-experience-v0.2.json",
            "configs/retrieval/adaptive_v2_v0.3.json",
            "configs/evidence/gate_calibrated_v0.3.json",
            "src/intern_rag/retrieval/adaptive.py",
            "src/intern_rag/agent/evidence.py",
            "src/intern_rag/evaluation/calibration.py",
            "src/intern_rag/evaluation/ci_gate.py",
        ]),
        "policy": "selector, thresholds and labels are immutable before release test",
    }
    final_path = ROOT / "configs/final/adaptive_v2_release_v0.3.json"
    final_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(final_path, manifest)
    print(json.dumps({"dev": comparison["adaptive_comparison"], "gate": new_gate, "ci_passed": ci.passed}, ensure_ascii=False, indent=2))
    return 0


def run_frozen() -> int:
    """验证锁定哈希后，仅运行一次历史 test split 并标记为 release test。"""

    manifest = _read_json(ROOT / "configs/final/adaptive_v2_release_v0.3.json")
    for path, expected in dict(manifest["sha256"]).items():
        if _sha256(ROOT / path) != expected:
            raise ValueError(f"locked release input changed: {path}")
    release_dir = ROOT / "reports/releases" / RELEASE_ID
    if release_dir.exists():
        raise FileExistsError(f"release test already exists: {release_dir}")
    chunks = load_chunks_jsonl(ROOT / "data/processed/chunks/evalrag_v0.3.jsonl")
    cases = load_knowledge_dataset(ROOT / "data/evaluation/evalrag_v0.3.jsonl")
    selected = {name: CONFIGS[name] for name in ("bm25", "graph_vector", "adaptive_v2")}
    rows, summaries = _run_matrix(cases, chunks, "test", RELEASE_ID, selected)
    gate = _evaluate_gate(cases, rows["adaptive_v2"], chunks, load_evidence_config(ROOT / str(manifest["evidence_config"])))
    gate_rows = gate.pop("case_results")
    payload = {
        "release_id": RELEASE_ID, "dataset_version": "evalrag_v0.3", "split": "test",
        "case_count": 80, "historical_test_disclosure": (
            "This split has historical runs; this is the release test for the newly locked config, not a never-seen blind test."
        ),
        "locked_manifest": "configs/final/adaptive_v2_release_v0.3.json",
        "strategies": summaries, "deterministic_gate_e2e": gate,
        "post_run_policy": "no selector, threshold or label tuning on this release",
    }
    release_dir.mkdir(parents=True)
    _write_json(release_dir / "summary.json", payload)
    _write_jsonl(release_dir / "gate_case_results.jsonl", gate_rows)
    _write_jsonl(release_dir / "gate_failures.jsonl", [row for row in gate_rows if not row["success"]])
    (release_dir / "report.md").write_text(_release_report(payload), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def _run_matrix(cases, chunks, split, prefix, configs):
    rows, summaries = {}, {}
    for name, relative in configs.items():
        raw = _read_json(ROOT / relative)
        run_id = f"{prefix}-{name}"
        run_dir = ROOT / "reports/runs" / run_id
        reusable = (
            ROOT / "reports/runs" / f"p1-adaptive-v2-v03-dev-20260903-{name}"
            if split == "dev" and name in {"bm25", "dense", "hybrid", "graph_vector"}
            else run_dir
        )
        if reusable != run_dir and (reusable / "summary.json").exists():
            summaries[name] = _read_json(reusable / "summary.json")
            rows[name] = _read_jsonl(reusable / "case_results.jsonl")
            continue
        if (run_dir / "summary.json").exists():
            summaries[name] = _read_json(run_dir / "summary.json")
            rows[name] = _read_jsonl(run_dir / "case_results.jsonl")
            continue
        result = run_knowledge_evaluation(
            cases, chunks,
            KnowledgeRunConfig(run_id, "evalrag_v0.3", "job-skill-experience-v0.2", split, name, 5,
                               f"PYTHONPATH=src python scripts/run_adaptive_v2_release.py {split}", raw),
            build_retriever_from_config(raw),
        )
        save_knowledge_run(result, run_dir)
        rows[name], summaries[name] = result.case_results, result.summary
    return rows, summaries


def _group_by_need(cases, rows, analyzer):
    case_map = {case.case_id: case for case in cases if case.split == "dev"}
    grouped = defaultdict(list)
    for row in rows:
        case = case_map[str(row["case_id"])]
        need = analyzer.classify_evidence_need(analyzer.analyze(case.query, set(case.expected_sources))).need_type
        grouped[need].append(row)
    return {name: _row_summary(items) for name, items in sorted(grouped.items())}


def _row_summary(rows):
    answerable = [r for r in rows if r["answerable"]]
    metric = lambda key: sum(float(r["metrics"][key]) for r in answerable) / len(answerable) if answerable else 0.0
    latency = sorted(float(r["latency_ms"]) for r in rows)
    pct = lambda q: latency[max(0, min(len(latency)-1, int(len(latency)*q + .999999)-1))] if latency else 0.0
    return {"case_count": len(rows), "answerable_case_count": len(answerable), "recall_at_5": metric("recall_at_5"),
            "mrr": metric("reciprocal_rank"), "ndcg_at_5": metric("ndcg_at_5"), "p50_ms": pct(.5), "p95_ms": pct(.95)}


def _semantic_mapping(rows):
    values = {}
    for name in ("dense", "hybrid"):
        selected = [r for r in rows[name] if r["category"] == "semantic_paraphrase"]
        values[name] = _row_summary(selected)
    tolerance = 1 / max(1, values["dense"]["answerable_case_count"])
    dense_selected = (
        values["dense"]["recall_at_5"] >= values["hybrid"]["recall_at_5"] - tolerance
        and values["dense"]["mrr"] >= values["hybrid"]["mrr"] - tolerance
        and values["dense"]["p95_ms"] < values["hybrid"]["p95_ms"]
    )
    return {"case_count": 20, "one_case_tolerance": tolerance, "dense": values["dense"],
            "hybrid": values["hybrid"], "locked_strategy": "dense" if dense_selected else "hybrid"}


def _selector_summary(rows):
    strategies = Counter()
    errors = []
    for row in rows:
        trace = row["predicted"]["trace"]
        strategy = str(trace.get("selected_strategy", trace.get("strategy", "unknown")))
        strategies[strategy] += 1
        if row["answerable"] and float(row["metrics"]["recall_at_5"] or 0) < 1:
            errors.append({"case_id": row["case_id"], "category": row["category"], "strategy": strategy,
                           "selection_rule": trace.get("selection_rule"), "query": row["query"]})
    return {"strategy_distribution": dict(strategies),
            "graph_invocation_rate": strategies["graph_hybrid"] / len(rows), "error_cases": errors}


def _evaluate_gate(cases, rows, chunks, config):
    case_map = {c.case_id: c for c in cases if c.split == rows[0]["split"]}
    chunk_map = {c.id: c for c in chunks}
    records = []
    for row in rows:
        case = case_map[row["case_id"]]
        predicted = row["predicted"]
        results = [RetrievalResult(x["chunk_id"], float(x["score"]), int(x["rank"]), chunk_map[x["chunk_id"]], x["reason"], x["details"])
                   for x in predicted["retrieved"]]
        trace = predicted["trace"]
        requirement = trace.get("evidence_requirement", {})
        route = RouteDecision("evaluation" if case.answerable else "unknown", list(case.expected_sources), [])
        decision = check_evidence(route, results, retriever_name="adaptive", retry_count=1, max_retries=1,
                                  config=config, evidence_requirement=requirement, retrieval_trace=trace)
        structurally_positive = is_structurally_positive(case, predicted["retrieved"][:5])
        accepted = decision.status == "sufficient"
        success = (case.answerable and structurally_positive and accepted) or (not case.answerable and not accepted)
        records.append({"case_id": case.case_id, "answerable": case.answerable,
                        "structurally_positive": structurally_positive, "accepted": accepted,
                        "success": success, "decision": asdict(decision)})
    pos = [r for r in records if r["structurally_positive"]]
    neg = [r for r in records if not r["structurally_positive"]]
    unanswerable = [r for r in records if not r["answerable"]]
    answerable = [r for r in records if r["answerable"]]
    return {"config_version": config.config_version, "case_count": len(records),
            "far": sum(r["accepted"] for r in neg)/len(neg) if neg else 0.0,
            "frr": sum(not r["accepted"] for r in pos)/len(pos) if pos else 0.0,
            "answerable_acceptance": sum(r["accepted"] for r in answerable)/len(answerable),
            "abstention_accuracy": sum(not r["accepted"] for r in unanswerable)/len(unanswerable),
            "deterministic_e2e_success": sum(r["success"] for r in records)/len(records),
            "failure_count": sum(not r["success"] for r in records),
            "case_results": records}


def _run_fixed_regressions():
    cases = load_regression_cases(ROOT / "tests/regression/cases_v0.2.jsonl")
    result = run_regression_suite(cases, {"route": lambda q: {"intent": route_query(q).intent, "sources": route_query(q).routed_sources}})
    return result.to_dict()


def _flat_metrics(summary):
    return {**summary["metrics"], "p95_ms": summary["latency_ms"]["p95"]}


def _differences(rows_by_name, limit):
    indexes = {name: {r["case_id"]: r for r in rows} for name, rows in rows_by_name.items()}
    output = []
    for case_id, row in indexes["adaptive_v2"].items():
        if not row["answerable"]:
            continue
        values = {name: {"recall_at_5": values[case_id]["metrics"]["recall_at_5"],
                         "mrr": values[case_id]["metrics"]["reciprocal_rank"]}
                  for name, values in indexes.items()}
        scores = [float(v["recall_at_5"] or 0) + float(v["mrr"] or 0) for v in values.values()]
        output.append({"case_id": case_id, "category": row["category"], "query": row["query"],
                       "difference": max(scores)-min(scores), "metrics": values})
    return sorted(output, key=lambda x: (-x["difference"], x["case_id"]))[:limit]


def _report(payload):
    lines = ["# Adaptive Retrieval v2 Dev Closure", "", "`evalrag_v0.3/dev`: 160 cases, 120 answerable. Frozen test was not read.", "",
             "| Strategy | Recall@5 | MRR | NDCG@5 | P95 ms |", "|---|---:|---:|---:|---:|"]
    for name, summary in payload["strategies"].items():
        m, l = summary["metrics"], summary["latency_ms"]
        lines.append(f"| {name} | {m['recall_at_5']:.4f} | {m['mrr']:.4f} | {m['ndcg_at_5']:.4f} | {l['p95']:.3f} |")
    gate = payload["gate_comparison"]["v2"]
    lines += ["", "## Gate v2", "", f"- FAR: {gate['far']:.4f}; FRR: {gate['frr']:.4f}; answerable acceptance: {gate['answerable_acceptance']:.4f}.",
              f"- Abstention Accuracy: {gate['abstention_accuracy']:.4f}; deterministic E2E success: {gate['deterministic_e2e_success']:.4f}.",
              "", "## Boundary", "", payload["boundary"] + "."]
    return "\n".join(lines) + "\n"


def _release_report(payload):
    lines = ["# Adaptive v2 Release Test", "", payload["historical_test_disclosure"], "",
             "| Strategy | Recall@5 | MRR | NDCG@5 | P95 ms |", "|---|---:|---:|---:|---:|"]
    for name, summary in payload["strategies"].items():
        m, l = summary["metrics"], summary["latency_ms"]
        lines.append(f"| {name} | {m['recall_at_5']:.4f} | {m['mrr']:.4f} | {m['ndcg_at_5']:.4f} | {l['p95']:.3f} |")
    return "\n".join(lines) + "\n"


def _hashes(paths):
    return {path: _sha256(ROOT / path) for path in paths}


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
