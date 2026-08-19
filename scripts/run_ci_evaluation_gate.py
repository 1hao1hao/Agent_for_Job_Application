from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from intern_rag.evaluation import (  # noqa: E402
    MetricGate,
    evaluate_ci_gate,
    load_regression_cases,
    run_regression_suite,
)
from intern_rag.routing import route_query  # noqa: E402
from intern_rag.routing.feedback import FeedbackRouter, JsonlRouterFeedbackStore  # noqa: E402


def main() -> int:
    """运行 fixed regression 和 dev/reference 指标门禁，不触碰 frozen test。"""

    config = _read("configs/ci/evaluation_gate_v0.1.json")
    router_report = _read(config["reference_artifacts"]["router"])
    retrieval = _read(config["reference_artifacts"]["retrieval"])
    grounding = _read(config["reference_artifacts"]["grounding"])
    e2e = _read(config["reference_artifacts"]["end_to_end"])
    baseline_router = router_report["results"]["hybrid-v0.2"]
    candidate_router = router_report["results"]["hybrid-feedback-v0.3"]

    reference = {
        "router_accuracy": float(baseline_router["accuracy"]),
        "recall_at_5": float(retrieval["metrics"]["recall_at_5"]),
        "ndcg_at_5": float(retrieval["metrics"]["ndcg_at_5"]),
        "grounding_support": 1.0 - float(grounding["metrics"]["unsupported_answer_rate"]),
        "end_to_end_success": float(e2e["metrics"]["end_to_end_success_rate"]),
        "p95_ms": float(baseline_router["latency_ms"]["p95"]),
    }
    # P1-D6 只修改 Router；未修改的 retrieval/E2E 使用同一版本化 reference 值。
    candidate = {
        **reference,
        "router_accuracy": float(candidate_router["accuracy"]),
        "p95_ms": float(candidate_router["latency_ms"]["p95"]),
    }

    feedback_router = FeedbackRouter(
        route_query,
        JsonlRouterFeedbackStore(ROOT / "data/evaluation/router_feedback_v0.1.jsonl").read_all(),
    )
    cases = []
    for path in config["fixed_regression_files"]:
        cases.extend(load_regression_cases(ROOT / path))
    regression = run_regression_suite(cases, {
        "route": lambda query: _route_result(route_query(query)),
        "route_feedback": lambda query: _route_result(feedback_router(query)),
    })
    failed_case_ids = [
        str(item["case_id"]) for item in regression.case_results
        if item.get("status") == "fixed" and not item.get("passed")
    ]
    thresholds = config["thresholds"]
    gate = evaluate_ci_gate(
        reference,
        candidate,
        [
            MetricGate("router_accuracy", "higher_is_better", float(thresholds["router_accuracy_max_drop"])),
            MetricGate("recall_at_5", "higher_is_better", float(thresholds["recall_at_5_max_drop"])),
            MetricGate("ndcg_at_5", "higher_is_better", float(thresholds["ndcg_at_5_max_drop"])),
            MetricGate("grounding_support", "higher_is_better", float(thresholds["grounding_support_max_drop"])),
            MetricGate("end_to_end_success", "higher_is_better", float(thresholds["end_to_end_success_max_drop"])),
            MetricGate("p95_ms", "lower_is_better", max_increase_ratio=float(thresholds["p95_max_increase_ratio"])),
        ],
        fixed_regression_pass_rate=regression.fixed_pass_rate,
        failed_case_ids=failed_case_ids,
    )
    output = {
        "gate_version": config["gate_version"],
        "dataset_versions": config["dataset_versions"],
        "split": config["split"],
        "frozen_test_used": False,
        "reference": reference,
        "candidate": candidate,
        "unchanged_subsystems": ["retrieval", "grounding", "end_to_end"],
        "regression": regression.to_dict(),
        "gate": gate.to_dict(),
    }
    output_dir = ROOT / "reports/ci/p1-d6-evaluation-gate-v01"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    failures = [item for item in regression.case_results if item.get("passed") is False]
    (output_dir / "failures.jsonl").write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in failures),
        encoding="utf-8",
    )
    print(json.dumps({
        "passed": gate.passed,
        "fixed_regression_pass_rate": regression.fixed_pass_rate,
        "failed_case_ids": failed_case_ids,
        "checks": gate.checks,
    }, ensure_ascii=False, indent=2))
    return 0 if gate.passed else 1


def _read(path: str) -> dict[str, object]:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def _route_result(decision) -> dict[str, object]:
    return {"intent": decision.intent, "sources": decision.routed_sources}


if __name__ == "__main__":
    raise SystemExit(main())
