from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from intern_rag.evaluation import (  # noqa: E402
    MetricGate, evaluate_ci_gate, load_chunks_jsonl,
    load_regression_cases, run_regression_suite,
)
from intern_rag.evaluation.knowledge_dataset import load_knowledge_dataset  # noqa: E402
from intern_rag.evaluation.knowledge_runner import KnowledgeRunConfig, run_knowledge_evaluation  # noqa: E402
from intern_rag.retrieval import build_retriever_from_config  # noqa: E402
from intern_rag.routing import route_query  # noqa: E402
from intern_rag.routing.feedback import FeedbackRouter, JsonlRouterFeedbackStore  # noqa: E402


def main() -> int:
    """真实运行 dev v1/v2 与 fixed Regression；禁止读取 frozen test。"""

    config = _read("configs/ci/evaluation_gate_v0.2.json")
    if config["split"] != "dev":
        raise ValueError("CI evaluation gate only permits dev split")
    cases = load_knowledge_dataset(ROOT / "data/evaluation/evalrag_v0.3.jsonl")
    chunks = load_chunks_jsonl(ROOT / "data/processed/chunks/evalrag_v0.3.jsonl")
    runs = {}
    for name, path in {
        "reference": "configs/retrieval/adaptive_v1_replay_v0.3.json",
        "candidate": "configs/retrieval/adaptive_v2_v0.3.json",
    }.items():
        raw = _read(path)
        runs[name] = run_knowledge_evaluation(
            cases, chunks,
            KnowledgeRunConfig(
                f"ci-adaptive-v2-{name}", "evalrag_v0.3",
                "job-skill-experience-v0.2", "dev", name, 5,
                "PYTHONPATH=src python scripts/run_ci_evaluation_gate.py", raw,
            ),
            build_retriever_from_config(raw),
        )
    regression = _regressions()
    failed_case_ids = [
        str(item["case_id"]) for item in regression.case_results
        if item.get("status") == "fixed" and item.get("passed") is False
    ]
    reference = _flat(runs["reference"].summary)
    candidate = _flat(runs["candidate"].summary)
    gate = evaluate_ci_gate(
        reference, candidate,
        [
            MetricGate("recall_at_5", "higher_is_better", dynamic_one_case_tolerance=True),
            MetricGate("mrr", "higher_is_better", dynamic_one_case_tolerance=True),
            MetricGate("p95_ms", "lower_is_better", max_increase_ratio=float(config["max_p95_increase_ratio"])),
            MetricGate("ndcg_at_5", "higher_is_better", blocking=False),
        ],
        fixed_regression_pass_rate=regression.fixed_pass_rate,
        failed_case_ids=failed_case_ids,
        case_count=160, answerable_case_count=120,
    )
    output = {
        "config_version": config["config_version"],
        "dataset_version": "evalrag_v0.3", "split": "dev",
        "case_count": 160, "answerable_case_count": 120,
        "frozen_test_used": False, "reference": reference, "candidate": candidate,
        "regression": regression.to_dict(), "gate": gate.to_dict(),
        "p95_scope": config["p95_scope"],
    }
    output_dir = ROOT / "reports/ci/evaluation-gate-v2"
    output_dir.mkdir(parents=True, exist_ok=True)
    _write(output_dir / "summary.json", output)
    _write_jsonl(output_dir / "reference_case_results.jsonl", runs["reference"].case_results)
    _write_jsonl(output_dir / "candidate_case_results.jsonl", runs["candidate"].case_results)
    _write_jsonl(output_dir / "failures.jsonl", [x for x in regression.case_results if x.get("passed") is False])
    print(json.dumps({"passed": gate.passed, "checks": gate.checks,
                      "fixed_regression_pass_rate": regression.fixed_pass_rate}, ensure_ascii=False, indent=2))
    return 0 if gate.passed else 1


def _regressions():
    cases = []
    for path in ("tests/regression/cases_v0.2.jsonl", "tests/regression/router_feedback_v0.1.jsonl"):
        cases.extend(load_regression_cases(ROOT / path))
    feedback = FeedbackRouter(
        route_query,
        JsonlRouterFeedbackStore(ROOT / "data/evaluation/router_feedback_v0.1.jsonl").read_all(),
    )
    handler = lambda router: lambda query: {
        "intent": router(query).intent, "sources": router(query).routed_sources
    }
    return run_regression_suite(cases, {"route": handler(route_query), "route_feedback": handler(feedback)})


def _flat(summary):
    return {**summary["metrics"], "p95_ms": summary["latency_ms"]["p95"]}


def _read(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def _write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
