from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from intern_rag.evaluation.dataset import load_evaluation_dataset  # noqa: E402
from intern_rag.retrieval import load_dense_index  # noqa: E402
from intern_rag.routing import (  # noqa: E402
    DEFAULT_INTENT_PROTOTYPES,
    HybridRouter,
    HybridRouterConfig,
    SemanticRouter,
    SemanticRouterConfig,
    route_query,
)
from intern_rag.routing.feedback import (  # noqa: E402
    FeedbackRouter,
    JsonlRouterFeedbackStore,
    RouterVersion,
    RouterVersionRegistry,
    compare_router_versions,
    evaluate_router_shadow,
    prototypes_from_feedback,
)


RUN_ID = "p1-d6-router-feedback-v01-dev-20260816"


def main() -> int:
    """运行 Rule/Semantic/Hybrid 与 feedback candidate 的同集 dev shadow 对照。"""

    config = json.loads((ROOT / "configs/routing/hybrid_feedback_v0.3.json").read_text(encoding="utf-8"))
    cases = [
        case for case in load_evaluation_dataset(ROOT / "data/evaluation/evalrag_v0.2.jsonl")
        if case.split == "dev"
    ]
    _, embedding_model = load_dense_index(ROOT / str(config["embedding_index_dir"]))
    semantic_config = SemanticRouterConfig(
        float(config["semantic_min_score"]), float(config["semantic_min_margin"])
    )
    hybrid_config = HybridRouterConfig(
        float(config["semantic_override_score"]),
        float(config["semantic_override_margin"]),
        int(config["max_weak_rule_keywords"]),
    )
    semantic = SemanticRouter(embedding_model, config=semantic_config)
    hybrid = HybridRouter(route_query, semantic, config=hybrid_config)
    feedback = JsonlRouterFeedbackStore(ROOT / str(config["feedback_dataset"])).read_all()
    candidate_prototypes = prototypes_from_feedback(DEFAULT_INTENT_PROTOTYPES, feedback)
    candidate_semantic = SemanticRouter(
        embedding_model, prototypes=candidate_prototypes, config=semantic_config
    )
    prototype_candidate = HybridRouter(
        route_query, candidate_semantic, config=hybrid_config
    )
    candidate = FeedbackRouter(prototype_candidate, feedback)

    results = {
        "rule-v0.2": evaluate_router_shadow(route_query, cases),
        "semantic-v0.2": evaluate_router_shadow(semantic, cases),
        "hybrid-v0.2": evaluate_router_shadow(hybrid, cases),
        "hybrid-feedback-v0.3": evaluate_router_shadow(candidate, cases),
    }
    gate = compare_router_versions(
        results["hybrid-v0.2"], results["hybrid-feedback-v0.3"],
        max_p95_ratio=1.25,
    )
    output_dir = ROOT / "reports/ablations" / RUN_ID
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": RUN_ID,
        "dataset_version": "evalrag_v0.2",
        "split": "dev",
        "case_count": len(cases),
        "feedback_count": len(feedback),
        "model": config["embedding_name"],
        "model_revision": config["embedding_revision"],
        "online_self_learning": False,
        "results": results,
        "shadow_gate": gate,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _write_report(output_dir / "report.md", payload)

    registry = RouterVersionRegistry(ROOT / "configs/routing/router_registry_v0.1.json")
    state = registry.load()
    if not state["versions"]:
        registry.publish(
            RouterVersion(
                "hybrid-v0.2", "hybrid", {**config, "router_version": "hybrid-v0.2"},
                "none", str((output_dir / "summary.json").relative_to(ROOT)), None, "2026-08-16T00:00:00+00:00",
            ),
            gate_passed=True,
        )
    if not any(item["version"] == "hybrid-feedback-v0.3" for item in registry.load()["versions"]):
        registry.publish(
            RouterVersion(
                "hybrid-feedback-v0.3", "hybrid", config,
                str(config["feedback_dataset"]), str((output_dir / "summary.json").relative_to(ROOT)),
                "hybrid-v0.2", "2026-08-16T00:00:00+00:00",
            ),
            gate_passed=bool(gate["passed"]),
        )
    print(json.dumps({
        "run_id": RUN_ID,
        "metrics": {
            key: {
                "accuracy": value["accuracy"],
                "unknown_precision": value["unknown_precision"],
                "unknown_recall": value["unknown_recall"],
                "p95_ms": value["latency_ms"]["p95"],
            }
            for key, value in results.items()
        },
        "gate_passed": gate["passed"],
        "drift_count": len(gate["differences"]),
    }, ensure_ascii=False, indent=2))
    return 0 if gate["passed"] else 1


def _write_report(path: Path, payload: dict[str, object]) -> None:
    results = payload["results"]
    lines = [
        "# P1-D6 Router Feedback Dev Shadow Report", "",
        f"- Dataset: `{payload['dataset_version']}` / `{payload['split']}` / {payload['case_count']} cases",
        f"- Feedback: {payload['feedback_count']} confirmed dev failures",
        f"- Model: `{payload['model']}@{payload['model_revision']}`", "",
        "| Version | Accuracy | Unknown Precision | Unknown Recall | P95 ms |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, value in results.items():
        lines.append(
            f"| {name} | {value['accuracy']:.4f} | {value['unknown_precision']:.4f} | "
            f"{value['unknown_recall']:.4f} | {value['latency_ms']['p95']:.3f} |"
        )
    lines.extend(["", "## Drift Cases", ""])
    for item in payload["shadow_gate"]["differences"]:
        lines.append(
            f"- `{item['case_id']}` {item['change']}: "
            f"{item['before']['predicted_intent']} -> {item['after']['predicted_intent']}"
        )
    lines.extend([
        "", "## Boundary", "",
        "Feedback 来自已确认 dev failure，只离线更新 prototype；没有在线自学习。",
        "候选版本通过同集 shadow gate 后才写入 active registry；frozen test 未参与调参。",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
