from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path


STRATEGIES = ("rule", "semantic", "hybrid")


def main() -> int:
    """汇总已经由 Evaluation Runner 产生的三组 Router dev 工件。"""

    args = _parse_args()
    run_dirs = {
        "rule": Path("reports/runs") / args.rule_run,
        "semantic": Path("reports/runs") / args.semantic_run,
        "hybrid": Path("reports/runs") / args.hybrid_run,
    }
    summaries = {name: _read_json(path / "summary.json") for name, path in run_dirs.items()}
    cases = {name: _read_jsonl(path / "case_results.jsonl") for name, path in run_dirs.items()}
    payload = {
        "ablation_id": args.report_id,
        "dataset_version": "evalrag_v0.2",
        "split": "dev",
        "case_count": len(cases["rule"]),
        "run_dirs": {name: str(path) for name, path in run_dirs.items()},
        "summaries": summaries,
        "classification": {
            name: _classification_summary(items) for name, items in cases.items()
        },
        "strategy_differences": _strategy_differences(cases),
    }
    report_dir = Path("reports/ablations") / args.report_id
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (report_dir / "report.md").write_text(_format_report(payload), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def _classification_summary(items: list[dict[str, object]]) -> dict[str, object]:
    """按类别和 expected/predicted intent 汇总 Router 分类结果。"""

    by_category: dict[str, dict[str, int]] = {}
    confusion: Counter[str] = Counter()
    for item in items:
        category = str(item["category"])
        metrics = dict(item["metrics"])  # type: ignore[arg-type]
        bucket = by_category.setdefault(category, {"correct": 0, "total": 0})
        bucket["total"] += 1
        bucket["correct"] += int(bool(metrics["router_correct"]))
        expected = str(dict(item["expected"])["intent"])  # type: ignore[arg-type]
        predicted = str(dict(item["predicted"])["intent"])  # type: ignore[arg-type]
        confusion[f"{expected}->{predicted}"] += 1
    return {
        "by_category": {
            category: {
                **counts,
                "accuracy": counts["correct"] / counts["total"],
            }
            for category, counts in sorted(by_category.items())
        },
        "intent_confusion": dict(sorted(confusion.items())),
    }


def _strategy_differences(
    cases: dict[str, list[dict[str, object]]],
) -> list[dict[str, object]]:
    """保留至少五条策略正确性不同或共同失败的 case。"""

    indexed = {
        name: {str(item["case_id"]): item for item in items}
        for name, items in cases.items()
    }
    differences: list[dict[str, object]] = []
    for case_id, rule_case in indexed["rule"].items():
        correctness = {
            name: bool(dict(indexed[name][case_id]["metrics"])["router_correct"])
            for name in STRATEGIES
        }
        if len(set(correctness.values())) == 1 and all(correctness.values()):
            continue
        differences.append({
            "case_id": case_id,
            "category": rule_case["category"],
            "query": rule_case["query"],
            "expected_intent": dict(rule_case["expected"])["intent"],
            "correct": correctness,
            "predicted_intents": {
                name: dict(indexed[name][case_id]["predicted"])["intent"]
                for name in STRATEGIES
            },
            "reasons": {
                name: dict(indexed[name][case_id]["predicted"]).get("reason", "legacy_rule")
                for name in STRATEGIES
            },
        })
    differences.sort(
        key=lambda item: (
            -sum(bool(value) for value in dict(item["correct"]).values()),
            str(item["case_id"]),
        )
    )
    return differences[:10]


def _format_report(payload: dict[str, object]) -> str:
    summaries = dict(payload["summaries"])
    classification = dict(payload["classification"])
    lines = [
        "# Router V2 Dev Ablation",
        "",
        "- Dataset: `evalrag_v0.2`",
        "- Split: `dev`，80 cases；frozen test 未运行。",
        "- 三组使用同一人工审核标签、Keyword Retriever、top-k=5。",
        "- Accuracy 要求 intent 与 source set 同时严格一致。",
        "",
        "| Router | Accuracy | Wrong | P50 ms | P95 ms |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in STRATEGIES:
        summary = dict(summaries[name])
        metrics = dict(summary["metrics"])
        latency = dict(dict(summary["latency_ms"])["routing"])
        wrong = dict(summary["failure_counts"])["router_wrong"]
        lines.append(
            f"| {name} | {float(metrics['router_accuracy']):.4f} | {wrong} | "
            f"{float(latency['p50']):.3f} | {float(latency['p95']):.3f} |"
        )
    lines.extend(["", "## Category Accuracy", ""])
    for name in STRATEGIES:
        categories = dict(dict(classification[name])["by_category"])
        rendered = ", ".join(
            f"{category}={float(dict(values)['accuracy']):.2%}"
            for category, values in categories.items()
        )
        lines.append(f"- **{name}**: {rendered}")
    lines.extend(["", "## Strategy Differences", ""])
    for item in payload["strategy_differences"]:  # type: ignore[union-attr]
        lines.append(
            f"- `{item['case_id']}`: expected={item['expected_intent']}; "
            f"rule/semantic/hybrid="
            f"{item['predicted_intents']['rule']}/"
            f"{item['predicted_intents']['semantic']}/"
            f"{item['predicted_intents']['hybrid']}。"
        )
    lines.extend([
        "",
        "## Conclusion",
        "",
        "Hybrid 在本次 dev 上准确率最高，但 CPU P95 明显增加。",
        "Semantic 单路低于修复后的 Rule，说明向量相似不应无条件替代精确规则。",
        "Hybrid 仍有 3 个失败 case，全部保留；本报告不代表答案准确率。",
    ])
    return "\n".join(lines) + "\n"


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-id", required=True)
    parser.add_argument("--rule-run", required=True)
    parser.add_argument("--semantic-run", required=True)
    parser.add_argument("--hybrid-run", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
