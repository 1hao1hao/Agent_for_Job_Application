from __future__ import annotations

import json
from pathlib import Path


OUTPUT_DIR = Path("reports/final/p0-d5-v0.2")


def main() -> int:
    """汇总 P0-D5 已落盘工件，不重新运行或修改任何 prediction。"""

    rerank = _read_json(
        Path("reports/ablations/p0-d5-reranker-dev-20260804/summary.json")
    )
    frozen = _read_json(
        Path("reports/comparisons/p0-d5-v02-frozen-test-20260804/summary.json")
    )
    e2e = _read_json(
        Path("reports/runs/p0-d5-v02-frozen-test-20260804-extractive-e2e/summary.json")
    )
    regression = _read_json(
        Path("reports/regression/p0-d5-v0.2/summary.json")
    )
    summary = {
        "task_id": "P0-D5",
        "dataset_version": "evalrag_v0.2",
        "frozen_test_case_count": frozen["case_count"],
        "selected_final_strategy": frozen["selected_final_strategy"],
        "reranker_enabled": frozen["final_reranker_enabled"],
        "reranker_dev": {
            "baseline": rerank["baseline"]["metrics"],
            "candidate": rerank["rerank"]["metrics"],
            "decision": rerank["decision"],
        },
        "frozen_retrieval": {
            name: run["summary"]["metrics"]
            for name, run in frozen["runs"].items()
        },
        "end_to_end_extractive": e2e["metrics"],
        "end_to_end_counts": e2e["counts"],
        "end_to_end_total_latency_ms": e2e["latency_ms"]["total"],
        "tokens": e2e["tokens"],
        "estimated_cost_usd": e2e["estimated_cost_usd"],
        "regression": {
            "fixed_count": regression["fixed_count"],
            "fixed_passed": regression["fixed_passed"],
            "fixed_pass_rate": regression["fixed_pass_rate"],
            "open_count": regression["open_count"],
        },
        "limitations": [
            "The neural CrossEncoder adapter was implemented but not used in the formal ablation because model weights were unavailable locally.",
            "The end-to-end run uses deterministic extractive generation, not a live LLM.",
            "Unsupported Answer Rate is contract-level and is not independent human semantic grading.",
            "Frozen test predictions were not rerun after inspection; a metric bug was fixed by recomputing from saved case results.",
        ],
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _write_json(OUTPUT_DIR / "summary.json", summary)
    (OUTPUT_DIR / "report.md").write_text(
        _build_markdown(summary), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _build_markdown(summary: dict[str, object]) -> str:
    retrieval = summary["frozen_retrieval"]
    e2e = summary["end_to_end_extractive"]
    regression = summary["regression"]
    latency = summary["end_to_end_total_latency_ms"]
    rows = "\n".join(
        f"| {name} | {values['recall_at_3']:.2%} | {values['recall_at_5']:.2%} | {values['mrr']:.2%} |"
        for name, values in retrieval.items()  # type: ignore[union-attr]
    )
    return f"""# P0-D5 最终报告

## 范围

- 数据集：`evalrag_v0.2`
- Frozen test：40 条，其中可回答 30 条、不可回答 10 条
- 执行规则：每种声明策略只运行一次，查看 test 后不调参
- 最终检索：`{summary['selected_final_strategy']}`；Reranker 在 dev 负向消融后关闭

## Frozen Retrieval

| 策略 | Recall@3 | Recall@5 | MRR |
|---|---:|---:|---:|
{rows}

Hybrid 的 frozen-test MRR 最高（{retrieval['hybrid']['mrr']:.2%}），Recall@5 与 Dense 并列最高（{retrieval['hybrid']['recall_at_5']:.2%}），因此被选为最终策略。

## Reranker 决策

仅在 dev 上运行的 token-overlap reranker 将 Recall@3 从 {summary['reranker_dev']['baseline']['recall_at_3']:.2%} 降至 {summary['reranker_dev']['candidate']['recall_at_3']:.2%}，Recall@5 从 {summary['reranker_dev']['baseline']['recall_at_5']:.2%} 降至 {summary['reranker_dev']['candidate']['recall_at_5']:.2%}，MRR 从 {summary['reranker_dev']['baseline']['mrr']:.2%} 降至 {summary['reranker_dev']['candidate']['mrr']:.2%}，因此最终关闭。仓库保留真实 CrossEncoder adapter 和 Fake scorer 测试，但本报告不声称获得了神经 Reranker 实测结果。

## 端到端 Extractive Baseline

| 指标 | 结果 |
|---|---:|
| Citation Validity | {e2e['citation_validity']:.2%} |
| Key-Point Coverage | {e2e['key_point_coverage']:.2%} |
| Abstention Accuracy | {e2e['abstention_accuracy']:.2%} |
| Unsupported Answer Rate | {e2e['unsupported_answer_rate']:.2%} |
| End-to-End Success Rate | {e2e['end_to_end_success_rate']:.2%} |
| 端到端延迟 P50 | {latency['p50']:.2f} ms |
| 端到端延迟 P95 | {latency['p95']:.2f} ms |

Generator 是 deterministic extractive 实现，因此 token 与 API 估算成本不可用。Citation Validity 只检查引用 ID 是否存在于当前 Context；Unsupported Answer Rate 是 extractive contract 级结果，不是独立语义核验。

## Regression

- Fixed：{regression['fixed_passed']}/{regression['fixed_count']} 通过（{regression['fixed_pass_rate']:.2%}）
- Open：{regression['open_count']} 条，不进入通过率分母

## 证据路径

- Reranker dev 消融：`reports/ablations/p0-d5-reranker-dev-20260804/`
- Frozen retrieval 对照：`reports/comparisons/p0-d5-v02-frozen-test-20260804/`
- 端到端 case results 与 traces：`reports/runs/p0-d5-v02-frozen-test-20260804-extractive-e2e/`
- Executable regression：`reports/regression/p0-d5-v0.2/`

## 审计说明

Frozen predictions 保存后发现指标实现将“可回答但拒答”误记为 unknown，而不是失败。系统没有重新运行 predictions，只从原始 `case_results.jsonl` 重算指标；修复前摘要保留为 `summary_before_metric_fix.json`。
"""


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
