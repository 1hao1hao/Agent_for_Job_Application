from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

from intern_rag.agent import build_context
from intern_rag.evaluation import load_chunks_jsonl
from intern_rag.retrieval import RetrievalResult


def main() -> int:
    """复用已保存的检索预测，对比两种 Context 预算选择策略。"""

    args = _parse_args()
    chunks = {
        chunk.id: chunk for chunk in load_chunks_jsonl(Path(args.chunks))
    }
    cases = _read_jsonl(Path(args.case_results))
    budgets = [int(value) for value in args.budgets.split(",")]
    comparisons = [
        _evaluate_strategy(cases, chunks, budget, strategy)
        for budget in budgets
        for strategy in ("rank_prefix", "source_balanced")
    ]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "source_run": args.case_results,
        "scope": "answerable dev cases with correct Router prediction",
        "comparisons": comparisons,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "report.md").write_text(
        _render_report(summary), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _evaluate_strategy(
    cases: list[dict[str, object]],
    chunks: dict[str, object],
    budget: int,
    strategy: str,
) -> dict[str, object]:
    """逐 Case 构建 Context，并汇总证据召回、来源覆盖与预算利用率。"""

    rows: list[dict[str, float | bool | int]] = []
    for case in cases:
        metrics = dict(case["metrics"])  # type: ignore[arg-type]
        if not bool(case["answerable"]) or not bool(metrics["router_correct"]):
            continue
        predicted = dict(case["predicted"])  # type: ignore[arg-type]
        expected = dict(case["expected"])  # type: ignore[arg-type]
        sources = list(predicted["sources"])  # type: ignore[arg-type]
        results = _retrieval_results(predicted, chunks)
        context = build_context(
            str(case["query"]),
            results,
            max_chars=budget,
            strategy=strategy,  # type: ignore[arg-type]
            required_source_types=sources,
        )
        relevant = set(expected["relevant_chunk_ids"])  # type: ignore[arg-type]
        used = set(context.used_chunk_ids)
        rows.append(
            {
                "relevant_recall": len(used & relevant) / len(relevant),
                "source_coverage": (
                    len(set(context.covered_source_types) & set(sources))
                    / len(set(sources))
                    if sources else 1.0
                ),
                "full_source_coverage": not context.missing_source_types,
                "used_chunks": len(context.used_chunk_ids),
                "budget_utilization": context.char_count / budget,
            }
        )
    return {
        "strategy": strategy,
        "budget": budget,
        "case_count": len(rows),
        "macro_relevant_recall": mean(float(row["relevant_recall"]) for row in rows),
        "macro_source_coverage": mean(float(row["source_coverage"]) for row in rows),
        "full_source_coverage_rate": mean(
            float(bool(row["full_source_coverage"])) for row in rows
        ),
        "mean_used_chunks": mean(float(row["used_chunks"]) for row in rows),
        "mean_budget_utilization": mean(
            float(row["budget_utilization"]) for row in rows
        ),
    }


def _retrieval_results(
    predicted: dict[str, object],
    chunks: dict[str, object],
) -> list[RetrievalResult]:
    """把保存的检索预测重新组装成 Context Builder 所需的统一结果。"""

    results: list[RetrievalResult] = []
    for item_value in predicted["retrieved"]:  # type: ignore[union-attr]
        item = dict(item_value)
        chunk_id = str(item["chunk_id"])
        results.append(
            RetrievalResult(
                chunk_id=chunk_id,
                score=float(item["score"]),
                rank=int(item["rank"]),
                chunk=chunks[chunk_id],  # type: ignore[arg-type]
                reason=str(item["reason"]),
                details=dict(item.get("details", {})),
            )
        )
    return results


def _render_report(summary: dict[str, object]) -> str:
    """生成可直接复核的 Markdown 消融报告。"""

    lines = [
        "# Context Builder Dev Ablation",
        "",
        f"- Source run: `{summary['source_run']}`",
        f"- Scope: {summary['scope']}",
        "- 唯一变量：Context 预算选择策略；不重新运行 Router/Retriever。",
        "",
        "| Budget | Strategy | Relevant Recall | Source Coverage | Full Source Rate | Mean Chunks | Budget Use |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for value in summary["comparisons"]:  # type: ignore[union-attr]
        row = dict(value)
        lines.append(
            f"| {row['budget']} | {row['strategy']} | "
            f"{float(row['macro_relevant_recall']):.2%} | "
            f"{float(row['macro_source_coverage']):.2%} | "
            f"{float(row['full_source_coverage_rate']):.2%} | "
            f"{float(row['mean_used_chunks']):.2f} | "
            f"{float(row['mean_budget_utilization']):.2%} |"
        )
    lines.extend(
        [
            "",
            "`source_balanced` 先保留每个 Router 必需来源的最高排名 Chunk，再按 rank 填充；",
            "`rank_prefix` 保留原实现。指标只描述 Context 选证据，不等于答案准确率。",
            "",
        ]
    )
    return "\n".join(lines)


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    """读取非空 JSONL 行。"""

    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _parse_args() -> argparse.Namespace:
    """解析 Context 消融输入路径与预算列表。"""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case-results",
        default=(
            "reports/runs/p0-d5-v02-dev-20260804-hybrid-baseline/"
            "case_results.jsonl"
        ),
    )
    parser.add_argument(
        "--chunks", default="data/processed/chunks/evalrag_v0.2.jsonl"
    )
    parser.add_argument("--budgets", default="800,1200,2000,4000")
    parser.add_argument(
        "--output-dir",
        default="reports/ablations/p1-context-builder-v02-dev-20260811",
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
