from __future__ import annotations

import hashlib
import json
from pathlib import Path


DEV_RUN = "p0-d6-v02-dev-20260804-deepseek-v4-flash-v2"
TEST_RUN = "p0-d6-v02-test-20260804-deepseek-v4-flash"
DEV_SOURCE = "p0-d5-v02-dev-20260804-deepseek-v4-flash"
TEST_SOURCE = "p0-d5-v02-frozen-test-20260804-deepseek-v4-flash"


def main() -> int:
    """汇总 P0-D6 dev/test 审核，并记录只读 source prediction 的 SHA-256。"""

    dev = _read_summary(DEV_RUN)
    test = _read_summary(TEST_RUN)
    source_hashes = {
        "dev_case_results": _sha256(_source_path(DEV_SOURCE, "case_results_before_support_review.jsonl")),
        "dev_traces": _sha256(_source_path(DEV_SOURCE, "traces.jsonl")),
        "test_case_results": _sha256(_source_path(TEST_SOURCE, "case_results_before_support_review.jsonl")),
        "test_traces": _sha256(_source_path(TEST_SOURCE, "traces.jsonl")),
    }
    output = Path("reports/final/p0-d6-semantic-grounding-v0.2")
    output.mkdir(parents=True, exist_ok=True)
    combined = {
        "dataset_version": "evalrag_v0.2",
        "grader_model": "deepseek-v4-flash",
        "dev_run_id": DEV_RUN,
        "test_run_id": TEST_RUN,
        "source_prediction_hashes": source_hashes,
        "dev": _compact(dev),
        "test": _compact(test),
        "frozen_predictions_regenerated": False,
        "independent_human_review": False,
    }
    (output / "summary.json").write_text(
        json.dumps(combined, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output / "report.md").write_text(
        _report(dev, test, source_hashes), encoding="utf-8"
    )
    return 0


def _compact(summary: dict[str, object]) -> dict[str, object]:
    return {
        "case_count": summary["case_count"],
        "metrics": summary["metrics"],
        "coverage_comparison": summary["coverage_comparison"],
        "grader_usage": summary["grader_usage"],
        "grounding_review": summary["grounding_review"],
    }


def _report(
    dev: dict[str, object],
    test: dict[str, object],
    hashes: dict[str, str],
) -> str:
    dev_comparison = dict(dev["coverage_comparison"])  # type: ignore[arg-type]
    test_comparison = dict(test["coverage_comparison"])  # type: ignore[arg-type]
    dev_metrics = dict(dev["metrics"])  # type: ignore[arg-type]
    test_metrics = dict(test["metrics"])  # type: ignore[arg-type]
    dev_usage = dict(dev["grader_usage"])  # type: ignore[arg-type]
    test_usage = dict(test["grader_usage"])  # type: ignore[arg-type]
    dev_review = dict(dev["grounding_review"])  # type: ignore[arg-type]
    test_review = dict(test["grounding_review"])  # type: ignore[arg-type]
    return f"""# P0-D6 Semantic Metrics 与 Claim-Level Grounding Audit

## 实验边界

- Dataset：`evalrag_v0.2`，80 dev / 40 frozen test。
- Grader：`deepseek-v4-flash`，Key-Point `p0-d6-key-point-v1` 与 Grounding
  `p0-d6-grounding-v1`。
- 只读取 P0-D5 已保存 predictions 与 Trace，没有重新运行 Router、Retriever、Generator。
- Judge 与 Generator 使用同一模型家族，不是独立人工审核，存在自评偏差。

## Key-Point Coverage 对照

| Split | Lexical | Semantic | Delta | Improved | Regressed | Unknown |
|---|---:|---:|---:|---:|---:|---:|
| dev | {_pct(dev_comparison['lexical_macro_coverage'])} | {_pct(dev_comparison['semantic_macro_coverage'])} | {_delta(dev_comparison)} | {dev_comparison['improved_case_count']} | {dev_comparison['regressed_case_count']} | {dev_comparison['unknown_case_count']} |
| frozen test | {_pct(test_comparison['lexical_macro_coverage'])} | {_pct(test_comparison['semantic_macro_coverage'])} | {_delta(test_comparison)} | {test_comparison['improved_case_count']} | {test_comparison['regressed_case_count']} | {test_comparison['unknown_case_count']} |

dev 与 frozen 各保存 10 条差异/一致 Case。语义评分能识别“岗位时效性/下架版本”、
“工具调用/外部操作”等同义表达；dev 的 `v02_multi_005` 出现一次退化，未删除。

## Claim-Level Grounding

| Split | Answered | Known | Unknown | Unsupported Answer Rate | E2E |
|---|---:|---:|---:|---:|---:|
| dev | {dev_review['answered_count']} | {dev_review['known_count']} | {dev_review['unknown_count']} | {_pct(dev_metrics['unsupported_answer_rate'])} | {_pct(dev_metrics['end_to_end_success_rate'])} |
| frozen test | {test_review['answered_count']} | {test_review['known_count']} | {test_review['unknown_count']} | {_pct(test_metrics['unsupported_answer_rate'])} | {_pct(test_metrics['end_to_end_success_rate'])} |

dev 找到 1 条 unsupported factual claim：回答断言未引用的 resume chunks 主题与问题
不同。frozen 的 UAR 0% 只表示 20 条已知 verdict 中没有 unsupported；另有 3 条
unknown，因此不能写成“全部 23 条回答零幻觉”，E2E 也按协议保持 unavailable。

## Judge 延迟、Token 与成本

| Split | Calls | Unavailable | P50 | P95 | Tokens | Estimated Cost |
|---|---:|---:|---:|---:|---:|---:|
| dev | {dev_usage['call_count']} | {dev_usage['unavailable_count']} | {float(dev_usage['p50_latency_ms']):.2f} ms | {float(dev_usage['p95_latency_ms']):.2f} ms | {dict(dev_usage['tokens'])['total_tokens']} | ${float(dev_usage['estimated_cost_usd']):.6f} |
| frozen test | {test_usage['call_count']} | {test_usage['unavailable_count']} | {float(test_usage['p50_latency_ms']):.2f} ms | {float(test_usage['p95_latency_ms']):.2f} ms | {dict(test_usage['tokens'])['total_tokens']} | ${float(test_usage['estimated_cost_usd']):.6f} |

## Source Prediction Hashes

```text
dev case results: {hashes['dev_case_results']}
dev traces:       {hashes['dev_traces']}
test case results:{hashes['test_case_results']}
test traces:      {hashes['test_traces']}
```

## 工件

- Dev：`reports/runs/{DEV_RUN}/`
- Frozen：`reports/runs/{TEST_RUN}/`
- 每个目录包含 `point_verdicts.jsonl`、`claim_verdicts.jsonl`、
  `grader_calls.jsonl`、`case_results.jsonl`、`failures.jsonl` 和报告。

## 结论边界

Semantic Coverage 是模型评分，不等于答案准确率。Citation Validity 仍只验证 ID；
Claim-Level Grounding 才核对事实与引用文本。unknown 不进入 UAR 已审核分母，并使
完整 E2E unavailable，避免把 Judge 失败伪装成 supported。
"""


def _delta(comparison: dict[str, object]) -> str:
    lexical = float(comparison["lexical_macro_coverage"])
    semantic = float(comparison["semantic_macro_coverage"])
    return f"{(semantic - lexical) * 100:+.2f} pp"


def _pct(value: object) -> str:
    return "unavailable" if value is None else f"{float(value):.2%}"


def _read_summary(run_id: str) -> dict[str, object]:
    path = Path("reports/runs") / run_id / "summary.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _source_path(run_id: str, name: str) -> Path:
    return Path("reports/runs") / run_id / name


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
