from __future__ import annotations

from collections import Counter
import json
from pathlib import Path


DEV_RUN = "p0-d5-v02-dev-20260804-deepseek-v4-flash"
TEST_RUN = "p0-d5-v02-frozen-test-20260804-deepseek-v4-flash"
OUTPUT_DIR = Path("reports/final/p0-d5-live-llm-v0.2")


def main() -> int:
    """汇总真实 DeepSeek Pipeline 的 dev、frozen、token、成本和失败证据。"""

    dev = _load_run(DEV_RUN)
    test = _load_run(TEST_RUN)
    summary = {
        "task": "P0-D5 live LLM completion",
        "model": "deepseek-v4-flash",
        "thinking": "disabled",
        "prompt_version": "p0-deepseek-json-v1",
        "dataset_version": "evalrag_v0.2",
        "dev": dev,
        "frozen_test": test,
        "review_boundary": (
            "Unsupported answers were reviewed by Codex against cited current "
            "context; this is not independent human review or LLM-as-a-Judge."
        ),
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (OUTPUT_DIR / "report.md").write_text(
        _build_markdown(summary), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _load_run(run_id: str) -> dict[str, object]:
    run_dir = Path("reports/runs") / run_id
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    cases = _read_jsonl(run_dir / "case_results.jsonl")
    failures = _read_jsonl(run_dir / "failures.jsonl")
    traces = _read_jsonl(run_dir / "traces.jsonl")
    return {
        "run_id": run_id,
        "case_count": summary["case_count"],
        "metrics": summary["metrics"],
        "counts": summary["counts"],
        "status_counts": dict(Counter(case["status"] for case in cases)),
        "error_counts": dict(Counter(
            str(case.get("error_type") or "none") for case in cases
        )),
        "failure_counts": dict(Counter(
            failure["failure_type"] for failure in failures
        )),
        "format_retry_count": sum(
            attempt.get("type") == "format_repair"
            for trace in traces
            for attempt in trace["attempts"]
        ),
        "citation_invalid_case_ids": [
            case["case_id"]
            for case in cases
            if case.get("error_type") == "citation_invalid"
        ],
        "latency_ms": summary["latency_ms"],
        "tokens": summary["tokens"],
        "estimated_cost_usd": summary["estimated_cost_usd"],
        "support_review": summary["support_review"],
        "run_path": str(run_dir),
    }


def _build_markdown(summary: dict[str, object]) -> str:
    dev = summary["dev"]
    test = summary["frozen_test"]
    return f"""# DeepSeek 真实 LLM 端到端报告

## 固定配置

- Model：`deepseek-v4-flash`
- Mode：非思考模式
- Prompt：`p0-deepseek-json-v1`
- Dataset：`evalrag_v0.2`，80 dev / 40 frozen test
- Pipeline：Hybrid Router -> Hybrid Retriever -> Evidence Gate -> Context -> DeepSeek JSON Generator -> Citation Validator -> Trace

## 结果

| Split | Cases | Citation Validity | Key-Point Coverage | Abstention Accuracy | Unsupported Answer Rate | E2E Success | P95 Total |
|---|---:|---:|---:|---:|---:|---:|---:|
| dev | {dev['case_count']} | {dev['metrics']['citation_validity']:.2%} | {dev['metrics']['key_point_coverage']:.2%} | {dev['metrics']['abstention_accuracy']:.2%} | {dev['metrics']['unsupported_answer_rate']:.2%} | {dev['metrics']['end_to_end_success_rate']:.2%} | {dev['latency_ms']['total']['p95']:.2f} ms |
| frozen test | {test['case_count']} | {test['metrics']['citation_validity']:.2%} | {test['metrics']['key_point_coverage']:.2%} | {test['metrics']['abstention_accuracy']:.2%} | {test['metrics']['unsupported_answer_rate']:.2%} | {test['metrics']['end_to_end_success_rate']:.2%} | {test['latency_ms']['total']['p95']:.2f} ms |

## Token 与成本

| Split | LLM Calls | Input | Output | Total | Estimated Cost |
|---|---:|---:|---:|---:|---:|
| dev | {dev['tokens']['llm_call_count']} | {dev['tokens']['input_tokens']} | {dev['tokens']['output_tokens']} | {dev['tokens']['total_tokens']} | ${dev['estimated_cost_usd']:.6f} |
| frozen test | {test['tokens']['llm_call_count']} | {test['tokens']['input_tokens']} | {test['tokens']['output_tokens']} | {test['tokens']['total_tokens']} | ${test['estimated_cost_usd']:.6f} |

价格按 2026-08-04 官方美元单价快照估算；实际账单以 Provider 为准。

## 真实失败

- dev：{dev['failure_counts']}
- frozen test：{test['failure_counts']}
- frozen `citation_invalid`：{test['citation_invalid_case_ids']}
- 两个 split 均未发生 JSON format retry；Validator 各拦截 1 条 `sufficient=false` 但仍携带 citations 的不一致输出。
- 当前首要失败是 Evidence Gate 的 required-source coverage 过严导致 unexpected abstention。

## 审核边界

Answered case 由 Codex 逐条对照引用 Context 检查事实支持性，不是独立人工审核，也不是 LLM-as-a-Judge。Citation Validity 只证明引用 ID 合法；Key-Point Coverage 使用规范化子串匹配。

## 工件

- Dev：`reports/runs/{DEV_RUN}/`
- Frozen test：`reports/runs/{TEST_RUN}/`
- 每个目录包含 config、case results、failures、traces、latency、support review 和 summary。
"""


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


if __name__ == "__main__":
    raise SystemExit(main())
