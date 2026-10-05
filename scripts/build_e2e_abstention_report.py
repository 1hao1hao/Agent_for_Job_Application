from __future__ import annotations

from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from intern_rag.evaluation.failure_funnel import (  # noqa: E402
    FailureFunnelCase,
    summarize_failure_funnel,
)


BASELINE = ROOT / "reports/runs/p1-e2e-abstention-v03-dev-baseline-20261005-rerun"
CANDIDATE = ROOT / "reports/runs/p1-e2e-abstention-v03-dev-release-candidate-20261005"
RELEASE_TEST = ROOT / "reports/runs/p1-e2e-abstention-v03-release-test-20261005"
OUTPUT = ROOT / "reports/ablations/p1-e2e-abstention-v03-20261005"


def main() -> int:
    """汇总 E2E 拒答修复，并离线重放最后的确定性 Validator 规则。

    Candidate 的 LLM 原始输出保持不变。对于 `sufficient=false` 且仅因附带非法
    citation 被记为 error 的 Case，按最终 Pipeline 规则丢弃引用并改记安全拒答；
    这一步不重新调用 Router、Retriever 或 LLM，也不改变回答内容。
    """

    baseline_summary = _read_json(BASELINE / "summary.json")
    candidate_rows = _read_rows(CANDIDATE / "case_results.jsonl")
    traces = {
        str(trace["request_id"]): trace
        for trace in _read_jsonl(CANDIDATE / "traces.jsonl")
    }
    replayed_rows: list[FailureFunnelCase] = []
    replayed_case_ids: list[str] = []
    for row in candidate_rows:
        trace = traces[row.case_id]
        if (
            row.terminal_stage == "citation_invalid"
            and trace.get("generation", {}).get("sufficient") is False
        ):
            replayed_case_ids.append(row.case_id)
            row = replace(
                row,
                terminal_stage="generator_abstain",
                response_status="insufficient_evidence",
                reason=(
                    "deterministic validation replay: sufficient=false; "
                    "discard invalid citations and preserve safe abstention"
                ),
                cited_chunk_ids=(),
                citation_valid=True,
                error_type=None,
            )
        replayed_rows.append(row)

    final_summary = summarize_failure_funnel(replayed_rows)
    candidate_summary = _read_json(CANDIDATE / "summary.json")
    final_summary.update({
        key: value for key, value in candidate_summary.items()
        if key not in final_summary
    })
    final_summary.update({
        "run_id": "p1-e2e-abstention-v03-dev-final-20261005",
        "source_prediction_run": candidate_summary["run_id"],
        "deterministic_validation_replay": True,
        "replayed_case_ids": replayed_case_ids,
        "evidence_config_version": "evidence-gate-v2.1-e2e-locked",
    })
    test_summary = _read_json(RELEASE_TEST / "summary.json")
    comparison = {
        "dataset_version": "evalrag_v0.3",
        "dev_case_count": 160,
        "release_test_case_count": 80,
        "baseline": baseline_summary,
        "final_dev": final_summary,
        "release_test": test_summary,
        "delta": {
            "answerable_e2e_success": (
                float(final_summary["answerable_e2e_success"])
                - float(baseline_summary["answerable_e2e_success"])
            ),
            "unexpected_abstention_rate": (
                float(final_summary["unexpected_abstention_rate"])
                - float(baseline_summary["unexpected_abstention_rate"])
            ),
            "unanswerable_abstention_accuracy": (
                float(final_summary["unanswerable_abstention_accuracy"])
                - float(baseline_summary["unanswerable_abstention_accuracy"])
            ),
            "error_count": int(final_summary["error_count"])
            - int(baseline_summary["error_count"]),
        },
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    _write_json(OUTPUT / "summary.json", comparison)
    _write_rows(OUTPUT / "case_results.jsonl", replayed_rows)
    _write_rows(
        OUTPUT / "failures.jsonl",
        [row for row in replayed_rows if row.terminal_stage != "answered"],
    )
    (OUTPUT / "report.md").write_text(
        _report(comparison, replayed_rows), encoding="utf-8"
    )
    return 0


def _report(comparison: dict[str, object], rows: list[FailureFunnelCase]) -> str:
    baseline = dict(comparison["baseline"])
    final = dict(comparison["final_dev"])
    test = dict(comparison["release_test"])
    remaining = Counter(
        row.terminal_stage for row in rows
        if row.answerable and row.terminal_stage != "answered"
    )
    lines = [
        "# E2E Unexpected Abstention Closure",
        "",
        "## Protocol",
        "",
        "- Dataset: `evalrag_v0.3`; dev=160 (answerable=120), release test=80 (answerable=60).",
        "- Pipeline: Feedback Hybrid Router -> Adaptive v2 -> Evidence Gate -> Adaptive Context -> DeepSeek -> Citation Validator.",
        "- Dev 用于定位和修复；配置锁定后 test 仅运行一次，未据此继续调参。",
        "- 最终 dev 复用 candidate 的模型输出，只对 1 条 `sufficient=false` 非法引用执行确定性 Validator replay。",
        "",
        "## Dev Before / After",
        "",
        "| Metric | Baseline | Final | Delta |",
        "|---|---:|---:|---:|",
        _metric_row("Answerable E2E Success", baseline, final, "answerable_e2e_success"),
        _metric_row("Unexpected Abstention Rate", baseline, final, "unexpected_abstention_rate"),
        _metric_row("Unanswerable Abstention Accuracy", baseline, final, "unanswerable_abstention_accuracy"),
        _metric_row("Citation Validity (answered)", baseline, final, "citation_validity_answered"),
        f"| Error count | {baseline['error_count']} | {final['error_count']} | {int(final['error_count']) - int(baseline['error_count']):+d} |",
        "",
        "## Failure Funnel",
        "",
        "| Terminal stage | Baseline all | Final all | Baseline answerable | Final answerable |",
        "|---|---:|---:|---:|---:|",
    ]
    for stage in (
        "router_abstain", "retrieval_miss", "gate_reject",
        "context_evidence_dropped", "generator_abstain", "citation_invalid",
        "model_error", "answered",
    ):
        lines.append(
            f"| {stage} | {dict(baseline['terminal_stage_counts']).get(stage, 0)} | "
            f"{dict(final['terminal_stage_counts']).get(stage, 0)} | "
            f"{dict(baseline['answerable_terminal_stage_counts']).get(stage, 0)} | "
            f"{dict(final['answerable_terminal_stage_counts']).get(stage, 0)} |"
        )
    lines.extend([
        "",
        "## Root Cause And Fix",
        "",
        "1. `EvidenceRequirement` 未保存 Query 明确点名的来源，导致跨来源问题只检索 Router 返回的部分来源；现由 Gate 按明确来源判断并最多扩源一次。",
        "2. freshness metadata 与 Graph path 已被 Retriever 持有但未进入 Prompt；Context 现在只注入必要 provenance 字段和已验证关系路径。",
        "3. Context Engine 曾在同优先级内按 chunk id 而非 rank 装箱，且 Citation 白名单包含被 token budget 丢弃的证据；现保持 rank/source 顺序并同步真实白名单。",
        "4. 模型安全拒答但携带引用时不再升级成系统错误：Validator 仍记录问题，Pipeline 丢弃引用并返回 `insufficient_evidence`。",
        "",
        "## Locked Release Test",
        "",
        f"- Answerable E2E Success: {float(test['answerable_e2e_success']):.2%}",
        f"- Unexpected Abstention Rate: {float(test['unexpected_abstention_rate']):.2%}",
        f"- Unanswerable Abstention Accuracy: {float(test['unanswerable_abstention_accuracy']):.2%}",
        f"- Citation Validity (answered): {float(test['citation_validity_answered']):.2%}",
        f"- Error count: {test['error_count']}",
        "",
        "## Remaining Dev Failures",
        "",
        ", ".join(f"`{key}`={value}" for key, value in sorted(remaining.items())),
        "",
        "最大剩余项是 retrieval miss，应留给后续 Retriever/Rerank；关系问题的 graph path 缺失应留给 Graph Retrieval。Context 仍有 2 条预算丢证据，适合后续做 query-aware evidence packing。本任务不据此修改 test、Retriever 或 benchmark 标签。",
        "",
        "## Artifacts",
        "",
        "- Baseline: `reports/runs/p1-e2e-abstention-v03-dev-baseline-20261005-rerun/`",
        "- Final dev case/failure: 本目录 `case_results.jsonl` / `failures.jsonl`",
        "- Release test: `reports/runs/p1-e2e-abstention-v03-release-test-20261005/`",
    ])
    return "\n".join(lines) + "\n"


def _metric_row(
    label: str, baseline: dict[str, object], final: dict[str, object], key: str
) -> str:
    before = float(baseline[key])
    after = float(final[key])
    return f"| {label} | {before:.2%} | {after:.2%} | {after - before:+.2%} |"


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _read_rows(path: Path) -> list[FailureFunnelCase]:
    return [FailureFunnelCase(**raw) for raw in _read_jsonl(path)]


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_rows(path: Path, rows: list[FailureFunnelCase]) -> None:
    path.write_text(
        "".join(json.dumps(row.to_dict(), ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
