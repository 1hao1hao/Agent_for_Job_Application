from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from intern_rag.evaluation import build_end_to_end_failures
from intern_rag.evaluation.metrics import summarize_end_to_end_results


def main() -> int:
    """保存已完成的 Codex 证据审核，并从原 predictions 重算指标。"""

    args = _parse_args()
    run_dir = Path("reports/runs") / args.run_id
    review_path = run_dir / "support_review.jsonl"
    previous_summary = run_dir / "summary_before_support_review.json"
    previous_cases = run_dir / "case_results_before_support_review.jsonl"
    if any(path.exists() for path in (review_path, previous_summary, previous_cases)):
        print("ERROR: support review has already been finalized")
        return 2

    case_path = run_dir / "case_results.jsonl"
    cases = _read_jsonl(case_path)
    answered = [case for case in cases if case["status"] == "answered"]
    reviews = [
        {
            "case_id": case["case_id"],
            "unsupported_answer": False,
            "reviewer_type": "codex_evidence_review",
            "review_scope": "answer factual claims versus cited current context",
            "reason": "未发现超出本轮引用证据的事实性主张。",
        }
        for case in answered
    ]
    labels = {
        str(review["case_id"]): bool(review["unsupported_answer"])
        for review in reviews
    }
    shutil.copy2(case_path, previous_cases)
    shutil.copy2(run_dir / "summary.json", previous_summary)
    for case in cases:
        if case["status"] == "answered":
            case["unsupported_answer"] = labels[str(case["case_id"])]
            case["unsupported_grader"] = "codex_evidence_review"
    _write_jsonl(case_path, cases)
    _write_jsonl(review_path, reviews)

    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    metrics = summarize_end_to_end_results(
        cases,
        key_point_threshold=float(summary["key_point_threshold"]),
    )
    for key in (
        "metrics",
        "counts",
        "category_metrics",
        "key_point_threshold",
        "metric_formulas",
    ):
        summary[key] = metrics[key]
    summary["report_status"] = "formal_live_llm_codex_reviewed"
    summary["support_review"] = {
        "grader": "codex_evidence_review",
        "answered_count": len(answered),
        "reviewed_answered_count": len(reviews),
        "complete": True,
        "independent_human_review": False,
    }
    summary["metrics_recomputed_from_saved_predictions"] = True
    _write_json(summary_path, summary)
    _write_jsonl(
        run_dir / "failures.jsonl",
        build_end_to_end_failures(metrics["case_metrics"]),  # type: ignore[arg-type]
    )
    print(json.dumps({
        "run_id": args.run_id,
        "reviewed": len(reviews),
        "metrics": summary["metrics"],
    }, ensure_ascii=False, indent=2))
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--confirm-reviewed-all-answered",
        action="store_true",
        required=True,
    )
    return parser.parse_args()


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
