from __future__ import annotations

import json
from pathlib import Path
import shutil

from intern_rag.evaluation import build_end_to_end_failures
from intern_rag.evaluation.metrics import summarize_end_to_end_results


RUN_DIR = Path("reports/runs/p0-d5-v02-frozen-test-20260804-extractive-e2e")


def main() -> int:
    """只从已保存 prediction 重算指标，不重新运行 frozen test。"""

    case_results = _read_jsonl(RUN_DIR / "case_results.jsonl")
    summary_path = RUN_DIR / "summary.json"
    previous_path = RUN_DIR / "summary_before_metric_fix.json"
    if previous_path.exists():
        print(f"ERROR: metric fix audit artifact already exists: {previous_path}")
        return 2

    previous = json.loads(summary_path.read_text(encoding="utf-8"))
    metrics = summarize_end_to_end_results(
        case_results,
        key_point_threshold=float(previous["key_point_threshold"]),
    )
    shutil.copy2(summary_path, previous_path)
    for key in (
        "metrics",
        "counts",
        "category_metrics",
        "key_point_threshold",
        "metric_formulas",
    ):
        previous[key] = metrics[key]
    previous["metric_recomputed_from_saved_predictions"] = True
    previous["metric_fix"] = (
        "answerable abstention is an explicit end-to-end failure"
    )
    _write_json(summary_path, previous)
    _write_jsonl(
        RUN_DIR / "failures.jsonl",
        build_end_to_end_failures(metrics["case_metrics"]),  # type: ignore[arg-type]
    )
    print(json.dumps(previous["metrics"], ensure_ascii=False, indent=2))
    return 0


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
