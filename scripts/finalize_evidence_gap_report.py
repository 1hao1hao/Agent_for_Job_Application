"""整理已保存 prediction 的实验结论，不重新调用 Retriever 或 LLM。"""

import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    """保存配置快照、完整改善/退化清单及中断 E2E 的真实状态。"""
    directory = ROOT / "reports/ablations/p1-evidence-gap-v03-dev-20261009-r2"
    summary = json.loads((directory / "summary.json").read_text())
    rows = {}
    for name in ("current_adaptive", "on_demand_rescue"):
        rows[name] = {
            row["case_id"]: row
            for row in _read(directory / f"{name}_case_results.jsonl")
        }
    improved, regressed, remaining = [], [], []
    for case_id, baseline in rows["current_adaptive"].items():
        if not baseline["answerable"]:
            continue
        candidate = rows["on_demand_rescue"][case_id]
        delta = candidate["recall_at_5"] - baseline["recall_at_5"]
        if delta > 0:
            improved.append(case_id)
        elif delta < 0:
            regressed.append(case_id)
        if candidate["recall_at_5"] == 0:
            remaining.append(case_id)
    summary.update({
        "recall_improved_ids": improved,
        "recall_regressed_ids": regressed,
        "remaining_zero_recall_ids": remaining,
        "reference_prediction_source": "p1-evidence-gap-v03-dev-20261009",
        "reference_scope": "current Adaptive first pass with live Router source filter",
        "command": "PYTHONPATH=src python scripts/run_evidence_gap_ablation.py --run-id p1-evidence-gap-v03-dev-20261009-r2 --reuse-reference-run p1-evidence-gap-v03-dev-20261009",
        "code_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip() + "-dirty",
        "deployment_status": "opt-in candidate; existing service default retained pending complete live E2E",
    })
    _write(directory / "summary.json", summary)
    for name in ("base_config", "rescue_config", "gate_config"):
        _write(directory / f"{name}_snapshot.json", json.loads((ROOT / summary[name]).read_text()))
    partial = ROOT / "reports/runs/p1-evidence-gap-e2e-v03-dev-20261009"
    cases = _read(partial / "case_results.jsonl")
    traces = _read(partial / "traces.jsonl")
    gateway_calls = [
        attempt["model_gateway"] for trace in traces
        for attempt in trace.get("attempts", []) if attempt.get("model_gateway")
    ]
    successful_calls = [
        attempt for call in gateway_calls for attempt in call.get("attempts", [])
        if attempt.get("status") == "succeeded"
    ]
    _write(partial / "status.json", {
        "status": "INTERRUPTED_CONNECTION_ERRORS",
        "completed_cases": len(cases), "planned_cases": 160,
        "generation_calls": len(gateway_calls),
        "successful_provider_calls": len(successful_calls),
        "reported_total_tokens": sum(int(item.get("tokens", {}).get("total_tokens", 0)) for item in successful_calls),
        "reason": "Repeated provider connection_error and circuit opening; stopped without rerunning predictions.",
        "complete_e2e_comparison": "NOT TESTED",
        "configuration": "pre-r2 candidate; partial results cannot validate final candidate",
    })
    with (directory / "report.md").open("a", encoding="utf-8") as stream:
        stream.write(
            "\n## Final interpretation\n\n"
            f"- Recall improved: {len(improved)} cases; regressed: {len(regressed)}; zero-recall remaining: {len(remaining)}.\n"
            "- Reference is current Adaptive first pass with the same live Router source filter, not the previous whole E2E run or full-library benchmark.\n"
            "- Reference/fixed predictions are reused from the first run; the candidate alone was rerun after two dev fixes.\n"
            "- Path completeness means all labeled edge IDs appear in returned evidence; it is not answer correctness.\n"
            "- P95 is CPU wall time; sequential runs are not controlled hardware repetitions.\n"
            "- Full live E2E: **NOT TESTED**. A pre-r2 run completed only 14/160 cases before connection failures; artifacts are preserved.\n"
            "- Keep rescue as opt-in; existing service default stays until live E2E verifies Gate/Context/citations and latency.\n"
            "- Remaining debt: incomplete entity linking, graph groups exceeding top-k/budget, strict anchor/source checks causing refusals, and high rescue P95.\n"
        )
    print(json.dumps({"improved": len(improved), "regressed": len(regressed), "remaining_zero_recall": len(remaining), "partial_e2e_cases": len(cases)}, indent=2))


def _read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
