from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys


FINAL_CONFIG = Path("configs/final/p0_v0.2.json")
COMPARISON_ID = "p0-d5-v02-frozen-test-20260804"
STRATEGIES = {
    "keyword": "configs/retrieval/keyword_v0.2.json",
    "dense": "configs/retrieval/dense_v0.2.json",
    "hybrid": "configs/retrieval/hybrid_v0.2.json",
    "hybrid_rerank_candidate": "configs/retrieval/hybrid_rerank_v0.2.json",
}


def main() -> int:
    """校验冻结配置后，每种声明策略只运行一次 test 并保存对照。"""

    report_dir = Path("reports/comparisons") / COMPARISON_ID
    if report_dir.exists():
        print(f"ERROR: frozen comparison already exists: {report_dir}")
        return 2
    final_config = json.loads(FINAL_CONFIG.read_text(encoding="utf-8"))
    _verify_frozen_inputs(final_config)
    run_dirs: dict[str, Path] = {}
    for strategy, config_path in STRATEGIES.items():
        run_id = f"{COMPARISON_ID}-{strategy}"
        command = [
            sys.executable,
            "scripts/run_evaluation.py",
            "--dataset-version", "evalrag_v0.2",
            "--split", "test",
            "--config", config_path,
            "--router-config", str(final_config["router_config"]),
            "--run-id", run_id,
            "--allow-frozen-test",
        ]
        completed = subprocess.run(command, check=False)
        if completed.returncode != 0:
            return completed.returncode
        run_dirs[strategy] = Path("reports/runs") / run_id

    payload = {
        "comparison_id": COMPARISON_ID,
        "config_version": final_config["config_version"],
        "dataset_version": "evalrag_v0.2",
        "split": "test",
        "case_count": 40,
        "test_execution_policy": "one run per declared strategy; no post-test tuning",
        "final_reranker_enabled": final_config["reranker_enabled"],
        "selected_final_strategy": "hybrid",
        "runs": {
            name: {
                "path": str(path),
                "summary": _read_json(path / "summary.json"),
            }
            for name, path in run_dirs.items()
        },
    }
    report_dir.mkdir(parents=True)
    (report_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (report_dir / "report.md").write_text(_format_report(payload), encoding="utf-8")
    print(f"Frozen comparison written to {report_dir}")
    return 0


def _verify_frozen_inputs(config: dict[str, object]) -> None:
    checks = {
        "data/evaluation/evalrag_v0.2.jsonl": config["dataset_sha256"],
        "data/processed/chunks/evalrag_v0.2.jsonl": config["chunks_sha256"],
        str(config["router_config"]): config["router_config_sha256"],
        str(config["retriever_config"]): config["retriever_config_sha256"],
        str(config["evidence_config"]): config["evidence_config_sha256"],
    }
    for path_text, expected in checks.items():
        actual = hashlib.sha256(Path(path_text).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"frozen input changed: {path_text}")


def _format_report(payload: dict[str, object]) -> str:
    lines = [
        "# EvalRAG v0.2 Frozen Test Retrieval Comparison",
        "",
        "- Split: `test`; 40 cases；每种策略只运行一次。",
        "- 配置在查看 test 前已冻结；查看结果后不继续调参。",
        "- dev 决策已禁用 token-overlap Reranker，candidate 仅作为固定负对照。",
        "",
        "| Strategy | Router Accuracy | Recall@3 | Recall@5 | MRR | Retrieval P95 ms |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, run in payload["runs"].items():  # type: ignore[union-attr]
        summary = run["summary"]
        metrics = summary["metrics"]
        latency = summary["latency_ms"]["retrieval"]
        lines.append(
            f"| {name} | {metrics['router_accuracy']:.4f} | "
            f"{metrics['recall_at_3']:.4f} | {metrics['recall_at_5']:.4f} | "
            f"{metrics['mrr']:.4f} | {latency['p95']:.3f} |"
        )
    lines.extend([
        "",
        "## Boundary",
        "",
        "这些指标只衡量 Router/Retrieval，不等于答案准确率。所有失败 case 均保留在各 run 的 failures.jsonl。",
    ])
    return "\n".join(lines) + "\n"


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    raise SystemExit(main())
