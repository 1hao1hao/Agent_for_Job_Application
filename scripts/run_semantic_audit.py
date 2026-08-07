from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess

from intern_rag.agent import DeepSeekChatClient
from intern_rag.evaluation import (
    LlmGroundingGrader,
    LlmKeyPointGrader,
    SemanticAuditConfig,
    run_saved_prediction_audit,
    save_semantic_audit_artifacts,
)


def main() -> int:
    """复用 P0-D5 predictions 运行语义要点和 claim-level Grounding 审核。

    脚本读取指定 split 的原始 case results 与 Trace，不会构造 RagPipeline，也不会
    再次调用 Generator。它仅调用 Judge，保存逐 point/claim verdict、重算后的
    Metrics 和差异报告。密钥缺失时在任何网络调用前退出。
    """

    args = _parse_args()
    config_data = json.loads(args.config.read_text(encoding="utf-8"))
    api_key_env = str(config_data["api_key_env"])
    if not os.environ.get(api_key_env, "").strip():
        print(f"ERROR: {api_key_env} is not configured")
        return 2

    source_run_id = str(config_data["source_prediction_runs"][args.split])
    source_dir = Path("reports/runs") / source_run_id
    source_cases = source_dir / "case_results_before_support_review.jsonl"
    trace_path = source_dir / "traces.jsonl"
    if not source_cases.exists() or not trace_path.exists():
        print(f"ERROR: saved prediction artifacts are missing under {source_dir}")
        return 2

    run_id = args.run_id or (
        f"p0-d6-v02-{args.split}-20260804-deepseek-v4-flash"
    )
    output_dir = args.output_dir or Path("reports/runs") / run_id
    if output_dir.exists() and any(output_dir.iterdir()):
        print(f"ERROR: output directory is not empty: {output_dir}")
        return 2


    pricing = config_data["pricing_usd_per_million_tokens"]
    command = (
        "PYTHONPATH=src python scripts/run_semantic_audit.py "
        f"--split {args.split} --run-id {run_id}"
    )
    run_config = SemanticAuditConfig(
        run_id=run_id,
        dataset_version=str(config_data["dataset_version"]),
        split=args.split,
        source_prediction_run_id=source_run_id,
        model=str(config_data["model"]),
        temperature=float(config_data["temperature"]),
        key_point_prompt_version=str(config_data["key_point_prompt_version"]),
        grounding_prompt_version=str(config_data["grounding_prompt_version"]),
        key_point_threshold=float(config_data["key_point_threshold"]),
        input_cache_hit_usd_per_million=float(pricing["input_cache_hit"]),
        input_cache_miss_usd_per_million=float(pricing["input_cache_miss"]),
        output_usd_per_million=float(pricing["output"]),
        pricing_source=str(config_data["pricing_source"]),
        pricing_checked_at=str(config_data["pricing_checked_at"]),
        command=command,
        grader_independence=str(config_data["independence"]),
    )
    client = DeepSeekChatClient(
        timeout_seconds=float(config_data["timeout_seconds"]),
        api_key_env=api_key_env,
        max_tokens=int(config_data["max_tokens"]),
        system_instruction=(
            "你是 EvalRAG 的离线评测器。只依据输入中的 Answer、expected points "
            "和 cited Context 审核，并且只输出合法 JSON。"
        ),
        max_retries=int(config_data["max_retries"]),
    )
    key_point_grader = LlmKeyPointGrader(
        client,
        model=run_config.model,
        temperature=run_config.temperature,
        prompt_version=run_config.key_point_prompt_version,
    )
    grounding_grader = LlmGroundingGrader(
        client,
        model=run_config.model,
        temperature=run_config.temperature,
        prompt_version=run_config.grounding_prompt_version,
    )
    result = run_saved_prediction_audit(
        _read_jsonl(source_cases),
        _read_jsonl(trace_path),
        run_config,
        key_point_grader,
        grounding_grader,
        progress_callback=lambda index, total, case_id: print(
            f"[{index}/{total}] {case_id}", flush=True
        ),
    )
    save_semantic_audit_artifacts(result, output_dir)
    print(json.dumps({
        "run_id": run_id,
        "source_prediction_run_id": source_run_id,
        "summary": result.summary,
        "git_revision": _git_revision(),
    }, ensure_ascii=False, indent=2))
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("dev", "test"), required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/evaluation/semantic_grounding_v1.json"),
    )
    parser.add_argument("--run-id")
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _git_revision() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


if __name__ == "__main__":
    raise SystemExit(main())
