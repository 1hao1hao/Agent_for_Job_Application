from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess

from intern_rag.evaluation import (
    EvaluationRunConfig,
    load_chunks_jsonl,
    load_evaluation_dataset,
    run_retrieval_evaluation,
    save_run_artifacts,
    validate_evaluation_dataset,
)
from intern_rag.retrieval import build_retriever_from_config
from intern_rag.routing import build_router_from_config


def main() -> int:
    """运行指定 Router/Retriever 配置并保存标准工件。"""

    args = _parse_args()
    if args.split == "test" and not args.allow_frozen_test:
        print(
            "ERROR: frozen test 在 P0-D5 前不可运行；"
            "如确需最终运行必须显式传入 --allow-frozen-test。"
        )
        return 2

    config_data = json.loads(
        Path(args.config).read_text(encoding="utf-8")
    )
    router_config_data = (
        json.loads(Path(args.router_config).read_text(encoding="utf-8"))
        if args.router_config
        else {"router_name": "rule"}
    )
    dataset_path = Path("data/evaluation") / f"{args.dataset_version}.jsonl"
    chunks_path = (
        Path("data/processed/chunks") / f"{args.dataset_version}.jsonl"
    )
    cases = load_evaluation_dataset(dataset_path)
    chunks = load_chunks_jsonl(chunks_path)
    validation = validate_evaluation_dataset(
        cases,
        available_chunk_ids={chunk.id for chunk in chunks},
        require_full_distribution=True,
        require_human_review=not args.candidate,
    )
    if not validation.is_valid:
        print(json.dumps(validation.to_dict(), ensure_ascii=False, indent=2))
        return 1

    run_id = args.run_id or _default_run_id(args.split, args.candidate)
    command = (
        "PYTHONPATH=src python scripts/run_evaluation.py "
        f"--dataset-version {args.dataset_version} --split {args.split} "
        f"--config {args.config}"
        + (" --candidate" if args.candidate else "")
        + (f" --run-id {args.run_id}" if args.run_id else "")
        + (" --allow-frozen-test" if args.allow_frozen_test else "")
        + (f" --router-config {args.router_config}" if args.router_config else "")
    )
    run_config = EvaluationRunConfig(
        run_id=run_id,
        dataset_version=args.dataset_version,
        split=args.split,
        retriever_name=str(config_data["retriever_name"]),
        top_k=int(config_data["top_k"]),
        chunk_max_chars=int(config_data["chunk_max_chars"]),
        git_commit=_git_revision(),
        command=command,
        candidate_run=args.candidate,
        retriever_config=config_data,
        router_name=str(router_config_data["router_name"]),
        router_config=router_config_data,
    )
    retriever = build_retriever_from_config(config_data)
    router = build_router_from_config(router_config_data)
    result = run_retrieval_evaluation(
        cases, chunks, run_config, retriever, router
    )
    run_dir = Path("reports/runs") / run_id
    save_run_artifacts(result, run_dir)
    print(json.dumps(result.summary, ensure_ascii=False, indent=2))
    print(f"Artifacts written to {run_dir}")
    return 0


def _parse_args() -> argparse.Namespace:
    """解析 baseline 运行参数。"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-version", default="evalrag_v0.1")
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument(
        "--config",
        default="configs/retrieval/keyword_v0.1.json",
    )
    parser.add_argument("--run-id")
    parser.add_argument("--router-config")
    parser.add_argument("--candidate", action="store_true")
    parser.add_argument("--allow-frozen-test", action="store_true")
    return parser.parse_args()


def _default_run_id(split: str, candidate: bool) -> str:
    """生成包含 split 与候选状态的稳定可读 run id。"""

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = "candidate" if candidate else "formal"
    return f"retrieval-{split}-{suffix}-{timestamp}"


def _git_revision() -> str:
    """记录当前代码版本，容器中优先使用构建时注入的版本。"""

    configured_revision = os.getenv("EVALRAG_GIT_COMMIT", "").strip()
    if configured_revision:
        return configured_revision

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        return f"{commit}-dirty" if dirty else commit
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unavailable"


if __name__ == "__main__":
    raise SystemExit(main())
