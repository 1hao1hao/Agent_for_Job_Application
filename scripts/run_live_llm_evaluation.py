from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess

from intern_rag.agent import DeepSeekChatClient, EvidenceConfig
from intern_rag.evaluation import (
    LiveLlmRunConfig,
    load_chunks_jsonl,
    load_evaluation_dataset,
    run_live_llm_end_to_end_evaluation,
    save_end_to_end_artifacts,
)
from intern_rag.retrieval import build_retriever_from_config
from intern_rag.routing import build_router_from_config


def main() -> int:
    """使用固定 DeepSeek 配置运行完整 Pipeline 并保存标准评测工件。"""

    args = _parse_args()
    llm_data = _read_json(Path(args.llm_config))
    api_key_env = str(llm_data["api_key_env"])
    if not os.environ.get(api_key_env, "").strip():
        print(f"ERROR: {api_key_env} is not configured")
        return 2
    run_dir = Path("reports/runs") / args.run_id
    if run_dir.exists():
        print(f"ERROR: run already exists: {run_dir}")
        return 2

    final = _read_json(Path("configs/final/p0_v0.2.json"))
    router_data = _read_json(Path(str(final["router_config"])))
    retriever_data = _read_json(Path(str(final["retriever_config"])))
    evidence_data = _read_json(Path(str(final["evidence_config"])))
    prices = dict(llm_data["pricing_usd_per_million_tokens"])
    config = LiveLlmRunConfig(
        run_id=args.run_id,
        dataset_version="evalrag_v0.2",
        split=args.split,
        router_name=str(router_data["router_name"]),
        retriever_name=str(retriever_data["retriever_name"]),
        top_k=int(retriever_data["top_k"]),
        context_max_chars=int(llm_data["context_max_chars"]),
        key_point_threshold=float(llm_data["key_point_threshold"]),
        model=str(llm_data["model"]),
        temperature=float(llm_data["temperature"]),
        prompt_version=str(llm_data["prompt_version"]),
        max_source_retries=int(llm_data["max_source_retries"]),
        max_format_retries=int(llm_data["max_format_retries"]),
        input_cache_hit_usd_per_million=float(prices["input_cache_hit"]),
        input_cache_miss_usd_per_million=float(prices["input_cache_miss"]),
        output_usd_per_million=float(prices["output"]),
        pricing_source=str(llm_data["pricing_source"]),
        pricing_checked_at=str(llm_data["pricing_checked_at"]),
        git_revision=_git_revision(),
        command=(
            "PYTHONPATH=src python scripts/run_live_llm_evaluation.py "
            f"--split {args.split} --run-id {args.run_id}"
        ),
    )
    support_labels = _load_support_labels(
        Path(args.support_review) if args.support_review else None
    )
    result = run_live_llm_end_to_end_evaluation(
        load_evaluation_dataset(Path("data/evaluation/evalrag_v0.2.jsonl")),
        load_chunks_jsonl(Path("data/processed/chunks/evalrag_v0.2.jsonl")),
        config,
        build_router_from_config(router_data),
        build_retriever_from_config(retriever_data),
        DeepSeekChatClient(
            timeout_seconds=float(llm_data["timeout_seconds"]),
            api_key_env=api_key_env,
            base_url=str(llm_data["base_url"]),
            max_tokens=int(llm_data["max_tokens"]),
        ),
        run_dir / "traces.jsonl",
        evidence_config=EvidenceConfig(
            min_results=int(evidence_data["min_results"]),
            min_scores=dict(evidence_data["min_scores"]),
            require_source_coverage=bool(evidence_data["require_source_coverage"]),
        ),
        support_labels=support_labels,
        progress_callback=lambda current, total, case_id: print(
            f"PROGRESS {current}/{total} {case_id}", flush=True
        ),
    )
    save_end_to_end_artifacts(result, run_dir)
    print(json.dumps(result.summary, ensure_ascii=False, indent=2))
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("dev", "test"), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--llm-config",
        default="configs/llm/deepseek_v4_flash_v1.json",
    )
    parser.add_argument("--support-review", default="")
    return parser.parse_args()


def _load_support_labels(path: Path | None) -> dict[str, bool]:
    if path is None:
        return {}
    return {
        str(record["case_id"]): bool(record["unsupported_answer"])
        for record in (
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }


def _git_revision() -> str:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return f"{commit}-dirty" if dirty else commit


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    raise SystemExit(main())
