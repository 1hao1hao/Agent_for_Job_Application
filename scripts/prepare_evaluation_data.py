from __future__ import annotations

import argparse
import json
from pathlib import Path

from intern_rag.evaluation import (
    build_corpus,
    export_chunks_jsonl,
    load_evaluation_dataset,
    validate_corpus_manifest,
    validate_evaluation_dataset,
    write_corpus_manifest,
    write_corpus_stats,
    write_dataset_validation,
)


def main() -> int:
    """生成 corpus/chunks 工件，并校验评测标签。"""

    args = _parse_args()
    raw_root = Path("data/raw")
    dataset_path = Path("data/evaluation") / f"{args.dataset_version}.jsonl"
    manifest_path = Path("data/evaluation/corpus_manifest.jsonl")
    chunks_path = (
        Path("data/processed/chunks") / f"{args.dataset_version}.jsonl"
    )
    stats_path = Path("data/evaluation/corpus_stats.json")
    validation_path = (
        Path("data/evaluation")
        / f"{args.dataset_version}_validation.json"
    )

    entries, chunks, stats = build_corpus(
        raw_root,
        max_chars=args.chunk_max_chars,
    )
    write_corpus_manifest(entries, manifest_path)
    export_chunks_jsonl(
        chunks,
        chunks_path,
        dataset_version=args.dataset_version,
    )
    write_corpus_stats(stats, stats_path)

    minimum_documents = 100 if args.dataset_version == "evalrag_v0.2" else 30
    corpus_errors = validate_corpus_manifest(
        entries,
        minimum_documents=minimum_documents,
    )
    if args.dataset_version == "evalrag_v0.2" and len(chunks) < 300:
        corpus_errors.append(
            f"evalrag_v0.2 requires at least 300 natural chunks, got {len(chunks)}"
        )
    pending_corpus_count = sum(not entry.human_reviewed for entry in entries)
    corpus_warnings = (
        [
            f"{pending_corpus_count} corpus documents are pending human review; "
            "they cannot be described as verified real or semi-real material"
        ]
        if pending_corpus_count
        else []
    )
    cases = load_evaluation_dataset(dataset_path)
    dataset_validation = validate_evaluation_dataset(
        cases,
        available_chunk_ids={chunk.id for chunk in chunks},
        require_full_distribution=True,
        require_human_review=not args.allow_pending_review,
    )
    write_dataset_validation(dataset_validation, validation_path)

    result = {
        "dataset_version": args.dataset_version,
        "candidate_mode": args.allow_pending_review,
        "corpus_errors": corpus_errors,
        "corpus_warnings": corpus_warnings,
        "dataset_validation": dataset_validation.to_dict(),
        "artifacts": {
            "manifest": str(manifest_path),
            "chunks": str(chunks_path),
            "corpus_stats": str(stats_path),
            "dataset_validation": str(validation_path),
        },
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if corpus_errors or not dataset_validation.is_valid else 0


def _parse_args() -> argparse.Namespace:
    """解析数据准备参数。"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-version", default="evalrag_v0.1")
    parser.add_argument("--chunk-max-chars", type=int, default=800)
    parser.add_argument(
        "--allow-pending-review",
        action="store_true",
        help="只生成 candidate 工件，不把未人工核验标签当正式数据。",
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
