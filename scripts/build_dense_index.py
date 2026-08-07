from __future__ import annotations

import argparse
import json
from pathlib import Path

from intern_rag.evaluation import load_chunks_jsonl
from intern_rag.retrieval import (
    build_dense_index,
    build_pretrained_dense_index,
    save_dense_index,
)


def main() -> int:
    """为版本化 Chunk 离线构建 Dense 索引和 Query encoder。"""

    args = _parse_args()
    chunks_path = Path("data/processed/chunks") / f"{args.dataset_version}.jsonl"
    chunks = load_chunks_jsonl(chunks_path)
    if args.model_kind == "sentence_transformer":
        revision = args.revision or _resolve_model_revision(args.model_name)
        index, model = build_pretrained_dense_index(
            chunks,
            dataset_version=args.dataset_version,
            model_name=args.model_name,
            revision=revision,
            device=args.device,
        )
    else:
        index, model = build_dense_index(
            chunks,
            dataset_version=args.dataset_version,
            dimensions=args.dimensions,
            max_features=args.max_features,
        )
    index_dir = Path(args.output_dir)
    save_dense_index(index, model, index_dir)
    print(json.dumps({
        "index_dir": str(index_dir),
        "dataset_version": args.dataset_version,
        "embedding_name": index.metadata.embedding_name,
        "embedding_version": index.metadata.embedding_version,
        "dimensions": index.metadata.dimensions,
        "chunk_count": index.metadata.chunk_count,
    }, ensure_ascii=False, indent=2))
    return 0


def _parse_args() -> argparse.Namespace:
    """解析离线索引构建参数。"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-version", default="evalrag_v0.2")
    parser.add_argument(
        "--output-dir",
        default="data/processed/indexes/evalrag_v0.2/lsa-v1",
    )
    parser.add_argument("--dimensions", type=int, default=128)
    parser.add_argument("--max-features", type=int, default=4096)
    parser.add_argument(
        "--model-kind",
        choices=("sentence_transformer", "sklearn_lsa"),
        default="sentence_transformer",
    )
    parser.add_argument("--model-name", default="BAAI/bge-small-zh-v1.5")
    parser.add_argument("--revision")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def _resolve_model_revision(model_name: str) -> str:
    """把可变的 main 解析为固定 commit，保证索引可复现。"""

    from huggingface_hub import model_info

    revision = model_info(model_name).sha
    if not revision:
        raise RuntimeError(f"cannot resolve model revision for {model_name}")
    return revision


if __name__ == "__main__":
    raise SystemExit(main())
