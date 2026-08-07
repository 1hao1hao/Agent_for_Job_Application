from __future__ import annotations

import argparse
import json
from pathlib import Path

from intern_rag.evaluation import load_chunks_jsonl
from intern_rag.retrieval import build_bm25_index, save_bm25_index


def main() -> int:
    """从版本化 Chunk export 构建可复现 BM25 index。"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-version", default="evalrag_v0.2")
    parser.add_argument(
        "--output",
        default="data/processed/indexes/evalrag_v0.2/bm25-v1/index.json",
    )
    args = parser.parse_args()
    chunks_path = Path("data/processed/chunks") / f"{args.dataset_version}.jsonl"
    chunks = load_chunks_jsonl(chunks_path)
    index = build_bm25_index(chunks, args.dataset_version)
    output_path = Path(args.output)
    save_bm25_index(index, output_path)
    summary = {
        "dataset_version": index.dataset_version,
        "chunk_count": len(index.chunk_ids),
        "vocabulary_size": len(index.document_frequencies),
        "average_document_length": index.average_document_length,
        "tokenizer_version": index.tokenizer_version,
        "index_path": str(output_path),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
