from __future__ import annotations

import argparse
import json
from pathlib import Path

from intern_rag.evaluation import load_chunks_jsonl
from intern_rag.graph import (
    DeterministicGraphExtractor,
    build_knowledge_graph,
    load_entity_catalog,
    save_knowledge_graph,
)


def main() -> int:
    """从版本化 Chunk 构建可重建的 Job-Skill-Experience 图工件。"""

    args = _parse_args()
    config_path = Path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    specs, config_hash = load_entity_catalog(config_path)
    chunks = load_chunks_jsonl(Path(args.chunks))
    allowed_sources = set(config["source_types"])
    selected = [chunk for chunk in chunks if chunk.source_type in allowed_sources]
    graph = build_knowledge_graph(
        selected,
        DeterministicGraphExtractor(specs),
        version=str(config["graph_version"]),
        dataset_version=str(config["dataset_version"]),
        config_hash=config_hash,
    )
    output_path = Path(args.output)
    save_knowledge_graph(graph, output_path)
    print(json.dumps(graph.stats, ensure_ascii=False, indent=2))
    print(f"Graph written to {output_path}")
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/graph/job_skill_v0.1.json"
    )
    parser.add_argument(
        "--chunks", default="data/processed/chunks/evalrag_v0.2.jsonl"
    )
    parser.add_argument(
        "--output",
        default=(
            "data/processed/graphs/evalrag_v0.2/"
            "job-skill-experience-v0.1.json"
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
