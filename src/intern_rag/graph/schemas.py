from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Literal


NodeType = Literal[
    "job",
    "requirement",
    "skill",
    "project",
    "experience",
    "company",
    "technology",
    "interview_question",
    "location",
]
EdgeType = Literal[
    "posts",
    "requires",
    "demonstrates",
    "uses",
    "belongs_to",
    "asks_about",
    "located_in",
    "related_to",
]


@dataclass(frozen=True)
class GraphNode:
    """图中的实体节点，`chunk_ids` 保存实体的原始证据来源。"""

    node_id: str
    node_type: NodeType
    name: str
    aliases: tuple[str, ...] = ()
    chunk_ids: tuple[str, ...] = ()
    provenance: tuple[str, ...] = ()


@dataclass(frozen=True)
class GraphEdge:
    """两个实体之间的有向关系，关系本身也必须能回到原始 Chunk。"""

    edge_id: str
    edge_type: EdgeType
    source_node_id: str
    target_node_id: str
    chunk_ids: tuple[str, ...] = ()
    provenance: tuple[str, ...] = ()


@dataclass(frozen=True)
class GraphExtraction:
    """单个 Chunk 的抽取结果，便于替换确定性或模型抽取器。"""

    nodes: tuple[GraphNode, ...]
    edges: tuple[GraphEdge, ...]


@dataclass(frozen=True)
class KnowledgeGraph:
    """可版本化、可重建的本地图工件。"""

    version: str
    dataset_version: str
    config_hash: str
    nodes: tuple[GraphNode, ...]
    edges: tuple[GraphEdge, ...]
    stats: dict[str, int] = field(default_factory=dict)

    def node_by_id(self) -> dict[str, GraphNode]:
        """按 ID 建立节点索引。"""

        return {node.node_id: node for node in self.nodes}

    def to_dict(self) -> dict[str, object]:
        """转换为稳定 JSON 结构。"""

        return asdict(self)


def save_knowledge_graph(graph: KnowledgeGraph, path: Path) -> None:
    """将图保存为单个版本化 JSON 文件。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(graph.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def load_knowledge_graph(path: Path) -> KnowledgeGraph:
    """读取图文件并恢复强类型节点和边。"""

    raw = json.loads(path.read_text(encoding="utf-8"))
    return KnowledgeGraph(
        version=str(raw["version"]),
        dataset_version=str(raw["dataset_version"]),
        config_hash=str(raw["config_hash"]),
        nodes=tuple(
            GraphNode(
                node_id=str(item["node_id"]),
                node_type=str(item["node_type"]),  # type: ignore[arg-type]
                name=str(item["name"]),
                aliases=tuple(str(value) for value in item.get("aliases", [])),
                chunk_ids=tuple(str(value) for value in item.get("chunk_ids", [])),
                provenance=tuple(str(value) for value in item.get("provenance", [])),
            )
            for item in raw["nodes"]
        ),
        edges=tuple(
            GraphEdge(
                edge_id=str(item["edge_id"]),
                edge_type=str(item["edge_type"]),  # type: ignore[arg-type]
                source_node_id=str(item["source_node_id"]),
                target_node_id=str(item["target_node_id"]),
                chunk_ids=tuple(str(value) for value in item.get("chunk_ids", [])),
                provenance=tuple(str(value) for value in item.get("provenance", [])),
            )
            for item in raw["edges"]
        ),
        stats={str(key): int(value) for key, value in raw.get("stats", {}).items()},
    )
