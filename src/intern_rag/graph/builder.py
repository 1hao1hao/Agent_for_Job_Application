from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from pathlib import Path
from typing import Protocol

from intern_rag.graph.schemas import (
    EdgeType,
    GraphEdge,
    GraphExtraction,
    GraphNode,
    KnowledgeGraph,
    NodeType,
)
from intern_rag.ingestion import Chunk


@dataclass(frozen=True)
class EntitySpec:
    """确定性实体词典中的一个规范实体。"""

    name: str
    node_type: NodeType
    aliases: tuple[str, ...]


class GraphExtractor(Protocol):
    """把单个 Chunk 转换为节点和边的可注入小接口。"""

    def extract(self, chunk: Chunk) -> GraphExtraction:
        """返回当前 Chunk 支撑的实体和关系。"""


def load_entity_catalog(path: Path) -> tuple[list[EntitySpec], str]:
    """读取版本化实体别名配置，并返回配置内容哈希。"""

    raw_bytes = path.read_bytes()
    raw = json.loads(raw_bytes)
    specs = [
        EntitySpec(
            name=str(item["name"]),
            node_type=str(item["node_type"]),  # type: ignore[arg-type]
            aliases=tuple(str(alias) for alias in item.get("aliases", [])),
        )
        for item in raw["entities"]
    ]
    return specs, hashlib.sha256(raw_bytes).hexdigest()


class DeterministicGraphExtractor:
    """使用标题、metadata 和实体别名构建可解释关系。

    输入一个 JD、简历、项目日志或用户画像 Chunk。先建立文档实体，再匹配配置中的
    Skill/Technology；JD 使用 `requires`，简历和画像使用 `demonstrates`，项目日志
    对 Technology 使用 `uses`、对 Skill 使用 `demonstrates`。输出的每个节点和边都
    保存当前 chunk id，因此图检索结果仍可作为 Citation 证据。
    """

    _document_types: dict[str, NodeType] = {
        "jd": "job",
        "resume": "experience",
        "project_logs": "project",
        "user_profile": "experience",
        "interview": "interview_question",
    }

    def __init__(self, entity_specs: list[EntitySpec]) -> None:
        self.entity_specs = list(entity_specs)

    def extract(self, chunk: Chunk) -> GraphExtraction:
        """抽取文档实体、概念实体以及它们之间的证据关系。"""

        document_type = self._document_types.get(chunk.source_type)
        if document_type is None:
            return GraphExtraction((), ())

        document_name = _document_name(chunk)
        provenance = _chunk_provenance(chunk)
        document = _node(document_type, document_name, (), (chunk.id,), provenance)
        nodes = [document]
        edges: list[GraphEdge] = []

        company = str(chunk.metadata.get("company", "")).strip()
        if chunk.source_type == "jd" and company not in {"", "unknown"}:
            company_node = _node("company", company, (), (chunk.id,), provenance)
            nodes.append(company_node)
            edges.append(
                _edge(
                    "belongs_to", document.node_id, company_node.node_id,
                    chunk.id, provenance
                )
            )
            edges.append(
                _edge("posts", company_node.node_id, document.node_id, chunk.id, provenance)
            )

        city = str(chunk.metadata.get("city", "")).strip()
        if chunk.source_type == "jd" and city not in {"", "unknown"}:
            location = _node("location", city, (), (chunk.id,), provenance)
            nodes.append(location)
            edges.append(
                _edge("located_in", document.node_id, location.node_id, chunk.id, provenance)
            )

        matched = [
            spec for spec in self.entity_specs if _matches_spec(chunk.text, spec)
        ]
        concept_nodes: list[GraphNode] = []
        for spec in matched:
            concept = _node(
                spec.node_type, spec.name, spec.aliases, (chunk.id,), provenance
            )
            concept_nodes.append(concept)
            nodes.append(concept)
            edge_type = _document_relation(chunk.source_type, spec.node_type)
            edges.append(
                _edge(
                    edge_type, document.node_id, concept.node_id, chunk.id, provenance
                )
            )

        skills = [node for node in concept_nodes if node.node_type == "skill"]
        technologies = [
            node for node in concept_nodes if node.node_type == "technology"
        ]
        for skill in skills:
            for technology in technologies:
                edges.append(
                    _edge(
                        "related_to",
                        skill.node_id,
                        technology.node_id,
                        chunk.id,
                        provenance,
                    )
                )
        return GraphExtraction(tuple(nodes), tuple(edges))


def build_knowledge_graph(
    chunks: list[Chunk],
    extractor: GraphExtractor,
    *,
    version: str,
    dataset_version: str,
    config_hash: str,
) -> KnowledgeGraph:
    """逐 Chunk 抽取并去重合并，生成顺序稳定的本地图。

    相同实体和关系会合并 `chunk_ids`，因此重复出现的技能不会产生重复节点。
    节点、边和证据 ID 最后统一排序，保证相同语料与配置可重建出相同工件。
    """

    node_values: dict[str, dict[str, object]] = {}
    edge_values: dict[str, dict[str, object]] = {}
    for chunk in chunks:
        extraction = extractor.extract(chunk)
        for node in extraction.nodes:
            current = node_values.setdefault(
                node.node_id,
                {
                    "node_type": node.node_type,
                    "name": node.name,
                    "aliases": set(),
                    "chunk_ids": set(),
                    "provenance": set(),
                },
            )
            current["aliases"].update(node.aliases)  # type: ignore[union-attr]
            current["chunk_ids"].update(node.chunk_ids)  # type: ignore[union-attr]
            current["provenance"].update(node.provenance)  # type: ignore[union-attr]
        for edge in extraction.edges:
            current = edge_values.setdefault(
                edge.edge_id,
                {
                    "edge_type": edge.edge_type,
                    "source_node_id": edge.source_node_id,
                    "target_node_id": edge.target_node_id,
                    "chunk_ids": set(),
                    "provenance": set(),
                },
            )
            current["chunk_ids"].update(edge.chunk_ids)  # type: ignore[union-attr]
            current["provenance"].update(edge.provenance)  # type: ignore[union-attr]

    nodes = tuple(
        GraphNode(
            node_id=node_id,
            node_type=value["node_type"],  # type: ignore[arg-type]
            name=str(value["name"]),
            aliases=tuple(sorted(value["aliases"])),  # type: ignore[arg-type]
            chunk_ids=tuple(sorted(value["chunk_ids"])),  # type: ignore[arg-type]
            provenance=tuple(sorted(value["provenance"])),  # type: ignore[arg-type]
        )
        for node_id, value in sorted(node_values.items())
    )
    edges = tuple(
        GraphEdge(
            edge_id=edge_id,
            edge_type=value["edge_type"],  # type: ignore[arg-type]
            source_node_id=str(value["source_node_id"]),
            target_node_id=str(value["target_node_id"]),
            chunk_ids=tuple(sorted(value["chunk_ids"])),  # type: ignore[arg-type]
            provenance=tuple(sorted(value["provenance"])),  # type: ignore[arg-type]
        )
        for edge_id, value in sorted(edge_values.items())
    )
    return KnowledgeGraph(
        version=version,
        dataset_version=dataset_version,
        config_hash=config_hash,
        nodes=nodes,
        edges=edges,
        stats={
            "node_count": len(nodes),
            "edge_count": len(edges),
            "source_chunk_count": len(chunks),
            "evidence_chunk_count": len(
                {chunk_id for node in nodes for chunk_id in node.chunk_ids}
            ),
        },
    )


def _document_name(chunk: Chunk) -> str:
    if chunk.source_type == "jd":
        job_title = str(chunk.metadata.get("job_title", "")).strip()
        if job_title not in {"", "unknown"}:
            return job_title
    heading = re.sub(r"^#+\s*", "", chunk.text.splitlines()[0]).strip()
    return heading or chunk.title


def _matches_spec(text: str, spec: EntitySpec) -> bool:
    normalized = text.lower()
    return any(alias.lower() in normalized for alias in (spec.name, *spec.aliases))


def _document_relation(source_type: str, node_type: NodeType) -> EdgeType:
    if source_type == "jd":
        return "requires"
    if source_type == "interview":
        return "asks_about"
    if source_type == "project_logs" and node_type == "technology":
        return "uses"
    return "demonstrates"


def _node(
    node_type: NodeType,
    name: str,
    aliases: tuple[str, ...],
    chunk_ids: tuple[str, ...],
    provenance: tuple[str, ...] = (),
) -> GraphNode:
    normalized = " ".join(name.lower().split())
    node_id = f"{node_type}-{hashlib.sha1(normalized.encode('utf-8')).hexdigest()[:12]}"
    return GraphNode(node_id, node_type, name, aliases, chunk_ids, provenance)


def _edge(
    edge_type: EdgeType,
    source_node_id: str,
    target_node_id: str,
    chunk_id: str,
    provenance: tuple[str, ...] = (),
) -> GraphEdge:
    raw = f"{edge_type}|{source_node_id}|{target_node_id}"
    edge_id = f"edge-{hashlib.sha1(raw.encode('utf-8')).hexdigest()[:12]}"
    return GraphEdge(
        edge_id, edge_type, source_node_id, target_node_id, (chunk_id,), provenance
    )


def _chunk_provenance(chunk: Chunk) -> tuple[str, ...]:
    """提取图节点/边需要保留的来源 URL、文档 ID 和时间。"""

    values = (
        str(chunk.metadata.get("document_id", "")),
        str(chunk.metadata.get("source_url", "")),
        str(chunk.metadata.get("published_at", "")),
        str(chunk.metadata.get("collected_at", "")),
    )
    return tuple(value for value in values if value and value != "unknown")
