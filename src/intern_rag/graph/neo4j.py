from __future__ import annotations

import json
from typing import Any

from intern_rag.graph.schemas import GraphEdge, GraphNode, KnowledgeGraph


class Neo4jGraphRepository:
    """持久化版本化知识图，并从 Neo4j 恢复统一 KnowledgeGraph。

    所有业务边在数据库中使用 `EVIDENCE_RELATION` 类型，实际 edge_type 作为属性保存，
    避免把外部输入拼接成 Cypher。节点、边均保留 chunk ids 与 provenance，替换某个
    dataset version 时不会删除其他版本。
    """

    def __init__(self, driver: Any, *, database: str = "neo4j") -> None:
        self.driver = driver
        self.database = database

    def migrate(self) -> None:
        """创建版本化 node id 唯一约束。"""

        with self.driver.session(database=self.database) as session:
            session.run(
                "CREATE CONSTRAINT evalrag_node_version IF NOT EXISTS "
                "FOR (n:EvalRAGNode) REQUIRE (n.dataset_version, n.node_id) IS UNIQUE"
            ).consume()

    def replace(self, graph: KnowledgeGraph) -> None:
        """事务性替换同 dataset version 图，并批量写入节点和关系。"""

        nodes = [
            {
                "node_id": node.node_id,
                "node_type": node.node_type,
                "name": node.name,
                "aliases": list(node.aliases),
                "chunk_ids": list(node.chunk_ids),
                "provenance": list(node.provenance),
            }
            for node in graph.nodes
        ]
        edges = [
            {
                "edge_id": edge.edge_id,
                "edge_type": edge.edge_type,
                "source_node_id": edge.source_node_id,
                "target_node_id": edge.target_node_id,
                "chunk_ids": list(edge.chunk_ids),
                "provenance": list(edge.provenance),
            }
            for edge in graph.edges
        ]
        with self.driver.session(database=self.database) as session:
            session.execute_write(self._replace_tx, graph, nodes, edges)

    @staticmethod
    def _replace_tx(
        tx: Any,
        graph: KnowledgeGraph,
        nodes: list[dict[str, object]],
        edges: list[dict[str, object]],
    ) -> None:
        tx.run(
            "MATCH (n:EvalRAGNode {dataset_version: $dataset_version}) DETACH DELETE n",
            dataset_version=graph.dataset_version,
        )
        tx.run(
            """
            UNWIND $nodes AS item
            CREATE (n:EvalRAGNode {
              dataset_version: $dataset_version,
              graph_version: $graph_version,
              node_id: item.node_id,
              node_type: item.node_type,
              name: item.name,
              aliases: item.aliases,
              chunk_ids: item.chunk_ids,
              provenance: item.provenance
            })
            """,
            dataset_version=graph.dataset_version,
            graph_version=graph.version,
            nodes=nodes,
        )
        tx.run(
            """
            UNWIND $edges AS item
            MATCH (source:EvalRAGNode {
              dataset_version: $dataset_version, node_id: item.source_node_id
            })
            MATCH (target:EvalRAGNode {
              dataset_version: $dataset_version, node_id: item.target_node_id
            })
            CREATE (source)-[:EVIDENCE_RELATION {
              edge_id: item.edge_id,
              edge_type: item.edge_type,
              chunk_ids: item.chunk_ids,
              provenance: item.provenance
            }]->(target)
            """,
            dataset_version=graph.dataset_version,
            edges=edges,
        )
        tx.run(
            "MERGE (m:EvalRAGGraphMeta {dataset_version: $dataset_version}) "
            "SET m.graph_version=$graph_version, m.config_hash=$config_hash, "
            "m.stats=$stats",
            dataset_version=graph.dataset_version,
            graph_version=graph.version,
            config_hash=graph.config_hash,
            stats=json.dumps(graph.stats, ensure_ascii=False, sort_keys=True),
        )

    def load(self, dataset_version: str) -> KnowledgeGraph:
        """读取指定 dataset version；不存在时明确报错。"""

        with self.driver.session(database=self.database) as session:
            meta = session.run(
                "MATCH (m:EvalRAGGraphMeta {dataset_version: $dataset_version}) RETURN m",
                dataset_version=dataset_version,
            ).single()
            if meta is None:
                raise KeyError(f"neo4j graph not found: {dataset_version}")
            node_rows = session.run(
                "MATCH (n:EvalRAGNode {dataset_version: $dataset_version}) "
                "RETURN n ORDER BY n.node_id",
                dataset_version=dataset_version,
            )
            edge_rows = session.run(
                "MATCH (s:EvalRAGNode {dataset_version: $dataset_version})"
                "-[r:EVIDENCE_RELATION]->(t:EvalRAGNode) "
                "RETURN r, s.node_id AS source_id, t.node_id AS target_id "
                "ORDER BY r.edge_id",
                dataset_version=dataset_version,
            )
            metadata = dict(meta["m"])
            nodes = tuple(_node_from_mapping(dict(row["n"])) for row in node_rows)
            edges = tuple(
                _edge_from_mapping(dict(row["r"]), row["source_id"], row["target_id"])
                for row in edge_rows
            )
        return KnowledgeGraph(
            version=str(metadata["graph_version"]),
            dataset_version=dataset_version,
            config_hash=str(metadata["config_hash"]),
            nodes=nodes,
            edges=edges,
            stats={
                str(key): int(value)
                for key, value in json.loads(str(metadata.get("stats", "{}"))).items()
            },
        )

    def close(self) -> None:
        """关闭 Neo4j driver。"""

        self.driver.close()


def create_neo4j_repository(
    uri: str, user: str, password: str, *, database: str = "neo4j"
) -> Neo4jGraphRepository:
    """从环境配置创建真实 Neo4j adapter。"""

    from neo4j import GraphDatabase

    return Neo4jGraphRepository(
        GraphDatabase.driver(uri, auth=(user, password)), database=database
    )


def _node_from_mapping(value: dict[str, object]) -> GraphNode:
    return GraphNode(
        node_id=str(value["node_id"]),
        node_type=str(value["node_type"]),  # type: ignore[arg-type]
        name=str(value["name"]),
        aliases=tuple(str(item) for item in value.get("aliases", [])),
        chunk_ids=tuple(str(item) for item in value.get("chunk_ids", [])),
        provenance=tuple(str(item) for item in value.get("provenance", [])),
    )


def _edge_from_mapping(
    value: dict[str, object], source_id: object, target_id: object
) -> GraphEdge:
    return GraphEdge(
        edge_id=str(value["edge_id"]),
        edge_type=str(value["edge_type"]),  # type: ignore[arg-type]
        source_node_id=str(source_id),
        target_node_id=str(target_id),
        chunk_ids=tuple(str(item) for item in value.get("chunk_ids", [])),
        provenance=tuple(str(item) for item in value.get("provenance", [])),
    )
