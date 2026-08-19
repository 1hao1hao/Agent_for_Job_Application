from __future__ import annotations

from dataclasses import asdict, dataclass

from intern_rag.graph.schemas import KnowledgeGraph


@dataclass(frozen=True)
class QueryDecomposition:
    """关系查询分解结果，保留原问题以支持失败回退。"""

    original_query: str
    subgoals: tuple[str, ...]
    entity_node_ids: tuple[str, ...]
    entity_names: tuple[str, ...]
    relations: tuple[str, ...]
    is_cross_document: bool
    reason: str

    def to_trace(self) -> dict[str, object]:
        """转换为 Trace 可序列化字典。"""

        return asdict(self)


class QueryDecomposer:
    """用实体别名和关系标记把复杂 Query 拆成可解释子目标。"""

    _cross_markers = (
        "哪些项目", "哪个项目", "哪些经历", "哪段经历", "能否证明", "可以证明",
        "是否匹配", "匹配起来", "对应起来", "结合岗位", "岗位要求与", "候选人是否",
        "简历经历", "项目记录", "共同体现",
    )
    _relation_markers = {
        "requires": ("要求", "需要", "岗位"),
        "demonstrates": ("证明", "经历", "能力"),
        "uses": ("项目", "使用", "实践"),
        "belongs_to": ("公司", "属于"),
        "posts": ("发布", "招聘", "公司岗位"),
        "asks_about": ("面试", "问题", "考察"),
        "located_in": ("城市", "地点", "工作地"),
        "related_to": ("相关", "对应", "匹配"),
    }

    def decompose(self, query: str, graph: KnowledgeGraph) -> QueryDecomposition:
        """匹配实体和关系，并生成不调用 LLM 的稳定子目标。"""

        normalized = query.strip().lower()
        matched = []
        for node in graph.nodes:
            aliases = (node.name, *node.aliases)
            if any(alias.lower() in normalized for alias in aliases):
                matched.append(node)
        matched.sort(key=lambda item: item.node_id)
        relations = tuple(
            relation
            for relation, markers in self._relation_markers.items()
            if any(marker in normalized for marker in markers)
        )
        cross_document = any(marker in normalized for marker in self._cross_markers)
        if not cross_document:
            cross_document = (
                "岗位" in normalized
                and any(marker in normalized for marker in ("项目", "经历", "简历"))
            )

        subgoals: list[str] = []
        if "requires" in relations:
            subgoals.append("定位岗位要求的技能或技术")
        if any(relation in relations for relation in ("demonstrates", "uses")):
            subgoals.append("定位能证明能力的项目或经历")
        if cross_document:
            subgoals.append("连接岗位要求与候选人证据")
        reason = (
            f"matched {len(matched)} entities and {len(relations)} relations"
            if matched
            else "no graph entity matched; preserve original query for vector fallback"
        )
        return QueryDecomposition(
            original_query=query,
            subgoals=tuple(subgoals),
            entity_node_ids=tuple(node.node_id for node in matched),
            entity_names=tuple(node.name for node in matched),
            relations=relations,
            is_cross_document=cross_document,
            reason=reason,
        )
