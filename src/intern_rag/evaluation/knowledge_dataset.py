from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Literal


KnowledgeCaseCategory = Literal[
    "single_source",
    "cross_source",
    "semantic_paraphrase",
    "hard_negative",
    "unanswerable",
    "freshness_conflict",
    "two_hop",
    "three_hop",
]


@dataclass(frozen=True)
class KnowledgeEvaluationCase:
    """v0.3 检索/图评测标签，不包含系统 prediction。"""

    case_id: str
    query: str
    category: KnowledgeCaseCategory
    split: Literal["dev", "test"]
    expected_sources: tuple[str, ...]
    relevant_chunk_ids: tuple[str, ...]
    expected_points: tuple[str, ...]
    answerable: bool
    expected_entities: tuple[str, ...] = ()
    expected_relations: tuple[str, ...] = ()
    graph_edge_ids: tuple[str, ...] = ()
    review_method: str = "ai_assisted"
    human_reviewed: bool = False

    def to_dict(self) -> dict[str, object]:
        """转换为 JSONL 字典。"""

        return asdict(self)


@dataclass(frozen=True)
class KnowledgeDatasetValidation:
    """v0.3 标签完整性、分布和证据引用校验。"""

    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    case_count: int
    split_counts: dict[str, int]
    category_counts: dict[str, int]
    review_method_counts: dict[str, int]
    human_reviewed_count: int

    @property
    def is_valid(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, object]:
        return {**asdict(self), "is_valid": self.is_valid}


def load_knowledge_dataset(path: Path) -> list[KnowledgeEvaluationCase]:
    """读取 v0.3 标签，并拒绝任何 predicted 字段。"""

    cases: list[KnowledgeEvaluationCase] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        raw = json.loads(line)
        if any(str(key).startswith("predicted") for key in raw):
            raise ValueError(f"{path}:{line_number} contains predicted fields")
        cases.append(
            KnowledgeEvaluationCase(
                case_id=str(raw["case_id"]),
                query=str(raw["query"]),
                category=str(raw["category"]),  # type: ignore[arg-type]
                split=str(raw["split"]),  # type: ignore[arg-type]
                expected_sources=tuple(str(item) for item in raw["expected_sources"]),
                relevant_chunk_ids=tuple(
                    str(item) for item in raw["relevant_chunk_ids"]
                ),
                expected_points=tuple(str(item) for item in raw["expected_points"]),
                answerable=bool(raw["answerable"]),
                expected_entities=tuple(
                    str(item) for item in raw.get("expected_entities", [])
                ),
                expected_relations=tuple(
                    str(item) for item in raw.get("expected_relations", [])
                ),
                graph_edge_ids=tuple(
                    str(item) for item in raw.get("graph_edge_ids", [])
                ),
                review_method=str(raw.get("review_method", "unknown")),
                human_reviewed=bool(raw.get("human_reviewed", False)),
            )
        )
    return cases


def validate_knowledge_dataset(
    cases: list[KnowledgeEvaluationCase],
    *,
    available_chunk_ids: set[str],
    available_edge_ids: set[str],
    expected_count: int = 240,
) -> KnowledgeDatasetValidation:
    """校验唯一 ID、160/80 split、八类分布以及真实 Chunk/Edge 引用。"""

    errors: list[str] = []
    warnings: list[str] = []
    seen_ids: set[str] = set()
    seen_queries: set[str] = set()
    for case in cases:
        if case.case_id in seen_ids:
            errors.append(f"duplicate case id: {case.case_id}")
        seen_ids.add(case.case_id)
        normalized_query = " ".join(case.query.lower().split())
        if normalized_query in seen_queries:
            errors.append(f"duplicate query: {case.case_id}")
        seen_queries.add(normalized_query)
        missing_chunks = set(case.relevant_chunk_ids) - available_chunk_ids
        if missing_chunks:
            errors.append(f"{case.case_id} references missing chunks")
        missing_edges = set(case.graph_edge_ids) - available_edge_ids
        if missing_edges:
            errors.append(f"{case.case_id} references missing graph edges")
        if case.answerable and not case.relevant_chunk_ids:
            errors.append(f"{case.case_id} answerable without evidence")
        if not case.answerable and case.relevant_chunk_ids:
            errors.append(f"{case.case_id} unanswerable with evidence")
        if case.category in {"two_hop", "three_hop"}:
            expected_hops = 2 if case.category == "two_hop" else 3
            if len(case.graph_edge_ids) != expected_hops:
                errors.append(f"{case.case_id} has invalid path length")
        if case.review_method == "human" and not case.human_reviewed:
            errors.append(f"{case.case_id} has inconsistent human review label")

    split_counts = dict(Counter(case.split for case in cases))
    category_counts = dict(Counter(case.category for case in cases))
    if len(cases) != expected_count:
        errors.append(f"expected {expected_count} cases, got {len(cases)}")
    if split_counts != {"dev": 160, "test": 80}:
        errors.append(f"expected 160 dev/80 test, got {split_counts}")
    expected_categories = {
        category: 30
        for category in (
            "single_source", "cross_source", "semantic_paraphrase",
            "hard_negative", "unanswerable", "freshness_conflict",
            "two_hop", "three_hop",
        )
    }
    if category_counts != expected_categories:
        errors.append(f"unexpected category distribution: {category_counts}")
    if not all(case.human_reviewed for case in cases):
        warnings.append(
            "labels are corpus-grounded and AI-assisted, not fully human reviewed"
        )
    return KnowledgeDatasetValidation(
        errors=tuple(errors),
        warnings=tuple(warnings),
        case_count=len(cases),
        split_counts=split_counts,
        category_counts=category_counts,
        review_method_counts=dict(Counter(case.review_method for case in cases)),
        human_reviewed_count=sum(case.human_reviewed for case in cases),
    )
