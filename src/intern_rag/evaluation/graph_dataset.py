from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Literal


GraphCaseCategory = Literal[
    "job_skill", "skill_project", "cross_source_two_hop", "hard_negative"
]
GraphExpectedStrategy = Literal["vector", "graph_hybrid"]


@dataclass(frozen=True)
class GraphChallengeCase:
    """关系型检索标签，不包含任何系统 prediction。"""

    case_id: str
    query: str
    split: Literal["dev", "test"]
    category: GraphCaseCategory
    expected_strategy: GraphExpectedStrategy
    expected_sources: list[str]
    relevant_chunk_ids: list[str]
    expected_entities: list[str]
    expected_relations: list[str]
    answerable: bool
    expected_points: list[str]
    label_reviewed: bool
    review_method: str

    def to_dict(self) -> dict[str, object]:
        """转换成可写入 JSONL 的标签字典。"""

        return asdict(self)


@dataclass(frozen=True)
class GraphDatasetValidation:
    """Graph challenge 的错误与分布统计。"""

    errors: list[str]
    case_count: int
    split_counts: dict[str, int]
    category_counts: dict[str, int]
    reviewed_count: int

    @property
    def is_valid(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, object]:
        return {**asdict(self), "is_valid": self.is_valid}


def load_graph_challenge(path: Path) -> list[GraphChallengeCase]:
    """读取 Graph challenge，拒绝标签文件中出现 predicted 字段。"""

    cases: list[GraphChallengeCase] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        raw = json.loads(line)
        if any(str(key).startswith("predicted") for key in raw):
            raise ValueError(f"{path}:{line_number} contains predicted fields")
        cases.append(
            GraphChallengeCase(
                case_id=str(raw["case_id"]),
                query=str(raw["query"]),
                split=str(raw["split"]),  # type: ignore[arg-type]
                category=str(raw["category"]),  # type: ignore[arg-type]
                expected_strategy=str(raw["expected_strategy"]),  # type: ignore[arg-type]
                expected_sources=list(raw["expected_sources"]),
                relevant_chunk_ids=list(raw["relevant_chunk_ids"]),
                expected_entities=list(raw["expected_entities"]),
                expected_relations=list(raw["expected_relations"]),
                answerable=bool(raw["answerable"]),
                expected_points=list(raw["expected_points"]),
                label_reviewed=bool(raw["label_reviewed"]),
                review_method=str(raw["review_method"]),
            )
        )
    return cases


def validate_graph_challenge(
    cases: list[GraphChallengeCase], available_chunk_ids: set[str]
) -> GraphDatasetValidation:
    """校验 40 Case、30/10 split、标签完整性和相关 Chunk 存在性。"""

    errors: list[str] = []
    seen_ids: set[str] = set()
    seen_queries: set[str] = set()
    valid_sources = {"jd", "resume", "project_logs", "user_profile"}
    valid_relations = {
        "requires", "demonstrates", "uses", "belongs_to", "related_to"
    }
    for case in cases:
        if case.case_id in seen_ids:
            errors.append(f"duplicate case id: {case.case_id}")
        seen_ids.add(case.case_id)
        normalized_query = " ".join(case.query.lower().split())
        if normalized_query in seen_queries:
            errors.append(f"duplicate query: {case.case_id}")
        seen_queries.add(normalized_query)
        if not set(case.expected_sources).issubset(valid_sources):
            errors.append(f"{case.case_id} has invalid sources")
        if not set(case.expected_relations).issubset(valid_relations):
            errors.append(f"{case.case_id} has invalid relations")
        missing = sorted(set(case.relevant_chunk_ids) - available_chunk_ids)
        if missing:
            errors.append(f"{case.case_id} misses chunks: {', '.join(missing)}")
        if case.answerable and not case.relevant_chunk_ids:
            errors.append(f"{case.case_id} answerable without relevant chunks")
        if not case.answerable and case.relevant_chunk_ids:
            errors.append(f"{case.case_id} unanswerable with relevant chunks")
        if not case.label_reviewed:
            errors.append(f"{case.case_id} label is not reviewed")

    split_counts = dict(Counter(case.split for case in cases))
    category_counts = dict(Counter(case.category for case in cases))
    if len(cases) != 40:
        errors.append(f"expected 40 cases, got {len(cases)}")
    if split_counts != {"dev": 30, "test": 10}:
        errors.append(f"expected 30 dev/10 test, got {split_counts}")
    if category_counts != {
        "job_skill": 10,
        "skill_project": 10,
        "cross_source_two_hop": 10,
        "hard_negative": 10,
    }:
        errors.append(f"unexpected category distribution: {category_counts}")
    return GraphDatasetValidation(
        errors=errors,
        case_count=len(cases),
        split_counts=split_counts,
        category_counts=category_counts,
        reviewed_count=sum(case.label_reviewed for case in cases),
    )
