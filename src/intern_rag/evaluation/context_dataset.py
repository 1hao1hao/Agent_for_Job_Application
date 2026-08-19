from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Literal


ContextCaseCategory = Literal[
    "reference", "ellipsis", "history_constraint", "cross_session",
    "memory_conflict", "topic_switch", "multi_source", "unanswerable",
]


@dataclass(frozen=True)
class ContextEvaluationCase:
    """五轮 Context/Memory 场景标签，不包含任何系统 prediction。"""

    case_id: str
    category: ContextCaseCategory
    split: Literal["dev"]
    user_id: str
    session_id: str
    messages: tuple[dict[str, str], ...]
    expected_point: str
    answerable: bool
    profile_facts: tuple[dict[str, object], ...]
    summary: str
    memories: tuple[dict[str, object], ...]
    evidence: tuple[dict[str, str], ...]
    review_method: str = "scenario_authored_ai_assisted"
    human_reviewed: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def load_context_dataset(path: Path) -> list[ContextEvaluationCase]:
    """读取 Context benchmark，并拒绝 predicted 字段。"""

    output: list[ContextEvaluationCase] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        if any(str(key).startswith("predicted") for key in raw):
            raise ValueError("context labels must not contain predictions")
        output.append(
            ContextEvaluationCase(
                case_id=str(raw["case_id"]), category=str(raw["category"]),  # type: ignore[arg-type]
                split=str(raw["split"]), user_id=str(raw["user_id"]),  # type: ignore[arg-type]
                session_id=str(raw["session_id"]),
                messages=tuple(dict(item) for item in raw["messages"]),
                expected_point=str(raw["expected_point"]), answerable=bool(raw["answerable"]),
                profile_facts=tuple(dict(item) for item in raw["profile_facts"]),
                summary=str(raw["summary"]),
                memories=tuple(dict(item) for item in raw["memories"]),
                evidence=tuple(dict(item) for item in raw["evidence"]),
                review_method=str(raw.get("review_method", "unknown")),
                human_reviewed=bool(raw.get("human_reviewed", False)),
            )
        )
    return output


def validate_context_dataset(cases: list[ContextEvaluationCase]) -> dict[str, object]:
    """校验 60 组、每组五轮、类别分布和标签边界。"""

    errors: list[str] = []
    ids = [case.case_id for case in cases]
    if len(cases) != 60:
        errors.append(f"expected 60 cases, got {len(cases)}")
    if len(ids) != len(set(ids)):
        errors.append("duplicate case ids")
    for case in cases:
        if len(case.messages) != 5:
            errors.append(f"{case.case_id} does not contain five turns")
        if case.answerable and not case.expected_point:
            errors.append(f"{case.case_id} answerable without expected point")
        if not case.answerable and case.expected_point:
            errors.append(f"{case.case_id} unanswerable with expected point")
    return {
        "case_count": len(cases),
        "turn_count": sum(len(case.messages) for case in cases),
        "split_counts": dict(Counter(case.split for case in cases)),
        "category_counts": dict(Counter(case.category for case in cases)),
        "review_method_counts": dict(Counter(case.review_method for case in cases)),
        "human_reviewed_count": sum(case.human_reviewed for case in cases),
        "errors": errors,
        "is_valid": not errors,
    }
