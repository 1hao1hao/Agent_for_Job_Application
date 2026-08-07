from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Literal


EvalCategory = Literal[
    "single_source",
    "multi_source",
    "semantic_paraphrase",
    "unanswerable",
]
EvalSplit = Literal["dev", "test"]

VALID_CATEGORIES = {
    "single_source",
    "multi_source",
    "semantic_paraphrase",
    "unanswerable",
}
VALID_SPLITS = {"dev", "test"}
VALID_SOURCES = {"interview", "jd", "project_logs", "resume", "user_profile"}
VALID_INTENTS = {
    "analyze_jd",
    "match_resume",
    "interview_prepare",
    "project_explain",
    "application_plan",
    "unknown",
}


@dataclass(frozen=True)
class EvaluationCase:
    """不包含系统 prediction 的人工标签契约。"""

    case_id: str
    query: str
    category: EvalCategory
    split: EvalSplit
    expected_intent: str
    expected_sources: list[str]
    relevant_chunk_ids: list[str]
    answerable: bool
    expected_points: list[str]
    notes: str
    human_reviewed: bool

    def to_dict(self) -> dict[str, object]:
        """转换成可写入 JSONL 的字典。"""

        return asdict(self)


@dataclass(frozen=True)
class DatasetValidation:
    """评测集校验结果和分布统计。"""

    errors: list[str]
    warnings: list[str]
    case_count: int
    split_counts: dict[str, int]
    category_counts: dict[str, int]
    reviewed_case_count: int
    human_review_required: bool

    @property
    def is_valid(self) -> bool:
        """没有错误时才允许作为正式评测集。"""

        return not self.errors

    def to_dict(self) -> dict[str, object]:
        """转换成校验报告字典。"""

        return {
            **asdict(self),
            "is_valid": self.is_valid,
        }


def load_evaluation_dataset(path: Path) -> list[EvaluationCase]:
    """读取不含 predicted 字段的 JSONL 评测标签。"""

    cases: list[EvaluationCase] = []
    with path.open("r", encoding="utf-8") as dataset_file:
        for line_number, line in enumerate(dataset_file, start=1):
            if not line.strip():
                continue
            raw_case = json.loads(line)
            if not isinstance(raw_case, dict):
                raise ValueError(f"{path}:{line_number} must be a JSON object")
            predicted_fields = sorted(
                key for key in raw_case if key.startswith("predicted_")
            )
            if predicted_fields:
                raise ValueError(
                    f"{path}:{line_number} contains forbidden predictions: "
                    f"{', '.join(predicted_fields)}"
                )
            cases.append(_case_from_dict(raw_case, path, line_number))
    return cases


def validate_evaluation_dataset(
    cases: list[EvaluationCase],
    *,
    available_chunk_ids: set[str],
    require_full_distribution: bool = True,
    require_human_review: bool = True,
) -> DatasetValidation:
    """校验 ID、分割、标签、相关 Chunk 和版本对应的分布。"""

    errors: list[str] = []
    warnings: list[str] = []
    seen_case_ids: set[str] = set()
    seen_queries: set[str] = set()

    for case in cases:
        if case.case_id in seen_case_ids:
            errors.append(f"duplicate case_id: {case.case_id}")
        seen_case_ids.add(case.case_id)
        if case.split not in VALID_SPLITS:
            errors.append(f"{case.case_id} has invalid split: {case.split}")
        if case.category not in VALID_CATEGORIES:
            errors.append(f"{case.case_id} has invalid category: {case.category}")
        if not case.query.strip():
            errors.append(f"{case.case_id} has empty query")
        normalized_query = " ".join(case.query.lower().split())
        if normalized_query in seen_queries:
            errors.append(f"duplicate query text: {case.case_id}")
        seen_queries.add(normalized_query)
        if case.expected_intent not in VALID_INTENTS:
            errors.append(
                f"{case.case_id} has invalid intent: {case.expected_intent}"
            )
        if not set(case.expected_sources).issubset(VALID_SOURCES):
            errors.append(f"{case.case_id} has invalid expected_sources")
        if len(case.expected_sources) != len(set(case.expected_sources)):
            errors.append(f"{case.case_id} has duplicate expected_sources")
        if len(case.relevant_chunk_ids) != len(set(case.relevant_chunk_ids)):
            errors.append(f"{case.case_id} has duplicate relevant_chunk_ids")

        missing_chunk_ids = sorted(
            set(case.relevant_chunk_ids) - available_chunk_ids
        )
        if missing_chunk_ids:
            errors.append(
                f"{case.case_id} references missing chunks: "
                f"{', '.join(missing_chunk_ids)}"
            )
        if case.answerable and not case.relevant_chunk_ids:
            errors.append(f"{case.case_id} is answerable but has no relevant chunks")
        if not case.answerable and case.relevant_chunk_ids:
            errors.append(
                f"{case.case_id} is unanswerable but has relevant chunks"
            )
        if case.answerable and not case.expected_points:
            errors.append(f"{case.case_id} is answerable but has no expected points")
        if require_human_review and not case.human_reviewed:
            errors.append(f"{case.case_id} has not been human reviewed")

    split_counts = dict(Counter(case.split for case in cases))
    category_counts = dict(Counter(case.category for case in cases))
    if require_full_distribution:
        expected_total = len(cases)
        if expected_total == 60:
            expected_dev, expected_test, expected_category = 40, 20, 15
        elif expected_total == 120:
            expected_dev, expected_test, expected_category = 80, 40, 30
        else:
            expected_dev, expected_test, expected_category = 0, 0, 0
            errors.append(
                "full distribution requires a supported 60-case or 120-case dataset, "
                f"got {expected_total}"
            )
        if split_counts != {"dev": expected_dev, "test": expected_test}:
            errors.append(
                f"expected split counts {expected_dev}/{expected_test}, "
                f"got {split_counts}"
            )
        expected_categories = {
            category: expected_category for category in VALID_CATEGORIES
        }
        if category_counts != expected_categories:
            errors.append(
                f"expected 15 cases per category, got {category_counts}"
            )
        for category in VALID_CATEGORIES:
            dev_count = sum(
                case.category == category and case.split == "dev" for case in cases
            )
            test_count = sum(
                case.category == category and case.split == "test" for case in cases
            )
            expected_category_dev = expected_category * 2 // 3
            expected_category_test = expected_category - expected_category_dev
            if (dev_count, test_count) != (
                expected_category_dev,
                expected_category_test,
            ):
                errors.append(
                    f"{category} requires {expected_category_dev} dev/"
                    f"{expected_category_test} test, "
                    f"got {dev_count}/{test_count}"
                )

    if not all(case.human_reviewed for case in cases):
        warnings.append(
            "dataset contains labels pending human review; "
            "results must be marked candidate"
        )
    return DatasetValidation(
        errors=errors,
        warnings=warnings,
        case_count=len(cases),
        split_counts=split_counts,
        category_counts=category_counts,
        reviewed_case_count=sum(case.human_reviewed for case in cases),
        human_review_required=require_human_review,
    )


def write_dataset_validation(validation: DatasetValidation, path: Path) -> None:
    """写出评测集校验报告。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(validation.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _case_from_dict(
    raw_case: dict[str, object],
    path: Path,
    line_number: int,
) -> EvaluationCase:
    """把一行 JSON 转换为 EvaluationCase，并检查必要字段。"""

    required_fields = {
        "case_id",
        "query",
        "category",
        "split",
        "expected_intent",
        "expected_sources",
        "relevant_chunk_ids",
        "answerable",
        "expected_points",
        "notes",
        "human_reviewed",
    }
    missing_fields = sorted(required_fields - raw_case.keys())
    if missing_fields:
        raise ValueError(
            f"{path}:{line_number} misses fields: {', '.join(missing_fields)}"
        )

    list_fields = ("expected_sources", "relevant_chunk_ids", "expected_points")
    for field_name in list_fields:
        value = raw_case[field_name]
        if not isinstance(value, list) or not all(
            isinstance(item, str) for item in value
        ):
            raise ValueError(
                f"{path}:{line_number} {field_name} must be list[string]"
            )
    if not isinstance(raw_case["answerable"], bool):
        raise ValueError(f"{path}:{line_number} answerable must be boolean")
    if not isinstance(raw_case["human_reviewed"], bool):
        raise ValueError(f"{path}:{line_number} human_reviewed must be boolean")

    return EvaluationCase(
        case_id=str(raw_case["case_id"]),
        query=str(raw_case["query"]),
        category=str(raw_case["category"]),  # type: ignore[arg-type]
        split=str(raw_case["split"]),  # type: ignore[arg-type]
        expected_intent=str(raw_case["expected_intent"]),
        expected_sources=list(raw_case["expected_sources"]),
        relevant_chunk_ids=list(raw_case["relevant_chunk_ids"]),
        answerable=raw_case["answerable"],
        expected_points=list(raw_case["expected_points"]),
        notes=str(raw_case["notes"]),
        human_reviewed=raw_case["human_reviewed"],
    )
