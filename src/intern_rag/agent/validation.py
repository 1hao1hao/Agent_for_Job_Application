from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

from intern_rag.agent.answer import Citation
from intern_rag.agent.generation import GenerationResult
from intern_rag.agent.schemas import BuiltContext, ContextItem


ValidationErrorType = Literal[
    "citation_not_found",
    "duplicate_citation",
    "missing_citation",
    "insufficient_with_citations",
]


@dataclass(frozen=True)
class ValidationIssue:
    """Citation Validator 发现的一条确定性问题。"""

    error_type: ValidationErrorType
    message: str
    chunk_id: str | None = None

    def to_dict(self) -> dict[str, str | None]:
        """转换成可写入 Trace 的普通字典。"""

        return asdict(self)


@dataclass(frozen=True)
class ValidationResult:
    """Citation Validator 的输出。

    只有 is_valid=True 时 citations 才包含可交给 RagResponse 的合法引用；
    失败时通过 issues 保留全部可确定的问题。
    """

    is_valid: bool
    citations: list[Citation]
    issues: list[ValidationIssue]


def validate_generation(
    generation: GenerationResult,
    context: BuiltContext,
) -> ValidationResult:
    """校验模型引用是否存在、重复，并检查 sufficient 组合是否合法。"""

    issues: list[ValidationIssue] = []
    context_items = {item.chunk_id: item for item in context.items}
    seen_ids: set[str] = set()

    if generation.sufficient and not generation.cited_chunk_ids:
        issues.append(
            ValidationIssue(
                error_type="missing_citation",
                message="sufficient=true requires at least one citation",
            )
        )
    if not generation.sufficient and generation.cited_chunk_ids:
        issues.append(
            ValidationIssue(
                error_type="insufficient_with_citations",
                message="sufficient=false must not include citations",
            )
        )

    for chunk_id in generation.cited_chunk_ids:
        if chunk_id in seen_ids:
            issues.append(
                ValidationIssue(
                    error_type="duplicate_citation",
                    message="cited chunk id is duplicated",
                    chunk_id=chunk_id,
                )
            )
            continue
        seen_ids.add(chunk_id)
        if chunk_id not in context_items:
            issues.append(
                ValidationIssue(
                    error_type="citation_not_found",
                    message="cited chunk id is not in the current context",
                    chunk_id=chunk_id,
                )
            )

    if issues:
        return ValidationResult(is_valid=False, citations=[], issues=issues)

    citations = [
        _citation_from_context_item(context_items[chunk_id])
        for chunk_id in generation.cited_chunk_ids
    ]
    return ValidationResult(is_valid=True, citations=citations, issues=[])


def _citation_from_context_item(item: ContextItem) -> Citation:
    """把通过校验的 ContextItem 转换为已有 Citation 契约。"""

    return Citation(
        chunk_id=item.chunk_id,
        source_path=item.source_path,
        source_type=item.source_type,
        title=item.title,
        rank=item.rank,
        score=item.score,
    )
