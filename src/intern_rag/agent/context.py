from __future__ import annotations

from typing import Literal, Sequence

from intern_rag.agent.schemas import BuiltContext, ContextItem
from intern_rag.retrieval import RetrievalResult


CONTEXT_SEPARATOR = "\n\n"
ContextStrategy = Literal["rank_prefix", "source_balanced"]


def build_context(
    query: str,
    retrieved_results: list[RetrievalResult],
    max_chars: int,
    *,
    strategy: ContextStrategy = "rank_prefix",
    required_source_types: Sequence[str] = (),
) -> BuiltContext:
    """在字符预算内构建不截断单条证据的模型上下文。

    Args:
        query: 当前用户问题，用于把上下文与请求关联。
        retrieved_results: 检索模块返回的候选证据。
        max_chars: 最终 context text 允许使用的最大字符数。
        strategy: `rank_prefix` 严格选择排名前缀；`source_balanced` 先为
            每个必需来源保留其最高排名证据，再按 rank 填满剩余预算。
        required_source_types: Router 判定本轮回答需要覆盖的来源类型。

    Returns:
        BuiltContext，包含模型输入文本、实际使用的证据和因预算跳过的
        chunk ids。

    Raises:
        ValueError: query 为空或 max_chars 不是正数。

    `rank_prefix` 保持原有行为：下一个完整 Chunk 放不下时立即停止。
    `source_balanced` 面向多来源紧预算场景，优先保留来源覆盖，并允许跳过
    放不下的候选继续尝试后续较短证据。两种策略都不会截断 Chunk。
    """

    if not query.strip():
        raise ValueError("query must not be empty")
    if max_chars <= 0:
        raise ValueError("max_chars must be greater than 0")
    if strategy not in {"rank_prefix", "source_balanced"}:
        raise ValueError(f"unknown context strategy: {strategy}")

    ranked_results = sorted(retrieved_results, key=lambda result: result.rank)
    if strategy == "rank_prefix":
        selected_results = _select_rank_prefix(ranked_results, max_chars)
    else:
        candidate_order = _source_balanced_order(
            ranked_results, required_source_types
        )
        selected_results = _pack_candidates(candidate_order, max_chars)
        selected_results.sort(key=lambda result: result.rank)

    selected_ids = {result.chunk_id for result in selected_results}
    skipped_chunk_ids = [
        result.chunk_id
        for result in ranked_results
        if result.chunk_id not in selected_ids
    ]
    selected_items = [context_item_from_result(result) for result in selected_results]
    context_text = CONTEXT_SEPARATOR.join(
        format_context_item(item) for item in selected_items
    )
    covered_source_types = sorted({item.source_type for item in selected_items})
    required_sources = list(dict.fromkeys(required_source_types))
    missing_source_types = [
        source for source in required_sources if source not in covered_source_types
    ]
    return BuiltContext(
        query=query,
        text=context_text,
        items=selected_items,
        used_chunk_ids=[item.chunk_id for item in selected_items],
        skipped_chunk_ids=skipped_chunk_ids,
        char_count=len(context_text),
        max_chars=max_chars,
        selection_strategy=strategy,
        covered_source_types=covered_source_types,
        missing_source_types=missing_source_types,
    )


def _select_rank_prefix(
    ranked_results: list[RetrievalResult],
    max_chars: int,
) -> list[RetrievalResult]:
    """保持原有严格 rank 前缀策略，遇到首条放不下的证据即停止。"""

    selected: list[RetrievalResult] = []
    current_char_count = 0
    for result in ranked_results:
        formatted = format_context_item(context_item_from_result(result))
        separator_length = len(CONTEXT_SEPARATOR) if selected else 0
        candidate_char_count = current_char_count + separator_length + len(formatted)
        if candidate_char_count > max_chars:
            break
        selected.append(result)
        current_char_count = candidate_char_count
    return selected


def _source_balanced_order(
    ranked_results: list[RetrievalResult],
    required_source_types: Sequence[str],
) -> list[RetrievalResult]:
    """先列出每个必需来源的最高排名证据，再追加其余 rank 候选。"""

    representatives: list[RetrievalResult] = []
    representative_ids: set[str] = set()
    for source_type in dict.fromkeys(required_source_types):
        representative = next(
            (
                result
                for result in ranked_results
                if result.chunk.source_type == source_type
            ),
            None,
        )
        if representative is not None and representative.chunk_id not in representative_ids:
            representatives.append(representative)
            representative_ids.add(representative.chunk_id)
    representatives.sort(key=lambda result: result.rank)
    return representatives + [
        result
        for result in ranked_results
        if result.chunk_id not in representative_ids
    ]


def _pack_candidates(
    candidates: list[RetrievalResult],
    max_chars: int,
) -> list[RetrievalResult]:
    """按候选优先级放入完整证据，单条放不下时继续尝试后续候选。"""

    selected_items: list[ContextItem] = []
    selected_results: list[RetrievalResult] = []
    current_char_count = 0
    for result in candidates:
        item = context_item_from_result(result)
        formatted_item = format_context_item(item)
        separator_length = len(CONTEXT_SEPARATOR) if selected_items else 0
        candidate_char_count = (
            current_char_count + separator_length + len(formatted_item)
        )
        if candidate_char_count > max_chars:
            continue
        selected_items.append(item)
        selected_results.append(result)
        current_char_count = candidate_char_count
    return selected_results


def context_item_from_result(result: RetrievalResult) -> ContextItem:
    """把 RetrievalResult 转换为不丢失来源和排名的 ContextItem。"""

    return ContextItem(
        chunk_id=result.chunk_id,
        source_type=result.chunk.source_type,
        source_path=result.chunk.source_path,
        title=result.chunk.title,
        text=result.chunk.text,
        rank=result.rank,
        score=result.score,
    )


def format_context_item(item: ContextItem) -> str:
    """把一条结构化证据格式化为模型可读文本。

    chunk id 与来源字段和原文一起进入上下文，后续模型才能返回可由程序
    校验的 cited_chunk_ids。
    """

    return (
        f"chunk_id: {item.chunk_id}\n"
        f"source_type: {item.source_type}\n"
        f"source_path: {item.source_path}\n"
        f"title: {item.title}\n"
        f"rank: {item.rank}\n"
        f"text:\n{item.text.strip()}"
    )
