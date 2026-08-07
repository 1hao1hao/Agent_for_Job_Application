from __future__ import annotations

from intern_rag.agent.schemas import BuiltContext, ContextItem
from intern_rag.retrieval import RetrievalResult


CONTEXT_SEPARATOR = "\n\n"


def build_context(
    query: str,
    retrieved_results: list[RetrievalResult],
    max_chars: int,
) -> BuiltContext:
    """在字符预算内按 rank 构建完整证据上下文。

    Args:
        query: 当前用户问题，用于把上下文与请求关联。
        retrieved_results: 检索模块返回的候选证据。
        max_chars: 最终 context text 允许使用的最大字符数。

    Returns:
        BuiltContext，包含模型输入文本、实际使用的证据和因预算跳过的
        chunk ids。

    Raises:
        ValueError: query 为空或 max_chars 不是正数。

    选择策略是严格的 rank 前缀：下一个完整 chunk 放不下时立即停止，不会
    截断该 chunk，也不会跳过高排名结果后改用更短的低排名结果。
    """

    if not query.strip():
        raise ValueError("query must not be empty")
    if max_chars <= 0:
        raise ValueError("max_chars must be greater than 0")

    ranked_results = sorted(retrieved_results, key=lambda result: result.rank)
    selected_items: list[ContextItem] = []
    context_parts: list[str] = []
    skipped_chunk_ids: list[str] = []
    current_char_count = 0

    for result_index, result in enumerate(ranked_results):
        item = context_item_from_result(result)
        formatted_item = format_context_item(item)
        separator_length = len(CONTEXT_SEPARATOR) if context_parts else 0
        candidate_char_count = (
            current_char_count + separator_length + len(formatted_item)
        )

        if candidate_char_count > max_chars:
            skipped_chunk_ids = [
                skipped_result.chunk_id
                for skipped_result in ranked_results[result_index:]
            ]
            break

        selected_items.append(item)
        context_parts.append(formatted_item)
        current_char_count = candidate_char_count

    context_text = CONTEXT_SEPARATOR.join(context_parts)
    return BuiltContext(
        query=query,
        text=context_text,
        items=selected_items,
        used_chunk_ids=[item.chunk_id for item in selected_items],
        skipped_chunk_ids=skipped_chunk_ids,
        char_count=len(context_text),
        max_chars=max_chars,
    )


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
