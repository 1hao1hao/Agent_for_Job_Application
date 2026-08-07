from __future__ import annotations

import re

from intern_rag.ingestion import Chunk
from intern_rag.retrieval.base import RetrievalResult


ENGLISH_OR_NUMBER_PATTERN = re.compile(r"[a-zA-Z0-9]+")


def tokenize_text(text: str) -> set[str]:
    """把中英文混合文本转换成可匹配的关键词集合。

    第一版不引入分词库：英文和数字按连续词提取，中文按单字和相邻双字
    提取。这样能覆盖中文 query 中常见的“检索”“实习”“向量”等短词。
    """

    normalized_text = text.lower()
    tokens = set(ENGLISH_OR_NUMBER_PATTERN.findall(normalized_text))
    chinese_chars = [
        char for char in normalized_text if "\u4e00" <= char <= "\u9fff"
    ]
    # 对于set集合, add()是加一个元素，update是加多个元素
    tokens.update(chinese_chars)
    tokens.update(
        chinese_chars[index] + chinese_chars[index + 1]
        for index in range(len(chinese_chars) - 1)
    )
    return {token for token in tokens if token.strip()}


def score_chunk(query_tokens: set[str], chunk: Chunk) -> tuple[float, set[str]]:
    """计算 query 和 chunk 的关键词匹配分数。（都先经过 tokenize_text 处理）

    分数采用最简单的重叠率：命中的 query token 数量 / query token 数量。
    返回命中的 token，方便测试和后续 trace 解释检索原因。
    """

    if not query_tokens:
        return 0.0, set()

    chunk_tokens = tokenize_text(chunk.text)
    matched_tokens = query_tokens & chunk_tokens #交集
    score = len(matched_tokens) / len(query_tokens)
    return score, matched_tokens


def retrieve_top_k(
    query: str,
    chunks: list[Chunk],
    top_k: int = 5,
    source_types: set[str] | None = None,
) -> list[RetrievalResult]:
    """从 chunks 中检索和 query 最相关的 top-k 结果。

    Args:
        query: 用户查询文本。
        chunks: ingestion 生成的统一 Chunk 列表。
        top_k: 最多返回多少条结果。
        source_types: 可选 source type 过滤，例如只检索 `{"jd"}`。

    Returns:
        按 score 从高到低排序的 RetrievalResult 列表。没有关键词命中的
        chunk 不会返回。
    """

    if top_k <= 0:
        return []

    query_tokens = tokenize_text(query)
    scored_results: list[RetrievalResult] = []

    for chunk in chunks:
        if source_types is not None and chunk.source_type not in source_types:
            continue

        score, matched_tokens = score_chunk(query_tokens, chunk)
        if score <= 0:
            continue

        scored_results.append(
            RetrievalResult(
                chunk_id=chunk.id,
                score=score,
                rank=0,
                chunk=chunk,
                reason=", ".join(sorted(matched_tokens)) if matched_tokens else None,
                details={"matched_token_count": len(matched_tokens)},
            )
        )

    sorted_results = sorted(
        scored_results,
        key=lambda result: (-result.score, result.chunk.source_type, result.chunk_id),# 按照比较键依次比较
    )
    return [
        RetrievalResult(
            chunk_id=result.chunk_id,
            score=result.score,
            rank=rank,
            chunk=result.chunk,
            reason=result.reason,
            details=result.details,
        )
        for rank, result in enumerate(sorted_results[:top_k], start=1)
    ]
