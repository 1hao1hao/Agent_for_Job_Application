from __future__ import annotations

from dataclasses import asdict, dataclass

from intern_rag.retrieval import RetrievalResult


CitationValue = str | int | float


@dataclass(frozen=True)
class Citation:
    """回答中一条引用的结构化来源。

    citation（引用）用来说明回答依据来自哪个 chunk 和原始文件。第一版
    只记录检索阶段已有的信息，不额外做引用准确率判断。
    """

    chunk_id: str
    source_path: str
    source_type: str
    title: str
    rank: int
    score: float

    def to_dict(self) -> dict[str, CitationValue]:
        """转换成普通 dict，方便写入 trace 或 JSON。"""

        return asdict(self)


@dataclass(frozen=True)
class AnswerResult:
    """一次回答组织的结果。

    answer 是面向用户的文本；citations 是回答所用证据来源；当证据不足时，
    `is_evidence_sufficient` 为 False，并且 citations 为空。
    """

    answer: str
    citations: list[Citation]
    used_chunk_ids: list[str]
    is_evidence_sufficient: bool


def compose_answer(
    query: str,
    retrieved_results: list[RetrievalResult],
    max_chunks: int = 3,
    snippet_chars: int = 220,
) -> AnswerResult:
    """基于 top chunks 组织第一版回答和引用。

    这个函数不调用 LLM,也不生成检索证据之外的新事实。它只把 top chunks
    中的原文片段整理成简短回答，并为每个使用的 chunk 生成 citation。

    实际操作：选取检索结果，构建引用 (检索结果中的前 max_chunks 个chunks,为其构建 citation )
    再提取依据(chunk 的 rank + 前snippet_char 个字符 +...);
    生成答案
    """

    if max_chunks <= 0 or snippet_chars <= 0 or not retrieved_results:
        return _insufficient_answer(query)

    selected_results = retrieved_results[:max_chunks]
    citations = [citation_from_result(result) for result in selected_results]
    evidence_lines = [
        f"{index}. {format_evidence_snippet(result, snippet_chars)}"
        for index, result in enumerate(selected_results, start=1)
    ]

    answer = (
        f"针对你的问题：{query}\n\n"
        "当前只能基于已检索到的证据回答：\n"
        + "\n".join(evidence_lines)
        + "\n\n"
        "以上内容均来自检索到的 chunks，具体来源见 citations。"
    )

    return AnswerResult(
        answer=answer,
        citations=citations,
        used_chunk_ids=[result.chunk_id for result in selected_results],
        is_evidence_sufficient=True,
    )


def citation_from_result(result: RetrievalResult) -> Citation:
    """从 RetrievalResult 生成一条 Citation。"""

    return Citation(
        chunk_id=result.chunk_id,
        source_path=result.chunk.source_path,
        source_type=result.chunk.source_type,
        title=result.chunk.title,
        rank=result.rank,
        score=result.score,
    )


def format_evidence_snippet(result: RetrievalResult, snippet_chars: int) -> str:
    """把检索结果中的 chunk 文本整理成回答里的证据片段。 
        实际操作：取chunk中的rank + 前 snippet_chars 个字符作为证据片段
    """

    text = " ".join(result.chunk.text.split())
    snippet = text[:snippet_chars]
    if len(text) > snippet_chars:
        snippet += "..."
    return f"[{result.rank}] {snippet}"


def _insufficient_answer(query: str) -> AnswerResult:
    """构造证据不足时的保守回答。"""

    return AnswerResult(
        answer=(
            f"针对你的问题：{query}\n\n"
            "当前证据不足：没有检索到可用于回答的相关 chunk。"
            "请补充岗位、简历或面试资料后再尝试。"
        ),
        citations=[],
        used_chunk_ids=[],
        is_evidence_sufficient=False,
    )
