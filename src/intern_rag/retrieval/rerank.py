from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence

from intern_rag.ingestion import Chunk
from intern_rag.retrieval.base import RetrievalResult, Retriever
from intern_rag.retrieval.keyword import tokenize_text


class RerankScorer(Protocol):
    """Reranker 依赖的最小 Query-Document 打分接口。"""

    name: str
    version: str

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        """返回每个 document 与 query 的相关性分数。"""


@dataclass
class FakeRerankScorer:
    """自动化测试使用的可预测 scorer，不访问网络。"""

    scores_by_text: dict[str, float]
    name: str = "fake-reranker"
    version: str = "v1"
    calls: list[tuple[str, list[str]]] = field(default_factory=list, init=False)

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        """按预设文本分数返回结果并记录调用。"""

        document_list = list(documents)
        self.calls.append((query, document_list))
        return [self.scores_by_text.get(text, 0.0) for text in document_list]


class CrossEncoderRerankScorer:
    """使用固定 revision 的多语言 CrossEncoder 重排中文候选。"""

    def __init__(
        self,
        model_name: str,
        revision: str,
        *,
        device: str = "cpu",
        local_files_only: bool = False,
        batch_size: int = 16,
    ) -> None:
        from sentence_transformers import CrossEncoder

        if not model_name.strip() or not revision.strip():
            raise ValueError("model_name and revision must not be empty")
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than 0")
        self.name = model_name
        self.version = revision
        self.batch_size = batch_size
        self.model = CrossEncoder(
            model_name,
            revision=revision,
            device=device,
            local_files_only=local_files_only,
        )

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        """批量计算 Query-Chunk pair 的 CrossEncoder 分数。"""

        if not documents:
            return []
        pairs = [(query, document) for document in documents]
        scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )
        return [float(score) for score in scores]


class ChineseTokenOverlapScorer:
    """使用中文单字/双字与英文 token overlap 精排候选。"""

    name = "chinese-token-overlap"
    version = "v1"

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        """计算文档覆盖 Query token 的比例，作为可解释精排分数。"""

        query_tokens = tokenize_text(query)
        if not query_tokens:
            return [0.0 for _ in documents]
        return [
            len(query_tokens & tokenize_text(document)) / len(query_tokens)
            for document in documents
        ]


class RerankRetriever:
    """先召回 top-N 候选，再使用 scorer 重排并返回统一 RetrievalResult。"""

    def __init__(
        self,
        candidate_retriever: Retriever,
        scorer: RerankScorer,
        *,
        candidate_k: int = 20,
    ) -> None:
        if candidate_k <= 0:
            raise ValueError("candidate_k must be greater than 0")
        self.candidate_retriever = candidate_retriever
        self.scorer = scorer
        self.candidate_k = candidate_k

    def __call__(
        self,
        query: str,
        chunks: list[Chunk],
        top_k: int = 5,
        source_types: set[str] | None = None,
    ) -> list[RetrievalResult]:
        """重排 Hybrid 候选；无法召回的 Chunk 不会被 Reranker 找回。"""

        if not query.strip() or top_k <= 0:
            return []
        candidate_k = max(top_k, self.candidate_k)
        candidates = self.candidate_retriever(
            query,
            chunks,
            top_k=candidate_k,
            source_types=source_types,
        )
        if not candidates:
            return []
        scores = self.scorer.score(
            query,
            [candidate.chunk.text for candidate in candidates],
        )
        if len(scores) != len(candidates):
            raise ValueError("reranker score count does not match candidates")

        reranked = sorted(
            zip(candidates, scores),
            key=lambda item: (-item[1], item[0].rank, item[0].chunk_id),
        )
        return [
            RetrievalResult(
                chunk_id=candidate.chunk_id,
                score=score,
                rank=rank,
                chunk=candidate.chunk,
                reason=(
                    f"rerank_score={score:.6f}, original_rank={candidate.rank}"
                ),
                details={
                    **candidate.details,
                    "original_rank": candidate.rank,
                    "original_score": candidate.score,
                    "rerank_score": score,
                },
            )
            for rank, (candidate, score) in enumerate(
                reranked[:top_k], start=1
            )
        ]
