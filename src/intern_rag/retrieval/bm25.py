from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re

from intern_rag.ingestion import Chunk
from intern_rag.retrieval.base import RetrievalResult


ENGLISH_OR_NUMBER_PATTERN = re.compile(r"[a-zA-Z0-9]+")


def tokenize_bm25(text: str) -> list[str]:
    """把中英文文本转换为保留词频的 BM25 token 序列。

    英文和数字保留连续词；中文同时保留单字与相邻双字。与旧 Keyword 的
    `set` 不同，这里返回 `list`，因为 BM25 必须使用词频和文档长度。
    """

    normalized = text.lower()
    tokens = ENGLISH_OR_NUMBER_PATTERN.findall(normalized)
    chinese_chars = [char for char in normalized if "\u4e00" <= char <= "\u9fff"]
    tokens.extend(chinese_chars)
    tokens.extend(
        chinese_chars[index] + chinese_chars[index + 1]
        for index in range(len(chinese_chars) - 1)
    )
    return [token for token in tokens if token.strip()]


@dataclass(frozen=True)
class BM25Index:
    """离线构建的 BM25 统计量，不保存无法复现的运行时对象。"""

    dataset_version: str
    chunk_ids: list[str]
    document_lengths: list[int]
    term_frequencies: list[dict[str, int]]
    document_frequencies: dict[str, int]
    average_document_length: float
    created_at: str
    tokenizer_version: str = "mixed-char-bigram-v1"

    def to_dict(self) -> dict[str, object]:
        """转换为可持久化 JSON 的普通字典。"""

        return asdict(self)


def build_bm25_index(chunks: list[Chunk], dataset_version: str) -> BM25Index:
    """离线统计 Chunk 词频、文档频率和平均长度，生成稳定 BM25 index。"""

    if not chunks:
        raise ValueError("cannot build BM25 index from empty chunks")
    if not dataset_version.strip():
        raise ValueError("dataset_version must not be empty")

    term_frequencies: list[dict[str, int]] = []
    document_frequencies: Counter[str] = Counter()
    document_lengths: list[int] = []
    for chunk in chunks:
        tokens = tokenize_bm25(chunk.text)
        frequencies = Counter(tokens)
        term_frequencies.append(dict(frequencies))
        document_lengths.append(len(tokens))
        document_frequencies.update(frequencies.keys())

    return BM25Index(
        dataset_version=dataset_version,
        chunk_ids=[chunk.id for chunk in chunks],
        document_lengths=document_lengths,
        term_frequencies=term_frequencies,
        document_frequencies=dict(document_frequencies),
        average_document_length=sum(document_lengths) / len(document_lengths),
        created_at=datetime.now(timezone.utc).isoformat(),
    )


def save_bm25_index(index: BM25Index, path: Path) -> None:
    """保存版本化 BM25 index，供 API、Worker 和评测进程重复加载。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(index.to_dict(), ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def load_bm25_index(path: Path) -> BM25Index:
    """读取 BM25 index，并校验各 Chunk 统计数组长度一致。"""

    payload = json.loads(path.read_text(encoding="utf-8"))
    index = BM25Index(
        dataset_version=str(payload["dataset_version"]),
        chunk_ids=[str(value) for value in payload["chunk_ids"]],
        document_lengths=[int(value) for value in payload["document_lengths"]],
        term_frequencies=[
            {str(token): int(count) for token, count in values.items()}
            for values in payload["term_frequencies"]
        ],
        document_frequencies={
            str(token): int(count)
            for token, count in payload["document_frequencies"].items()
        },
        average_document_length=float(payload["average_document_length"]),
        created_at=str(payload["created_at"]),
        tokenizer_version=str(payload.get("tokenizer_version", "mixed-char-bigram-v1")),
    )
    lengths = {
        len(index.chunk_ids),
        len(index.document_lengths),
        len(index.term_frequencies),
    }
    if len(lengths) != 1:
        raise ValueError("BM25 index arrays have inconsistent lengths")
    return index


class BM25Retriever:
    """使用标准 Okapi BM25 公式对离线 Chunk 统计进行检索。"""

    def __init__(self, index: BM25Index, *, k1: float = 1.5, b: float = 0.75) -> None:
        if k1 <= 0:
            raise ValueError("k1 must be greater than 0")
        if not 0 <= b <= 1:
            raise ValueError("b must be between 0 and 1")
        self.index = index
        self.k1 = k1
        self.b = b

    def __call__(
        self,
        query: str,
        chunks: list[Chunk],
        top_k: int = 5,
        source_types: set[str] | None = None,
    ) -> list[RetrievalResult]:
        """计算 Query 的 BM25 分数，过滤来源后稳定返回 top-k。

        输入是统一 Chunk 列表；处理时只读取离线 index 中对应 Chunk 的词频和
        长度；输出继续使用 `RetrievalResult`，因此 Pipeline 无需了解 BM25 细节。
        """

        query_tokens = list(dict.fromkeys(tokenize_bm25(query)))
        if not query_tokens or top_k <= 0:
            return []
        chunk_by_id = {chunk.id: chunk for chunk in chunks}
        document_count = len(self.index.chunk_ids)
        average_length = self.index.average_document_length or 1.0
        scored: list[tuple[float, str, Chunk, list[str]]] = []

        for position, chunk_id in enumerate(self.index.chunk_ids):
            chunk = chunk_by_id.get(chunk_id)
            if chunk is None:
                continue
            if source_types is not None and chunk.source_type not in source_types:
                continue
            frequencies = self.index.term_frequencies[position]
            document_length = self.index.document_lengths[position]
            score = 0.0
            matched_tokens: list[str] = []
            for token in query_tokens:
                term_frequency = frequencies.get(token, 0)
                if term_frequency <= 0:
                    continue
                matched_tokens.append(token)
                document_frequency = self.index.document_frequencies.get(token, 0)
                inverse_document_frequency = math.log(
                    1.0
                    + (document_count - document_frequency + 0.5)
                    / (document_frequency + 0.5)
                )
                length_normalization = self.k1 * (
                    1.0 - self.b + self.b * document_length / average_length
                )
                score += inverse_document_frequency * (
                    term_frequency * (self.k1 + 1.0)
                    / (term_frequency + length_normalization)
                )
            if score > 0:
                scored.append((score, chunk_id, chunk, matched_tokens))

        scored.sort(key=lambda item: (-item[0], item[2].source_type, item[1]))
        return [
            RetrievalResult(
                chunk_id=chunk_id,
                score=score,
                rank=rank,
                chunk=chunk,
                reason=f"bm25={score:.6f}; matched={','.join(matched_tokens)}",
                details={
                    "bm25_score": score,
                    "matched_token_count": len(matched_tokens),
                },
            )
            for rank, (score, chunk_id, chunk, matched_tokens) in enumerate(
                scored[:top_k], start=1
            )
        ]
