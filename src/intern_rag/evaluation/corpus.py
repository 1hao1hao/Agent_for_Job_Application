from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
from statistics import mean
from typing import Iterable

from intern_rag.ingestion import (
    Chunk,
    build_document_from_file,
    load_chunks_from_raw_dir,
)
from intern_rag.ingestion.chunking import SUPPORTED_SUFFIXES


SOURCE_TYPES = {"interview", "jd", "project_logs", "resume", "user_profile"}


@dataclass(frozen=True)
class CorpusManifestEntry:
    """语料清单中的一份文档记录。"""

    document_id: str
    source_type: str
    source_path: str
    source_platform: str
    source_url: str
    collected_at: str
    public_status: str
    anonymized: bool
    content_hash: str
    content_origin: str
    human_reviewed: bool

    def to_dict(self) -> dict[str, object]:
        """转换成 JSONL 可写入的字典。"""

        return asdict(self)


@dataclass(frozen=True)
class CorpusStats:
    """语料和 Chunk 的可复现统计结果。"""

    document_count: int
    source_counts: dict[str, int]
    chunk_count: int
    chunk_length_min: int
    chunk_length_max: int
    chunk_length_mean: float
    chunk_length_p50: int
    chunk_length_p95: int
    empty_documents: list[str]
    duplicate_content_hashes: dict[str, list[str]]
    human_reviewed_documents: int

    def to_dict(self) -> dict[str, object]:
        """转换成报告可写入的字典。"""

        return asdict(self)


def build_corpus(
    raw_root: Path,
    *,
    max_chars: int = 800,
) -> tuple[list[CorpusManifestEntry], list[Chunk], CorpusStats]:
    """从 raw 目录生成 manifest、统一 Chunks 和语料统计。"""

    entries: list[CorpusManifestEntry] = []
    empty_documents: list[str] = []
    hashes_to_documents: dict[str, list[str]] = defaultdict(list)

    for path in sorted(raw_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        document = build_document_from_file(path, raw_root)
        raw_metadata = _read_front_matter(path)
        relative_path = path.relative_to(raw_root).as_posix()
        document_id = f"doc-{sha256(relative_path.encode('utf-8')).hexdigest()[:12]}"
        content_hash = sha256(document.text.strip().encode("utf-8")).hexdigest()
        entry = CorpusManifestEntry(
            document_id=document_id,
            source_type=document.metadata.source_type,
            source_path=str(path),
            source_platform=raw_metadata.get("source_platform", ""),
            source_url=raw_metadata.get("source_url", ""),
            collected_at=raw_metadata.get(
                "collected_at",
                raw_metadata.get("first_seen_at", ""),
            ),
            public_status=raw_metadata.get("public_status", ""),
            anonymized=_parse_bool(raw_metadata.get("anonymized", "")),
            content_hash=content_hash,
            content_origin=raw_metadata.get("content_origin", ""),
            human_reviewed=_parse_bool(raw_metadata.get("human_reviewed", "")),
        )
        entries.append(entry)
        hashes_to_documents[content_hash].append(document_id)
        if not document.text.strip():
            empty_documents.append(relative_path)

    chunks = load_chunks_from_raw_dir(raw_root, max_chars=max_chars)
    chunk_lengths = [len(chunk.text) for chunk in chunks]
    duplicate_hashes = {
        content_hash: document_ids
        for content_hash, document_ids in hashes_to_documents.items()
        if len(document_ids) > 1
    }
    stats = CorpusStats(
        document_count=len(entries),
        source_counts=dict(Counter(entry.source_type for entry in entries)),
        chunk_count=len(chunks),
        chunk_length_min=min(chunk_lengths, default=0),
        chunk_length_max=max(chunk_lengths, default=0),
        chunk_length_mean=mean(chunk_lengths) if chunk_lengths else 0.0,
        chunk_length_p50=_nearest_rank_percentile(chunk_lengths, 0.50),
        chunk_length_p95=_nearest_rank_percentile(chunk_lengths, 0.95),
        empty_documents=empty_documents,
        duplicate_content_hashes=duplicate_hashes,
        human_reviewed_documents=sum(entry.human_reviewed for entry in entries),
    )
    return entries, chunks, stats


def validate_corpus_manifest(
    entries: list[CorpusManifestEntry],
    *,
    minimum_documents: int = 30,
) -> list[str]:
    """校验语料规模、来源覆盖和必要的 provenance 字段。"""

    errors: list[str] = []
    if len(entries) < minimum_documents:
        errors.append(
            f"corpus requires at least {minimum_documents} documents, "
            f"got {len(entries)}"
        )

    covered_sources = {entry.source_type for entry in entries}
    missing_sources = sorted(SOURCE_TYPES - covered_sources)
    if missing_sources:
        errors.append(f"corpus misses source types: {', '.join(missing_sources)}")

    required_text_fields = (
        "source_platform",
        "source_url",
        "collected_at",
        "public_status",
        "content_origin",
    )
    for entry in entries:
        for field_name in required_text_fields:
            if not str(getattr(entry, field_name)).strip():
                errors.append(f"{entry.source_path} misses {field_name}")
    return errors


def write_corpus_manifest(
    entries: list[CorpusManifestEntry],
    path: Path,
) -> None:
    """把 corpus manifest 写成一行一条记录的 JSONL。"""

    _write_jsonl((entry.to_dict() for entry in entries), path)


def export_chunks_jsonl(
    chunks: list[Chunk],
    path: Path,
    *,
    dataset_version: str,
) -> None:
    """导出带 dataset version 的统一 Chunk JSONL。"""

    records = (
        {
            "dataset_version": dataset_version,
            "chunk_id": chunk.id,
            "source_type": chunk.source_type,
            "source_path": chunk.source_path,
            "title": chunk.title,
            "text": chunk.text,
            "metadata": chunk.metadata,
        }
        for chunk in chunks
    )
    _write_jsonl(records, path)


def load_chunks_jsonl(path: Path) -> list[Chunk]:
    """读取 export_chunks_jsonl 生成的版本化 Chunks。"""

    chunks: list[Chunk] = []
    for record in _read_jsonl(path):
        chunks.append(
            Chunk(
                id=str(record["chunk_id"]),
                source_type=str(record["source_type"]),
                source_path=str(record["source_path"]),
                title=str(record["title"]),
                text=str(record["text"]),
                metadata=dict(record.get("metadata", {})),
            )
        )
    return chunks


def write_corpus_stats(stats: CorpusStats, path: Path) -> None:
    """把 CorpusStats 写成便于检查的 JSON 文件。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(stats.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _read_front_matter(path: Path) -> dict[str, str]:
    """读取当前项目支持的简单 Markdown front matter。"""

    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0].strip() != "---":
        return {}

    metadata: dict[str, str] = {}
    for line in lines[1:]:
        if line.strip() == "---":
            return metadata
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        metadata[key.strip()] = value.strip()
    return {}


def _parse_bool(value: str) -> bool:
    """把 front matter 中的常见布尔文本转换为 bool。"""

    return value.strip().lower() in {"1", "true", "yes"}


def _nearest_rank_percentile(values: list[int], percentile: float) -> int:
    """使用最近秩方法计算整数百分位。"""

    if not values:
        return 0
    sorted_values = sorted(values)
    rank = max(1, int(len(sorted_values) * percentile + 0.999999))
    return sorted_values[rank - 1]


def _write_jsonl(
    records: Iterable[dict[str, object]],
    path: Path,
) -> None:
    """把可迭代字典记录写成 JSONL。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        for record in records:
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    """读取 JSONL，忽略空行。"""

    records: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number} must be a JSON object")
            records.append(record)
    return records
