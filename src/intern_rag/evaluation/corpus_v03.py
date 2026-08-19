from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from hashlib import sha1, sha256
import json
import math
from pathlib import Path
import re
from statistics import mean
from typing import Iterable
from urllib.parse import urlsplit

from intern_rag.ingestion import Chunk
from intern_rag.ingestion.chunking import split_text


LEGACY_SOURCE_MAP = {
    "job_posting": "jd",
    "interview_knowledge": "interview",
    "project_documentation": "project_logs",
    "candidate_experience": "resume",
    "candidate_profile": "user_profile",
}
CANONICAL_SOURCE_MAP = {value: key for key, value in LEGACY_SOURCE_MAP.items()}


@dataclass(frozen=True)
class CorpusDocumentV03:
    """v0.3 统一文档，正文与来源审计字段一起参与版本化。"""

    document_id: str
    source_type: str
    canonical_source_type: str
    title: str
    text: str
    source_url: str
    source_domain: str
    source_method: str
    owner_scope: str
    collected_at: str
    published_at: str
    public_status: str
    license_status: str
    anonymized: bool
    review_status: str
    content_hash: str
    metadata: dict[str, str | int | bool] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        """转换为 JSONL 可写入的稳定字典。"""

        return asdict(self)


@dataclass(frozen=True)
class RejectedDocument:
    """被质量门禁删除的文档及可审计原因。"""

    document_id: str
    reason: str
    duplicate_of: str = ""
    hamming_distance: int | None = None


@dataclass(frozen=True)
class CorpusQualityReport:
    """v0.3 去重前后规模、来源分布和文本质量统计。"""

    dataset_version: str
    input_documents: int
    retained_documents: int
    rejected_documents: int
    exact_duplicates: int
    near_duplicates: int
    empty_documents: int
    source_counts: dict[str, int]
    canonical_source_counts: dict[str, int]
    method_counts: dict[str, int]
    owner_scope_counts: dict[str, int]
    domain_counts: dict[str, int]
    review_counts: dict[str, int]
    document_length_min: int
    document_length_mean: float
    document_length_p50: int
    document_length_p95: int
    document_length_max: int
    template_line_ratio: float
    chunk_count: int = 0
    chunk_length_p50: int = 0
    chunk_length_p95: int = 0

    def to_dict(self) -> dict[str, object]:
        """转换为报告字典。"""

        return asdict(self)


def normalize_corpus_record(
    raw: dict[str, object], *, dataset_version: str = "evalrag_v0.3"
) -> CorpusDocumentV03:
    """把采集器或本地导入器记录归一化成 v0.3 Document。

    输入记录至少需要 source type、title、text 和 source URL。函数统一旧/新
    source type、规范正文空白，并根据来源与正文生成稳定 document_id/content_hash。
    provenance 缺失时保留 unknown，而不是伪造公开、许可或人工审核状态。
    """

    canonical = str(raw.get("canonical_source_type", "")).strip()
    legacy = str(raw.get("source_type", "")).strip()
    if canonical in LEGACY_SOURCE_MAP:
        legacy = LEGACY_SOURCE_MAP[canonical]
    elif legacy in CANONICAL_SOURCE_MAP:
        canonical = CANONICAL_SOURCE_MAP[legacy]
    else:
        raise ValueError(f"unsupported source type: {canonical or legacy}")

    text = _normalize_text(str(raw.get("text", "")))
    title = " ".join(str(raw.get("title", "untitled")).split())
    source_url = str(raw.get("source_url", "")).strip()
    source_method = str(raw.get("source_method", "unknown")).strip() or "unknown"
    source_key = str(raw.get("source_key", source_url or title)).strip()
    raw_id = f"{dataset_version}|{canonical}|{source_method}|{source_key}"
    document_id = str(raw.get("document_id", "")).strip() or (
        f"doc-{sha256(raw_id.encode('utf-8')).hexdigest()[:16]}"
    )
    content_hash = sha256(text.encode("utf-8")).hexdigest()
    reserved = {
        "document_id", "source_type", "canonical_source_type", "title", "text",
        "source_url", "source_domain", "source_method", "owner_scope",
        "collected_at", "published_at", "public_status", "license_status",
        "anonymized", "review_status", "content_hash", "source_key",
    }
    metadata = {
        str(key): value
        for key, value in raw.items()
        if key not in reserved and isinstance(value, (str, int, bool))
    }
    return CorpusDocumentV03(
        document_id=document_id,
        source_type=legacy,
        canonical_source_type=canonical,
        title=title,
        text=text,
        source_url=source_url,
        source_domain=str(raw.get("source_domain", "")).strip()
        or urlsplit(source_url).netloc,
        source_method=source_method,
        owner_scope=str(raw.get("owner_scope", "public")),
        collected_at=str(raw.get("collected_at", "")),
        published_at=str(raw.get("published_at", "")),
        public_status=str(raw.get("public_status", "unknown")),
        license_status=str(raw.get("license_status", "unknown")),
        anonymized=_to_bool(raw.get("anonymized", False)),
        review_status=str(raw.get("review_status", "unreviewed")),
        content_hash=content_hash,
        metadata=metadata,
    )


def deduplicate_corpus(
    documents: list[CorpusDocumentV03],
    *,
    near_duplicate_distance: int = 3,
    minimum_chars: int = 80,
) -> tuple[list[CorpusDocumentV03], list[RejectedDocument]]:
    """按 exact hash 和 SimHash 删除空/重复文本并保留删除原因。

    先删除过短或完全相同的正文，再仅在同一 canonical source 内比较 SimHash。
    近重复判断同时要求长度比不低于 0.85，避免把共享少量技术术语的不同文档误删。
    输出顺序按 document_id 稳定，保证相同输入得到相同版本工件。
    """

    if near_duplicate_distance < 0:
        raise ValueError("near_duplicate_distance must not be negative")
    retained: list[CorpusDocumentV03] = []
    rejected: list[RejectedDocument] = []
    exact_owner: dict[str, str] = {}
    for document in documents:
        current = exact_owner.get(document.content_hash)
        if current is None or document.document_id < current:
            exact_owner[document.content_hash] = document.document_id
    simhash_buckets: dict[str, list[tuple[int, int, str]]] = defaultdict(list)

    for document in sorted(documents, key=lambda item: item.document_id):
        if len(document.text) < minimum_chars:
            rejected.append(RejectedDocument(document.document_id, "too_short"))
            continue
        duplicate_of = exact_owner[document.content_hash]
        if duplicate_of != document.document_id:
            rejected.append(
                RejectedDocument(document.document_id, "exact_duplicate", duplicate_of)
            )
            continue

        fingerprint = simhash64(document.text)
        length = len(document.text)
        near_match: tuple[str, int] | None = None
        for known_hash, known_length, known_id in simhash_buckets[
            document.canonical_source_type
        ]:
            length_ratio = min(length, known_length) / max(length, known_length)
            if length_ratio < 0.85:
                continue
            distance = (fingerprint ^ known_hash).bit_count()
            if distance <= near_duplicate_distance:
                near_match = (known_id, distance)
                break
        if near_match:
            rejected.append(
                RejectedDocument(
                    document.document_id,
                    "near_duplicate",
                    near_match[0],
                    near_match[1],
                )
            )
            continue

        retained.append(document)
        simhash_buckets[document.canonical_source_type].append(
            (fingerprint, length, document.document_id)
        )
    return retained, rejected


def build_v03_chunks(
    documents: list[CorpusDocumentV03],
    *,
    max_chars: int = 420,
) -> list[Chunk]:
    """把 v0.3 Document 切成稳定 Chunk，并完整继承 provenance。"""

    chunks: list[Chunk] = []
    for document in sorted(documents, key=lambda item: item.document_id):
        parts = split_text(
            document.text,
            max_chars=max_chars,
            strategy="sentence_boundary",
            min_chunk_ratio=0.5,
        )
        for index, text in enumerate(parts):
            digest = sha1(
                f"{document.document_id}|{index}|{text}".encode("utf-8")
            ).hexdigest()[:12]
            metadata: dict[str, str | int | bool] = {
                **document.metadata,
                "document_id": document.document_id,
                "canonical_source_type": document.canonical_source_type,
                "source_url": document.source_url,
                "source_domain": document.source_domain,
                "source_method": document.source_method,
                "owner_scope": document.owner_scope,
                "collected_at": document.collected_at,
                "published_at": document.published_at,
                "public_status": document.public_status,
                "license_status": document.license_status,
                "anonymized": document.anonymized,
                "review_status": document.review_status,
                "content_hash": document.content_hash,
                "schema_version": "v0.3",
                "chunk_index": index,
                "char_count": len(text),
            }
            chunks.append(
                Chunk(
                    id=f"{document.source_type}-{document.document_id}-{index}-{digest}",
                    source_type=document.source_type,
                    source_path=document.source_url or document.document_id,
                    title=document.title,
                    text=text,
                    metadata=metadata,
                )
            )
    return chunks


def build_quality_report(
    original: list[CorpusDocumentV03],
    retained: list[CorpusDocumentV03],
    rejected: list[RejectedDocument],
    chunks: list[Chunk],
    *,
    dataset_version: str,
) -> CorpusQualityReport:
    """汇总去重、来源、长度、模板行和 Chunk 统计。"""

    lengths = [len(item.text) for item in retained]
    chunk_lengths = [len(item.text) for item in chunks]
    line_counts: Counter[str] = Counter()
    total_lines = 0
    for document in retained:
        normalized_lines = {
            " ".join(line.split()).lower()
            for line in document.text.splitlines()
            if len(" ".join(line.split())) >= 12
        }
        line_counts.update(normalized_lines)
        total_lines += len(normalized_lines)
    repeated_lines = sum(count for count in line_counts.values() if count >= 5)
    return CorpusQualityReport(
        dataset_version=dataset_version,
        input_documents=len(original),
        retained_documents=len(retained),
        rejected_documents=len(rejected),
        exact_duplicates=sum(item.reason == "exact_duplicate" for item in rejected),
        near_duplicates=sum(item.reason == "near_duplicate" for item in rejected),
        empty_documents=sum(item.reason == "too_short" for item in rejected),
        source_counts=dict(Counter(item.source_type for item in retained)),
        canonical_source_counts=dict(
            Counter(item.canonical_source_type for item in retained)
        ),
        method_counts=dict(Counter(item.source_method for item in retained)),
        owner_scope_counts=dict(Counter(item.owner_scope for item in retained)),
        domain_counts=dict(Counter(item.source_domain for item in retained)),
        review_counts=dict(Counter(item.review_status for item in retained)),
        document_length_min=min(lengths, default=0),
        document_length_mean=mean(lengths) if lengths else 0.0,
        document_length_p50=_percentile(lengths, 0.50),
        document_length_p95=_percentile(lengths, 0.95),
        document_length_max=max(lengths, default=0),
        template_line_ratio=repeated_lines / total_lines if total_lines else 0.0,
        chunk_count=len(chunks),
        chunk_length_p50=_percentile(chunk_lengths, 0.50),
        chunk_length_p95=_percentile(chunk_lengths, 0.95),
    )


def validate_v03_corpus(
    documents: list[CorpusDocumentV03],
    chunks: list[Chunk],
    *,
    minimum_documents: int = 500,
    minimum_chunks: int = 1500,
) -> list[str]:
    """校验规模、五类来源覆盖、唯一 ID、provenance 和 Chunk 引用。"""

    errors: list[str] = []
    if len(documents) < minimum_documents:
        errors.append(f"requires {minimum_documents} documents, got {len(documents)}")
    if len(chunks) < minimum_chunks:
        errors.append(f"requires {minimum_chunks} chunks, got {len(chunks)}")
    covered = {item.canonical_source_type for item in documents}
    missing = sorted(set(LEGACY_SOURCE_MAP) - covered)
    if missing:
        errors.append(f"missing canonical sources: {', '.join(missing)}")
    if len({item.document_id for item in documents}) != len(documents):
        errors.append("duplicate document ids")
    if len({item.content_hash for item in documents}) != len(documents):
        errors.append("duplicate content hashes remain")
    if len({item.id for item in chunks}) != len(chunks):
        errors.append("duplicate chunk ids")
    document_ids = {item.document_id for item in documents}
    if any(str(chunk.metadata.get("document_id", "")) not in document_ids for chunk in chunks):
        errors.append("chunk references missing document")
    for item in documents:
        if item.owner_scope == "public" and (
            not item.source_url or item.license_status == "unknown"
        ):
            errors.append(f"{item.document_id} misses public provenance")
    return errors


def simhash64(text: str) -> int:
    """用字符三元组和英文词构造 64 位 SimHash。"""

    tokens = _quality_tokens(text)
    if not tokens:
        return 0
    weights = [0] * 64
    for token in tokens:
        digest = int.from_bytes(sha256(token.encode("utf-8")).digest()[:8], "big")
        for bit in range(64):
            weights[bit] += 1 if digest & (1 << bit) else -1
    fingerprint = 0
    for bit, weight in enumerate(weights):
        if weight >= 0:
            fingerprint |= 1 << bit
    return fingerprint


def write_v03_artifacts(
    documents: list[CorpusDocumentV03],
    chunks: list[Chunk],
    rejected: list[RejectedDocument],
    report: CorpusQualityReport,
    *,
    documents_path: Path,
    manifest_path: Path,
    chunks_path: Path,
    rejected_path: Path,
    stats_path: Path,
) -> None:
    """一次写出 documents、manifest、chunks、删除清单与质量报告。"""

    _write_jsonl((item.to_dict() for item in documents), documents_path)
    _write_jsonl(
        (
            {
                key: value
                for key, value in item.to_dict().items()
                if key not in {"text", "metadata"}
            }
            for item in documents
        ),
        manifest_path,
    )
    _write_jsonl(
        (
            {
                "dataset_version": report.dataset_version,
                "chunk_id": item.id,
                "source_type": item.source_type,
                "source_path": item.source_path,
                "title": item.title,
                "text": item.text,
                "metadata": item.metadata,
            }
            for item in chunks
        ),
        chunks_path,
    )
    _write_jsonl((asdict(item) for item in rejected), rejected_path)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _normalize_text(text: str) -> str:
    lines = [" ".join(line.split()) for line in text.replace("\x00", "").splitlines()]
    compact: list[str] = []
    for line in lines:
        if not line and (not compact or not compact[-1]):
            continue
        compact.append(line)
    return "\n".join(compact).strip()


def _quality_tokens(text: str) -> list[str]:
    normalized = re.sub(r"\s+", "", text.lower())
    chinese = [
        normalized[index : index + 3]
        for index in range(max(0, len(normalized) - 2))
        if all("\u4e00" <= char <= "\u9fff" for char in normalized[index : index + 3])
    ]
    words = re.findall(r"[a-z][a-z0-9_+#.-]{1,}", normalized)
    return list(dict.fromkeys([*chinese, *words]))


def _percentile(values: list[int], percentile: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = max(0, math.ceil(len(ordered) * percentile) - 1)
    return ordered[index]


def _to_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def _write_jsonl(records: Iterable[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
