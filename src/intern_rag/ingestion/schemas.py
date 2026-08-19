from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, cast


JobStatus = Literal["active", "expired", "unknown"]
MetadataValue = str | int | bool


DEFAULT_METADATA: dict[str, MetadataValue] = {
    "source_type": "unknown",
    "job_id": "",
    "company": "unknown",
    "job_title": "unknown",
    "city": "unknown",
    "source_platform": "unknown",
    "source_url": "",
    "first_seen_at": "",
    "last_seen_at": "",
    "status": "unknown",
    "version": 1,
    "source_priority": 0,
    "content_hash": "",
    "document_id": "",
    "canonical_source_type": "unknown",
    "owner_scope": "public",
    "source_method": "unknown",
    "source_domain": "",
    "collected_at": "",
    "published_at": "",
    "public_status": "unknown",
    "license_status": "unknown",
    "anonymized": False,
    "review_status": "unreviewed",
    "near_duplicate_group": "",
    "schema_version": "v1",
}


@dataclass(frozen=True)
class DocumentMetadata:
    """Document 和 Chunk 共享的元数据。

    这里保存的是“这份资料从哪里来、对应什么岗位、现在是否还有效”
    这类信息。Chunk 从 Document 切出来后会继承这些字段，方便后续检索
    阶段判断岗位是否过期、哪个来源更可信。
    """

    source_type: str = "unknown"
    job_id: str = ""
    company: str = "unknown"
    job_title: str = "unknown"
    city: str = "unknown"
    source_platform: str = "unknown"
    source_url: str = ""
    first_seen_at: str = ""
    last_seen_at: str = ""
    status: JobStatus = "unknown"
    version: int = 1
    source_priority: int = 0
    content_hash: str = ""
    document_id: str = ""
    canonical_source_type: str = "unknown"
    owner_scope: str = "public"
    source_method: str = "unknown"
    source_domain: str = ""
    collected_at: str = ""
    published_at: str = ""
    public_status: str = "unknown"
    license_status: str = "unknown"
    anonymized: bool = False
    review_status: str = "unreviewed"
    near_duplicate_group: str = ""
    schema_version: str = "v1"

    def to_dict(self) -> dict[str, MetadataValue]:
        """转换成普通 dict，便于直接放进 Chunk.metadata。"""

        return {
            "source_type": self.source_type,
            "job_id": self.job_id,
            "company": self.company,
            "job_title": self.job_title,
            "city": self.city,
            "source_platform": self.source_platform,
            "source_url": self.source_url,
            "first_seen_at": self.first_seen_at,
            "last_seen_at": self.last_seen_at,
            "status": self.status,
            "version": self.version,
            "source_priority": self.source_priority,
            "content_hash": self.content_hash,
            "document_id": self.document_id,
            "canonical_source_type": self.canonical_source_type,
            "owner_scope": self.owner_scope,
            "source_method": self.source_method,
            "source_domain": self.source_domain,
            "collected_at": self.collected_at,
            "published_at": self.published_at,
            "public_status": self.public_status,
            "license_status": self.license_status,
            "anonymized": self.anonymized,
            "review_status": self.review_status,
            "near_duplicate_group": self.near_duplicate_group,
            "schema_version": self.schema_version,
        }


@dataclass(frozen=True)
class Document:
    """切分前的一整份资料。

    例如一份岗位 JD、一份简历或一篇面经。Document 负责保留完整文本和
    文档级 metadata；chunking 模块再把它切成多个 Chunk。
    """

    source_path: str
    title: str
    text: str
    metadata: DocumentMetadata


@dataclass(frozen=True)
class Chunk:
    """检索系统实际使用的最小文本证据单元。

    RAG 通常不会把一整份长文档直接丢给检索或生成模型，而是先切成
    Chunk。每个 Chunk 都带有原文档的 metadata，保证引用和时效性判断
    不会丢失上下文。
    """

    id: str
    source_type: str
    source_path: str
    title: str
    text: str
    metadata: dict[str, MetadataValue] = field(default_factory=dict)


def build_document_metadata(
    raw_metadata: dict[str, str], source_type: str = "unknown"
) -> DocumentMetadata:
    """把原始 metadata 规范化为 DocumentMetadata。

    缺失字段会使用默认值，避免旧数据因为没写岗位时效性字段而加载失败。
    非法 status 会被归一化为 unknown。
    """

    values = DEFAULT_METADATA | raw_metadata | {"source_type": source_type}
    status = str(values["status"])
    if status not in {"active", "expired", "unknown"}:
        status = "unknown"

    return DocumentMetadata(
        source_type=str(values["source_type"]),
        job_id=str(values["job_id"]),
        company=str(values["company"]),
        job_title=str(values["job_title"]),
        city=str(values["city"]),
        source_platform=str(values["source_platform"]),
        source_url=str(values["source_url"]),
        first_seen_at=str(values["first_seen_at"]),
        last_seen_at=str(values["last_seen_at"]),
        status=cast(JobStatus, status),
        version=_to_int(values["version"], default=1),
        source_priority=_to_int(values["source_priority"], default=0),
        content_hash=str(values["content_hash"]),
        document_id=str(values["document_id"]),
        canonical_source_type=str(values["canonical_source_type"]),
        owner_scope=str(values["owner_scope"]),
        source_method=str(values["source_method"]),
        source_domain=str(values["source_domain"]),
        collected_at=str(values["collected_at"]),
        published_at=str(values["published_at"]),
        public_status=str(values["public_status"]),
        license_status=str(values["license_status"]),
        anonymized=_to_bool(values["anonymized"]),
        review_status=str(values["review_status"]),
        near_duplicate_group=str(values["near_duplicate_group"]),
        schema_version=str(values["schema_version"]),
    )


def _to_int(value: MetadataValue, default: int) -> int:
    """把 metadata 字段转成 int；失败时使用默认值。"""

    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _to_bool(value: MetadataValue) -> bool:
    """把 metadata 常见布尔表示归一化为 bool。"""

    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}
