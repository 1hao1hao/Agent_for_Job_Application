from __future__ import annotations

import argparse
from datetime import date
import json
from pathlib import Path
import sys
from time import sleep
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from intern_rag.evaluation.corpus_v03 import (  # noqa: E402
    build_quality_report,
    build_v03_chunks,
    deduplicate_corpus,
    normalize_corpus_record,
    validate_v03_corpus,
    write_v03_artifacts,
)
from intern_rag.ingestion import build_document_from_file  # noqa: E402
from intern_rag.ingestion.chunking import SUPPORTED_SUFFIXES  # noqa: E402
from intern_rag.ingestion.public_corpus import (  # noqa: E402
    DatasetSource,
    GithubSource,
    collect_github_documents,
    collect_huggingface_jobs,
)


USER_AGENT = "EvalRAG-corpus-builder/0.3 (+public-research; rate-limited)"


def fetch_json(url: str) -> dict[str, object]:
    """带有限重试读取公开 JSON API，不处理登录或反爬页面。"""

    return json.loads(fetch_text(url))


def fetch_text(url: str) -> str:
    """读取公开文本；429/5xx 最多重试两次并保留最终错误。"""

    for attempt in range(3):
        try:
            request = Request(url, headers={"User-Agent": USER_AGENT})
            with urlopen(request, timeout=45) as response:
                charset = response.headers.get_content_charset() or "utf-8"
                return response.read().decode(charset, errors="replace")
        except HTTPError as error:
            if error.code not in {429, 500, 502, 503, 504} or attempt == 2:
                raise
        except (URLError, TimeoutError):
            if attempt == 2:
                raise
        sleep(1.0 * (attempt + 1))
    raise RuntimeError("unreachable fetch retry state")


def build_corpus(config_path: Path, raw_root: Path) -> dict[str, object]:
    """采集公开来源、合并本地私有资料、去重切分并输出 v0.3 工件。

    输入是固定 revision 的 collection config 与本地 raw 根目录。处理顺序为：
    分页导入公开 JD、读取 GitHub 文档、导入已有脱敏私有文档、统一 schema、
    exact/SimHash 去重、句子边界切分、质量校验。网络失败会进入 collection report；
    最终规模或 provenance 不达标时脚本返回错误，不用复制文本补齐数量。
    """

    config = json.loads(config_path.read_text(encoding="utf-8"))
    collected_at = date.today().isoformat()
    raw_records: list[dict[str, object]] = []
    attempts: list[dict[str, object]] = []

    hf = config["huggingface_jobs"]
    job_records, job_attempts = collect_huggingface_jobs(
        DatasetSource(
            dataset=str(hf["dataset"]),
            config=str(hf["config"]),
            split=str(hf["split"]),
            license_status=str(hf["license_status"]),
            quota=int(hf["quota"]),
            page_size=int(hf["page_size"]),
            maximum_pages=int(hf["maximum_pages"]),
        ),
        fetch_json,
        collected_at=collected_at,
    )
    for record in job_records:
        record["dataset_revision"] = str(hf["revision"])
    raw_records.extend(job_records)
    attempts.extend(job_attempts)

    for raw_source in config["github_sources"]:
        source = GithubSource(
            repository=str(raw_source["repository"]),
            revision=str(raw_source["revision"]),
            license_status=str(raw_source["license_status"]),
            canonical_source_type=str(raw_source["canonical_source_type"]),
            quota=int(raw_source["quota"]),
            include_prefixes=tuple(raw_source.get("include_prefixes", [])),
            exclude_fragments=tuple(raw_source.get("exclude_fragments", [])),
            split_headings=bool(raw_source.get("split_headings", False)),
        )
        records, source_attempts = collect_github_documents(
            source,
            fetch_json,
            fetch_text,
            collected_at=collected_at,
        )
        raw_records.extend(records)
        attempts.extend(source_attempts)

    raw_records.extend(_load_local_records(raw_root, collected_at=collected_at))
    documents = [
        normalize_corpus_record(record, dataset_version=str(config["dataset_version"]))
        for record in raw_records
    ]
    retained, rejected = deduplicate_corpus(
        documents,
        near_duplicate_distance=int(config["near_duplicate_distance"]),
    )
    chunks = build_v03_chunks(retained, max_chars=int(config["chunk_max_chars"]))
    report = build_quality_report(
        documents,
        retained,
        rejected,
        chunks,
        dataset_version=str(config["dataset_version"]),
    )
    errors = validate_v03_corpus(
        retained,
        chunks,
        minimum_documents=int(config["minimum_documents"]),
        minimum_chunks=int(config["minimum_chunks"]),
    )

    write_v03_artifacts(
        retained,
        chunks,
        rejected,
        report,
        documents_path=ROOT / "data/processed/documents/evalrag_v0.3.jsonl",
        manifest_path=ROOT / "data/evaluation/corpus_manifest_v0.3.jsonl",
        chunks_path=ROOT / "data/processed/chunks/evalrag_v0.3.jsonl",
        rejected_path=ROOT / "reports/data_quality/evalrag_v0.3/rejected.jsonl",
        stats_path=ROOT / "data/evaluation/corpus_stats_v0.3.json",
    )
    collection_report = {
        "dataset_version": config["dataset_version"],
        "collected_at": collected_at,
        "config_path": str(config_path),
        "configured_revisions": {
            "huggingface": hf["revision"],
            "github": {
                item["repository"]: item["revision"]
                for item in config["github_sources"]
            },
        },
        "attempt_count": len(attempts),
        "failed_attempts": [item for item in attempts if item["status"] == "failed"],
        "input_record_count": len(raw_records),
        "quality": report.to_dict(),
        "validation_errors": errors,
    }
    report_path = ROOT / "reports/data_quality/evalrag_v0.3/collection_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(collection_report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if errors:
        raise ValueError("; ".join(errors))
    return collection_report


def _load_local_records(raw_root: Path, *, collected_at: str) -> list[dict[str, object]]:
    """把本地资料作为 user_owned/private 记录导入，不公开原始路径和正文。"""

    records: list[dict[str, object]] = []
    for path in sorted(raw_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        document = build_document_from_file(path, raw_root)
        canonical = {
            "jd": "job_posting",
            "interview": "interview_knowledge",
            "project_logs": "project_documentation",
            "resume": "candidate_experience",
            "user_profile": "candidate_profile",
        }[document.metadata.source_type]
        relative = path.relative_to(raw_root).as_posix()
        records.append(
            {
                **document.metadata.to_dict(),
                "source_key": f"local:{relative}",
                "canonical_source_type": canonical,
                "title": document.title,
                "text": document.text,
                "source_url": f"private://{relative}",
                "source_domain": "local-private",
                "source_method": "user_owned",
                "owner_scope": "private",
                "collected_at": collected_at,
                "published_at": document.metadata.first_seen_at,
                "public_status": "private",
                "license_status": "user-owned-private",
                "anonymized": True,
                "review_status": "user_owned",
            }
        )
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description="构建 EvalRAG Corpus v0.3")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/collection/corpus_v0.3.json",
    )
    parser.add_argument("--raw-root", type=Path, default=ROOT / "data/raw")
    args = parser.parse_args()
    try:
        report = build_corpus(args.config, args.raw_root)
    except ValueError as error:
        print(json.dumps({"status": "failed", "error": str(error)}, ensure_ascii=False))
        return 1
    print(
        json.dumps(
            {"status": "succeeded", **report["quality"]}, ensure_ascii=False
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
