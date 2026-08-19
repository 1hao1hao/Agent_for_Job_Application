from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Callable
from urllib.parse import quote


JsonFetcher = Callable[[str], dict[str, object]]
TextFetcher = Callable[[str], str]


@dataclass(frozen=True)
class DatasetSource:
    """公开数据集分页导入配置。"""

    dataset: str
    config: str
    split: str
    license_status: str
    quota: int
    page_size: int = 100
    maximum_pages: int = 80


@dataclass(frozen=True)
class GithubSource:
    """GitHub 仓库文档导入配置。"""

    repository: str
    revision: str
    license_status: str
    canonical_source_type: str
    quota: int
    include_prefixes: tuple[str, ...] = ()
    exclude_fragments: tuple[str, ...] = ()
    split_headings: bool = False


def collect_huggingface_jobs(
    source: DatasetSource,
    fetch_json: JsonFetcher,
    *,
    collected_at: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """分页读取公开招聘描述，输出规范记录和逐页 attempt。

    岗位按稳定的 offset 跨区间采样，既保留 AI/后端相关职位，也保留其他职位作为
    hard negatives。解析失败、过短文本和重复正文由后续 corpus quality gate 统一处理。
    """

    records: list[dict[str, object]] = []
    attempts: list[dict[str, object]] = []
    seen_source_keys: set[str] = set()
    for page in range(source.maximum_pages):
        if len(records) >= source.quota:
            break
        offset = page * source.page_size
        url = (
            "https://datasets-server.huggingface.co/rows?"
            f"dataset={quote(source.dataset)}&config={quote(source.config)}&"
            f"split={quote(source.split)}&offset={offset}&length={source.page_size}"
        )
        try:
            payload = fetch_json(url)
            rows = payload.get("rows", [])
            if not isinstance(rows, list):
                raise ValueError("dataset rows must be a list")
            accepted = 0
            for wrapper in rows:
                if not isinstance(wrapper, dict) or not isinstance(wrapper.get("row"), dict):
                    continue
                parsed = parse_job_dataset_row(
                    wrapper["row"],
                    dataset=source.dataset,
                    row_index=int(wrapper.get("row_idx", offset + accepted)),
                    collected_at=collected_at,
                    license_status=source.license_status,
                )
                if parsed is None or parsed["source_key"] in seen_source_keys:
                    continue
                seen_source_keys.add(str(parsed["source_key"]))
                records.append(parsed)
                accepted += 1
                if len(records) >= source.quota:
                    break
            attempts.append(
                {"url": url, "status": "collected", "rows": len(rows), "accepted": accepted}
            )
            if not rows:
                break
        except (OSError, TimeoutError, ValueError, json.JSONDecodeError) as error:
            attempts.append(
                {
                    "url": url,
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
    return records, attempts


def parse_job_dataset_row(
    row: dict[str, object],
    *,
    dataset: str,
    row_index: int,
    collected_at: str,
    license_status: str,
) -> dict[str, object] | None:
    """解析招聘数据集中的 XML-like 岗位文本，不推断不存在的公司和城市。"""

    raw = str(row.get("user_short") or row.get("user") or "").strip()
    title = _extract_tag(raw, "岗位名称")
    description = _extract_tag(raw, "岗位描述")
    education = _extract_tag(raw, "学历描述") or str(row.get("assistant", ""))
    if not title or len(description) < 60:
        return None
    text = f"# {title}\n\n## 岗位描述\n{description}"
    if education:
        text += f"\n\n## 学历要求\n{education}"
    source_url = f"https://huggingface.co/datasets/{dataset}"
    source_key = f"hf:{dataset}:{row_index}"
    return {
        "source_key": source_key,
        "canonical_source_type": "job_posting",
        "title": title,
        "text": text,
        "source_url": source_url,
        "source_domain": "huggingface.co",
        "source_method": "dataset_import",
        "owner_scope": "public",
        "collected_at": collected_at,
        "published_at": "",
        "public_status": "public_dataset",
        "license_status": license_status,
        "anonymized": True,
        "review_status": "ai_assisted",
        "job_id": source_key,
        "job_title": title,
        "company": "unknown",
        "city": "unknown",
        "status": "unknown",
        "education_requirement": education,
    }


def collect_github_documents(
    source: GithubSource,
    fetch_json: JsonFetcher,
    fetch_text: TextFetcher,
    *,
    collected_at: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """从固定 revision 的 GitHub tree 选择 Markdown，并保存逐文件 provenance。"""

    tree_url = (
        f"https://api.github.com/repos/{source.repository}/git/trees/"
        f"{source.revision}?recursive=1"
    )
    payload = fetch_json(tree_url)
    tree = payload.get("tree", [])
    if not isinstance(tree, list):
        raise ValueError("github tree must be a list")
    paths = sorted(
        str(item["path"])
        for item in tree
        if isinstance(item, dict)
        and item.get("type") == "blob"
        and str(item.get("path", "")).lower().endswith((".md", ".mdx"))
        and _path_allowed(str(item.get("path", "")), source)
    )
    records: list[dict[str, object]] = []
    attempts: list[dict[str, object]] = []
    for path in paths:
        if len(records) >= source.quota:
            break
        raw_url = (
            f"https://raw.githubusercontent.com/{source.repository}/"
            f"{source.revision}/{path}"
        )
        try:
            text = fetch_text(raw_url)
            sections = (
                split_markdown_sections(text, minimum_chars=180)
                if source.split_headings
                else [(markdown_title(text, path), text)]
            )
            accepted = 0
            for section_index, (title, section) in enumerate(sections):
                if len(records) >= source.quota:
                    break
                cleaned = _strip_markdown_noise(section)
                if len(cleaned) < 180:
                    continue
                source_key = f"github:{source.repository}:{source.revision}:{path}:{section_index}"
                records.append(
                    {
                        "source_key": source_key,
                        "canonical_source_type": source.canonical_source_type,
                        "title": title,
                        "text": cleaned,
                        "source_url": (
                            f"https://github.com/{source.repository}/blob/"
                            f"{source.revision}/{path}"
                        ),
                        "source_domain": "github.com",
                        "source_method": "repository_import",
                        "owner_scope": "public",
                        "collected_at": collected_at,
                        "published_at": "",
                        "public_status": "public_repository",
                        "license_status": source.license_status,
                        "anonymized": True,
                        "review_status": "ai_assisted",
                        "repository": source.repository,
                        "revision": source.revision,
                        "repository_path": path,
                    }
                )
                accepted += 1
            attempts.append(
                {"url": raw_url, "status": "collected", "accepted": accepted}
            )
        except (OSError, TimeoutError, UnicodeError, ValueError) as error:
            attempts.append(
                {
                    "url": raw_url,
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
    return records, attempts


def split_markdown_sections(
    text: str, *, minimum_chars: int = 180
) -> list[tuple[str, str]]:
    """按二/三级标题切出完整知识条目，过短相邻段不会被计作文档。"""

    heading_pattern = re.compile(r"^(#{2,3})\s+(.+?)\s*$", re.MULTILINE)
    matches = list(heading_pattern.finditer(text))
    sections: list[tuple[str, str]] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = text[match.start():end].strip()
        if len(body) >= minimum_chars:
            sections.append((" ".join(match.group(2).split()), body))
    if not sections and len(text.strip()) >= minimum_chars:
        return [("文档", text.strip())]
    return sections


def markdown_title(text: str, path: str) -> str:
    """优先读取 Markdown 一级标题，缺失时使用文件名。"""

    match = re.search(r"^#\s+(.+?)\s*$", text, flags=re.MULTILINE)
    return " ".join(match.group(1).split()) if match else path.rsplit("/", 1)[-1]


def _path_allowed(path: str, source: GithubSource) -> bool:
    if source.include_prefixes and not any(
        path.startswith(prefix) for prefix in source.include_prefixes
    ):
        return False
    return not any(fragment in path for fragment in source.exclude_fragments)


def _extract_tag(text: str, tag: str) -> str:
    match = re.search(fr"<{tag}>(.*?)</{tag}>", text, flags=re.DOTALL)
    return " ".join(match.group(1).split()) if match else ""


def _strip_markdown_noise(text: str) -> str:
    text = re.sub(r"<img\b[^>]*>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"!\[[^]]*]\([^)]*\)", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
