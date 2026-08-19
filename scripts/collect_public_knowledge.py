from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from hashlib import sha256
from pathlib import Path
from time import sleep
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen
from urllib.robotparser import RobotFileParser


ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from intern_rag.ingestion.web_importer import (  # noqa: E402
    PublicSourceSpec,
    collect_public_source,
    extract_text_segment,
    html_to_visible_text,
)


USER_AGENT = "EvalRAG-public-data-research/0.1"


def fetch_public_url(url: str, *, timeout: float = 20.0) -> str:
    """在 robots 允许时下载公开页面，并按响应 charset 解码。"""

    if not robots_allows(url, timeout=timeout):
        raise PermissionError("robots.txt does not allow this URL")
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=timeout) as response:
        charset = response.headers.get_content_charset() or "utf-8"
        return response.read().decode(charset, errors="replace")


def robots_allows(url: str, *, timeout: float = 20.0) -> bool:
    """读取站点 robots.txt；不存在时把公开页面视为允许访问。"""

    parts = urlsplit(url)
    robots_url = f"{parts.scheme}://{parts.netloc}/robots.txt"
    request = Request(robots_url, headers={"User-Agent": USER_AGENT})
    try:
        with urlopen(request, timeout=timeout) as response:
            content = response.read().decode("utf-8", errors="replace")
    except HTTPError as error:
        if error.code == 404:
            return True
        raise

    parser = RobotFileParser()
    parser.set_url(robots_url)
    parser.parse(content.splitlines())
    return parser.can_fetch(USER_AGENT, url)


def collect_sources(config_path: Path, output_dir: Path) -> dict[str, Any]:
    """采集配置中的公开来源，输出候选数据和可复查的成功/失败报告。"""

    config = json.loads(config_path.read_text(encoding="utf-8"))
    collected_at = date.today().isoformat()
    jobs: list[dict[str, Any]] = []
    interviews: list[dict[str, Any]] = []
    projects: list[dict[str, Any]] = []
    attempts: list[dict[str, Any]] = []

    for raw_spec in config["sources"]:
        spec = PublicSourceSpec(
            source_id=raw_spec["source_id"],
            source_type=raw_spec["source_type"],
            url=raw_spec["url"],
            title=raw_spec["title"],
            content_type=raw_spec.get("content_type", "html"),
            start_marker=raw_spec.get("start_marker", ""),
            end_marker=raw_spec.get("end_marker", ""),
            max_chars=int(raw_spec.get("max_chars", 20_000)),
        )
        attempt = {
            "source_id": spec.source_id,
            "source_type": spec.source_type,
            "source_url": spec.url,
        }
        try:
            raw_content = fetch_public_url(spec.url)
            collected = collect_public_source(spec, raw_content)
            if spec.source_type == "jd":
                visible_text = html_to_visible_text(raw_content)
                jobs.append(_build_job_record(raw_spec, visible_text, collected_at))
            else:
                record = {
                    "source_id": collected.source_id,
                    "source_type": collected.source_type,
                    "title": collected.title,
                    "source_platform": raw_spec.get("source_platform", "public_web"),
                    "source_url": collected.source_url,
                    "collected_at": collected_at,
                    "public_status": "public_web",
                    "anonymized": True,
                    "human_reviewed": False,
                    "candidate_only": True,
                    "content_hash": sha256(collected.text.encode("utf-8")).hexdigest(),
                    "text": collected.text,
                }
                target = interviews if spec.source_type == "interview" else projects
                target.append(record)
            attempt.update(status="collected", char_count=len(collected.text))
        except (HTTPError, URLError, PermissionError, TimeoutError, ValueError) as error:
            attempt.update(
                status="failed",
                error_type=type(error).__name__,
                error_message=str(error),
            )
        attempts.append(attempt)
        sleep(0.5)

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "jobs.json", {"jobs": jobs})
    _write_jsonl(output_dir / "interview_sources.jsonl", interviews)
    _write_jsonl(output_dir / "project_sources.jsonl", projects)
    report = {
        "collection_version": config["version"],
        "collected_at": collected_at,
        "candidate_only": True,
        "counts": {
            "configured": len(config["sources"]),
            "collected": sum(item["status"] == "collected" for item in attempts),
            "failed": sum(item["status"] == "failed" for item in attempts),
            "jd": len(jobs),
            "interview": len(interviews),
            "project_logs": len(projects),
        },
        "private_sources": {
            "resume": "不从互联网猜测；继续使用项目作者明确提供并脱敏的本地简历。",
            "user_profile": "不从互联网推断；继续使用项目作者明确提供的偏好与约束。",
            "personal_project_experience": "仅采集项目作者公开 GitHub README，不推断未公开经历。",
        },
        "attempts": attempts,
    }
    _write_json(output_dir / "collection_report.json", report)
    return report


def _build_job_record(
    raw_spec: dict[str, Any], visible_text: str, collected_at: str
) -> dict[str, Any]:
    description = extract_text_segment(
        visible_text,
        start_marker=raw_spec["description_start"],
        end_marker=raw_spec["description_end"],
    )
    requirements = extract_text_segment(
        visible_text,
        start_marker=raw_spec["requirements_start"],
        end_marker=raw_spec["requirements_end"],
    )
    content = f"{description}\n{requirements}"
    status = "expired" if "已过期" in visible_text else "unknown"
    return {
        "job_id": raw_spec["job_id"],
        "company": raw_spec["company"],
        "job_title": raw_spec["title"],
        "city": raw_spec["city"],
        "source_platform": "official_careers",
        "source_url": raw_spec["url"],
        "description": description,
        "requirements": requirements,
        "first_seen_at": collected_at,
        "last_seen_at": collected_at,
        "status": status,
        "version": 1,
        "content_hash": sha256(content.encode("utf-8")).hexdigest(),
        "published_at": raw_spec.get("published_at", ""),
        "human_reviewed": False,
        "candidate_only": True,
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="采集公开候选知识源")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/collection/public_sources_v0.1.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data/import_candidates/public_web_v0.1",
    )
    args = parser.parse_args()
    report = collect_sources(args.config, args.output_dir)
    print(json.dumps(report["counts"], ensure_ascii=False))


if __name__ == "__main__":
    main()
