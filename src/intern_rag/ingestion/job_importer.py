from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from intern_rag.ingestion.schemas import Document, build_document_metadata


JobStatus = Literal["active", "expired", "unknown"]
REQUIRED_JOB_FIELDS = {
    "job_id",
    "company",
    "job_title",
    "city",
    "source_platform",
    "source_url",
    "description",
    "requirements",
    "first_seen_at",
    "last_seen_at",
    "status",
    "version",
    "content_hash",
}


@dataclass(frozen=True)
class JobPosting:
    """A normalized job posting imported from manual JSON or CSV data."""

    job_id: str
    company: str
    job_title: str
    city: str
    source_platform: str
    source_url: str
    description: str
    requirements: str
    first_seen_at: str
    last_seen_at: str
    status: JobStatus
    version: int
    content_hash: str


def load_job_postings(path: Path) -> list[JobPosting]:
    """Load job postings from a JSON or CSV file.

    JSON files must contain either a list of posting objects or an object with a
    `jobs` list. CSV files must contain one posting per row with the required
    column names.
    """

    suffix = path.suffix.lower()
    if suffix == ".json":
        return load_job_postings_from_json(path)
    if suffix == ".csv":
        return load_job_postings_from_csv(path)
    raise ValueError(f"Unsupported job import file type: {path.suffix}")


def load_job_postings_from_json(path: Path) -> list[JobPosting]:
    """Load job postings from a JSON file."""

    raw_data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw_data, dict):
        raw_jobs = raw_data.get("jobs")
    else:
        raw_jobs = raw_data

    if not isinstance(raw_jobs, list):
        raise ValueError("JSON job import must be a list or contain a jobs list")

    return [job_posting_from_dict(raw_job) for raw_job in raw_jobs]


def load_job_postings_from_csv(path: Path) -> list[JobPosting]:
    """Load job postings from a CSV file with required job columns."""

    with path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        return [job_posting_from_dict(dict(row)) for row in reader]


def job_posting_from_dict(raw_job: dict[str, Any]) -> JobPosting:
    """Validate and normalize one raw job posting dictionary."""

    missing_fields = sorted(field for field in REQUIRED_JOB_FIELDS if field not in raw_job)
    if missing_fields:
        raise ValueError(f"Missing job posting fields: {', '.join(missing_fields)}")

    status = str(raw_job["status"])
    if status not in {"active", "expired", "unknown"}:
        status = "unknown"

    return JobPosting(
        job_id=str(raw_job["job_id"]),
        company=str(raw_job["company"]),
        job_title=str(raw_job["job_title"]),
        city=str(raw_job["city"]),
        source_platform=str(raw_job["source_platform"]),
        source_url=str(raw_job["source_url"]),
        description=str(raw_job["description"]),
        requirements=str(raw_job["requirements"]),
        first_seen_at=str(raw_job["first_seen_at"]),
        last_seen_at=str(raw_job["last_seen_at"]),
        status=status,  # type: ignore[arg-type]
        version=_to_int(raw_job["version"], field_name="version"),
        content_hash=str(raw_job["content_hash"]),
    )


def job_posting_to_document(job: JobPosting, source_path: str = "job_import") -> Document:
    """Convert a JobPosting to the existing Document structure."""

    text = f"# {job.job_title}\n\n## Description\n{job.description}\n\n## Requirements\n{job.requirements}"
    metadata = build_document_metadata(
        {
            "job_id": job.job_id,
            "company": job.company,
            "job_title": job.job_title,
            "city": job.city,
            "source_platform": job.source_platform,
            "source_url": job.source_url,
            "first_seen_at": job.first_seen_at,
            "last_seen_at": job.last_seen_at,
            "status": job.status,
            "version": str(job.version),
            "content_hash": job.content_hash,
        },
        source_type="jd",
    )
    return Document(
        source_path=source_path,
        title=f"{job.company} - {job.job_title}",
        text=text,
        metadata=metadata,
    )


def job_postings_to_documents(
    jobs: list[JobPosting], source_path: str = "job_import"
) -> list[Document]:
    """Convert imported job postings to Documents."""

    return [job_posting_to_document(job, source_path=source_path) for job in jobs]


def _to_int(value: Any, field_name: str) -> int:
    """Convert an imported field to int with a clear error message."""

    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an integer") from exc
