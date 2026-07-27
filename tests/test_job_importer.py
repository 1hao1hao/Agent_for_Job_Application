import csv
import json
from pathlib import Path
import tempfile
import unittest

from intern_rag.ingestion import (
    JobPosting,
    job_posting_from_dict,
    job_posting_to_document,
    job_postings_to_documents,
    load_job_postings,
)


def _sample_job() -> dict[str, str | int]:
    return {
        "job_id": "job-001",
        "company": "Example Cloud",
        "job_title": "Backend Intern",
        "city": "Shanghai",
        "source_platform": "manual",
        "source_url": "https://jobs.example.com/job-001",
        "description": "Build Python services for retrieval workflows.",
        "requirements": "Python, tests, basic RAG knowledge.",
        "first_seen_at": "2026-06-01",
        "last_seen_at": "2026-06-13",
        "status": "active",
        "version": 2,
        "content_hash": "abc123",
    }


class JobImporterTests(unittest.TestCase):
    def test_load_job_postings_from_json(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "jobs.json"
            path.write_text(json.dumps({"jobs": [_sample_job()]}), encoding="utf-8")

            jobs = load_job_postings(path)

        self.assertEqual(len(jobs), 1)
        self.assertIsInstance(jobs[0], JobPosting)
        self.assertEqual(jobs[0].job_id, "job-001")
        self.assertEqual(jobs[0].version, 2)
        self.assertEqual(jobs[0].status, "active")

    def test_load_job_postings_from_csv(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "jobs.csv"
            with path.open("w", encoding="utf-8", newline="") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=list(_sample_job().keys()))
                writer.writeheader()
                writer.writerow(_sample_job())

            jobs = load_job_postings(path)

        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0].company, "Example Cloud")
        self.assertEqual(jobs[0].source_platform, "manual")

    def test_job_posting_to_document_preserves_metadata(self) -> None:
        job = job_posting_from_dict(_sample_job())

        document = job_posting_to_document(job, source_path="tests/sample_jobs.json")

        self.assertEqual(document.source_path, "tests/sample_jobs.json")
        self.assertEqual(document.title, "Example Cloud - Backend Intern")
        self.assertIn("Build Python services", document.text)
        self.assertIn("Python, tests", document.text)
        self.assertEqual(document.metadata.source_type, "jd")
        self.assertEqual(document.metadata.job_id, "job-001")
        self.assertEqual(document.metadata.company, "Example Cloud")
        self.assertEqual(document.metadata.job_title, "Backend Intern")
        self.assertEqual(document.metadata.city, "Shanghai")
        self.assertEqual(document.metadata.source_platform, "manual")
        self.assertEqual(document.metadata.source_url, "https://jobs.example.com/job-001")
        self.assertEqual(document.metadata.first_seen_at, "2026-06-01")
        self.assertEqual(document.metadata.last_seen_at, "2026-06-13")
        self.assertEqual(document.metadata.status, "active")
        self.assertEqual(document.metadata.version, 2)
        self.assertEqual(document.metadata.content_hash, "abc123")

    def test_job_postings_to_documents_keeps_count(self) -> None:
        jobs = [job_posting_from_dict(_sample_job())]

        documents = job_postings_to_documents(jobs, source_path="import.csv")

        self.assertEqual(len(documents), 1)
        self.assertEqual(documents[0].metadata.job_id, "job-001")

    def test_missing_required_field_raises_value_error(self) -> None:
        raw_job = _sample_job()
        del raw_job["content_hash"]

        with self.assertRaises(ValueError):
            job_posting_from_dict(raw_job)

    def test_invalid_status_is_normalized_to_unknown(self) -> None:
        raw_job = _sample_job()
        raw_job["status"] = "paused"

        job = job_posting_from_dict(raw_job)

        self.assertEqual(job.status, "unknown")

    def test_unsupported_import_suffix_raises_value_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "jobs.xlsx"
            path.write_text("", encoding="utf-8")

            with self.assertRaises(ValueError):
                load_job_postings(path)


if __name__ == "__main__":
    unittest.main()
