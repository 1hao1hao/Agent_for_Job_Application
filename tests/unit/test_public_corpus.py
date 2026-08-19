import unittest

from intern_rag.ingestion.public_corpus import (
    DatasetSource,
    GithubSource,
    collect_github_documents,
    collect_huggingface_jobs,
    split_markdown_sections,
)


class PublicCorpusTests(unittest.TestCase):
    def test_huggingface_job_parser_preserves_dataset_provenance(self) -> None:
        payload = {
            "rows": [
                {
                    "row_idx": 7,
                    "row": {
                        "user": (
                            "<岗位名称>大模型实习生</岗位名称>"
                            "<岗位描述>负责 RAG、Agent、检索和评测工程。" * 4
                            + "</岗位描述><学历描述>本科</学历描述>"
                        )
                    },
                }
            ]
        }
        source = DatasetSource("owner/dataset", "default", "train", "Apache-2.0", 1)

        records, attempts = collect_huggingface_jobs(
            source, lambda url: payload, collected_at="2026-08-16"
        )

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["job_title"], "大模型实习生")
        self.assertEqual(records[0]["license_status"], "Apache-2.0")
        self.assertEqual(attempts[0]["status"], "collected")

    def test_github_heading_split_creates_grounded_records(self) -> None:
        markdown = (
            "# 题库\n\n## RAG 如何评测\n" + "召回率衡量证据覆盖。" * 20
            + "\n\n## Agent 如何追踪\n" + "Trace 保存每个阶段。" * 20
        )
        source = GithubSource(
            repository="owner/repo",
            revision="abc123",
            license_status="CC-BY-SA-4.0",
            canonical_source_type="interview_knowledge",
            quota=10,
            include_prefixes=("modules/",),
            split_headings=True,
        )

        records, _ = collect_github_documents(
            source,
            lambda url: {"tree": [{"type": "blob", "path": "modules/rag.md"}]},
            lambda url: markdown,
            collected_at="2026-08-16",
        )

        self.assertEqual(len(records), 2)
        self.assertIn("abc123", records[0]["source_url"])
        self.assertEqual(records[0]["review_status"], "ai_assisted")

    def test_short_heading_is_not_counted_as_document(self) -> None:
        sections = split_markdown_sections("## 短段\n一句话", minimum_chars=30)
        self.assertEqual(sections, [])


if __name__ == "__main__":
    unittest.main()
