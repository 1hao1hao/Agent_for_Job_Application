import unittest

from intern_rag.evaluation.corpus_v03 import (
    build_v03_chunks,
    deduplicate_corpus,
    normalize_corpus_record,
    simhash64,
    validate_v03_corpus,
)


class CorpusV03Tests(unittest.TestCase):
    def _record(self, key: str, text: str, source: str = "job_posting"):
        return normalize_corpus_record(
            {
                "source_key": key,
                "canonical_source_type": source,
                "title": key,
                "text": text,
                "source_url": f"https://example.com/{key}",
                "source_method": "dataset_import",
                "owner_scope": "public",
                "collected_at": "2026-08-16",
                "public_status": "public_dataset",
                "license_status": "Apache-2.0",
                "anonymized": True,
                "review_status": "ai_assisted",
            }
        )

    def test_normalize_and_chunk_preserve_provenance(self) -> None:
        document = self._record("job-1", "岗位要求包括 Python、RAG 和评测能力。" * 12)

        chunks = build_v03_chunks([document], max_chars=120)

        self.assertGreater(len(chunks), 1)
        self.assertEqual(chunks[0].source_type, "jd")
        self.assertEqual(chunks[0].metadata["document_id"], document.document_id)
        self.assertEqual(chunks[0].metadata["license_status"], "Apache-2.0")
        self.assertTrue(chunks[0].metadata["anonymized"])

    def test_exact_and_near_duplicate_are_rejected(self) -> None:
        base = "岗位职责是实现检索增强生成系统并完成引用评测。" * 8
        documents = [
            self._record("a", base),
            self._record("b", base),
            self._record("c", base + "新增一个标点。"),
            self._record("d", "完全不同的项目文档，介绍图数据库和向量索引。" * 8),
        ]

        retained, rejected = deduplicate_corpus(documents, near_duplicate_distance=6)

        self.assertEqual(len(retained), 2)
        self.assertEqual({item.reason for item in rejected}, {"exact_duplicate", "near_duplicate"})

    def test_simhash_is_stable(self) -> None:
        text = "中文 SimHash 用于识别模板化近重复文档。"
        self.assertEqual(simhash64(text), simhash64(text))

    def test_validation_reports_scale_and_missing_sources(self) -> None:
        document = self._record("only-one", "有效岗位描述。" * 20)
        chunks = build_v03_chunks([document], max_chars=120)

        errors = validate_v03_corpus(
            [document], chunks, minimum_documents=2, minimum_chunks=1
        )

        self.assertTrue(any("requires 2 documents" in error for error in errors))
        self.assertTrue(any("missing canonical sources" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
