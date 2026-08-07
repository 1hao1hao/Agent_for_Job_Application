from pathlib import Path
import tempfile
import unittest

from intern_rag.evaluation import (
    build_corpus,
    export_chunks_jsonl,
    load_chunks_jsonl,
    validate_corpus_manifest,
    write_corpus_manifest,
)


class EvaluationCorpusTests(unittest.TestCase):
    def test_build_corpus_records_provenance_stats_and_chunk_export(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            raw_root = Path(temp_dir) / "raw"
            source_dir = raw_root / "jd"
            source_dir.mkdir(parents=True)
            source_file = source_dir / "sample.md"
            source_file.write_text(
                "---\n"
                "source_platform: project_authored\n"
                "source_url: unknown\n"
                "collected_at: 2026-07-29\n"
                "public_status: project_owned\n"
                "anonymized: true\n"
                "content_origin: test_fixture\n"
                "human_reviewed: true\n"
                "---\n"
                "# 测试岗位\n\n岗位要求熟悉 Python。",
                encoding="utf-8",
            )

            entries, chunks, stats = build_corpus(raw_root, max_chars=800)
            manifest_path = Path(temp_dir) / "manifest.jsonl"
            chunks_path = Path(temp_dir) / "chunks.jsonl"
            write_corpus_manifest(entries, manifest_path)
            export_chunks_jsonl(
                chunks,
                chunks_path,
                dataset_version="test-v1",
            )
            loaded_chunks = load_chunks_jsonl(chunks_path)

        self.assertEqual(len(entries), 1)
        self.assertTrue(entries[0].anonymized)
        self.assertTrue(entries[0].human_reviewed)
        self.assertEqual(stats.document_count, 1)
        self.assertEqual(stats.chunk_count, 1)
        self.assertGreater(stats.chunk_length_p50, 0)
        self.assertEqual(loaded_chunks, chunks)

    def test_manifest_validation_checks_scale_sources_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            raw_root = Path(temp_dir) / "raw"
            source_dir = raw_root / "jd"
            source_dir.mkdir(parents=True)
            (source_dir / "missing.md").write_text(
                "# 缺少 provenance 的文档",
                encoding="utf-8",
            )
            entries, _, _ = build_corpus(raw_root)

        errors = validate_corpus_manifest(entries, minimum_documents=30)

        self.assertIn("corpus requires at least 30 documents", errors[0])
        self.assertTrue(any("misses source types" in error for error in errors))
        self.assertTrue(any("misses source_platform" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
