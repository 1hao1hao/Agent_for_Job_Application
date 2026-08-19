from pathlib import Path
import tempfile
import unittest

from intern_rag.ingestion import (
    Chunk,
    build_document_from_file,
    build_chunks_from_file,
    load_chunks_from_raw_dir,
    read_text_file,
    split_text,
)


class IngestionTests(unittest.TestCase):
    def test_read_and_build_chunks_from_supported_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            raw_root = Path(temp_dir)
            jd_dir = raw_root / "jd"
            jd_dir.mkdir()
            source_file = jd_dir / "backend.md"
            source_file.write_text(
                "# Backend Intern\n\nPython services and retrieval quality.\n\nWrite tests.",
                encoding="utf-8",
            )

            chunks = build_chunks_from_file(source_file, raw_root, max_chars=60)

        self.assertGreaterEqual(len(chunks), 2)
        self.assertTrue(all(isinstance(chunk, Chunk) for chunk in chunks))
        self.assertEqual(chunks[0].source_type, "jd")
        self.assertEqual(chunks[0].source_path, str(source_file))
        self.assertIn("Backend Intern", chunks[0].text)
        self.assertIn("chunk_index", chunks[0].metadata)
        self.assertIn("source_file_name", chunks[0].metadata)
        self.assertIn("char_count", chunks[0].metadata)
        self.assertEqual(chunks[0].metadata["company"], "unknown")
        self.assertEqual(chunks[0].metadata["status"], "unknown")
        self.assertEqual(chunks[0].metadata["version"], 1)
        self.assertEqual(chunks[0].metadata["source_priority"], 0)

    def test_document_metadata_is_inherited_by_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            raw_root = Path(temp_dir)
            jd_dir = raw_root / "jd"
            jd_dir.mkdir()
            source_file = jd_dir / "fresh_job.md"
            source_file.write_text(
                "\n".join(
                    [
                        "---",
                        "company: Example AI",
                        "job_title: RAG Intern",
                        "city: Beijing",
                        "source_url: https://example.com/rag-intern",
                        "first_seen_at: 2026-06-01",
                        "last_seen_at: 2026-06-13",
                        "status: active",
                        "version: 3",
                        "source_priority: 9",
                        "---",
                        "# RAG Intern",
                        "",
                        "Build freshness-aware retrieval.",
                        "",
                        "Explain metadata inheritance.",
                    ]
                ),
                encoding="utf-8",
            )

            document = build_document_from_file(source_file, raw_root)
            chunks = build_chunks_from_file(source_file, raw_root, max_chars=60)

        self.assertEqual(document.metadata.company, "Example AI")
        self.assertEqual(document.metadata.status, "active")
        self.assertEqual(document.metadata.version, 3)
        self.assertGreaterEqual(len(chunks), 2)
        for chunk in chunks:
            self.assertEqual(chunk.metadata["source_type"], "jd")
            self.assertEqual(chunk.metadata["company"], "Example AI")
            self.assertEqual(chunk.metadata["job_title"], "RAG Intern")
            self.assertEqual(chunk.metadata["city"], "Beijing")
            self.assertEqual(chunk.metadata["source_url"], "https://example.com/rag-intern")
            self.assertEqual(chunk.metadata["first_seen_at"], "2026-06-01")
            self.assertEqual(chunk.metadata["last_seen_at"], "2026-06-13")
            self.assertEqual(chunk.metadata["status"], "active")
            self.assertEqual(chunk.metadata["version"], 3)
            self.assertEqual(chunk.metadata["source_priority"], 9)

    def test_missing_document_metadata_uses_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            raw_root = Path(temp_dir)
            profile_dir = raw_root / "user_profile"
            profile_dir.mkdir()
            source_file = profile_dir / "profile.md"
            source_file.write_text("Backend internship target.", encoding="utf-8")

            document = build_document_from_file(source_file, raw_root)
            chunks = build_chunks_from_file(source_file, raw_root)

        self.assertEqual(document.metadata.source_type, "user_profile")
        self.assertEqual(document.metadata.company, "unknown")
        self.assertEqual(document.metadata.job_title, "unknown")
        self.assertEqual(document.metadata.city, "unknown")
        self.assertEqual(document.metadata.source_url, "")
        self.assertEqual(document.metadata.first_seen_at, "")
        self.assertEqual(document.metadata.last_seen_at, "")
        self.assertEqual(document.metadata.status, "unknown")
        self.assertEqual(document.metadata.version, 1)
        self.assertEqual(document.metadata.source_priority, 0)
        self.assertEqual(chunks[0].metadata["company"], "unknown")
        self.assertEqual(chunks[0].metadata["source_type"], "user_profile")

    def test_empty_file_produces_no_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            raw_root = Path(temp_dir)
            resume_dir = raw_root / "resume"
            resume_dir.mkdir()
            source_file = resume_dir / "empty.txt"
            source_file.write_text("   \n\n", encoding="utf-8")

            chunks = build_chunks_from_file(source_file, raw_root)

        self.assertEqual(chunks, [])

    def test_load_chunks_from_raw_dir_uses_supported_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            raw_root = Path(temp_dir)
            interview_dir = raw_root / "interview"
            interview_dir.mkdir()
            (interview_dir / "notes.md").write_text("RAG needs citations.", encoding="utf-8")
            (interview_dir / "ignore.csv").write_text("not supported", encoding="utf-8")

            chunks = load_chunks_from_raw_dir(raw_root)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].source_type, "interview")
        self.assertEqual(chunks[0].text, "RAG needs citations.")

    def test_read_text_file_rejects_unsupported_suffix(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            source_file = Path(temp_dir) / "bad.csv"
            source_file.write_text("unsupported", encoding="utf-8")

            with self.assertRaises(ValueError):
                read_text_file(source_file)

    def test_split_text_rejects_invalid_max_chars(self) -> None:
        with self.assertRaises(ValueError):
            split_text("hello", max_chars=0)

    def test_sentence_boundary_merges_short_sentences_before_cutting(self) -> None:
        text = "第一句很短。第二句也不长。第三句提供完整的检索与评测说明。最后一句收尾。"

        chunks = split_text(
            text,
            max_chars=24,
            strategy="sentence_boundary",
            min_chunk_ratio=0.5,
        )

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk) <= 24 for chunk in chunks))
        self.assertEqual("".join(chunks), text)
        self.assertGreater(len(chunks[0]), len("第一句很短。"))
        self.assertTrue(chunks[0].endswith("。"))

    def test_sentence_boundary_uses_fixed_fallback_without_punctuation(self) -> None:
        text = "没有任何句末标点的连续文本" * 5

        chunks = split_text(
            text,
            max_chars=20,
            strategy="sentence_boundary",
            min_chunk_ratio=0.5,
        )

        self.assertTrue(all(10 <= len(chunk) <= 20 for chunk in chunks))
        self.assertEqual("".join(chunks), text)

    def test_split_text_rejects_invalid_strategy_and_ratio(self) -> None:
        with self.assertRaises(ValueError):
            split_text("hello", strategy="unknown")  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            split_text("hello", min_chunk_ratio=0)


if __name__ == "__main__":
    unittest.main()
