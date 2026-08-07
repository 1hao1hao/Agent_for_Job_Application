import unittest

from intern_rag.agent import (
    BuiltContext,
    ContextItem,
    GenerationResult,
    validate_generation,
)


def _context() -> BuiltContext:
    item = ContextItem(
        chunk_id="jd-1",
        source_type="jd",
        source_path="data/raw/jd/test.md",
        title="测试岗位",
        text="岗位要求熟悉 Python。",
        rank=1,
        score=0.9,
    )
    return BuiltContext(
        query="岗位要求是什么？",
        text="chunk_id: jd-1",
        items=[item],
        used_chunk_ids=["jd-1"],
        skipped_chunk_ids=[],
        char_count=14,
        max_chars=1000,
    )


def _generation(
    cited_chunk_ids: list[str],
    sufficient: bool = True,
) -> GenerationResult:
    return GenerationResult(
        answer="岗位要求熟悉 Python。",
        cited_chunk_ids=cited_chunk_ids,
        sufficient=sufficient,
        reason="测试",
    )


class ValidationTests(unittest.TestCase):
    def test_valid_citation_is_converted_to_existing_citation_schema(self) -> None:
        result = validate_generation(_generation(["jd-1"]), _context())

        self.assertTrue(result.is_valid)
        self.assertEqual(result.issues, [])
        self.assertEqual(result.citations[0].chunk_id, "jd-1")
        self.assertEqual(result.citations[0].source_path, "data/raw/jd/test.md")

    def test_nonexistent_citation_is_rejected(self) -> None:
        result = validate_generation(_generation(["missing-id"]), _context())

        self.assertFalse(result.is_valid)
        self.assertEqual(result.citations, [])
        self.assertEqual(result.issues[0].error_type, "citation_not_found")

    def test_duplicate_citation_is_rejected(self) -> None:
        result = validate_generation(_generation(["jd-1", "jd-1"]), _context())

        self.assertFalse(result.is_valid)
        self.assertEqual(result.citations, [])
        self.assertIn(
            "duplicate_citation",
            [issue.error_type for issue in result.issues],
        )

    def test_sufficient_answer_requires_non_empty_citation(self) -> None:
        result = validate_generation(_generation([]), _context())

        self.assertFalse(result.is_valid)
        self.assertEqual(result.issues[0].error_type, "missing_citation")

    def test_insufficient_result_with_empty_citations_is_valid(self) -> None:
        result = validate_generation(
            _generation([], sufficient=False),
            _context(),
        )

        self.assertTrue(result.is_valid)
        self.assertEqual(result.citations, [])

    def test_insufficient_result_must_not_include_citations(self) -> None:
        result = validate_generation(
            _generation(["jd-1"], sufficient=False),
            _context(),
        )

        self.assertFalse(result.is_valid)
        self.assertEqual(
            result.issues[0].error_type,
            "insufficient_with_citations",
        )


if __name__ == "__main__":
    unittest.main()
