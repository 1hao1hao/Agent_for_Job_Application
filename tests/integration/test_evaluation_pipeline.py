from pathlib import Path
import tempfile
import unittest

from intern_rag.evaluation import (
    EvaluationCase,
    EvaluationRunConfig,
    build_corpus,
    export_chunks_jsonl,
    load_chunks_jsonl,
    run_keyword_evaluation,
    save_run_artifacts,
    validate_evaluation_dataset,
)


class EvaluationPipelineIntegrationTests(unittest.TestCase):
    def test_corpus_to_system_predictions_and_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            raw_root = temp_root / "raw"
            source_dir = raw_root / "jd"
            source_dir.mkdir(parents=True)
            (source_dir / "job.md").write_text(
                "---\n"
                "source_platform: test\n"
                "source_url: unknown\n"
                "collected_at: 2026-07-29\n"
                "public_status: test_fixture\n"
                "anonymized: true\n"
                "content_origin: test_fixture\n"
                "human_reviewed: true\n"
                "---\n"
                "# Python 实习岗位\n\n岗位要求熟悉 Python 和单元测试。",
                encoding="utf-8",
            )
            _, chunks, _ = build_corpus(raw_root)
            chunks_path = temp_root / "chunks.jsonl"
            export_chunks_jsonl(
                chunks,
                chunks_path,
                dataset_version="test-v1",
            )
            loaded_chunks = load_chunks_jsonl(chunks_path)
            case = EvaluationCase(
                case_id="integration-1",
                query="分析岗位 Python 要求",
                category="single_source",
                split="dev",
                expected_intent="analyze_jd",
                expected_sources=["jd"],
                relevant_chunk_ids=[loaded_chunks[0].id],
                answerable=True,
                expected_points=["Python", "单元测试"],
                notes="人工测试标签",
                human_reviewed=True,
            )
            validation = validate_evaluation_dataset(
                [case],
                available_chunk_ids={loaded_chunks[0].id},
                require_full_distribution=False,
                require_human_review=True,
            )
            result = run_keyword_evaluation(
                [case],
                loaded_chunks,
                EvaluationRunConfig(
                    run_id="integration-run",
                    dataset_version="test-v1",
                    split="dev",
                    retriever_name="keyword",
                    top_k=5,
                    chunk_max_chars=800,
                    git_commit="test",
                    command="test",
                ),
            )
            run_dir = temp_root / "run"
            save_run_artifacts(result, run_dir)

            self.assertTrue(validation.is_valid)
            self.assertEqual(
                result.case_results[0]["predicted"]["retrieved"][0]["chunk_id"],
                loaded_chunks[0].id,
            )
            self.assertTrue((run_dir / "summary.json").exists())
            self.assertTrue((run_dir / "failures.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
