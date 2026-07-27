from pathlib import Path
import tempfile
import unittest

from intern_rag.evaluation import (
    RetrievalEvalCase,
    RouterEvalCase,
    calculate_average_recall_at_k,
    calculate_recall_at_k,
    calculate_router_accuracy,
    evaluate_cases,
    load_evaluation_cases,
)


class EvaluationTests(unittest.TestCase):
    def test_calculate_recall_at_k(self) -> None:
        recall = calculate_recall_at_k(
            retrieved_chunk_ids=["chunk-a", "chunk-b", "chunk-c"],
            relevant_chunk_ids=["chunk-b", "chunk-d"],
            k=2,
        )

        self.assertEqual(recall, 0.5)

    def test_calculate_recall_at_k_returns_zero_without_labels(self) -> None:
        self.assertEqual(calculate_recall_at_k(["chunk-a"], [], k=1), 0.0)
        self.assertEqual(calculate_recall_at_k(["chunk-a"], ["chunk-a"], k=0), 0.0)

    def test_calculate_average_recall_at_k(self) -> None:
        cases = [
            RetrievalEvalCase(
                query="岗位要求",
                relevant_chunk_ids=["jd-1"],
                retrieved_chunk_ids=["jd-1", "resume-1"],
            ),
            RetrievalEvalCase(
                query="面试问题",
                relevant_chunk_ids=["interview-1", "interview-2"],
                retrieved_chunk_ids=["resume-1", "interview-2"],
            ),
        ]

        self.assertEqual(calculate_average_recall_at_k(cases, k=2), 0.75)

    def test_calculate_router_accuracy_requires_intent_and_sources(self) -> None:
        cases = [
            RouterEvalCase(
                query="分析岗位",
                expected_intent="analyze_jd",
                predicted_intent="analyze_jd",
                expected_sources=["jd"],
                predicted_sources=["jd"],
            ),
            RouterEvalCase(
                query="准备面试",
                expected_intent="interview_prepare",
                predicted_intent="interview_prepare",
                expected_sources=["interview", "jd", "resume"],
                predicted_sources=["interview"],
            ),
        ]

        self.assertEqual(calculate_router_accuracy(cases), 0.5)

    def test_evaluate_cases_builds_report(self) -> None:
        retrieval_cases = [
            RetrievalEvalCase(
                query="岗位要求",
                relevant_chunk_ids=["jd-1"],
                retrieved_chunk_ids=["jd-1"],
            )
        ]
        router_cases = [
            RouterEvalCase(
                query="分析岗位",
                expected_intent="analyze_jd",
                predicted_intent="analyze_jd",
                expected_sources=["jd"],
                predicted_sources=["jd"],
            )
        ]

        report = evaluate_cases(retrieval_cases, router_cases, k=1)

        self.assertEqual(report.recall_at_k, 1.0)
        self.assertEqual(report.router_accuracy, 1.0)
        self.assertEqual(report.retrieval_case_count, 1)
        self.assertEqual(report.router_case_count, 1)

    def test_load_evaluation_cases_from_json(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "cases.json"
            path.write_text(
                """
                {
                  "retrieval_cases": [
                    {
                      "query": "岗位要求",
                      "relevant_chunk_ids": ["jd-1"],
                      "retrieved_chunk_ids": ["jd-1"]
                    }
                  ],
                  "router_cases": [
                    {
                      "query": "分析岗位",
                      "expected_intent": "analyze_jd",
                      "predicted_intent": "analyze_jd",
                      "expected_sources": ["jd"],
                      "predicted_sources": ["jd"]
                    }
                  ]
                }
                """,
                encoding="utf-8",
            )

            retrieval_cases, router_cases = load_evaluation_cases(path)

        self.assertEqual(len(retrieval_cases), 1)
        self.assertEqual(retrieval_cases[0].query, "岗位要求")
        self.assertEqual(len(router_cases), 1)
        self.assertEqual(router_cases[0].expected_sources, ["jd"])


if __name__ == "__main__":
    unittest.main()
