from __future__ import annotations

import unittest

from loadtest.full_e2e import percentile, summarize_tier


class FullE2ELoadTestTests(unittest.TestCase):
    def test_percentile_uses_nearest_rank(self) -> None:
        self.assertEqual(percentile([1, 2, 3, 4], 0.50), 2.0)
        self.assertEqual(percentile([1, 2, 3, 4], 0.95), 4.0)

    def test_summary_separates_generation_from_gate_abstention(self) -> None:
        ledger = [
            {
                "request_id": "r1", "latency_ms": 100.0, "http_status": 200,
                "response_status": "answered", "session_request": True,
                "case_type": "semantic",
            },
            {
                "request_id": "r2", "latency_ms": 10.0, "http_status": 200,
                "response_status": "insufficient_evidence", "session_request": False,
                "case_type": "relation_reasoning",
            },
        ]
        traces = [
            {
                "request_id": "r1",
                "latency_ms": {"routing": 1, "retrieval": 20, "evidence": 1,
                               "context": 3, "generation": 60, "validation": 1,
                               "total": 90},
                "retrieval": {"decision": {"selected_strategy": "hybrid",
                                              "rerank_invoked": True}},
                "attempts": [
                    {"type": "initial_retrieval"},
                    {"type": "initial_generation", "token_usage": {
                        "input_tokens": 30, "output_tokens": 10, "total_tokens": 40,
                    }},
                ],
            },
            {
                "request_id": "r2",
                "latency_ms": {"routing": 1, "retrieval": 5, "evidence": 1,
                               "context": 0, "generation": 0, "validation": 0,
                               "total": 8},
                "retrieval": {"decision": {"selected_strategy": "graph_hybrid",
                                              "rerank_invoked": False}},
                "attempts": [{"type": "initial_retrieval"}],
            },
        ]

        summary = summarize_tier("real", 2, ledger, traces, rps=3.5)

        self.assertEqual(summary.request_count, 2)
        self.assertEqual(summary.generation_requests, 1)
        self.assertEqual(summary.generation_calls, 1)
        self.assertEqual(summary.strategy_counts, {"hybrid": 1, "graph_hybrid": 1})
        self.assertEqual(summary.error_type_counts, {})
        self.assertEqual(
            summary.case_type_counts, {"semantic": 1, "relation_reasoning": 1}
        )
        self.assertEqual(summary.rerank_count, 1)
        self.assertEqual(summary.token_usage["total_tokens"], 40)
        self.assertEqual(summary.generation_request_p50_ms, 100.0)
        self.assertEqual(summary.stage_latency_ms["generation"]["p95"], 60.0)
        self.assertEqual(summary.session_request_ratio, 0.5)


if __name__ == "__main__":
    unittest.main()
