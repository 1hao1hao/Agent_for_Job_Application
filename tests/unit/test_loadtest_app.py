from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from loadtest.app import create_loadtest_app
from loadtest.scenarios import QUERY_CASES


class LoadTestAppTests(unittest.TestCase):
    def test_query_catalog_covers_four_representative_needs(self) -> None:
        self.assertEqual(
            {str(item["name"]) for item in QUERY_CASES},
            {"exact_fact", "semantic", "multi_source", "relation_reasoning"},
        )
        self.assertGreaterEqual(len(QUERY_CASES), 12)
        self.assertTrue(all("retriever" not in item for item in QUERY_CASES))

    def test_deterministic_app_runs_real_pipeline_without_external_llm(self) -> None:
        client = TestClient(create_loadtest_app())

        response = client.post(
            "/v1/query",
            json={
                **{
                    key: value
                    for key, value in QUERY_CASES[0].items()
                    if key != "name"
                },
                "retriever": "bm25",
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn(response.json()["status"], {"answered", "insufficient_evidence"})
        self.assertTrue(response.json()["trace_id"])


if __name__ == "__main__":
    unittest.main()
