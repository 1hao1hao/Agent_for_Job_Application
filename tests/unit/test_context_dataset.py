from pathlib import Path
import unittest

from intern_rag.evaluation.context_dataset import (
    load_context_dataset,
    validate_context_dataset,
)


class ContextDatasetTests(unittest.TestCase):
    def test_v01_contains_sixty_five_turn_dev_cases(self) -> None:
        cases = load_context_dataset(Path("data/evaluation/evalrag_context_v0.1.jsonl"))
        report = validate_context_dataset(cases)

        self.assertTrue(report["is_valid"])
        self.assertEqual(report["case_count"], 60)
        self.assertEqual(report["turn_count"], 300)
        self.assertEqual(report["split_counts"], {"dev": 60})
        self.assertEqual(report["human_reviewed_count"], 0)


if __name__ == "__main__":
    unittest.main()
