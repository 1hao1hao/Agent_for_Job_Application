from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.export_fixed_demos import DEMO_CASES, export_fixed_demos


class FixedDemoExportTests(unittest.TestCase):
    """验证固定 Demo 能稳定配对 prediction、citation 和 Trace。"""

    def test_export_three_demos_without_raw_chunk_text(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "run"
            output_dir = root / "demos"
            run_dir.mkdir()
            cases = []
            traces = []
            for index, (name, case_id) in enumerate(DEMO_CASES.items(), start=1):
                answered = name != "abstention"
                chunk_id = f"chunk-{index}"
                cases.append({
                    "case_id": case_id,
                    "category": name,
                    "query": f"问题 {index}",
                    "answerable": answered,
                    "status": "answered" if answered else "insufficient_evidence",
                    "answer": "回答" if answered else "证据不足",
                    "citation_ids": [chunk_id] if answered else [],
                    "error_type": None,
                })
                traces.append({
                    "request_id": case_id,
                    "query": f"问题 {index}",
                    "retrieved_chunks": [{
                        "chunk_id": chunk_id,
                        "rank": 1,
                        "score": 1.0,
                        "source_type": "jd",
                        "source_path": "data/raw/jd/sample.md",
                        "reason": "fixture",
                        "text": "不应进入公开 Demo 的原始正文",
                    }],
                    "response_status": "answered" if answered else "insufficient_evidence",
                })
            self._write_jsonl(run_dir / "case_results.jsonl", cases)
            self._write_jsonl(run_dir / "traces.jsonl", traces)
            (run_dir / "run_config.json").write_text(
                json.dumps({"dataset_version": "fixture-v1"}), encoding="utf-8"
            )

            paths = export_fixed_demos(run_dir, output_dir)

            self.assertEqual(len(paths), 3)
            single = json.loads((output_dir / "single_source.json").read_text())
            self.assertEqual(single["response"]["citations"][0]["chunk_id"], "chunk-1")
            self.assertNotIn("text", single["trace"]["retrieved_chunks"][0])
            abstention = json.loads((output_dir / "abstention.json").read_text())
            self.assertEqual(abstention["response"]["citations"], [])

    @staticmethod
    def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )


if __name__ == "__main__":
    unittest.main()
