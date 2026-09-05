import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from intern_rag.serving.runtime import load_runtime_retriever


class FakeRetriever:
    def __call__(self, query, chunks, top_k=5, source_types=None):
        del query, chunks, top_k, source_types
        return []

    def get_last_trace(self):
        return {"selected_strategy": "graph_hybrid"}


class RuntimeRetrieverConfigTests(unittest.TestCase):
    def _write_config(self, root: Path, name: str, retriever_name: str) -> Path:
        path = root / name
        path.write_text(json.dumps({
            "retriever_name": retriever_name,
            "dataset_version": "evalrag_v0.3",
            "config_version": f"{retriever_name}-v1",
        }), encoding="utf-8")
        return path

    def test_locked_config_is_exposed_in_trace(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = self._write_config(root, "adaptive.json", "graph_adaptive")
            with patch(
                "intern_rag.serving.runtime.build_retriever_from_config",
                return_value=FakeRetriever(),
            ):
                binding, loaded = load_runtime_retriever(root, config)

        trace = binding.get_last_trace()
        self.assertEqual(loaded["retriever_name"], "graph_adaptive")
        self.assertEqual(trace["runtime_effective_retriever"], "graph_adaptive")
        self.assertEqual(trace["runtime_config_version"], "graph_adaptive-v1")
        self.assertIsNone(trace["runtime_fallback_reason"])

    def test_missing_capability_fails_fast_without_explicit_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = self._write_config(root, "adaptive.json", "graph_adaptive")
            with patch.dict(os.environ, {}, clear=False), patch(
                "intern_rag.serving.runtime.build_retriever_from_config",
                side_effect=ValueError("missing graph"),
            ):
                os.environ.pop("EVALRAG_RETRIEVER_FALLBACK_CONFIG", None)
                with self.assertRaisesRegex(RuntimeError, "failed to build"):
                    load_runtime_retriever(root, config)

    def test_explicit_fallback_is_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            primary = self._write_config(root, "adaptive.json", "graph_adaptive")
            fallback = self._write_config(root, "bm25.json", "bm25")
            with patch.dict(os.environ, {
                "EVALRAG_RETRIEVER_FALLBACK_CONFIG": str(fallback),
            }), patch(
                "intern_rag.serving.runtime.build_retriever_from_config",
                side_effect=[ValueError("missing graph"), FakeRetriever()],
            ):
                binding, loaded = load_runtime_retriever(root, primary)

        trace = binding.get_last_trace()
        self.assertEqual(loaded["retriever_name"], "bm25")
        self.assertEqual(trace["runtime_configured_retriever"], "graph_adaptive")
        self.assertEqual(trace["runtime_effective_retriever"], "bm25")
        self.assertIn("ValueError", trace["runtime_fallback_reason"])


if __name__ == "__main__":
    unittest.main()
