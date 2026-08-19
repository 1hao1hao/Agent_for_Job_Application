from __future__ import annotations

import unittest

from intern_rag.ingestion.web_importer import (
    PublicSourceSpec,
    collect_public_source,
    extract_text_segment,
    html_to_visible_text,
)


class WebImporterTests(unittest.TestCase):
    def test_html_to_visible_text_skips_script_and_preserves_blocks(self) -> None:
        html = "<h1>岗位</h1><script>secret()</script><p>职责一</p><p>职责二</p>"

        text = html_to_visible_text(html)

        self.assertEqual(text, "岗位\n职责一\n职责二")

    def test_collect_public_source_extracts_configured_segment(self) -> None:
        spec = PublicSourceSpec(
            source_id="jd-1",
            source_type="jd",
            url="https://example.com/job/1",
            title="测试岗位",
            start_marker="岗位职责",
            end_marker="申请岗位",
        )

        result = collect_public_source(
            spec,
            "<nav>首页</nav><h2>岗位职责</h2><p>开发 RAG。</p><div>申请岗位</div>",
        )

        self.assertEqual(result.text, "岗位职责\n开发 RAG。")

    def test_missing_marker_is_controlled_failure(self) -> None:
        with self.assertRaisesRegex(ValueError, "start marker not found"):
            extract_text_segment("只有导航", start_marker="岗位职责")


if __name__ == "__main__":
    unittest.main()
