from __future__ import annotations

from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Literal


WebContentType = Literal["html", "text"]


@dataclass(frozen=True)
class PublicSourceSpec:
    """描述一条允许公开采集的网页及其正文边界。"""

    source_id: str
    source_type: str
    url: str
    title: str
    content_type: WebContentType = "html"
    start_marker: str = ""
    end_marker: str = ""
    max_chars: int = 20_000


@dataclass(frozen=True)
class CollectedPublicSource:
    """公开网页经过正文抽取后的候选知识记录。"""

    source_id: str
    source_type: str
    title: str
    source_url: str
    text: str


class _VisibleTextParser(HTMLParser):
    """提取 HTML 可见文本，并跳过脚本、样式和 SVG。"""

    _SKIPPED_TAGS = {"script", "style", "noscript", "svg"}
    _BLOCK_TAGS = {
        "article",
        "br",
        "div",
        "h1",
        "h2",
        "h3",
        "h4",
        "li",
        "main",
        "p",
        "section",
    }

    def __init__(self) -> None:
        super().__init__()
        self._skip_depth = 0
        self.parts: list[str] = []

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        del attrs
        if tag in self._SKIPPED_TAGS:
            self._skip_depth += 1
        elif self._skip_depth == 0 and tag in self._BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIPPED_TAGS and self._skip_depth:
            self._skip_depth -= 1
        elif self._skip_depth == 0 and tag in self._BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0:
            self.parts.append(data)


def html_to_visible_text(html: str) -> str:
    """把 HTML 转成按块换行的可读文本，不执行 JavaScript。"""

    parser = _VisibleTextParser()
    parser.feed(html)
    lines = [
        " ".join(line.split())
        for line in "".join(parser.parts).splitlines()
        if line.strip()
    ]
    return "\n".join(lines)


def extract_text_segment(
    text: str,
    *,
    start_marker: str = "",
    end_marker: str = "",
    max_chars: int = 20_000,
) -> str:
    """按配置边界截取正文；缺少边界时明确报错，避免保存导航噪声。"""

    if max_chars <= 0:
        raise ValueError("max_chars must be greater than 0")

    start = 0
    if start_marker:
        start = text.find(start_marker)
        if start < 0:
            raise ValueError(f"start marker not found: {start_marker}")

    end = len(text)
    if end_marker:
        end = text.find(end_marker, start + len(start_marker))
        if end < 0:
            raise ValueError(f"end marker not found: {end_marker}")

    segment = text[start:end].strip()
    if not segment:
        raise ValueError("extracted text is empty")
    return segment[:max_chars].rstrip()


def collect_public_source(
    spec: PublicSourceSpec, raw_content: str
) -> CollectedPublicSource:
    """将已下载网页转换为候选知识记录，网络请求由外层脚本负责。"""

    visible_text = (
        html_to_visible_text(raw_content)
        if spec.content_type == "html"
        else raw_content.strip()
    )
    text = extract_text_segment(
        visible_text,
        start_marker=spec.start_marker,
        end_marker=spec.end_marker,
        max_chars=spec.max_chars,
    )
    return CollectedPublicSource(
        source_id=spec.source_id,
        source_type=spec.source_type,
        title=spec.title,
        source_url=spec.url,
        text=text,
    )
