from __future__ import annotations

from hashlib import sha1
from pathlib import Path
from typing import Literal

from intern_rag.ingestion.schemas import Chunk, Document, build_document_metadata


SUPPORTED_SUFFIXES = {".md", ".txt"}
SOURCE_TYPES = {"interview", "jd", "project_logs", "resume", "user_profile"}
ChunkingStrategy = Literal["paragraph_fixed", "sentence_boundary"]
SENTENCE_ENDINGS = frozenset("。！？!?；;")


def read_text_file(path: Path) -> str:
    """读取一个本地文本文件。

    目前只支持 `.md` 和 `.txt`。这里不做复杂解析，只负责把文件内容
    按 UTF-8 读出来；文件不存在时保留 Python 原生 FileNotFoundError。
    """

    if path.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise ValueError(f"Unsupported source file type: {path.suffix}")
    return path.read_text(encoding="utf-8")


def build_document_from_file(path: Path, raw_root: Path) -> Document:
    """从本地文件构建 Document。

    文件正文前可以有简单的 Markdown front matter，例如 `company: xxx`。
    如果没有这些字段，也会通过默认 metadata 正常加载旧数据。
    """

    raw_text = read_text_file(path)
    front_matter, body = _split_front_matter(raw_text)
    source_type = infer_source_type(path, raw_root)
    metadata = build_document_metadata(front_matter, source_type=source_type)
    return Document(
        source_path=str(path),
        title=path.stem,
        text=body,
        metadata=metadata,
    )


def split_text(
    text: str,
    max_chars: int = 800,
    *,
    strategy: ChunkingStrategy = "paragraph_fixed",
    min_chunk_ratio: float = 0.5,
) -> list[str]:
    """把长文本切成多个较短的文本片段。

    两种策略都先按空行保留段落，再把短段落合并到 `max_chars`。区别只在单个段落
    超长时：baseline 按固定字符切；`sentence_boundary` 优先选择靠近长度上限的
    句末标点。标点只是候选边界，不会遇到一句就立即生成 Chunk；找不到合适标点时
    仍按字符兜底。`min_chunk_ratio` 用于避免选择过早的标点产生过短片段。
    """

    if max_chars <= 0:
        raise ValueError("max_chars must be greater than 0")
    if strategy not in {"paragraph_fixed", "sentence_boundary"}:
        raise ValueError(f"unsupported chunking strategy: {strategy}")
    if not 0 < min_chunk_ratio <= 1:
        raise ValueError("min_chunk_ratio must be between 0 and 1")

    paragraphs = [paragraph.strip() for paragraph in text.split("\n\n")]
    chunks: list[str] = []
    current_parts: list[str] = []
    current_length = 0

    for paragraph in paragraphs:
        if not paragraph:
            continue

        if len(paragraph) > max_chars:
            if current_parts:
                chunks.append("\n\n".join(current_parts))
                current_parts = []
                current_length = 0
            if strategy == "sentence_boundary":
                chunks.extend(
                    _split_long_paragraph_at_sentence_boundary(
                        paragraph,
                        max_chars,
                        min_chunk_ratio=min_chunk_ratio,
                    )
                )
            else:
                chunks.extend(_split_long_paragraph(paragraph, max_chars))
            continue

        separator_length = 2 if current_parts else 0
        next_length = current_length + separator_length + len(paragraph)

        if current_parts and next_length > max_chars:
            chunks.append("\n\n".join(current_parts))
            current_parts = [paragraph]
            current_length = len(paragraph)
        else:
            current_parts.append(paragraph)
            current_length = next_length

    if current_parts:
        chunks.append("\n\n".join(current_parts))

    return chunks


def infer_source_type(path: Path, raw_root: Path) -> str:
    """从 `data/raw/` 下的一级目录推断 source_type。"""

    relative_parts = path.relative_to(raw_root).parts
    if not relative_parts:
        raise ValueError(f"Cannot infer source type from path: {path}")

    source_type = relative_parts[0]
    if source_type not in SOURCE_TYPES:
        raise ValueError(f"Unknown source type: {source_type}")
    return source_type


def build_chunks_from_file(
    path: Path,
    raw_root: Path,
    max_chars: int = 800,
    *,
    strategy: ChunkingStrategy = "paragraph_fixed",
    min_chunk_ratio: float = 0.5,
) -> list[Chunk]:
    """从一个本地文件生成 Chunk 列表。

    流程是：先构建 Document，再切分 Document.text，最后让每个 Chunk
    继承 Document.metadata，并追加 chunk 自己的索引、文件名和字符数。
    空文件或空正文会得到空列表。
    """

    document = build_document_from_file(path, raw_root)
    source_type = document.metadata.source_type
    chunk_texts = split_text(
        document.text,
        max_chars=max_chars,
        strategy=strategy,
        min_chunk_ratio=min_chunk_ratio,
    )
    chunks: list[Chunk] = []

    for chunk_index, chunk_text in enumerate(chunk_texts):
        chunk_id = _make_chunk_id(source_type, path, chunk_index, chunk_text)
        # metadata 先完整继承文档级信息，再追加 chunk 级信息。
        chunk_metadata = document.metadata.to_dict() | {
            "chunk_index": chunk_index,
            "source_file_name": path.name,
            "char_count": len(chunk_text),
        }
        chunks.append(
            Chunk(
                id=chunk_id,
                source_type=source_type,
                source_path=document.source_path,
                title=document.title,
                text=chunk_text,
                metadata=chunk_metadata,
            )
        )

    return chunks


def load_chunks_from_raw_dir(
    raw_root: Path,
    max_chars: int = 800,
    *,
    strategy: ChunkingStrategy = "paragraph_fixed",
    min_chunk_ratio: float = 0.5,
) -> list[Chunk]:
    """加载 raw 目录下所有支持的文本文件，并统一生成 Chunk。

    为了让测试和 demo 输出稳定，文件会按路径排序处理；不支持的后缀会被忽略。
    """

    chunks: list[Chunk] = []
    for path in sorted(raw_root.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            chunks.extend(
                build_chunks_from_file(
                    path,
                    raw_root,
                    max_chars=max_chars,
                    strategy=strategy,
                    min_chunk_ratio=min_chunk_ratio,
                )
            )
    return chunks


def _split_long_paragraph(paragraph: str, max_chars: int) -> list[str]:
    """当单个段落太长时，按固定字符窗口切开。"""

    return [
        paragraph[start : start + max_chars].strip()
        for start in range(0, len(paragraph), max_chars)
        if paragraph[start : start + max_chars].strip()
    ]


def _split_long_paragraph_at_sentence_boundary(
    paragraph: str,
    max_chars: int,
    *,
    min_chunk_ratio: float,
) -> list[str]:
    """优先在接近长度上限的句末切分，并避免短句独立成块。

    每轮只在 `[min_chars, max_chars]` 区间寻找最后一个句末标点，因此前面的短句
    会自然合并。若最后余量会过短，则选择更早的合法标点；没有合适标点时使用
    固定字符边界兜底，从而同时保持长度上限和尽可能完整的句子。
    """

    min_chars = max(1, int(max_chars * min_chunk_ratio))
    remaining = paragraph.strip()
    parts: list[str] = []
    while len(remaining) > max_chars:
        candidate_positions = [
            index + 1
            for index, char in enumerate(remaining[:max_chars])
            if char in SENTENCE_ENDINGS and index + 1 >= min_chars
        ]
        viable_positions = [
            position
            for position in candidate_positions
            if len(remaining) - position >= min_chars
        ]
        if viable_positions:
            cut_at = viable_positions[-1]
        elif candidate_positions and len(remaining) > max_chars + min_chars:
            cut_at = candidate_positions[-1]
        else:
            # 末尾过短时适当左移硬边界，避免产生很短的尾块。
            cut_at = min(max_chars, max(min_chars, len(remaining) - min_chars))

        part = remaining[:cut_at].strip()
        if part:
            parts.append(part)
        remaining = remaining[cut_at:].strip()

    if remaining:
        parts.append(remaining)
    return parts


def _make_chunk_id(source_type: str, path: Path, chunk_index: int, text: str) -> str:
    """生成可复现的 chunk_id。

    只要 source_type、路径、chunk_index 和文本内容不变，chunk_id 就不变。
    这对后续 citation、回归测试和失败样例定位很重要。
    """

    raw_id = f"{source_type}:{path}:{chunk_index}:{text}"
    digest = sha1(raw_id.encode("utf-8")).hexdigest()[:12]
    return f"{source_type}-{path.stem}-{chunk_index}-{digest}"


def _split_front_matter(text: str) -> tuple[dict[str, str], str]:
    """解析最简单的 Markdown front matter。

    支持形如 `key: value` 的行；解析不到 front matter 时返回空 metadata
    和原始正文。
    """

    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, text

    raw_metadata: dict[str, str] = {}
    for line_index, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            body = "\n".join(lines[line_index + 1 :])
            return raw_metadata, body
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        raw_metadata[key.strip()] = value.strip()

    return {}, text
