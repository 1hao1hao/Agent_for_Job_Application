from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import re
from typing import Literal, Protocol, Sequence

from intern_rag.agent.context import build_context
from intern_rag.agent.schemas import BuiltContext
from intern_rag.retrieval import RetrievalResult


ContextMode = Literal[
    "no_memory", "full_history", "recent_window", "summary_recent", "semantic_memory"
]
SegmentKind = Literal["system", "query", "profile", "history", "summary", "memory", "evidence"]


class TokenEstimator(Protocol):
    """估算文本 token 数；生产和实验可以注入模型对应 tokenizer。"""

    @property
    def version(self) -> str: ...

    def count(self, text: str) -> int:
        """返回非负 token 估计值。"""


class TextSummarizer(Protocol):
    """把历史消息压缩成保留事实的摘要。"""

    def summarize(self, messages: Sequence["ConversationMessage"]) -> str:
        """返回摘要；失败时由 Context Engine 回退。"""


class EvidenceCompressor(Protocol):
    """压缩单条证据，但不能生成输入中不存在的事实。"""

    def compress(self, text: str, query: str) -> str:
        """返回压缩文本；空结果或异常触发原文回退。"""


@dataclass(frozen=True)
class MixedTokenEstimator:
    """面向中英文混合文本的确定性近似 tokenizer。

    中文字符、英文单词、数字和标点分别计为一个 token。它不冒充具体模型 tokenizer，
    但适合离线同集比较；version 会进入 Trace 和实验配置。
    """

    version: str = "mixed-lexical-token-v1"

    def count(self, text: str) -> int:
        return len(re.findall(r"[\u4e00-\u9fff]|[A-Za-z]+|\d+|[^\s]", text))


@dataclass(frozen=True)
class ConversationMessage:
    """一条带用户和会话边界的历史消息。"""

    message_id: str
    session_id: str
    user_id: str
    role: Literal["user", "assistant", "system"]
    content: str
    created_at: str


@dataclass(frozen=True)
class ProfileFact:
    """用户显式确认或从确认材料导入的稳定画像事实。"""

    key: str
    value: str
    source: str
    confirmed: bool = True


@dataclass(frozen=True)
class UserProfile:
    """带乐观锁版本的用户画像；摘要不能反向覆盖它。"""

    user_id: str
    facts: tuple[ProfileFact, ...]
    version: int
    updated_at: str


@dataclass(frozen=True)
class MemoryItem:
    """可检索长期记忆，保存作用域、来源、版本、TTL 和确认状态。"""

    memory_id: str
    user_id: str
    memory_type: Literal["fact", "preference", "experience", "decision"]
    content: str
    source: str
    importance: float
    created_at: str
    version: int = 1
    session_id: str | None = None
    expires_at: str | None = None
    confirmed: bool = True
    active: bool = True

    def __post_init__(self) -> None:
        if not self.memory_id.strip() or not self.user_id.strip() or not self.content.strip():
            raise ValueError("memory id, user id and content must not be empty")
        if not 0.0 <= self.importance <= 1.0:
            raise ValueError("memory importance must be between 0 and 1")
        if self.version <= 0:
            raise ValueError("memory version must be greater than 0")

    @property
    def is_available(self) -> bool:
        """未确认、已删除或超过 TTL 的记忆不得进入 Context。"""

        if not self.confirmed or not self.active:
            return False
        if self.expires_at is None:
            return True
        expires_at = datetime.fromisoformat(self.expires_at.replace("Z", "+00:00"))
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)
        return expires_at > datetime.now(timezone.utc)


@dataclass(frozen=True)
class ContextSegment:
    """ManagedContext 中一个可审计片段。"""

    segment_id: str
    kind: SegmentKind
    text: str
    token_count: int
    priority: int
    reason: str


@dataclass(frozen=True)
class ManagedContext:
    """Context Engine 输出，包含完整模型文本、证据契约和预算决策。"""

    query: str
    text: str
    segments: tuple[ContextSegment, ...]
    evidence: BuiltContext
    token_count: int
    token_budget: int
    reserved_token_count: int
    token_estimator_version: str
    mode: ContextMode
    kept_ids: tuple[str, ...]
    dropped: tuple[dict[str, str], ...]
    recalled_memory_ids: tuple[str, ...]
    compression_fallbacks: tuple[str, ...] = ()

    def as_built_context(self) -> BuiltContext:
        """复用 Citation Validator 所需的 BuiltContext 证据字段。"""

        return BuiltContext(
            query=self.query,
            text=self.text,
            items=self.evidence.items,
            used_chunk_ids=self.evidence.used_chunk_ids,
            skipped_chunk_ids=self.evidence.skipped_chunk_ids,
            char_count=len(self.text),
            max_chars=len(self.text),
            selection_strategy=f"managed:{self.mode}",
            covered_source_types=self.evidence.covered_source_types,
            missing_source_types=self.evidence.missing_source_types,
        )


@dataclass(frozen=True)
class ContextEngineConfig:
    """完整 Prompt 预算、历史窗口与压缩策略配置。"""

    token_budget: int = 1800
    recent_message_count: int = 4
    evidence_char_budget: int = 4000
    mode: ContextMode = "recent_window"
    evidence_strategy: str = "source_balanced"
    compress_evidence: bool = False
    reserved_token_count: int = 0

    def __post_init__(self) -> None:
        if self.token_budget <= 0 or self.recent_message_count < 0:
            raise ValueError("invalid context engine budget")
        if self.reserved_token_count < 0 or self.reserved_token_count >= self.token_budget:
            raise ValueError("reserved token count must be within the total token budget")
        if self.mode not in {
            "no_memory", "full_history", "recent_window", "summary_recent", "semantic_memory"
        }:
            raise ValueError("unknown context mode")
        if self.evidence_strategy not in {"rank_prefix", "source_balanced"}:
            raise ValueError("unknown evidence strategy")


@dataclass(frozen=True)
class ContextInputs:
    """Pipeline 从会话服务取得的 Context Engine 可选输入。"""

    profile: UserProfile | None = None
    history: tuple[ConversationMessage, ...] = ()
    memories: tuple[MemoryItem, ...] = ()
    history_summary: str | None = None
    history_source: str = "none"


class ContextBudgetError(ValueError):
    """系统约束和当前 Query 已超过预算，不能静默裁剪。"""


class ContextEngine:
    """在完整 Prompt token 预算内编排画像、历史、记忆和检索证据。

    处理顺序是：固定保留 system/query；按 mode 选择 Profile、摘要/最近历史和已确认
    Memory；去重并按来源/分数/rank 选择完整证据；最后按优先级装入预算。可选压缩器
    失败时回退原文并记录原因。输出 ManagedContext，供 Generator 和 Trace 共同消费。
    """

    def __init__(
        self,
        estimator: TokenEstimator | None = None,
        *,
        summarizer: TextSummarizer | None = None,
        evidence_compressor: EvidenceCompressor | None = None,
    ) -> None:
        self.estimator = estimator or MixedTokenEstimator()
        self.summarizer = summarizer
        self.evidence_compressor = evidence_compressor

    def build(
        self,
        *,
        query: str,
        system_prompt: str,
        retrieved_results: list[RetrievalResult],
        config: ContextEngineConfig,
        required_source_types: Sequence[str] = (),
        profile: UserProfile | None = None,
        history: Sequence[ConversationMessage] = (),
        memories: Sequence[MemoryItem] = (),
        history_summary: str | None = None,
    ) -> ManagedContext:
        """构造完整上下文并记录每个保留、裁剪和压缩回退决定。"""

        fixed = [
            self._segment("system", "system", system_prompt, 100, "required"),
            self._segment("query", "query", query, 100, "required"),
        ]
        fixed_tokens = sum(item.token_count for item in fixed)
        managed_budget = config.token_budget - config.reserved_token_count
        if fixed_tokens > managed_budget:
            raise ContextBudgetError("system prompt and current query exceed token budget")

        candidates: list[ContextSegment] = []
        dropped: list[dict[str, str]] = []
        fallbacks: list[str] = []
        if profile is not None:
            confirmed = [fact for fact in profile.facts if fact.confirmed]
            if confirmed:
                text = "\n".join(f"{fact.key}: {fact.value}" for fact in confirmed)
                candidates.append(self._segment("profile", f"profile:{profile.version}", text, 90, "confirmed_profile"))

        history_candidates = self._history_segments(
            history, config, history_summary, dropped, fallbacks
        )
        candidates.extend(history_candidates)
        available_memories = sorted(
            (item for item in memories if item.is_available),
            key=lambda item: (-item.importance, item.created_at, item.memory_id),
        )
        if config.mode == "semantic_memory":
            candidates.extend(
                self._segment("memory", item.memory_id, item.content, 80, f"semantic_memory:{item.source}")
                for item in available_memories
            )

        evidence = build_context(
            query,
            _deduplicate_results(retrieved_results),
            max_chars=config.evidence_char_budget,
            strategy=config.evidence_strategy,  # type: ignore[arg-type]
            required_source_types=required_source_types,
        )
        for item in evidence.items:
            evidence_body = item.text.strip()
            if config.compress_evidence and self.evidence_compressor is not None:
                try:
                    compressed = self.evidence_compressor.compress(evidence_body, query).strip()
                    if compressed:
                        evidence_body = compressed
                    else:
                        fallbacks.append(f"evidence:{item.chunk_id}:empty_compression")
                except Exception:
                    fallbacks.append(f"evidence:{item.chunk_id}:compression_error")
            # 压缩器只处理正文，citation 所需的结构化头始终由 Engine 重建。
            text = (
                f"chunk_id: {item.chunk_id}\nsource_type: {item.source_type}\n"
                f"title: {item.title}\nrank: {item.rank}\nscore: {item.score:.6f}\n"
                f"text:\n{evidence_body}"
            )
            candidates.append(
                self._segment("evidence", item.chunk_id, text, 70, f"rank={item.rank},score={item.score:.6f}")
            )

        kept = list(fixed)
        used_tokens = fixed_tokens
        for segment in sorted(candidates, key=lambda item: (-item.priority, item.segment_id)):
            if used_tokens + segment.token_count > managed_budget:
                dropped.append({"segment_id": segment.segment_id, "reason": "token_budget"})
                continue
            kept.append(segment)
            used_tokens += segment.token_count
        text = "\n\n".join(_format_segment(segment) for segment in kept)
        return ManagedContext(
            query=query,
            text=text,
            segments=tuple(kept),
            evidence=evidence,
            token_count=used_tokens + config.reserved_token_count,
            token_budget=config.token_budget,
            reserved_token_count=config.reserved_token_count,
            token_estimator_version=self.estimator.version,
            mode=config.mode,
            kept_ids=tuple(item.segment_id for item in kept),
            dropped=tuple(dropped),
            recalled_memory_ids=tuple(
                item.segment_id for item in kept if item.kind == "memory"
            ),
            compression_fallbacks=tuple(fallbacks),
        )

    def _history_segments(
        self,
        history: Sequence[ConversationMessage],
        config: ContextEngineConfig,
        history_summary: str | None,
        dropped: list[dict[str, str]],
        fallbacks: list[str],
    ) -> list[ContextSegment]:
        if config.mode in {"no_memory", "semantic_memory"}:
            return []
        recent = (
            list(history)
            if config.mode == "full_history"
            else list(history[-config.recent_message_count :])
        )
        output: list[ContextSegment] = []
        if config.mode == "summary_recent":
            summary = history_summary
            if summary is None and self.summarizer is not None:
                try:
                    summary = self.summarizer.summarize(history[:-len(recent)] if recent else history)
                except Exception:
                    fallbacks.append("history:summary_error")
            if summary:
                output.append(self._segment("summary", "history-summary", summary, 60, "compressed_history"))
        for message in recent:
            output.append(
                self._segment(
                    "history", message.message_id, f"{message.role}: {message.content}", 50,
                    "full_history" if config.mode == "full_history" else "recent_window",
                )
            )
        recent_ids = {item.message_id for item in recent}
        dropped.extend(
            {"segment_id": item.message_id, "reason": "outside_recent_window"}
            for item in history
            if item.message_id not in recent_ids and config.mode == "recent_window"
        )
        return output

    def _segment(self, kind: SegmentKind, segment_id: str, text: str, priority: int, reason: str) -> ContextSegment:
        normalized = text.strip()
        return ContextSegment(segment_id, kind, normalized, self.estimator.count(normalized), priority, reason)


def _deduplicate_results(results: Sequence[RetrievalResult]) -> list[RetrievalResult]:
    """按 chunk id 和规范化文本去重，保留排名更高的候选。"""

    output: list[RetrievalResult] = []
    ids: set[str] = set()
    texts: set[str] = set()
    for result in sorted(results, key=lambda item: item.rank):
        normalized = " ".join(result.chunk.text.lower().split())
        if result.chunk_id in ids or normalized in texts:
            continue
        ids.add(result.chunk_id)
        texts.add(normalized)
        output.append(result)
    return output


def _format_segment(segment: ContextSegment) -> str:
    return f"[{segment.kind}:{segment.segment_id}]\n{segment.text}"
