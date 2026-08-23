from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import re
from typing import TYPE_CHECKING, Callable, Literal, Protocol, Sequence

from intern_rag.agent.context import build_context
from intern_rag.agent.schemas import BuiltContext
from intern_rag.retrieval import RetrievalResult


ContextMode = Literal[
    "no_memory", "full_history", "recent_window", "summary_recent", "semantic_memory",
    "adaptive",
]
SegmentKind = Literal["system", "query", "profile", "history", "summary", "memory", "evidence"]

if TYPE_CHECKING:
    from intern_rag.agent.context_policy import ContextPlan, ContextSignals


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
    similarity: float | None = None

    def __post_init__(self) -> None:
        if not self.memory_id.strip() or not self.user_id.strip() or not self.content.strip():
            raise ValueError("memory id, user id and content must not be empty")
        if not 0.0 <= self.importance <= 1.0:
            raise ValueError("memory importance must be between 0 and 1")
        if self.version <= 0:
            raise ValueError("memory version must be greater than 0")
        if self.similarity is not None and not 0.0 <= self.similarity <= 1.0:
            raise ValueError("memory similarity must be between 0 and 1")

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
    context_signals: "ContextSignals | None" = None
    context_plan: "ContextPlan | None" = None

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
            "no_memory", "full_history", "recent_window", "summary_recent", "semantic_memory",
            "adaptive",
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
        semantic_similarity: Callable[[str, Sequence[str]], list[float]] | None = None,
        deduplication_threshold: float = 0.92,
    ) -> None:
        self.estimator = estimator or MixedTokenEstimator()
        self.summarizer = summarizer
        self.evidence_compressor = evidence_compressor
        self.semantic_similarity = semantic_similarity
        self.deduplication_threshold = deduplication_threshold

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
        plan: "ContextPlan | None" = None,
        signals: "ContextSignals | None" = None,
    ) -> ManagedContext:
        """按固定 baseline mode 或自适应 ContextPlan 构造完整上下文。

        输入包括当前 Query、分层记忆候选、RAG 检索证据、统一 token 预算和可选 Plan；
        先由 Plan 决定启用层与数量，再对 Profile/Memory/Summary/History 跨层去重，
        最后按优先级装入完整片段并编排 Evidence。预算不足只丢弃完整片段，不静默
        截断事实。输出 ManagedContext，同时保留 signals/plan、kept/dropped 供 Trace。
        """

        if config.mode == "adaptive" and plan is None:
            raise ValueError("adaptive context mode requires a ContextPlan")

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
        if profile is not None and (plan is None or plan.use_profile):
            confirmed = [fact for fact in profile.facts if fact.confirmed]
            if confirmed:
                text = "\n".join(f"{fact.key}: {fact.value}" for fact in confirmed)
                candidates.append(self._segment("profile", f"profile:{profile.version}", text, 90, "confirmed_profile"))

        history_candidates = self._history_segments(
            history, config, history_summary, dropped, fallbacks, plan
        )
        candidates.extend(history_candidates)
        available_memories = sorted(
            (item for item in memories if item.is_available),
            key=lambda item: (
                -float(item.similarity or 0.0) * item.importance,
                -item.importance,
                item.created_at,
                item.memory_id,
            ),
        )
        memory_limit = (
            plan.memory_top_k
            if plan is not None
            else (len(available_memories) if config.mode == "semantic_memory" else 0)
        )
        if memory_limit > 0:
            candidates.extend(
                self._segment(
                    "memory", item.memory_id, item.content, 80,
                    f"semantic_memory:{item.source}:similarity={float(item.similarity or 0.0):.4f}",
                )
                for item in available_memories[:memory_limit]
            )

        candidates = deduplicate_memory_layers(
            candidates,
            dropped,
            similarity=self.semantic_similarity,
            threshold=self.deduplication_threshold,
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
            context_signals=signals,
            context_plan=plan,
        )

    def _history_segments(
        self,
        history: Sequence[ConversationMessage],
        config: ContextEngineConfig,
        history_summary: str | None,
        dropped: list[dict[str, str]],
        fallbacks: list[str],
        plan: "ContextPlan | None",
    ) -> list[ContextSegment]:
        if plan is not None:
            recent_count = plan.recent_history_count
            recent = list(history[-recent_count:]) if recent_count else []
            use_summary = plan.use_summary
        elif config.mode in {"no_memory", "semantic_memory"}:
            return []
        else:
            recent = (
                list(history)
                if config.mode == "full_history"
                else list(history[-config.recent_message_count :])
            )
            use_summary = config.mode == "summary_recent"
        output: list[ContextSegment] = []
        if use_summary:
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
                    "adaptive_recent" if plan is not None else (
                        "full_history" if config.mode == "full_history" else "recent_window"
                    ),
                )
            )
        recent_ids = {item.message_id for item in recent}
        dropped.extend(
            {"segment_id": item.message_id, "reason": "outside_recent_window"}
            for item in history
            if item.message_id not in recent_ids
            and (plan is not None or config.mode == "recent_window")
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


def deduplicate_memory_layers(
    segments: Sequence[ContextSegment],
    dropped: list[dict[str, str]],
    *,
    similarity: Callable[[str, Sequence[str]], list[float]] | None = None,
    threshold: float = 0.92,
) -> list[ContextSegment]:
    """在 Profile/Memory/Summary/History 之间去重并保留优先级更高的片段。

    先按 priority 排序；规范化文本完全相同直接去重，配置 embedding 相似度函数时，
    相似度超过阈值也视为重复。Evidence 不参与此处去重，避免删除 Citation 所需证据。
    """

    ordered = sorted(segments, key=lambda item: (-item.priority, item.segment_id))
    kept: list[ContextSegment] = []
    for segment in ordered:
        if segment.kind == "evidence":
            kept.append(segment)
            continue
        comparable = [item for item in kept if item.kind != "evidence"]
        normalized = " ".join(segment.text.lower().split())
        duplicate = next(
            (
                item for item in comparable
                if " ".join(item.text.lower().split()) == normalized
            ),
            None,
        )
        if duplicate is None and similarity is not None and comparable:
            scores = similarity(segment.text, [item.text for item in comparable])
            best_index = max(range(len(scores)), key=scores.__getitem__)
            if scores[best_index] >= threshold:
                duplicate = comparable[best_index]
        if duplicate is not None:
            dropped.append({
                "segment_id": segment.segment_id,
                "reason": f"cross_layer_duplicate:{duplicate.segment_id}",
            })
            continue
        kept.append(segment)
    return kept
