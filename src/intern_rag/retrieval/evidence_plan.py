"""证据任务契约、轻量规划、完整组装和结构就绪验证。

验证的是证据是否具备回答所需结构，不冒充自然语言事实蕴含判分。
在线流程不接收 benchmark 标签。授权来自服务端，路由仅表达检索偏好。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
import re
from typing import Callable, Literal

from intern_rag.ingestion import Chunk
from intern_rag.retrieval.adaptive import EvidenceRequirement, QueryAnalyzer
from intern_rag.retrieval.base import RetrievalResult


@dataclass(frozen=True)
class EvidenceScope:
    """服务端可信授权；空集合拒绝所有，None 表示当前单库不额外限制。"""

    user_id: str | None = None
    tenant_id: str | None = None
    allowed_sources: frozenset[str] | None = None
    allowed_chunk_ids: frozenset[str] | None = None

    def permits(self, chunk: Chunk) -> bool:
        """同时检查来源、文档 ACL 和用户/租户归属，不接受 Query 修改权限。"""
        meta = chunk.metadata
        return (
            (self.allowed_sources is None or chunk.source_type in self.allowed_sources)
            and (self.allowed_chunk_ids is None or chunk.id in self.allowed_chunk_ids)
            and (not meta.get("tenant_id") or meta["tenant_id"] == self.tenant_id)
            and (not meta.get("user_id") or meta["user_id"] == self.user_id)
        )


@dataclass(frozen=True)
class EvidenceSlot:
    """一项可追踪证据需求；满足条件不是简单的相关分数阈值。"""

    slot_id: str
    kind: Literal["fact", "relation", "source", "freshness"]
    subquery: str
    anchors: tuple[str, ...] = ()
    required_sources: tuple[str, ...] = ()
    min_hops: int = 1
    conditions: tuple[str, ...] = ("substantive_text", "semantic_or_anchor_support")


@dataclass(frozen=True)
class EvidencePlan:
    """问题、软路由、可信授权和真正必要证据分别存放。"""

    query: str
    slots: tuple[EvidenceSlot, ...]
    requirement: EvidenceRequirement
    route_preferences: tuple[str, ...]
    scope: EvidenceScope
    version: str = "evidence-plan-v1"
    planner: str = "deterministic"
    fallback_reason: str | None = None

    def to_trace(self) -> dict[str, object]:
        """保存规划和权限边界，但不暴露用户身份或完整 ACL 列表。"""
        result = asdict(self)
        result["scope"] = {
            "source_restricted": self.scope.allowed_sources is not None,
            "chunk_restricted": self.scope.allowed_chunk_ids is not None,
            "allowed_chunk_count": (
                len(self.scope.allowed_chunk_ids)
                if self.scope.allowed_chunk_ids is not None else None
            ),
        }
        return result


class EvidencePlanner:
    """从 Query 生成证据槽；可注入最多一次结构化分解，失败回退原规划。"""

    def __init__(self, analyzer: QueryAnalyzer | None = None,
                 decompose: Callable[[str], str] | None = None) -> None:
        self.analyzer = analyzer or QueryAnalyzer()
        self.decompose = decompose

    def plan(self, query: str, route_preferences: set[str] | None = None,
             scope: EvidenceScope | None = None) -> EvidencePlan:
        """输入问题和软路由，提取标题/来源/关系/时效槽，输出有界计划。

        必要来源只从 Query 得到；复合问题最多增加三条模型子问题。非法 JSON、
        非字符串或异常不影响主链，原始确定性计划保留且记录 fallback。
        """
        features = self.analyzer.analyze(query, None)
        requirement = self.analyzer.classify_evidence_need(features)
        anchors = tuple(re.findall(r"[《“\"]([^》”\"]{2,100})[》”\"]", query))
        kind = "relation" if requirement.graph_required else "fact"
        hops = 3 if any(word in query for word in ("三跳", "3-hop")) else (
            2 if any(word in query for word in ("两跳", "二跳", "2-hop")) else 1
        )
        slots = [EvidenceSlot("primary", kind, query, anchors, min_hops=hops)]
        for source in requirement.required_source_types:
            slots.append(EvidenceSlot(
                f"source:{source}", "source", query, (), (source,)
            ))
        if any(word in query for word in ("最新", "当前", "仍在招聘", "有效岗位", "时效")):
            slots.append(EvidenceSlot("freshness", "freshness", query, anchors,
                                      ("jd",), conditions=("active_status", "seen_date")))
        plan = EvidencePlan(query, tuple(slots), requirement,
                            tuple(sorted(route_preferences or ())), scope or EvidenceScope())
        if self.decompose is not None and requirement.graph_required:
            try:
                raw = json.loads(self.decompose(query))
                subqueries = raw["subqueries"]
                if not isinstance(subqueries, list) or not 1 <= len(subqueries) <= 3:
                    raise ValueError("subqueries must contain 1..3 strings")
                if any(not isinstance(q, str) or not q.strip() or len(q) > 500 for q in subqueries):
                    raise ValueError("invalid subquery")
                extra = tuple(EvidenceSlot(f"decomposed:{i}", "fact", q)
                              for i, q in enumerate(dict.fromkeys(subqueries)))
                plan = replace(plan, slots=(*plan.slots, *extra), planner="structured_once")
            except Exception as error:
                plan = replace(plan, fallback_reason=type(error).__name__)
        return plan


@dataclass(frozen=True)
class EvidenceGroup:
    """关系路径或跨源组合，预算裁剪时作为不可拆分单位。"""

    group_id: str
    chunk_ids: tuple[str, ...]
    path: str = ""
    edge_ids: tuple[str, ...] = ()
    complete: bool = True


@dataclass(frozen=True)
class EvidenceBundle:
    """已装箱证据、槽绑定和未装入原因；出处仍在原 RetrievalResult 中。"""

    results: tuple[RetrievalResult, ...]
    bindings: dict[str, tuple[str, ...]]
    groups: tuple[EvidenceGroup, ...]
    dropped: tuple[dict[str, object], ...]
    token_count: int

    def to_trace(self) -> dict[str, object]:
        """只序列化证据索引和组，不重复保存整份原文。"""
        return {"chunk_ids": [r.chunk_id for r in self.results],
                "bindings": self.bindings, "groups": [asdict(g) for g in self.groups],
                "dropped": list(self.dropped), "token_count": self.token_count,
                "provenance": {r.chunk_id: r.chunk.source_path for r in self.results}}


def slot_candidates(slot: EvidenceSlot, results: list[RetrievalResult]) -> list[RetrievalResult]:
    """按显式来源和标题锚点绑定候选；不使用评测 relevant ids。"""
    return [r for r in results
            if (not slot.required_sources or r.chunk.source_type in slot.required_sources)
            and (slot.kind == "relation" or not slot.anchors or any(a.lower() in (r.chunk.title or "").lower()
                                        or a.lower() in r.chunk.text.lower() for a in slot.anchors))]


class EvidenceAssembler:
    """将结果绑定到槽，并按完整路径/跨源组进行 Token 装箱。"""

    def assemble(self, plan: EvidencePlan, results: list[RetrievalResult],
                 token_budget: int, count_tokens: Callable[[str], int]) -> EvidenceBundle:
        """输入计划和候选，去重、分组、完整性检查后整组保留或整组丢弃。

        预算包括 ID、标题、来源和正文的保守格式开销；ContextEngine 仍负责最终
        Prompt 预算，Pipeline 在其裁剪后再验证，防止装箱后证据再次丢失。
        """
        unique = {r.chunk_id: r for r in reversed(results) if plan.scope.permits(r.chunk)}
        ordered = sorted(unique.values(), key=lambda r: (r.rank, r.chunk_id))
        by_group: dict[str, list[RetrievalResult]] = {}
        for r in ordered:
            key = str(r.details.get("path_group_id") or r.chunk_id)
            by_group.setdefault(key, []).append(r)
        if plan.requirement.multi_source_required and not plan.requirement.graph_required:
            selected = {r.chunk_id for slot in plan.slots if slot.kind == "source"
                        for r in slot_candidates(slot, ordered)[:1]}
            if selected:
                by_group = {key: [r for r in rows if r.chunk_id not in selected]
                            for key, rows in by_group.items()}
                by_group["cross-source"] = [r for r in ordered if r.chunk_id in selected]
        kept: list[RetrievalResult] = []
        groups: list[EvidenceGroup] = []
        dropped: list[dict[str, object]] = []
        tokens = 0
        for group_id, rows in sorted(by_group.items(), key=lambda p: min(
                (r.rank for r in p[1]), default=10**9)):
            if not rows:
                continue
            expected = max(int(r.details.get("path_group_size") or 1) for r in rows)
            complete = len(rows) >= expected
            group = EvidenceGroup(group_id, tuple(r.chunk_id for r in rows),
                                  str(rows[0].details.get("graph_path") or ""),
                                  tuple(str(rows[0].details.get("graph_edge_ids") or "").split("|"))
                                  if rows[0].details.get("graph_edge_ids") else (), complete)
            cost = sum(count_tokens(f"[{r.chunk_id}] {r.chunk.source_type} {r.chunk.title}\n"
                                    f"{r.chunk.source_path}\n{r.chunk.text}\n") + 16 for r in rows)
            groups.append(group)
            if not complete or tokens + cost > token_budget:
                dropped.append({"group_id": group_id, "chunk_ids": list(group.chunk_ids),
                                "reason": "incomplete_group" if not complete else "token_budget"})
                continue
            tokens += cost
            kept.extend(rows)
        kept = [replace(r, rank=i) for i, r in enumerate(kept, 1)]
        return EvidenceBundle(tuple(kept), {
            slot.slot_id: tuple(r.chunk_id for r in slot_candidates(slot, kept))
            for slot in plan.slots
        }, tuple(groups), tuple(dropped), tokens)


@dataclass(frozen=True)
class SlotVerdict:
    """结构就绪判定与原始证据索引；unknown 不会自动变为满足。"""

    slot_id: str
    status: Literal["satisfied", "missing", "conflicting", "unknown"]
    evidence_ids: tuple[str, ...]
    reason: str


@dataclass(frozen=True)
class EvidenceVerification:
    """逐槽结果和可补救性，交给 Controller 决定生成/补检索/拒答。"""

    verdicts: tuple[SlotVerdict, ...]
    ready: bool
    retryable: bool


class EvidenceVerifier:
    """校验内容、来源、完整图组、显式冲突与岗位时效，不把相关性当证明。"""

    def verify(self, plan: EvidencePlan, bundle: EvidenceBundle,
               exhausted: bool = False) -> EvidenceVerification:
        """逐槽检查；显式冲突不可通过加 Chunk 消除，缺失可在预算内补一次。

        事实槽要求有实质正文，并有标题实体锚点或语义通道召回依据；这是生成
        前的结构门禁，不是答案蕴含验证。正文是否支持最终 claim 仍由 Grounding 审核。
        """
        results = list(bundle.results)
        verdicts = []
        for slot in plan.slots:
            rows = slot_candidates(slot, results)
            ids = tuple(r.chunk_id for r in rows)
            status, reason = "satisfied", "structural evidence ready; not a factual entailment verdict"
            if not rows:
                status, reason = "missing", "no evidence bound to slot"
            elif any(r.chunk.metadata.get("conflicting") is True for r in rows):
                status, reason = "conflicting", "explicit provenance conflict"
            elif self._conflicting_versions(rows):
                status, reason = "conflicting", "same posting/version has conflicting content or status"
            elif slot.kind == "freshness":
                active = [r for r in rows if r.chunk.metadata.get("status") == "active"
                          and r.chunk.metadata.get("last_seen_at")]
                if any(r.chunk.metadata.get("status") == "expired" for r in rows):
                    status, reason = "conflicting", "expired posting cannot establish current availability"
                elif not active:
                    status, reason = "unknown", "active status and observation date are required"
            elif slot.kind == "relation":
                valid = [g for g in bundle.groups if g.complete and len(g.edge_ids) >= slot.min_hops
                         and set(g.chunk_ids) <= set(ids) and g.path]
                if not valid:
                    status, reason = "missing", "complete relation group of required hop length missing"
            elif not any(len(r.chunk.text.strip()) >= 20 for r in rows):
                status, reason = "unknown", "evidence contains only labels or fragments"
            elif slot.kind == "fact" and not slot.anchors and not any(
                    r.details.get("semantic_channel") == 1 or r.details.get("dense_rank")
                    for r in rows):
                status, reason = "unknown", "lexical relevance alone cannot establish factual readiness"
            verdicts.append(SlotVerdict(slot.slot_id, status, ids, reason))
        ready = bool(verdicts) and all(v.status == "satisfied" for v in verdicts)
        retryable = not exhausted and not ready and not any(v.status == "conflicting" for v in verdicts)
        return EvidenceVerification(tuple(verdicts), ready, retryable)

    @staticmethod
    def _conflicting_versions(rows: list[RetrievalResult]) -> bool:
        """仅比较同岗位同版本且不同来源的显式摘要，避免把不同 Chunk 当冲突。"""
        seen: dict[tuple[str, str], set[tuple[str, str]]] = {}
        for r in rows:
            m = r.chunk.metadata
            if m.get("job_id") and m.get("content_hash"):
                key = (str(m["job_id"]), str(m.get("version", "")))
                seen.setdefault(key, set()).add((str(m["content_hash"]), str(m.get("status", "unknown"))))
        return any(len(values) > 1 for values in seen.values())
