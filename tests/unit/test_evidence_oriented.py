import json
import tempfile
from dataclasses import replace
from pathlib import Path

from intern_rag.agent import FakeLlmClient, PipelineConfig, RagPipeline, RagRequest
from intern_rag.ingestion import Chunk
from intern_rag.retrieval import AdaptiveRetriever, FakeRerankScorer, RetrievalResult
from intern_rag.retrieval.evidence_oriented import (
    EvidenceOrientedRetriever,
    OrchestratorConfig,
    _RequestState,
)
from intern_rag.retrieval.evidence_plan import (
    EvidenceAssembler,
    EvidencePlanner,
    EvidenceScope,
    EvidenceVerifier,
)
from intern_rag.routing import RouteDecision


def chunk(cid="a", source="project_logs", **meta):
    return Chunk(cid, source, f"{cid}.md", "证据测试项目",
                 "证据测试项目使用 Python 实现检索，并且记录完整链路和验证结果。", meta)


def result(c, **details):
    return RetrievalResult(c.id, 1.0, 1, c, details=details)


class Corpus:
    def __init__(self):
        self.calls = 0

    def __call__(self, query, chunks, top_k=5, source_types=None):
        self.calls += 1
        return [result(c) for c in chunks if source_types is None or c.source_type in source_types][:top_k]


def retriever(config=None):
    backend = Corpus()
    base = AdaptiveRetriever({k: backend for k in ("bm25", "dense", "hybrid")},
                             FakeRerankScorer({}))
    return EvidenceOrientedRetriever(base, config=config or OrchestratorConfig(rerank=False)), backend


def test_route_preferences_are_not_required_evidence():
    plan = EvidencePlanner().plan("解释如何优化召回", {"jd", "resume", "project_logs"})
    assert not plan.requirement.multi_source_required
    assert not plan.requirement.required_source_types
    assert len(plan.slots) == 1
    assert EvidencePlanner().plan("结合 JD 与简历比较匹配度").requirement.required_source_types == ("jd", "resume")


def test_planner_one_call_controlled_fallback():
    calls = []
    def invalid(q):
        calls.append(q)
        return "invalid"
    plan = EvidencePlanner(decompose=invalid).plan("哪个项目证明岗位技能要求？")
    assert len(calls) == 1 and plan.fallback_reason == "JSONDecodeError"


def test_acl_is_enforced_before_and_after_rescue():
    engine, _ = retriever()
    chunks = [chunk("public"), chunk("mine", user_id="u"), chunk("other", user_id="v")]
    scope = EvidenceScope(user_id="u", allowed_chunk_ids=frozenset({"public", "mine"}))
    rows = engine.retrieve_evidence("如何优化检索", chunks, source_types={"jd"}, scope=scope)
    rows = engine.rescue("如何优化检索", chunks, initial_results=rows)
    assert {r.chunk_id for r in rows} == {"public", "mine"}
    assert "other" not in json.dumps(engine.get_last_trace())


def test_empty_acl_and_duplicate_call_budget():
    engine, backend = retriever(OrchestratorConfig(max_calls=1, rerank=False))
    rows = engine.retrieve_evidence("Redis 是什么", [chunk()], scope=EvidenceScope(allowed_sources=frozenset()))
    engine.rescue("Redis 是什么", [chunk()], initial_results=rows)
    engine.rescue("Redis 是什么", [chunk()], initial_results=rows)
    assert not rows and backend.calls == 1


def test_complete_path_atomic_budget_and_post_context_verification():
    plan = EvidencePlanner().plan("哪些项目通过两跳关系证明技能？")
    rows = [result(chunk(cid), path_group_id="g", path_group_size=2,
                   graph_edge_ids="e1|e2", graph_path="Job->Skill->Project", path_valid=1)
            for cid in ("a", "b")]
    bundle = EvidenceAssembler().assemble(plan, rows, 1000, len)
    assert EvidenceVerifier().verify(plan, bundle).ready
    tiny = EvidenceAssembler().assemble(plan, rows, 1, len)
    assert not tiny.results and not EvidenceVerifier().verify(plan, tiny).ready
    half = replace(bundle, results=bundle.results[:1])
    assert not EvidenceVerifier().verify(plan, half).ready


def test_unknown_lexical_and_expired_conflict_not_bypassed():
    plan = EvidencePlanner().plan("Redis是什么")
    bundle = EvidenceAssembler().assemble(plan, [result(chunk())], 1000, len)
    assert EvidenceVerifier().verify(plan, bundle).verdicts[0].status == "unknown"
    plan = EvidencePlanner().plan("当前有效岗位有哪些")
    rows = [result(chunk("old", "jd", status="expired", last_seen_at="2026-01-01"), semantic_channel=1),
            result(chunk("new", "jd", status="active", last_seen_at="2026-10-09"), semantic_channel=1)]
    verdict = EvidenceVerifier().verify(plan, EvidenceAssembler().assemble(plan, rows, 1000, len))
    assert not verdict.ready and not verdict.retryable


def test_conflicting_versions_are_not_sufficient():
    plan = EvidencePlanner().plan("最新岗位职责")
    rows = [result(chunk(cid, "jd", job_id="job", version=1, content_hash=cid), semantic_channel=1)
            for cid in ("a", "b")]
    verification = EvidenceVerifier().verify(plan, EvidenceAssembler().assemble(plan, rows, 1000, len))
    assert any(v.status == "conflicting" for v in verification.verdicts)


def test_pipeline_uses_plan_bundle_verifier_and_one_trace():
    engine, _ = retriever()
    llm = FakeLlmClient([json.dumps({"answer": "项目实现检索。", "cited_chunk_ids": ["a"],
                                   "sufficient": True, "reason": "有证据"})])
    with tempfile.TemporaryDirectory() as directory:
        pipeline = RagPipeline([chunk()], llm, PipelineConfig(model="fake"),
            trace_path=Path(directory) / "trace.jsonl", retriever=engine,
            retrievers={"adaptive": engine}, router=lambda q: RouteDecision("project_explanation", ["project_logs"], []))
        response = pipeline.run(RagRequest(query="解释项目如何检索", retriever="adaptive"))
        assert response.status == "answered"
        trace = pipeline.last_trace
        assert trace.retrieval["decision"]["evidence_plan"]["slots"]
        assert trace.evidence["slot_verification"]["ready"]
        assert trace.context["slot_verification"]["ready"]
        assert len((Path(directory) / "trace.jsonl").read_text().splitlines()) == 1


def test_context_drop_stops_before_llm():
    engine, _ = retriever()
    with tempfile.TemporaryDirectory() as directory:
        pipeline = RagPipeline([chunk()], FakeLlmClient([]), PipelineConfig(model="fake", context_max_chars=1),
            trace_path=Path(directory) / "trace.jsonl", retriever=engine,
            retrievers={"adaptive": engine}, router=lambda q: RouteDecision("project_explanation", ["project_logs"], []))
        response = pipeline.run(RagRequest(query="解释项目如何检索", retriever="adaptive"))
        assert response.status == "insufficient_evidence"
        assert pipeline.last_trace.context["evidence_dropped"]
        assert not pipeline.last_trace.generation


def test_semantic_fact_can_be_ready_but_partial_citations_refuse():
    engine, _ = retriever()
    chunks = [chunk("a", "jd"), chunk("b", "resume")]
    llm = FakeLlmClient([json.dumps({"answer": "两份资料相关。", "cited_chunk_ids": ["a"],
                                   "sufficient": True, "reason": "有证据"})])
    with tempfile.TemporaryDirectory() as directory:
        pipeline = RagPipeline(chunks, llm, PipelineConfig(model="fake"),
            trace_path=Path(directory) / "trace.jsonl", retriever=engine,
            retrievers={"adaptive": engine}, router=lambda q: RouteDecision("resume_match", ["jd", "resume"], []))
        response = pipeline.run(RagRequest(query="结合 JD 和简历比较", retriever="adaptive"))
        assert response.status == "insufficient_evidence"
        assert pipeline.last_trace.validation["citation_slot_failure"]


def test_private_identity_requires_trusted_scope_provider():
    engine, _ = retriever()
    with tempfile.TemporaryDirectory() as directory:
        pipeline = RagPipeline([chunk("private", user_id="u")], FakeLlmClient([]), PipelineConfig(model="fake"),
            trace_path=Path(directory) / "trace.jsonl", retriever=engine,
            retrievers={"adaptive": engine}, router=lambda q: RouteDecision("project_explanation", ["project_logs"], []))
        response = pipeline.run(RagRequest(query="解释项目", retriever="adaptive", user_id="u", session_id="s"))
        assert response.status == "insufficient_evidence"
        assert not pipeline.last_trace.retrieval["chunk_ids"]


def test_empty_results_rescue_is_bounded_and_final_unknown_refuses():
    engine, backend = retriever()
    with tempfile.TemporaryDirectory() as directory:
        pipeline = RagPipeline([], FakeLlmClient([]), PipelineConfig(model="fake"),
            trace_path=Path(directory) / "trace.jsonl", retriever=engine,
            retrievers={"adaptive": engine}, router=lambda q: RouteDecision("project_explanation", ["project_logs"], []))
        response = pipeline.run(RagRequest(query="解释项目", retriever="adaptive"))
        assert response.status == "insufficient_evidence"
        assert sum(a["type"] == "evidence_gap_rescue" for a in pipeline.last_trace.attempts) == 1
        assert backend.calls <= 2


def test_conflict_is_not_rescued_or_generated():
    engine, backend = retriever()
    with tempfile.TemporaryDirectory() as directory:
        pipeline = RagPipeline([chunk(conflicting=True)], FakeLlmClient([]), PipelineConfig(model="fake"),
            trace_path=Path(directory) / "trace.jsonl", retriever=engine,
            retrievers={"adaptive": engine}, router=lambda q: RouteDecision("project_explanation", ["project_logs"], []))
        response = pipeline.run(RagRequest(query="解释项目", retriever="adaptive"))
        assert response.status == "insufficient_evidence"
        assert pipeline.last_trace.evidence["reason"] == "slot_conflicting"
        assert backend.calls == 1


def test_shared_paths_pay_once_and_bad_path_does_not_delete_good_path():
    plan = EvidencePlanner().plan("哪些项目通过两跳关系证明技能？")
    groups = [
        {"group_id": "p1", "chunk_ids": ["a", "b"], "path": "J->S->P", "edge_ids": ["e1", "e2"]},
        {"group_id": "p2", "chunk_ids": ["b", "c"], "path": "S->P->E", "edge_ids": ["e2", "e3"]},
        {"group_id": "bad", "chunk_ids": ["b", "absent"], "path": "S->X", "edge_ids": ["e4"]},
    ]
    rows = [result(chunk(cid), path_groups_json=json.dumps(groups)) for cid in ("a", "b", "c")]
    bundle = EvidenceAssembler().assemble(plan, rows, 1000, len, format_result=lambda r: r.chunk.text)
    assert {g.group_id for g in bundle.groups} == {"p1", "p2"}
    assert bundle.token_count == sum(len(f"[evidence:{r.chunk_id}]\n{r.chunk.text}") for r in rows)
    assert EvidenceVerifier().verify(plan, bundle).ready


def test_topk_does_not_discard_members_before_group_recognition():
    engine, _ = retriever()
    rows = [replace(result(chunk(str(i)), semantic_channel=1), rank=i + 1) for i in range(7)]
    group = {"group_id": "tail", "chunk_ids": ["5", "6"], "path": "J->S->P", "edge_ids": ["e1", "e2"]}
    rows[5:] = [replace(r, details={"path_groups_json": json.dumps([group])}) for r in rows[5:]]
    state = _RequestState(EvidencePlanner().plan("哪些项目通过两跳关系证明技能？"), candidates=rows)
    engine._state.set(state)
    out = engine._finish(state, 1, exhausted=True)
    assert {"5", "6"} <= {r.chunk_id for r in out}
    assert len(engine.get_candidates()) == 7


def test_rerank_head_does_not_delete_tail_candidates():
    engine, _ = retriever(OrchestratorConfig(candidate_k=2, rerank=True))
    rows = [replace(result(chunk(str(i)), semantic_channel=1), rank=i + 1) for i in range(4)]
    state = _RequestState(EvidencePlanner().plan("解释检索"), candidates=rows)
    engine._state.set(state)
    engine._finish(state, 1, exhausted=True)
    assert len(engine.get_candidates()) == 4
