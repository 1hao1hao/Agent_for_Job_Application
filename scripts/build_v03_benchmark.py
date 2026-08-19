from __future__ import annotations

from collections import defaultdict
from itertools import combinations
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from intern_rag.evaluation import load_chunks_jsonl  # noqa: E402
from intern_rag.evaluation.knowledge_dataset import (  # noqa: E402
    KnowledgeEvaluationCase,
    validate_knowledge_dataset,
)
from intern_rag.graph import load_knowledge_graph  # noqa: E402


CATEGORIES = (
    "single_source",
    "cross_source",
    "semantic_paraphrase",
    "hard_negative",
    "unanswerable",
    "freshness_conflict",
    "two_hop",
    "three_hop",
)


def main() -> int:
    """从真实 v0.3 Chunk 与图路径构造 240 条可校验候选标签。"""

    chunks = load_chunks_jsonl(ROOT / "data/processed/chunks/evalrag_v0.3.jsonl")
    graph = load_knowledge_graph(
        ROOT / "data/processed/graphs/evalrag_v0.3/job-skill-experience-v0.2.json"
    )
    chunk_by_id = {chunk.id: chunk for chunk in chunks}
    paths = _enumerate_paths(graph)
    cases: list[KnowledgeEvaluationCase] = []
    for category in CATEGORIES:
        category_cases = _build_category_cases(
            category, chunks, chunk_by_id, graph, paths
        )
        if len(category_cases) < 30:
            raise ValueError(f"{category} only produced {len(category_cases)} cases")
        cases.extend(category_cases[:30])

    ordered: list[KnowledgeEvaluationCase] = []
    for index, case in enumerate(cases):
        ordered.append(
            KnowledgeEvaluationCase(
                **{
                    **case.to_dict(),
                    "split": "test" if index % 3 == 2 else "dev",
                }
            )
        )
    validation = validate_knowledge_dataset(
        ordered,
        available_chunk_ids=set(chunk_by_id),
        available_edge_ids={edge.edge_id for edge in graph.edges},
    )
    output = ROOT / "data/evaluation/evalrag_v0.3.jsonl"
    output.write_text(
        "".join(json.dumps(case.to_dict(), ensure_ascii=False) + "\n" for case in ordered),
        encoding="utf-8",
    )
    validation_path = ROOT / "data/evaluation/evalrag_v0.3_validation.json"
    validation_path.write_text(
        json.dumps(validation.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(validation.to_dict(), ensure_ascii=False, indent=2))
    return 0 if validation.is_valid else 1


def _build_category_cases(category, chunks, chunk_by_id, graph, paths):
    cases: list[KnowledgeEvaluationCase] = []
    if category in {"two_hop", "three_hop"}:
        hop_count = 2 if category == "two_hop" else 3
        for path_index, path in enumerate(paths[hop_count]):
            node_ids, edges = path
            nodes = graph.node_by_id()
            evidence = _path_chunks(edges, chunk_by_id)
            if not evidence:
                continue
            start, end = nodes[node_ids[0]], nodes[node_ids[-1]]
            cases.append(
                _case(
                    category,
                    len(cases),
                    f"请沿知识关系说明“{start.name}”与“{end.name}”之间的{hop_count}跳联系（路径{path_index + 1}）。",
                    evidence,
                    (start.name, end.name),
                    entities=(start.name, end.name),
                    relations=tuple(edge.edge_type for edge in edges),
                    edge_ids=tuple(edge.edge_id for edge in edges),
                )
            )
            if len(cases) >= 30:
                break
        return cases

    eligible = [
        chunk for chunk in chunks
        if len(chunk.text) >= 120 and int(chunk.metadata.get("chunk_index", 0)) == 0
    ]
    if category == "freshness_conflict":
        eligible = [chunk for chunk in eligible if chunk.source_type == "jd"]
    if category == "cross_source":
        node_sources: list[tuple[str, tuple[str, str], list]] = []
        for node in graph.nodes:
            values = [chunk_by_id[item] for item in node.chunk_ids if item in chunk_by_id]
            by_source = defaultdict(list)
            for chunk in values:
                by_source[chunk.source_type].append(chunk)
            if len(by_source) >= 2:
                for source_pair in combinations(sorted(by_source), 2):
                    selected = [by_source[key][0] for key in source_pair]
                    node_sources.append((node.name, source_pair, selected))
        for name, source_pair, selected in node_sources[:30]:
            cases.append(
                _case(
                    category,
                    len(cases),
                    f"{source_pair[0]} 与 {source_pair[1]} 分别如何描述“{name}”，它们能提供哪些互补证据？",
                    selected,
                    (name,),
                    entities=(name,),
                )
            )
        return cases

    for chunk in eligible:
        if len(cases) >= 30:
            break
        if category == "single_source":
            query = f"根据《{chunk.title}》说明这份资料的核心要求或结论。"
            answerable = True
        elif category == "semantic_paraphrase":
            query = f"不使用原文标题措辞，概括《{chunk.title}》主要解决的实际问题。"
            answerable = True
        elif category == "freshness_conflict":
            query = f"《{chunk.title}》这条岗位资料的状态、发布时间或采集时间是什么？"
            answerable = True
        elif category == "hard_negative":
            query = f"《{chunk.title}》是否证明候选人已经部署量子芯片驱动的十亿节点生产图？"
            answerable = False
        else:
            query = f"资料库是否公开了编号 UNPUBLISHED-{len(cases):03d} 的内部薪资审批名单？"
            answerable = False
        points = (
            tuple(part for part in (chunk.title, str(chunk.metadata.get("status", ""))) if part)
            if answerable else ()
        )
        cases.append(
            _case(
                category,
                len(cases),
                query,
                [chunk] if answerable else [],
                points,
                answerable=answerable,
            )
        )
    return cases


def _case(
    category,
    index,
    query,
    evidence,
    points,
    *,
    answerable=True,
    entities=(),
    relations=(),
    edge_ids=(),
):
    return KnowledgeEvaluationCase(
        case_id=f"v03_{category}_{index + 1:03d}",
        query=query,
        category=category,
        split="dev",
        expected_sources=tuple(sorted({chunk.source_type for chunk in evidence})),
        relevant_chunk_ids=tuple(chunk.id for chunk in evidence),
        expected_points=tuple(points),
        answerable=answerable,
        expected_entities=tuple(entities),
        expected_relations=tuple(relations),
        graph_edge_ids=tuple(edge_ids),
        review_method="corpus_grounded_ai_assisted",
        human_reviewed=False,
    )


def _enumerate_paths(graph):
    adjacency = defaultdict(list)
    for edge in graph.edges:
        adjacency[edge.source_node_id].append((edge.target_node_id, edge))
        adjacency[edge.target_node_id].append((edge.source_node_id, edge))
    output = {2: [], 3: []}
    seen = {2: set(), 3: set()}
    for start in sorted(adjacency):
        stack = [(start, (start,), ())]
        while stack:
            node, node_path, edge_path = stack.pop()
            hops = len(edge_path)
            if hops in output:
                signature = tuple(sorted(edge.edge_id for edge in edge_path))
                if signature not in seen[hops]:
                    output[hops].append((node_path, edge_path))
                    seen[hops].add(signature)
            if hops >= 3:
                continue
            for next_node, edge in reversed(adjacency[node]):
                if next_node not in node_path:
                    stack.append((next_node, (*node_path, next_node), (*edge_path, edge)))
    return output


def _path_chunks(edges, chunk_by_id):
    selected = []
    seen = set()
    for edge in edges:
        for chunk_id in edge.chunk_ids:
            if chunk_id in chunk_by_id and chunk_id not in seen:
                selected.append(chunk_by_id[chunk_id])
                seen.add(chunk_id)
                break
    return selected


if __name__ == "__main__":
    raise SystemExit(main())
