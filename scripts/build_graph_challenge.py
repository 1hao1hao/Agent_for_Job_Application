from __future__ import annotations

import json
from pathlib import Path

from intern_rag.evaluation import load_chunks_jsonl
from intern_rag.evaluation.graph_dataset import (
    GraphChallengeCase,
    validate_graph_challenge,
)


TOPICS = [
    ("01", "混合检索", "RRF", "单路召回偏科"),
    ("02", "引用校验", "Citation Validator", "模型编造证据编号"),
    ("03", "请求追踪", "Agent Trace", "错误阶段无法定位"),
    ("04", "意图路由", "Semantic Router", "多意图规则冲突"),
    ("05", "上下文预算", "Context Builder", "关键证据被截断"),
    ("06", "失败回归", "Regression Suite", "旧失败重新出现"),
    ("07", "岗位时效性", "freshness filter", "过期岗位污染回答"),
    ("08", "工具调用", "tool validator", "参数错误产生副作用"),
    ("09", "模型适配", "LlmClient Protocol", "供应商响应格式不同"),
    ("10", "证据门控", "Evidence Gate", "证据不足仍强行回答"),
]


def main() -> int:
    """从明确对应的 v0.2 跨来源材料生成 40 条独立关系标签。"""

    chunks = load_chunks_jsonl(
        Path("data/processed/chunks/evalrag_v0.2.jsonl")
    )
    chunk_ids = _index_relevant_chunks(chunks)
    cases: list[GraphChallengeCase] = []
    for topic_index, (group, topic, solution, risk) in enumerate(TOPICS):
        variants = _topic_cases(group, topic, solution, risk, chunk_ids)
        frozen_index = topic_index % len(variants)
        for index, case in enumerate(variants):
            cases.append(
                GraphChallengeCase(
                    **{
                        **case.to_dict(),
                        "split": "test" if index == frozen_index else "dev",
                    }
                )
            )

    validation = validate_graph_challenge(
        cases, available_chunk_ids={chunk.id for chunk in chunks}
    )
    if not validation.is_valid:
        print(json.dumps(validation.to_dict(), ensure_ascii=False, indent=2))
        return 1
    output_path = Path("data/evaluation/evalrag_graph_v0.1.jsonl")
    output_path.write_text(
        "".join(
            json.dumps(case.to_dict(), ensure_ascii=False) + "\n" for case in cases
        ),
        encoding="utf-8",
    )
    validation_path = Path(
        "data/evaluation/evalrag_graph_v0.1_validation.json"
    )
    validation_path.write_text(
        json.dumps(validation.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(validation.to_dict(), ensure_ascii=False, indent=2))
    return 0


def _index_relevant_chunks(chunks) -> dict[tuple[str, str], str]:
    """选择每组材料的“能力/实现”段，避免把同文档四段全部标为相关。"""

    indexed: dict[tuple[str, str], str] = {}
    for chunk in chunks:
        if int(chunk.metadata.get("chunk_index", -1)) != 1:
            continue
        for group, *_ in TOPICS:
            expected_title = {
                "jd": f"v02_{group}_jd",
                "resume": f"v02_{group}_resume",
                "project_logs": f"v02_{group}_project_logs",
            }.get(chunk.source_type)
            if expected_title == chunk.title:
                indexed[(group, chunk.source_type)] = chunk.id
    expected_count = len(TOPICS) * 3
    if len(indexed) != expected_count:
        raise ValueError(
            f"expected {expected_count} grounded chunks, got {len(indexed)}"
        )
    return indexed


def _topic_cases(
    group: str,
    topic: str,
    solution: str,
    risk: str,
    chunk_ids: dict[tuple[str, str], str],
) -> list[GraphChallengeCase]:
    jd = chunk_ids[(group, "jd")]
    resume = chunk_ids[(group, "resume")]
    project = chunk_ids[(group, "project_logs")]
    common = {
        "expected_entities": [topic],
        "label_reviewed": True,
        "review_method": (
            "corpus-grounded deterministic review against the selected v0.2 chunks"
        ),
    }
    return [
        GraphChallengeCase(
            case_id=f"graph_{group}_job_skill",
            query=f"{topic}岗位材料把哪种方案作为核心，并重点防范什么风险？",
            split="dev",
            category="job_skill",
            expected_strategy="vector",
            expected_sources=["jd"],
            relevant_chunk_ids=[jd],
            expected_relations=["requires"],
            answerable=True,
            expected_points=[solution, risk],
            **common,
        ),
        GraphChallengeCase(
            case_id=f"graph_{group}_skill_project",
            query=f"候选人的简历经历和项目记录如何共同体现处理“{risk}”的能力？",
            split="dev",
            category="skill_project",
            expected_strategy="graph_hybrid",
            expected_sources=["resume", "project_logs"],
            relevant_chunk_ids=[resume, project],
            expected_relations=["demonstrates", "uses"],
            answerable=True,
            expected_points=[topic, solution],
            **common,
        ),
        GraphChallengeCase(
            case_id=f"graph_{group}_two_hop",
            query=f"把围绕{topic}的岗位要求，与候选人的哪段经历和项目实践对应起来。",
            split="dev",
            category="cross_source_two_hop",
            expected_strategy="graph_hybrid",
            expected_sources=["jd", "resume", "project_logs"],
            relevant_chunk_ids=[jd, resume, project],
            expected_relations=["requires", "demonstrates", "uses"],
            answerable=True,
            expected_points=[topic, solution, risk],
            **common,
        ),
        GraphChallengeCase(
            case_id=f"graph_{group}_hard_negative",
            query=(
                f"围绕{topic}的岗位与项目材料能否证明候选人已经用量子图数据库"
                "上线十亿级生产系统？"
            ),
            split="dev",
            category="hard_negative",
            expected_strategy="graph_hybrid",
            expected_sources=["jd", "resume", "project_logs"],
            relevant_chunk_ids=[],
            expected_relations=["requires", "demonstrates", "uses"],
            answerable=False,
            expected_points=[],
            **common,
        ),
    ]


if __name__ == "__main__":
    raise SystemExit(main())
