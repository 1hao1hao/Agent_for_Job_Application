from __future__ import annotations


QUERY_CASES: tuple[dict[str, object], ...] = (
    {
        "name": "exact_fact",
        "query": "RRF 的定义和计算方式是什么？",
        "top_k": 5,
        "retriever": "bm25",
    },
    {
        "name": "semantic",
        "query": "如何减少知识库问答中没有证据支撑的内容？",
        "top_k": 5,
        "retriever": "bm25",
    },
    {
        "name": "multi_source",
        "query": "结合岗位要求和我的简历分析能力匹配度。",
        "top_k": 5,
        "retriever": "bm25",
    },
    {
        "name": "relation_reasoning",
        "query": "哪个项目能证明我符合 RAG 岗位要求？",
        "top_k": 5,
        "retriever": "bm25",
    },
)
