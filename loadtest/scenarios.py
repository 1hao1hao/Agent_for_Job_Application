from __future__ import annotations


# 每类保留多种问法，防止压测结果只代表一个恰好命中/拒答的 Query。
QUERY_CASES: tuple[dict[str, object], ...] = (
    {
        "name": "exact_fact",
        "query": "RRF 的定义和计算方式是什么？",
        "top_k": 5,
    },
    {
        "name": "exact_fact",
        "query": "Redis Stream 的 consumer group 是什么？",
        "top_k": 5,
    },
    {
        "name": "exact_fact",
        "query": "pgvector HNSW 索引有什么作用？",
        "top_k": 5,
    },
    {
        "name": "semantic",
        "query": "如何减少知识库问答中没有证据支撑的内容？",
        "top_k": 5,
    },
    {
        "name": "semantic",
        "query": "怎样判断一段回答是否真正依据了检索材料？",
        "top_k": 5,
    },
    {
        "name": "semantic",
        "query": "如何降低检索增强生成中的无依据回答？",
        "top_k": 5,
    },
    {
        "name": "multi_source",
        "query": "结合岗位要求和我的简历分析能力匹配度。",
        "top_k": 5,
    },
    {
        "name": "multi_source",
        "query": "结合岗位 JD、项目日志和个人经历说明我还缺少哪些能力。",
        "top_k": 5,
    },
    {
        "name": "multi_source",
        "query": "对照大模型应用岗位要求和简历，给出匹配证据。",
        "top_k": 5,
    },
    {
        "name": "relation_reasoning",
        "query": "哪个项目能证明我符合 RAG 岗位要求？",
        "top_k": 5,
    },
    {
        "name": "relation_reasoning",
        "query": "岗位要求的 Python 和 RAG 技能分别被哪些项目经历证明？",
        "top_k": 5,
    },
    {
        "name": "relation_reasoning",
        "query": "沿岗位、技能和项目关系说明我的经历为什么匹配该岗位。",
        "top_k": 5,
    },
)
