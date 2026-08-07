# Retrieval Dev Ablation

- Dataset: `evalrag_v0.2`
- Split: `dev`
- 三组运行使用相同 Query、Chunks、Router source filter 和 top-k。
- 本报告只衡量检索，不代表最终答案准确率。

| Retriever | Recall@3 | Recall@5 | MRR | Retrieval P50 ms | Retrieval P95 ms |
|---|---:|---:|---:|---:|---:|
| keyword | 0.7333 | 0.8056 | 0.8597 | 9.233 | 14.086 |
| dense | 0.7278 | 0.8083 | 0.6742 | 198.931 | 1391.526 |
| hybrid | 0.8083 | 0.8556 | 0.8319 | 202.901 | 1095.153 |

## Strategy Differences

- `v02_semantic_010` (semantic_paraphrase): 简历优势：怎样生成前判断材料是否足够，并避免证据不足仍强行回答？ MRR keyword/dense/hybrid = 1.000/0.000/0.250。
- `v02_semantic_017` (semantic_paraphrase): 面试怎么回答：怎样统一接收手工、CSV 和 JSON 岗位，并避免平台字段不一致？ MRR keyword/dense/hybrid = 1.000/0.000/0.500。
- `v02_single_015` (single_source): 面试准备中，请求追踪应该怎样解释和排查？ MRR keyword/dense/hybrid = 1.000/0.000/0.500。
- `v02_multi_012` (multi_source): 结合岗位、简历和面试资料准备请求追踪追问 MRR keyword/dense/hybrid = 1.000/0.250/1.000。
- `v02_multi_017` (multi_source): 结合岗位、简历和面试资料准备意图路由追问 MRR keyword/dense/hybrid = 0.500/1.000/1.000。
- `v02_multi_011` (multi_source): 结合岗位和简历分析请求追踪匹配度 MRR keyword/dense/hybrid = 1.000/0.500/1.000。
- `v02_multi_004` (multi_source): 结合用户画像、岗位和简历制定混合检索投递计划 MRR keyword/dense/hybrid = 0.500/1.000/1.000。
- `v02_semantic_016` (semantic_paraphrase): 面试怎么回答：怎样生成前判断材料是否足够，并避免证据不足仍强行回答？ MRR keyword/dense/hybrid = 1.000/0.200/0.333。
- `v02_multi_015` (multi_source): 面试时如何结合岗位和个人经历回答请求追踪问题 MRR keyword/dense/hybrid = 1.000/0.250/1.000。
- `v02_multi_002` (multi_source): 结合岗位、简历和面试资料准备混合检索追问 MRR keyword/dense/hybrid = 0.500/0.500/0.500。

## Boundary

Dense 使用固定 commit 的 BAAI/bge-small-zh-v1.5 预训练中文 embedding；文档向量离线构建。
RRF 只融合排名，不直接相加不可比的 Keyword/Dense 原始分数。
本次 Hybrid 提升 Recall@3/Recall@5，但 MRR 下降；说明候选覆盖改善不等于首位排序改善。
CPU 语义编码显著增加 P95；后续可评估批处理、模型量化或缓存，而不能忽略延迟代价。
