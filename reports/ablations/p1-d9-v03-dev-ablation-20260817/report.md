# P1 v0.3 Dev Unified Ablation Report

- Run: `p1-d9-v03-dev-ablation-20260817`
- Corpus/benchmark: `evalrag_v0.3/dev`, 160 cases; frozen test 未重跑。
- Router 只报告来源路由指标，因为 v0.3 没有 expected_intent 标签。

## 1. Router Source Routing

| Router | Source Exact | Answerable Exact | Source Precision | Source Recall | Unanswerable Acc | P95 ms |
|---|---:|---:|---:|---:|---:|---:|
| rule | 43.75% | 26.67% | 49.06% | 47.50% | 95.00% | 0.009 |
| semantic | 25.00% | 15.00% | 27.19% | 27.81% | 55.00% | 500.659 |
| hybrid | 43.75% | 26.67% | 49.06% | 47.50% | 95.00% | 396.603 |
| feedback_hybrid | 43.75% | 26.67% | 49.06% | 47.50% | 95.00% | 496.285 |

## 2. Retrieval Strategies

| Retriever | Recall@3 | Recall@5 | MRR | NDCG@5 | P95 ms |
|---|---:|---:|---:|---:|---:|
| keyword | 0.2625 | 0.2750 | 0.2931 | 0.2679 | 92.496 |
| bm25 | 0.3208 | 0.3708 | 0.3004 | 0.3047 | 14.591 |
| dense | 0.4042 | 0.4375 | 0.4024 | 0.3884 | 1422.115 |
| keyword_dense_rrf | 0.3667 | 0.4167 | 0.3839 | 0.3667 | 1427.806 |
| bm25_dense_rrf | 0.4167 | 0.5000 | 0.4403 | 0.4307 | 1115.262 |
| graph_only | 0.2958 | 0.3542 | 0.3729 | 0.3274 | 14.055 |
| graph_vector_rrf | 0.5542 | 0.5833 | 0.5210 | 0.5103 | 1186.649 |
| adaptive_graph | 0.4375 | 0.4833 | 0.4221 | 0.4103 | 1904.738 |

## 3. Rerank Policy

三路固定 BM25 + Dense + RRF、同一 MiniLM revision、candidate_k=10，只改变 rerank policy。

| Policy | Recall@3 | Recall@5 | MRR | NDCG@5 | Invocation | P95 ms |
|---|---:|---:|---:|---:|---:|---:|
| never | 0.4375 | 0.5083 | 0.4431 | 0.4367 | 0.00% | 1252.334 |
| always | 0.4708 | 0.5083 | 0.4922 | 0.4695 | 100.00% | 2799.094 |
| on_demand | 0.4458 | 0.5000 | 0.4518 | 0.4389 | 18.12% | 2174.664 |

## 4. Context Engine / Memory

- Dataset: `evalrag_context_v0.1/dev`; 60 groups / 300 turns。
- 该部分复用已完成的正式工件，没有因代码未变化重复运行。

| Strategy | Follow-up | Key-point | Prompt Tokens | Token Reduction vs Raw | Repeat Reads | P95 ms |
|---|---:|---:|---:|---:|---:|---:|
| no_memory | 36.67% | 28.30% | 38.18 | 31.50% | 0.00 | 0.101 |
| recent_window | 36.67% | 28.30% | 67.18 | 0.00% | 3.00 | 0.134 |
| summary_recent | 100.00% | 100.00% | 76.62 | 0.00% | 3.00 | 0.145 |
| semantic_memory | 100.00% | 100.00% | 66.22 | 4.18% | 0.00 | 296.933 |

## 5. Findings and Configuration Decision

- Router：Rule 与 Feedback Hybrid 的 Source Exact 均为 43.75%；Feedback 在 v0.3 没有改善，因为旧反馈锚点没有覆盖新 Query 分布。v0.3 缺少 intent 标签，因此不能用该结果重新声称 Router Intent Accuracy。
- Retrieval：固定 Graph+Vector RRF 的 Recall@5/MRR 为 58.33%/52.10%，高于 Adaptive Graph 的 48.33%/42.21%。当前 selector 漏触发部分关系型 Query，Graph+Vector 是新的 quality-first dev candidate；不重跑 frozen test。
- Rerank：always 将 MRR 从 44.31% 提高到 49.22%，但 P95 从 1252.33 ms 增至 2799.09 ms；on-demand 调用率 18.12%，MRR 45.18%。三者没有形成统一 Pareto 最优，默认不把 Reranker 包装成质量提升。
- Context：Semantic Memory 与 Summary+Recent 均保持 100.00% Follow-up Success，平均 Prompt Token 从 76.62 降至 66.22、重复读取从 3 降至 0，但 P95 增至 296.93 ms。

## 6. Additional Existing Ablations

- Chunking：段落固定切分 vs 句子边界切分。
- Context Builder：rank-prefix vs source-balanced，不同预算。
- Model Gateway：timeout、429/5xx、熔断、并发和 fallback 故障注入。

## Boundary

dev-only ablation; v0.3 labels are corpus-grounded AI-assisted; Router reports source routing rather than intent accuracy; retrieval metrics do not equal answer correctness; frozen test was not rerun
