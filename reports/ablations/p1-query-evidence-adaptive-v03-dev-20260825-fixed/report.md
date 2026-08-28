# Query Evidence Adaptive Retrieval Dev Ablation

- Dataset: `evalrag_v0.3`; split: `dev`; cases: 160。
- 80 条 frozen test 未运行，也未用于配置选择。
- Config: `configs/retrieval/graph_adaptive_v0.3.json`；top-k=5。
- 新版链路：`QueryFeatures -> EvidenceRequirement -> Retriever Selection`。

| Version | Recall@3 | Recall@5 | MRR | NDCG@5 | P50 ms | P95 ms |
|---|---:|---:|---:|---:|---:|---:|
| 旧 `is_cross_document` selector | 0.4375 | 0.4833 | 0.4221 | 0.4103 | - | 1904.738 |
| 首轮 Evidence Need selector | 0.3917 | 0.4375 | 0.3829 | 0.3785 | 14.565 | 1696.075 |
| 修复后 Evidence Need selector | 0.4292 | 0.4875 | 0.3901 | 0.3993 | 395.777 | 1322.255 |

旧版数字来自 `reports/ablations/p1-d9-v03-dev-ablation-20260817/`，运行日期不同，延迟只作
参考，不能当成严格同机性能提升。新版 Recall@5 比旧版高 0.42 个百分点，但 MRR 低 3.20
个百分点，因此结论是**召回覆盖略升、首条相关证据排序退化**，不是全面提升。

## Failure-Driven Fix

首轮 Trace 显示 two-hop/three-hop Query 被误分到 `exact_fact/bm25`，原因是关系词表遗漏
“知识关系、2/3 跳联系”；语义改写 Query 中 13/20 被标题内英文误判为精确事实。修复只增加：

- 关系表达：`知识关系 / 关系链 / 跳联系 / 是否证明`；
- 语义表达：`不使用原文 / 概括 / 转述`。

修复后策略分布：20 条 two-hop 和 20 条 three-hop 全部走 `graph_hybrid`，20 条
semantic_paraphrase 全部走 `dense`，20 条 cross_source 全部走 `hybrid`。逐 Case Trace 位于
`reports/runs/p1-query-evidence-adaptive-v03-dev-20260825-fixed-graph_vector/case_results.jsonl`。

## Boundary

本报告只衡量检索和图路径，不代表最终回答准确率；该规则版 Evidence Need Classifier 仍会受
词表覆盖影响，MRR 退化已保留，不通过删除失败 Case 或修改 frozen test 隐藏。
