# P1-D4 Corpus v0.3 Dev Retrieval Ablation

- Dataset: `evalrag_v0.3`; split: `dev`; cases: 160。
- 80 条 frozen test 未运行，也未用于配置选择。

| Strategy | Recall@3 | Recall@5 | MRR | NDCG@5 | P50 ms | P95 ms |
|---|---:|---:|---:|---:|---:|---:|
| bm25 | 0.3208 | 0.3708 | 0.3004 | 0.3047 | 11.363 | 14.190 |
| dense | 0.4042 | 0.4375 | 0.4024 | 0.3884 | 309.195 | 914.349 |
| graph_only | 0.2958 | 0.3542 | 0.3729 | 0.3274 | 8.560 | 16.131 |

## Strategy Differences

- `v03_cross_source_002` (cross_source): interview 与 project_logs 分别如何描述“岗位时效”，它们能提供哪些互补证据？
- `v03_cross_source_011` (cross_source): interview 与 jd 分别如何描述“证据门控”，它们能提供哪些互补证据？
- `v03_cross_source_013` (cross_source): interview 与 resume 分别如何描述“证据门控”，它们能提供哪些互补证据？
- `v03_cross_source_014` (cross_source): interview 与 user_profile 分别如何描述“证据门控”，它们能提供哪些互补证据？
- `v03_cross_source_016` (cross_source): jd 与 resume 分别如何描述“证据门控”，它们能提供哪些互补证据？
- `v03_cross_source_017` (cross_source): jd 与 user_profile 分别如何描述“证据门控”，它们能提供哪些互补证据？
- `v03_cross_source_019` (cross_source): project_logs 与 user_profile 分别如何描述“证据门控”，它们能提供哪些互补证据？
- `v03_cross_source_020` (cross_source): resume 与 user_profile 分别如何描述“证据门控”，它们能提供哪些互补证据？
- `v03_freshness_conflict_002` (freshness_conflict): 《软件与人工智能学院专任教师》这条岗位资料的状态、发布时间或采集时间是什么？
- `v03_freshness_conflict_004` (freshness_conflict): 《通信系统（测控数传系统）设计(博士)-2025》这条岗位资料的状态、发布时间或采集时间是什么？
- `v03_freshness_conflict_005` (freshness_conflict): 《行业咨询规划岗实习生(002090)》这条岗位资料的状态、发布时间或采集时间是什么？
- `v03_freshness_conflict_007` (freshness_conflict): 《文员(J10243)》这条岗位资料的状态、发布时间或采集时间是什么？
- `v03_freshness_conflict_008` (freshness_conflict): 《研究生教育管理主管》这条岗位资料的状态、发布时间或采集时间是什么？
- `v03_freshness_conflict_010` (freshness_conflict): 《热喷涂涂层材料研究》这条岗位资料的状态、发布时间或采集时间是什么？
- `v03_freshness_conflict_011` (freshness_conflict): 《产品管理》这条岗位资料的状态、发布时间或采集时间是什么？

## Boundary

本报告只衡量检索和图路径，不代表最终回答准确率。
