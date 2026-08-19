# P1-D3 Graph + Vector Dev Ablation

- Dataset: `evalrag_graph_v0.1`; split: `dev`; cases: 30。
- 10 条 frozen test 未运行，也未用于配置选择。
- 三种策略使用相同 Chunk、source filter、top-k 和标签。

| Strategy | Recall@5 | MRR | NDCG@5 | Path Validity | Selector Accuracy | P50 ms | P95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| adaptive_vector | 0.7121 | 0.3098 | 0.4196 | N/A | 0.2333 | 199.175 | 921.244 |
| graph_only | 0.5985 | 0.4545 | 0.4059 | 1.0000 | N/A | 0.542 | 0.882 |
| graph_vector | 0.7727 | 0.5227 | 0.5381 | 1.0000 | 1.0000 | 397.774 | 1190.255 |

## Largest Strategy Differences

- `graph_03_job_skill` (job_skill): Recall@5 vector/graph/graph+vector=0.000/1.000/0.000。
- `graph_05_skill_project` (skill_project): Recall@5 vector/graph/graph+vector=0.000/0.500/1.000。
- `graph_03_skill_project` (skill_project): Recall@5 vector/graph/graph+vector=1.000/0.500/1.000。
- `graph_04_skill_project` (skill_project): Recall@5 vector/graph/graph+vector=1.000/0.500/1.000。
- `graph_09_skill_project` (skill_project): Recall@5 vector/graph/graph+vector=1.000/0.500/1.000。
- `graph_01_skill_project` (skill_project): Recall@5 vector/graph/graph+vector=1.000/0.500/1.000。
- `graph_07_skill_project` (skill_project): Recall@5 vector/graph/graph+vector=1.000/0.500/1.000。
- `graph_08_job_skill` (job_skill): Recall@5 vector/graph/graph+vector=1.000/1.000/1.000。
- `graph_08_skill_project` (skill_project): Recall@5 vector/graph/graph+vector=1.000/0.500/1.000。
- `graph_02_job_skill` (job_skill): Recall@5 vector/graph/graph+vector=1.000/1.000/1.000。

## Boundary

这些指标只衡量关系证据召回和路径结构，不代表最终答案准确率。
