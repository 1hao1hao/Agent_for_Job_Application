# P1-D3 Graph + Vector Dev Ablation

- Dataset: `evalrag_graph_v0.1`; split: `dev`; cases: 30。
- 10 条 frozen test 未运行，也未用于配置选择。
- 三种策略使用相同 Chunk、source filter、top-k 和标签。

| Strategy | Recall@5 | MRR | NDCG@5 | Path Validity | Selector Accuracy | P50 ms | P95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| adaptive_vector | 0.7121 | 0.3098 | 0.4196 | N/A | 0.2333 | 297.949 | 2176.368 |
| graph_only | 0.5985 | 0.4545 | 0.4059 | 1.0000 | 0.2333 | 0.517 | 0.716 |
| graph_vector | 0.7273 | 0.3788 | 0.4577 | 1.0000 | 0.8000 | 303.047 | 1099.558 |

## Largest Strategy Differences

- `graph_03_job_skill` (job_skill): Recall@5 vector/graph/graph+vector=0.000/1.000/0.000。
- `graph_03_skill_project` (skill_project): Recall@5 vector/graph/graph+vector=1.000/0.500/1.000。
- `graph_04_skill_project` (skill_project): Recall@5 vector/graph/graph+vector=1.000/0.500/1.000。
- `graph_09_skill_project` (skill_project): Recall@5 vector/graph/graph+vector=1.000/0.500/1.000。
- `graph_01_skill_project` (skill_project): Recall@5 vector/graph/graph+vector=1.000/0.500/1.000。
- `graph_07_skill_project` (skill_project): Recall@5 vector/graph/graph+vector=1.000/0.500/1.000。
- `graph_08_job_skill` (job_skill): Recall@5 vector/graph/graph+vector=1.000/1.000/1.000。
- `graph_08_skill_project` (skill_project): Recall@5 vector/graph/graph+vector=1.000/0.500/1.000。
- `graph_02_job_skill` (job_skill): Recall@5 vector/graph/graph+vector=1.000/1.000/1.000。
- `graph_04_job_skill` (job_skill): Recall@5 vector/graph/graph+vector=1.000/1.000/1.000。

## Boundary

这些指标只衡量关系证据召回和路径结构，不代表最终答案准确率。
