# P1-D4 Corpus v0.3 Dev Retrieval Ablation

- Dataset: `evalrag_v0.3`; split: `dev`; cases: 160。
- 80 条 frozen test 未运行，也未用于配置选择。

| Strategy | Recall@3 | Recall@5 | MRR | NDCG@5 | P50 ms | P95 ms |
|---|---:|---:|---:|---:|---:|---:|
| graph_vector | 0.3917 | 0.4375 | 0.3829 | 0.3785 | 14.565 | 1696.075 |

## Strategy Differences


## Boundary

本报告只衡量检索和图路径，不代表最终回答准确率。
