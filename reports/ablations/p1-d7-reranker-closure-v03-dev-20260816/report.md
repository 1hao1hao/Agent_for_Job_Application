# P1-D7 Reranker Closure

| Variant | Recall@3 | Recall@5 | MRR | NDCG@5 | Invoke | P95 ms |
|---|---:|---:|---:|---:|---:|---:|
| bge_base_k20 | 0.4083 | 0.4458 | 0.4026 | 0.3914 | 13.12% | 5566.707 |
| bge_base_k10 | 0.3958 | 0.4375 | 0.3958 | 0.3834 | 21.25% | 3157.577 |
| minilm_k10 | 0.4042 | 0.4375 | 0.4021 | 0.3869 | 21.25% | 2100.939 |

Decision: `on_demand_minilm_k10`。P95 明显下降，MRR 基本持平且 Recall@5 退化不超过 1 个百分点。

只使用 dev；未读取 frozen test。
