# P1 Final Frozen Release

| Strategy | Recall@3 | Recall@5 | MRR | NDCG@5 | P95 ms |
|---|---:|---:|---:|---:|---:|
| bm25_baseline | 0.3917 | 0.4667 | 0.3519 | 0.3649 | 15.502 |
| p1_final_graph_vector | 0.4917 | 0.6333 | 0.5758 | 0.5508 | 1209.401 |

## Graph Challenge

10 frozen cases; metrics: `{"recall_at_5": 0.9166666666666666, "mrr": 0.5833333333333334, "ndcg_at_5": 0.6348114004692932, "path_validity": 1.0, "selector_accuracy": 1.0}`。

## Boundary

Context benchmark has no untouched test split, so it remains a dev-only reference rather than a fabricated frozen result.
