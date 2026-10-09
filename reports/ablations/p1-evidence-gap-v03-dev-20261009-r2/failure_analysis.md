# Historical Retrieval Miss Analysis

- Source run: `reports/runs/p1-task2-evidence-rerank-context-v03-dev-20261006`
- Retrieval misses: 45
- Gold absent from Top-20: 27
- Corpus/label missing: 0

| Root cause | Cases |
|---|---:|
| candidate_pool_miss | 12 |
| graph_path_missing | 15 |
| ranking_or_single_route_insufficient | 7 |
| route_source_filter_mismatch | 11 |

Gold ids were used only after prediction for offline diagnosis.
