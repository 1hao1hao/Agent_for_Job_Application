# Adaptive v2 Release Test

This split has historical runs; this is the release test for the newly locked config, not a never-seen blind test.

| Strategy | Recall@5 | MRR | NDCG@5 | P95 ms |
|---|---:|---:|---:|---:|
| bm25 | 0.4667 | 0.3519 | 0.3649 | 15.262 |
| graph_vector | 0.6917 | 0.6292 | 0.6138 | 1889.444 |
| adaptive_v2 | 0.6917 | 0.5311 | 0.5469 | 2031.671 |
