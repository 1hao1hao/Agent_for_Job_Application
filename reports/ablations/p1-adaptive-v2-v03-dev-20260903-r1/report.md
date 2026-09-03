# Adaptive Retrieval v2 Dev Closure

`evalrag_v0.3/dev`: 160 cases, 120 answerable. Frozen test was not read.

| Strategy | Recall@5 | MRR | NDCG@5 | P95 ms |
|---|---:|---:|---:|---:|
| bm25 | 0.3708 | 0.3004 | 0.3047 | 15.982 |
| dense | 0.4375 | 0.4024 | 0.3884 | 1597.459 |
| hybrid | 0.5000 | 0.4403 | 0.4307 | 1836.161 |
| graph_vector | 0.5833 | 0.5210 | 0.5103 | 1696.573 |
| adaptive_v1 | 0.5458 | 0.4776 | 0.4701 | 1518.740 |
| adaptive_v2 | 0.5542 | 0.4988 | 0.4878 | 1847.355 |

## Gate v2

- FAR: 0.3274; FRR: 0.0000; answerable acceptance: 0.7000.
- Abstention Accuracy: 1.0000; deterministic E2E success: 0.5437.

## Boundary

dev-only configuration selection; no frozen test was read.
