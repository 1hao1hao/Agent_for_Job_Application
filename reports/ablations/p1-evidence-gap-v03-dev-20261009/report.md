# Evidence-Gap Guided Retrieval Dev Ablation

- Dataset: `evalrag_v0.3`; split: `dev`; 160 cases / 120 answerable.
- Gold ids are used only for offline metrics; selector and rescue never receive labels.
- No LLM and no frozen test in this stage.

| Strategy | Recall@5 | MRR | NDCG@5 | Path completeness | P95 ms | Extra-call rate | Historical miss recovered |
|---|---:|---:|---:|---:|---:|---:|---:|
| current_adaptive | 0.5125 | 0.5107 | 0.4811 | 0.0000 | 4800.03 | 0.00% | 0 |
| fixed_multi_route | 0.6708 | 0.6118 | 0.5895 | 0.5750 | 5508.03 | 100.00% | 21 |
| on_demand_rescue | 0.6875 | 0.6750 | 0.6361 | 0.5000 | 6530.66 | 64.38% | 23 |

## Decision

- deterministic dev retrieval/path metric improved.

## Representative differences

- `v03_cross_source_002` (cross_source): Recall@5 +1.000, MRR +1.000, paths=['bm25', 'dense'].
- `v03_cross_source_004` (cross_source): Recall@5 +1.000, MRR +1.000, paths=['bm25', 'dense'].
- `v03_cross_source_008` (cross_source): Recall@5 +1.000, MRR +1.000, paths=['bm25', 'dense'].
- `v03_cross_source_010` (cross_source): Recall@5 +1.000, MRR +1.000, paths=['bm25', 'dense'].
- `v03_freshness_conflict_007` (freshness_conflict): Recall@5 +1.000, MRR +1.000, paths=['bm25', 'dense'].
- `v03_freshness_conflict_011` (freshness_conflict): Recall@5 +1.000, MRR +1.000, paths=['bm25', 'dense'].
- `v03_semantic_paraphrase_008` (semantic_paraphrase): Recall@5 -1.000, MRR -1.000, paths=['bm25', 'dense'].
- `v03_semantic_paraphrase_028` (semantic_paraphrase): Recall@5 +1.000, MRR +1.000, paths=['bm25', 'dense'].
- `v03_single_source_001` (single_source): Recall@5 +1.000, MRR +1.000, paths=['bm25', 'dense'].
- `v03_single_source_010` (single_source): Recall@5 +1.000, MRR +1.000, paths=['bm25', 'dense'].
