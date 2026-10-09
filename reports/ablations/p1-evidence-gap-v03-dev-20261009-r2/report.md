# Evidence-Gap Guided Retrieval Dev Ablation

- Dataset: `evalrag_v0.3`; split: `dev`; 160 cases / 120 answerable.
- Gold ids are used only for offline metrics; selector and rescue never receive labels.
- No LLM and no frozen test in this stage.

| Strategy | Recall@5 | MRR | NDCG@5 | Path completeness | P95 ms | Extra-call rate | Historical miss recovered |
|---|---:|---:|---:|---:|---:|---:|---:|
| current_adaptive | 0.5125 | 0.5107 | 0.4811 | 0.0000 | 4800.03 | 0.00% | 0 |
| fixed_multi_route | 0.6708 | 0.6118 | 0.5895 | 0.5750 | 5508.03 | 100.00% | 21 |
| on_demand_rescue | 0.7292 | 0.6757 | 0.6598 | 0.5750 | 7212.52 | 64.38% | 21 |

## Decision

- deterministic dev retrieval/path metric improved.

## Representative differences

- `v03_freshness_conflict_007` (freshness_conflict): Recall@5 +1.000, MRR +1.000, paths=['bm25', 'dense'].
- `v03_freshness_conflict_011` (freshness_conflict): Recall@5 +1.000, MRR +1.000, paths=['bm25', 'dense'].
- `v03_single_source_001` (single_source): Recall@5 +1.000, MRR +1.000, paths=['bm25', 'dense'].
- `v03_single_source_010` (single_source): Recall@5 +1.000, MRR +1.000, paths=['bm25', 'dense'].
- `v03_single_source_014` (single_source): Recall@5 +1.000, MRR +1.000, paths=['bm25', 'dense'].
- `v03_single_source_017` (single_source): Recall@5 +1.000, MRR +1.000, paths=['bm25', 'dense'].
- `v03_single_source_019` (single_source): Recall@5 +1.000, MRR +1.000, paths=['bm25', 'dense'].
- `v03_single_source_023` (single_source): Recall@5 +1.000, MRR +1.000, paths=['bm25', 'dense'].
- `v03_single_source_026` (single_source): Recall@5 +1.000, MRR +1.000, paths=['bm25', 'dense'].
- `v03_three_hop_005` (three_hop): Recall@5 +1.000, MRR +1.000, paths=['graph_path', 'bm25'].

## Final interpretation

- Recall improved: 34 cases; regressed: 0; zero-recall remaining: 25.
- Reference is current Adaptive first pass with the same live Router source filter, not the previous whole E2E run or full-library benchmark.
- Reference/fixed predictions are reused from the first run; the candidate alone was rerun after two dev fixes.
- Path completeness means all labeled edge IDs appear in returned evidence; it is not answer correctness.
- P95 is CPU wall time; sequential runs are not controlled hardware repetitions.
- Full live E2E: **NOT TESTED**. A pre-r2 run completed only 14/160 cases before connection failures; artifacts are preserved.
- Keep rescue as opt-in; existing service default stays until live E2E verifies Gate/Context/citations and latency.
- Remaining debt: incomplete entity linking, graph groups exceeding top-k/budget, strict anchor/source checks causing refusals, and high rescue P95.
