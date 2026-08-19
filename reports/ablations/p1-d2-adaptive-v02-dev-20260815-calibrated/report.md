# P1-D2 Adaptive Retrieval Dev Ablation

- Dataset: `evalrag_v0.2`; split: `dev`; cases: 80。
- 固定相同标签、Router source filter、top-k、Embedding 和 CrossEncoder revision。
- 本报告只衡量检索质量与延迟，不代表最终答案准确率。

| Strategy | Recall@3 | Recall@5 | MRR | NDCG@5 | P50 ms | P95 ms |
|---|---:|---:|---:|---:|---:|---:|
| fixed_hybrid | 0.8083 | 0.8556 | 0.8319 | 0.7831 | 109.895 | 591.723 |
| always_rerank | 0.6306 | 0.7667 | 0.6317 | 0.6447 | 3994.202 | 5306.240 |
| adaptive | 0.8083 | 0.8611 | 0.8319 | 0.7861 | 107.600 | 4896.945 |

## Adaptive Decisions

- Strategy counts: `{'hybrid': 59, 'dense': 1, 'bm25': 20}`。
- Rerank invoked/applied: 6/6，调用率 7.50%。
- Adaptive 保持 Recall@3/MRR，Recall@5 小幅改善，但 CPU P95 仍被少量 CrossEncoder 调用拉高，因此不是完整 Pareto 改善。

## Category Metrics

| Strategy | Category | Recall@3 | Recall@5 | MRR | NDCG@5 |
|---|---|---:|---:|---:|---:|
| fixed_hybrid | multi_source | 0.6250 | 0.7167 | 0.9250 | 0.7070 |
| fixed_hybrid | semantic_paraphrase | 0.8000 | 0.8500 | 0.6542 | 0.7043 |
| fixed_hybrid | single_source | 1.0000 | 1.0000 | 0.9167 | 0.9381 |
| fixed_hybrid | unanswerable | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| always_rerank | multi_source | 0.5417 | 0.7500 | 0.6500 | 0.6132 |
| always_rerank | semantic_paraphrase | 0.4500 | 0.6000 | 0.4325 | 0.4733 |
| always_rerank | single_source | 0.9000 | 0.9500 | 0.8125 | 0.8477 |
| always_rerank | unanswerable | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| adaptive | multi_source | 0.6250 | 0.7333 | 0.9250 | 0.7160 |
| adaptive | semantic_paraphrase | 0.8000 | 0.8500 | 0.6542 | 0.7043 |
| adaptive | single_source | 1.0000 | 1.0000 | 0.9167 | 0.9381 |
| adaptive | unanswerable | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Adaptive Reranked Cases

- `v02_single_014` (single_source), confidence=0.633: ΔRecall@5=+0.000, ΔMRR=+0.000, ΔNDCG@5=+0.000。
- `v02_single_015` (single_source), confidence=0.635: ΔRecall@5=+0.000, ΔMRR=+0.000, ΔNDCG@5=+0.000。
- `v02_single_017` (single_source), confidence=0.633: ΔRecall@5=+0.000, ΔMRR=+0.000, ΔNDCG@5=+0.000。
- `v02_single_018` (single_source), confidence=0.637: ΔRecall@5=+0.000, ΔMRR=+0.000, ΔNDCG@5=+0.000。
- `v02_multi_004` (multi_source), confidence=0.636: ΔRecall@5=+0.333, ΔMRR=+0.000, ΔNDCG@5=+0.182。
- `v02_multi_014` (multi_source), confidence=0.636: ΔRecall@5=+0.000, ΔMRR=+0.000, ΔNDCG@5=+0.000。

## Largest Strategy Differences

- `v02_semantic_009` (semantic_paraphrase): MRR fixed/always/adaptive=1.000/0.000/1.000; adaptive=hybrid, rerank=False。
- `v02_semantic_014` (semantic_paraphrase): MRR fixed/always/adaptive=1.000/0.000/1.000; adaptive=hybrid, rerank=False。
- `v02_semantic_015` (semantic_paraphrase): MRR fixed/always/adaptive=1.000/0.000/1.000; adaptive=hybrid, rerank=False。
- `v02_semantic_017` (semantic_paraphrase): MRR fixed/always/adaptive=0.500/0.000/0.500; adaptive=hybrid, rerank=False。
- `v02_semantic_018` (semantic_paraphrase): MRR fixed/always/adaptive=1.000/0.000/1.000; adaptive=hybrid, rerank=False。
- `v02_single_019` (single_source): MRR fixed/always/adaptive=0.500/0.000/0.500; adaptive=hybrid, rerank=False。
- `v02_semantic_011` (semantic_paraphrase): MRR fixed/always/adaptive=0.500/0.200/0.500; adaptive=hybrid, rerank=False。
- `v02_semantic_012` (semantic_paraphrase): MRR fixed/always/adaptive=1.000/0.200/1.000; adaptive=hybrid, rerank=False。
- `v02_multi_009` (multi_source): MRR fixed/always/adaptive=1.000/0.250/1.000; adaptive=hybrid, rerank=False。
- `v02_semantic_003` (semantic_paraphrase): MRR fixed/always/adaptive=1.000/0.250/1.000; adaptive=hybrid, rerank=False。
