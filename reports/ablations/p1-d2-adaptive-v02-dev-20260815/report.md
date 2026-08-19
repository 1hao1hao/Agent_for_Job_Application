# P1-D2 Adaptive Retrieval Dev Ablation

- Dataset: `evalrag_v0.2`; split: `dev`; cases: 80。
- 固定相同标签、Router source filter、top-k、Embedding 和 CrossEncoder revision。
- 本报告只衡量检索质量与延迟，不代表最终答案准确率。

| Strategy | Recall@3 | Recall@5 | MRR | NDCG@5 | P50 ms | P95 ms |
|---|---:|---:|---:|---:|---:|---:|
| fixed_hybrid | 0.8083 | 0.8556 | 0.8319 | 0.7831 | 109.895 | 591.723 |
| always_rerank | 0.6306 | 0.7667 | 0.6317 | 0.6447 | 3994.202 | 5306.240 |
| adaptive | 0.7417 | 0.8556 | 0.7986 | 0.7575 | 102.338 | 595.263 |

## Adaptive Decisions

- Strategy counts: `{'hybrid': 52, 'bm25': 27, 'dense': 1}`。
- Rerank invoked/applied: 0/0。

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
- `v02_semantic_001` (semantic_paraphrase): MRR fixed/always/adaptive=1.000/1.000/0.250; adaptive=bm25, rerank=False。
