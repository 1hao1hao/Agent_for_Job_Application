# Context Builder Dev Ablation

- Source run: `reports/runs/p0-d5-v02-dev-20260804-hybrid-baseline/case_results.jsonl`
- Scope: answerable dev cases with correct Router prediction
- 唯一变量：Context 预算选择策略；不重新运行 Router/Retriever。

| Budget | Strategy | Relevant Recall | Source Coverage | Full Source Rate | Mean Chunks | Budget Use |
|---:|---|---:|---:|---:|---:|---:|
| 800 | rank_prefix | 66.98% | 56.92% | 24.53% | 1.28 | 65.83% |
| 800 | source_balanced | 68.55% | 61.01% | 30.19% | 1.40 | 71.25% |
| 1200 | rank_prefix | 80.19% | 61.32% | 30.19% | 2.30 | 78.89% |
| 1200 | source_balanced | 79.25% | 78.93% | 54.72% | 2.36 | 80.00% |
| 2000 | rank_prefix | 93.08% | 70.75% | 47.17% | 4.32 | 88.74% |
| 2000 | source_balanced | 92.45% | 80.82% | 60.38% | 4.32 | 88.52% |
| 4000 | rank_prefix | 94.34% | 80.82% | 60.38% | 5.00 | 51.48% |
| 4000 | source_balanced | 94.34% | 80.82% | 60.38% | 5.00 | 51.48% |

`source_balanced` 先保留每个 Router 必需来源的最高排名 Chunk，再按 rank 填充；
`rank_prefix` 保留原实现。指标只描述 Context 选证据，不等于答案准确率。
