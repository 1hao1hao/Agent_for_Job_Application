# P0-D5 Reranker Dev Ablation

- Dataset: `evalrag_v0.2`; split: `dev`; 80 cases。
- 唯一主要变量：是否对 Hybrid top-20 使用中文 token-overlap 重排。
- CrossEncoder adapter 已实现，但外部权重下载未完成，本报告不是神经 Reranker 结果。

| Strategy | Recall@3 | Recall@5 | MRR | Retrieval P95 ms |
|---|---:|---:|---:|---:|
| Hybrid | 0.8083 | 0.8556 | 0.8319 | 700.713 |
| Hybrid + token rerank | 0.7333 | 0.8389 | 0.7944 | 902.752 |

## Difference Cases

- `v02_multi_004` regressed: ΔMRR=-0.500, ΔRecall@5=+0.000。
- `v02_multi_009` regressed: ΔMRR=-0.500, ΔRecall@5=+0.000。
- `v02_multi_014` regressed: ΔMRR=-0.500, ΔRecall@5=+0.000。
- `v02_single_013` regressed: ΔMRR=-0.500, ΔRecall@5=+0.000。
- `v02_multi_002` regressed: ΔMRR=+0.000, ΔRecall@5=-0.333。
- `v02_multi_012` regressed: ΔMRR=+0.000, ΔRecall@5=-0.333。
- `v02_multi_019` regressed: ΔMRR=+0.000, ΔRecall@5=-0.333。
- `v02_single_019` regressed: ΔMRR=-0.250, ΔRecall@5=+0.000。

## Frozen Decision

最终配置禁用该 Reranker。它只能重排已有候选，且本次 dev 同时损害召回、首位排序和延迟。
