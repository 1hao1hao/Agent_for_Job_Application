# BGE CrossEncoder Reranker Dev Ablation

- Dataset: `evalrag_v0.2`; split: `dev`; 80 cases。
- 唯一变量：对相同 Hybrid top-20 使用 BGE CrossEncoder 重排。
- Model: `BAAI/bge-reranker-base`; revision: `2cfc18c9415c912f9d8155881c133215df768a70`。

| Strategy | Recall@3 | Recall@5 | MRR | Retrieval P95 ms |
|---|---:|---:|---:|---:|
| Hybrid | 0.8083 | 0.8556 | 0.8319 | 700.713 |
| Hybrid + BGE CrossEncoder | 0.6306 | 0.7667 | 0.6317 | 7511.436 |

## Difference Cases

- `v02_semantic_009` regressed: ΔMRR=-1.000, ΔRecall@5=-1.000。
- `v02_semantic_014` regressed: ΔMRR=-1.000, ΔRecall@5=-1.000。
- `v02_semantic_015` regressed: ΔMRR=-1.000, ΔRecall@5=-1.000。
- `v02_semantic_018` regressed: ΔMRR=-1.000, ΔRecall@5=-1.000。
- `v02_semantic_017` regressed: ΔMRR=-0.500, ΔRecall@5=-1.000。
- `v02_single_019` regressed: ΔMRR=-0.500, ΔRecall@5=-1.000。
- `v02_semantic_012` regressed: ΔMRR=-0.800, ΔRecall@5=+0.000。
- `v02_multi_009` regressed: ΔMRR=-0.750, ΔRecall@5=+0.000。
- `v02_semantic_003` regressed: ΔMRR=-0.750, ΔRecall@5=+0.000。
- `v02_semantic_010` improved: ΔMRR=+0.750, ΔRecall@5=+0.000。

## Decision

当前默认仍关闭 Reranker。通用模型把同主题近似段落排在 benchmark 指定的精确证据前，
同时 CPU P95 显著上升；本结果不修改 frozen test，也不包装成质量提升。
