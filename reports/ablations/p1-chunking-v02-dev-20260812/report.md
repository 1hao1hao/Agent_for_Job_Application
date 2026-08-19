# Chunking Dev 消融实验

- Dataset: `evalrag_v0.2` / dev (80 cases)
- Max chars: `220`; min ratio: `0.5`
- 只改变超长文本边界策略；Router、Retriever、模型和 top-k 保持不变。
- Relevant ids 按来源和 expected point/文本范围自动映射，因此这是 dev-only
  candidate 实验，不是新的 frozen ground truth。

| 策略 | Chunks | 短块率 | 疑似句中截断率 | Recall@3 | Recall@5 | MRR | P95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| paragraph_fixed | 666 | 52.85% | 62.72% | 47.08% | 53.19% | 63.94% | 402.17 |
| sentence_boundary | 666 | 10.81% | 20.32% | 45.00% | 51.67% | 57.08% | 693.31 |

## 结论

句子边界策略明显减少短 Chunk 和句中截断，但整体检索退化，因此不切换默认配置：

- Recall@5: 53.19% -> 51.67%.
- MRR: 63.94% -> 57.08%.
- Single-source Recall@5: 85.00% -> 90.00%.
- Multi-source Recall@5: 32.08% -> 22.50%.

可能的取舍是：完整句子有利于单证据问题，但更多竞争片段使多来源问题更难在 top-5 内覆盖完整证据。当前保留为可选策略；后续若继续实验，应联合比较 candidate_k/top_k 或 parent-child retrieval，不能只因结构统计改善就启用。

## 指标变化 Case

- `v02_single_009`: Recall@5 0.00 -> 1.00, MRR 0.00 -> 1.00.
- `v02_semantic_009`: Recall@5 0.50 -> 0.50, MRR 1.00 -> 0.50.
- `v02_semantic_010`: Recall@5 0.50 -> 0.50, MRR 0.33 -> 0.25.
- `v02_semantic_017`: Recall@5 0.50 -> 0.50, MRR 1.00 -> 0.25.
- `v02_semantic_020`: Recall@5 0.50 -> 0.50, MRR 1.00 -> 0.33.
- `v02_multi_001`: Recall@5 0.50 -> 0.00, MRR 0.33 -> 0.00.
- `v02_multi_002`: Recall@5 0.33 -> 0.00, MRR 0.25 -> 0.00.
- `v02_multi_003`: Recall@5 0.50 -> 0.50, MRR 0.50 -> 1.00.
- `v02_multi_004`: Recall@5 0.17 -> 0.17, MRR 0.33 -> 0.25.
- `v02_multi_005`: Recall@5 0.33 -> 0.00, MRR 1.00 -> 0.00.

## 口径边界

本实验测试 220 字符压力预算下的句子边界行为，不替换 frozen 420 字符数据，
也不能证明最终答案准确率发生变化。
