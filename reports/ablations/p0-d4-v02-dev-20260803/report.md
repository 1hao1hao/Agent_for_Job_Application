# Router V2 Dev Ablation

- Dataset: `evalrag_v0.2`
- Split: `dev`，80 cases；frozen test 未运行。
- 三组使用同一人工审核标签、Keyword Retriever、top-k=5。
- Accuracy 要求 intent 与 source set 同时严格一致。

| Router | Accuracy | Wrong | P50 ms | P95 ms |
|---|---:|---:|---:|---:|
| rule | 0.9125 | 7 | 0.008 | 0.011 |
| semantic | 0.8750 | 10 | 1611.488 | 2414.531 |
| hybrid | 0.9625 | 3 | 1508.828 | 2600.273 |

## Category Accuracy

- **rule**: multi_source=80.00%, semantic_paraphrase=85.00%, single_source=100.00%, unanswerable=100.00%
- **semantic**: multi_source=100.00%, semantic_paraphrase=70.00%, single_source=100.00%, unanswerable=80.00%
- **hybrid**: multi_source=100.00%, semantic_paraphrase=85.00%, single_source=100.00%, unanswerable=100.00%

## Strategy Differences

- `v02_multi_005`: expected=interview_prepare; rule/semantic/hybrid=analyze_jd/interview_prepare/interview_prepare。
- `v02_multi_010`: expected=interview_prepare; rule/semantic/hybrid=analyze_jd/interview_prepare/interview_prepare。
- `v02_multi_015`: expected=interview_prepare; rule/semantic/hybrid=analyze_jd/interview_prepare/interview_prepare。
- `v02_multi_020`: expected=interview_prepare; rule/semantic/hybrid=analyze_jd/interview_prepare/interview_prepare。
- `v02_semantic_001`: expected=analyze_jd; rule/semantic/hybrid=analyze_jd/unknown/analyze_jd。
- `v02_semantic_002`: expected=analyze_jd; rule/semantic/hybrid=analyze_jd/match_resume/analyze_jd。
- `v02_semantic_005`: expected=analyze_jd; rule/semantic/hybrid=analyze_jd/unknown/analyze_jd。
- `v02_semantic_007`: expected=match_resume; rule/semantic/hybrid=analyze_jd/match_resume/match_resume。
- `v02_semantic_013`: expected=interview_prepare; rule/semantic/hybrid=analyze_jd/interview_prepare/interview_prepare。
- `v02_unanswerable_002`: expected=unknown; rule/semantic/hybrid=unknown/application_plan/unknown。

## Conclusion

Hybrid 在本次 dev 上准确率最高，但 CPU P95 明显增加。
Semantic 单路低于修复后的 Rule，说明向量相似不应无条件替代精确规则。
Hybrid 仍有 3 个失败 case，全部保留；本报告不代表答案准确率。
