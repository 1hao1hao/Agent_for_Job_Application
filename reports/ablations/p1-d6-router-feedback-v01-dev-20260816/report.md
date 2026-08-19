# P1-D6 Router Feedback Dev Shadow Report

- Dataset: `evalrag_v0.2` / `dev` / 80 cases
- Feedback: 3 confirmed dev failures
- Model: `BAAI/bge-small-zh-v1.5@7999e1d3359715c523056ef9478215996d62a620`

| Version | Accuracy | Unknown Precision | Unknown Recall | P95 ms |
|---|---:|---:|---:|---:|
| rule-v0.2 | 0.9125 | 1.0000 | 1.0000 | 0.009 |
| semantic-v0.2 | 0.8750 | 0.8889 | 0.8000 | 394.259 |
| hybrid-v0.2 | 0.9625 | 1.0000 | 1.0000 | 389.733 |
| hybrid-feedback-v0.3 | 1.0000 | 1.0000 | 1.0000 | 200.071 |

## Drift Cases

- `v02_semantic_003` improved: interview_prepare -> analyze_jd
- `v02_semantic_006` improved: interview_prepare -> analyze_jd
- `v02_semantic_019` improved: analyze_jd -> project_explain

## Boundary

Feedback 来自已确认 dev failure，只离线更新 prototype；没有在线自学习。
候选版本通过同集 shadow gate 后才写入 active registry；frozen test 未参与调参。
